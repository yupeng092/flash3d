#!/usr/bin/env python3
"""Single-image Flash3D inference and multi-view rendering on Ascend NPU.

This entry point is for checkpoints pre-trained with
``+experiment=layered_re10k_npu``: Depth Anything V2 Base supplies frozen
depth, while Flash3D's ResNet/Gaussian heads are learned.  Novel views use the
portable PyTorch/NPU tile rasterizer, not the CUDA extension.

Example (after sourcing CANN):
  python scripts/infer_npu.py --image data/DS0304.jpg \
    --checkpoint outputs/layered_re10k_npu/checkpoints --output outputs/npu_demo
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.decoder.npu_differentiable_renderer import render_predicted_torch
from models.model import GaussianPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="One RGB source image")
    parser.add_argument("--checkpoint", type=Path, required=True, help="NPU pre-training checkpoint file or its checkpoints directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256, help="Flash3D input/render height; multiple of 32")
    parser.add_argument("--width", type=int, default=384, help="Flash3D input/render width; multiple of 32")
    parser.add_argument(
        "--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"), default=None,
        help="Calibrated source intrinsics in original-image pixels. Recommended for RE10K frames.",
    )
    parser.add_argument(
        "--default-focal-ratio", type=float, default=0.9,
        help="Fallback focal/max(image dimension) when --intrinsics is unavailable",
    )
    parser.add_argument("--rig", choices=("cross5", "arc5", "grid9"), default="cross5")
    parser.add_argument("--views", type=int, default=5, help="Only used with --rig arc5")
    parser.add_argument("--baseline", type=float, default=0.15, help="Horizontal physical camera offset in predicted depth units")
    parser.add_argument("--vertical-baseline", type=float, default=0.10)
    parser.add_argument("--scale-modifier", type=float, default=0.55)
    parser.add_argument("--max-gaussians", type=int, default=65536, help="Top-opacity visible Gaussian cap; 0 keeps all")
    parser.add_argument("--max-gaussians-per-tile", type=int, default=128, help="Per-tile cap; 0 keeps all")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--tile-span", type=int, default=5, help="Odd tile neighbourhood size")
    parser.add_argument("--max-radius", type=float, default=32.0)
    parser.add_argument("--sigma-cutoff", type=float, default=3.0)
    parser.add_argument("--min-variance", type=float, default=0.30)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-gaussians", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def compose_cfg(args: argparse.Namespace):
    if args.height % 32 or args.width % 32:
        raise ValueError("--height and --width must be multiples of 32")
    overrides = [
        "+experiment=layered_re10k_npu",
        "data_loader.batch_size=1",
        "data_loader.num_workers=0",
        f"dataset.height={args.height}",
        f"dataset.width={args.width}",
        "model.gaussian_rendering=false",
        "model.randomise_bg_colour=false",
        # Loading the trained checkpoint replaces these weights. Avoid any
        # ImageNet download in an offline NPU inference environment.
        "model.backbone.weights_init=scratch",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def make_intrinsics(
    original_width: int, original_height: int, height: int, width: int, pad: int,
    supplied: tuple[float, float, float, float] | None, focal_ratio: float, device: torch.device,
) -> tuple[torch.Tensor, str]:
    matrix = torch.eye(3, dtype=torch.float32, device=device)
    if supplied is None:
        focal = max(original_width, original_height) * focal_ratio
        fx, fy = focal, focal
        cx, cy = (original_width - 1) * 0.5, (original_height - 1) * 0.5
        source = "default_focal_ratio"
    else:
        fx, fy, cx, cy = supplied
        source = "provided"
    matrix[0, 0] = fx * width / original_width
    matrix[1, 1] = fy * height / original_height
    matrix[0, 2] = cx * width / original_width + pad
    matrix[1, 2] = cy * height / original_height + pad
    return matrix.unsqueeze(0), source


def load_input(args: argparse.Namespace, cfg, device: torch.device) -> tuple[dict, str]:
    with Image.open(args.image) as source:
        source = source.convert("RGB")
        original_width, original_height = source.size
        image = TVF.resize(source, [cfg.dataset.height, cfg.dataset.width], interpolation=InterpolationMode.LANCZOS)
    color = TVF.to_tensor(image).unsqueeze(0).to(device)
    pad = int(cfg.dataset.pad_border_aug)
    color_aug = F.pad(color, (pad, pad, pad, pad)) if pad else color
    intrinsics, intrinsics_mode = make_intrinsics(
        original_width, original_height, cfg.dataset.height, cfg.dataset.width, pad,
        None if args.intrinsics is None else tuple(args.intrinsics), args.default_focal_ratio, device,
    )
    return {
        ("color", 0, 0): color,
        ("color_aug", 0, 0): color_aug,
        ("K_src", 0): intrinsics,
    }, intrinsics_mode


def flatten_gaussians(outputs: dict, layers: int) -> dict[str, torch.Tensor]:
    """Flatten Flash3D's BCHW Gaussian tensors to native rasterizer fields."""
    means = outputs["gauss_means"][:, :3, :]
    if means.shape[0] != layers:
        raise RuntimeError(f"Expected one image and {layers} Gaussian layers, got {means.shape[0]}")
    result = {
        "xyz": means.permute(0, 2, 1).reshape(-1, 3).contiguous(),
        "opacity": outputs["gauss_opacity"].permute(0, 2, 3, 1).reshape(-1, 1).contiguous(),
        "scaling": outputs["gauss_scaling"].permute(0, 2, 3, 1).reshape(-1, 3).contiguous(),
        "rotation": outputs["gauss_rotation"].permute(0, 2, 3, 1).reshape(-1, 4).contiguous(),
        "features_dc": outputs["gauss_features_dc"].permute(0, 2, 3, 1).reshape(-1, 1, 3).contiguous(),
    }
    if "gauss_features_rest" in outputs:
        rest = outputs["gauss_features_rest"].permute(0, 2, 3, 1).reshape(result["xyz"].shape[0], -1)
        if rest.shape[1] % 3:
            raise RuntimeError("gauss_features_rest channel count must be divisible by 3")
        result["features_rest"] = rest.reshape(rest.shape[0], -1, 3).contiguous()
    return result


def preset_cameras(args: argparse.Namespace) -> list[dict]:
    if args.rig == "cross5":
        return [
            {"name": "center", "position_xyz": [0.0, 0.0, 0.0], "source_camera": True},
            {"name": "left", "position_xyz": [-args.baseline, 0.0, 0.0]},
            {"name": "right", "position_xyz": [args.baseline, 0.0, 0.0]},
            {"name": "up", "position_xyz": [0.0, -args.vertical_baseline, 0.0]},
            {"name": "down", "position_xyz": [0.0, args.vertical_baseline, 0.0]},
        ]
    if args.rig == "arc5":
        if args.views < 1:
            raise ValueError("--views must be >= 1")
        return [
            {"name": f"arc_{index:02d}", "position_xyz": [offset, 0.0, 0.0]}
            for index, offset in enumerate(torch.linspace(-args.baseline, args.baseline, args.views).tolist())
        ]
    cameras = []
    for row, dy in enumerate((args.vertical_baseline, 0.0, -args.vertical_baseline)):
        for column, dx in enumerate((-args.baseline, 0.0, args.baseline)):
            cameras.append({"name": f"r{row}_c{column}", "position_xyz": [dx, dy, 0.0]})
    return cameras


def world_to_camera(camera: dict, target: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    centre = torch.tensor(camera["position_xyz"], dtype=torch.float32, device=device)
    if camera.get("source_camera", False):
        return torch.eye(4, dtype=torch.float32, device=device), centre
    forward = F.normalize(target - centre, dim=0)
    reference_up = torch.tensor((0.0, -1.0, 0.0), dtype=torch.float32, device=device)
    if torch.abs(torch.dot(forward, reference_up)) > 0.98:
        reference_up = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32, device=device)
    right = F.normalize(torch.linalg.cross(forward, reference_up), dim=0)
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack((right, down, forward))
    transform = torch.eye(4, dtype=torch.float32, device=device)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ centre
    return transform, centre


def render_config(args: argparse.Namespace, cfg) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=SimpleNamespace(znear=cfg.dataset.znear, zfar=cfg.dataset.zfar),
        model=SimpleNamespace(
            npu_renderer_min_variance=args.min_variance,
            npu_renderer_sigma_cutoff=args.sigma_cutoff,
            npu_renderer_max_radius=args.max_radius,
            npu_renderer_max_gaussians=args.max_gaussians,
            npu_renderer_tile_size=args.tile_size,
            npu_renderer_tile_span=args.tile_span,
            npu_renderer_max_gaussians_per_tile=args.max_gaussians_per_tile,
        ),
    )


def save_rgb(rgb: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    TVF.to_pil_image(rgb.detach().float().cpu().clamp(0, 1)).save(path)


def save_grid(entries: list[tuple[str, Path]], output: Path) -> Path:
    images = [(name, Image.open(path).convert("RGB")) for name, path in entries]
    width, height = images[0][1].size
    sheet = Image.new("RGB", (len(images) * (width + 4) + 4, height + 76), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    title = "Flash3D NPU Torch"
    box = draw.textbbox((0, 0), title, font=font)
    draw.text(((sheet.width - (box[2] - box[0])) / 2, 6), title, fill="black", font=font)
    for index, (name, image) in enumerate(images):
        x = 4 + index * (width + 4)
        draw.text((x, 51), name, fill="black")
        sheet.paste(image, (x, 72))
    destination = output / "comparison_grid.png"
    sheet.save(destination)
    return destination


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if args.tile_span < 1 or args.tile_span % 2 == 0:
        raise ValueError("--tile-span must be a positive odd number")
    try:
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise SystemExit("torch_npu is required. Source CANN and install requirements_npu.txt first.") from error
    if not torch.npu.is_available():
        raise SystemExit("No Ascend NPU is available. Check CANN and ASCEND_RT_VISIBLE_DEVICES.")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    cfg = compose_cfg(args)
    model = GaussianPredictor(cfg)
    model.load_model(args.checkpoint, device="cpu")
    model = model.to(device).eval()
    depth_module = model.models["depth_anything_extended"].depth_model
    if any(parameter.requires_grad for parameter in depth_module.parameters()):
        raise RuntimeError("The NPU inference config must use model.depth.freeze=true")
    inputs, intrinsics_mode = load_input(args, cfg, device)
    args.output.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode(), torch.autocast(device_type="npu", dtype=torch.float16, enabled=args.amp):
        torch.npu.synchronize()
        started = time.perf_counter()
        outputs = model(inputs)
        torch.npu.synchronize()
        inference_seconds = time.perf_counter() - started
        pc = flatten_gaussians(outputs, cfg.model.gaussians_per_pixel)
        source_k = outputs[("K_src", 0)][0].float()
        render_k = source_k.clone()
        render_k[0, 2] -= cfg.dataset.pad_border_aug
        render_k[1, 2] -= cfg.dataset.pad_border_aug
        # Mean is universally supported by torch_npu and avoids bringing a
        # CPU-only quantile kernel into the NPU inference path.
        target = pc["xyz"].mean(dim=0)
        renderer_cfg = render_config(args, cfg)
        bg = torch.tensor(cfg.model.bg_colour, dtype=torch.float32, device=device)
        fov_x = 2.0 * math.atan(args.width / (2.0 * render_k[0, 0].item()))
        fov_y = 2.0 * math.atan(args.height / (2.0 * render_k[1, 1].item()))
        entries, report_views = [], []
        for index, camera in enumerate(preset_cameras(args)):
            transform, centre = world_to_camera(camera, target, device)
            torch.npu.synchronize()
            started = time.perf_counter()
            result = render_predicted_torch(
                renderer_cfg, pc, transform.T.contiguous(), transform.T, transform.T,
                centre, (fov_x, fov_y), (args.height, args.width), bg,
                cfg.model.max_sh_degree, args.scale_modifier,
                principal_point=(render_k[0, 2].item(), render_k[1, 2].item()),
            )
            torch.npu.synchronize()
            elapsed = time.perf_counter() - started
            stem = f"{index:02d}_{camera['name']}"
            path = args.output / "rgb" / f"{stem}.png"
            save_rgb(result["render"], path)
            entries.append((camera["name"], path))
            report_views.append({
                **camera,
                "world_to_camera": transform.detach().cpu().tolist(),
                "visible_gaussians": int(result["visibility_filter"].sum().item()),
                "seconds": elapsed,
                "rgb": str(path.resolve()),
            })

    if args.save_gaussians:
        torch.save({"gaussians": {key: value.detach().cpu() for key, value in pc.items()}, "metadata": {
            "format_version": 1,
            "coordinate_system": "Flash3D source-camera coordinates: x right, y down, z forward",
            "input_image": str(args.image.resolve()),
            "input_size_hw": [args.height, args.width],
            "padded_size_hw": [args.height + 2 * cfg.dataset.pad_border_aug, args.width + 2 * cfg.dataset.pad_border_aug],
            "gaussians_per_pixel": int(cfg.model.gaussians_per_pixel),
            "max_sh_degree": int(cfg.model.max_sh_degree),
            "camera": {"intrinsics": render_k.detach().cpu().tolist()},
            "intrinsics_mode": intrinsics_mode,
        }}, args.output / "gaussians.pt")
    grid = save_grid(entries, args.output)
    report = {
        "device": str(device), "torch": torch.__version__, "amp": args.amp,
        "checkpoint": str(args.checkpoint.resolve()), "image": str(args.image.resolve()),
        "input_size_hw": [args.height, args.width], "inference_seconds": inference_seconds,
        "gaussians": int(pc["xyz"].shape[0]), "intrinsics_mode": intrinsics_mode,
        "intrinsics_at_render_size": render_k.detach().cpu().tolist(),
        "renderer": "Flash3D portable PyTorch/NPU tile renderer",
        "renderer_limits": {"max_gaussians": args.max_gaussians, "max_gaussians_per_tile": args.max_gaussians_per_tile},
        "comparison_grid": str(grid.resolve()), "views": report_views,
    }
    (args.output / "inference_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
