#!/usr/bin/env python3
"""CPU-only single-image Flash3D benchmark and reference multi-view renderer.

The project's native Gaussian rasterizer is CUDA-only.  This script therefore
disables that branch, obtains Flash3D's predicted Gaussians, and renders a
small camera trajectory with a deterministic CPU reference splatter.  It is
intended to profile reconstruction and validate outputs, not to replace a
production Vulkan/Metal renderer.

Example:
  python benchmark_cpu.py --image assets/example.jpg --checkpoint exp/re10k_v2/checkpoints \
      --output outputs/cpu_benchmark --views 9 --render-height 192 --render-width 288
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
import tracemalloc
from pathlib import Path

import hydra
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

from models.model import GaussianPredictor
from misc.visualise_3d import save_ply


def make_input(image_path: Path, height: int, width: int, pad: int, device: torch.device) -> dict:
    """Match Flash3D's RE10K loader: RGB [0,1], resize, then zero padding."""
    image = Image.open(image_path).convert("RGB")
    image = TVF.resize(image, [height, width], interpolation=InterpolationMode.LANCZOS)
    color = TVF.to_tensor(image).unsqueeze(0).to(device)
    color_aug = F.pad(color, (pad, pad, pad, pad)) if pad else color
    return {("color", 0, 0): color, ("color_aug", 0, 0): color_aug}


def percentile_clip(values: torch.Tensor, quantile: float) -> torch.Tensor:
    if quantile >= 1.0:
        return torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
    threshold = torch.quantile(values, 1.0 - quantile)
    return values >= threshold


def collect_gaussians(outputs: dict, gaussians_per_pixel: int, keep_ratio: float) -> dict:
    """Flatten Flash3D tensors and retain the most opaque Gaussians."""
    means = outputs["gauss_means"][:, :3, :]
    b_times_g, _, n = means.shape
    if b_times_g != gaussians_per_pixel:
        raise RuntimeError(f"Expected batch=1 and {gaussians_per_pixel} layers, got {b_times_g} tensors")
    xyz = means.permute(0, 2, 1).reshape(-1, 3).contiguous()
    opacity = outputs["gauss_opacity"].permute(0, 2, 3, 1).reshape(-1).contiguous()
    scaling = outputs["gauss_scaling"].permute(0, 2, 3, 1).reshape(-1, 3).contiguous()
    colors = outputs["gauss_features_dc"].permute(0, 2, 3, 1).reshape(-1, 3).contiguous()
    keep = percentile_clip(opacity, keep_ratio)
    # Flash3D stores DC SH coefficients. The native CUDA rasterizer converts
    # them to RGB; the reference renderer follows the degree-0 conversion.
    colors = torch.clamp(0.5 + 0.28209479177387814 * colors, 0.0, 1.0)
    return {"xyz": xyz[keep], "opacity": opacity[keep], "scaling": scaling[keep], "color": colors[keep]}


def flatten_raw_gaussians(outputs: dict, gaussians_per_pixel: int) -> dict[str, torch.Tensor]:
    """Return all native Flash3D Gaussian attributes without viewer transforms."""
    means = outputs["gauss_means"][:, :3, :]
    if means.shape[0] != gaussians_per_pixel:
        raise RuntimeError(
            f"Expected batch=1 and {gaussians_per_pixel} layers, got {means.shape[0]} tensors"
        )
    raw = {
        "xyz": means.permute(0, 2, 1).reshape(-1, 3),
        "opacity": outputs["gauss_opacity"].permute(0, 2, 3, 1).reshape(-1, 1),
        "scaling": outputs["gauss_scaling"].permute(0, 2, 3, 1).reshape(-1, 3),
        "rotation": outputs["gauss_rotation"].permute(0, 2, 3, 1).reshape(-1, 4),
        "features_dc": outputs["gauss_features_dc"].permute(0, 2, 3, 1).reshape(-1, 3),
    }
    if "gauss_features_rest" in outputs:
        raw["features_rest"] = (
            outputs["gauss_features_rest"].permute(0, 2, 3, 1).reshape(raw["xyz"].shape[0], -1)
        )
    return {key: value.detach().cpu().contiguous() for key, value in raw.items()}


def save_gaussian_data(
    outputs: dict,
    output_dir: Path,
    gaussians_per_pixel: int,
    metadata: dict,
) -> None:
    """Save exact native parameters as PT/NPZ plus a viewer-compatible PLY."""
    raw = flatten_raw_gaussians(outputs, gaussians_per_pixel)
    torch.save({"gaussians": raw, "metadata": metadata}, output_dir / "gaussians.pt")
    np.savez_compressed(
        output_dir / "gaussians.npz",
        **{key: value.numpy() for key, value in raw.items()},
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    # This PLY is transformed and normalized by the repository helper for viewing.
    # Use gaussians.pt or gaussians.npz for numeric evaluation and re-rendering.
    save_ply(outputs, output_dir / "gaussians_viewer.ply", gaussians_per_pixel)


def make_trajectory(views: int, translation: float, yaw_deg: float, device: torch.device) -> list[torch.Tensor]:
    """Return source-to-target transforms for a horizontal camera arc."""
    if views < 1:
        raise ValueError("--views must be >= 1")
    angles = torch.linspace(-yaw_deg, yaw_deg, views, device=device) * np.pi / 180.0
    positions = torch.linspace(-translation, translation, views, device=device)
    transforms = []
    for angle, x in zip(angles, positions):
        c, s = torch.cos(angle), torch.sin(angle)
        transform = torch.eye(4, dtype=torch.float32, device=device)
        transform[:3, :3] = torch.stack((torch.stack((c, torch.tensor(0., device=device), s)),
                                          torch.tensor([0., 1., 0.], device=device),
                                          torch.stack((-s, torch.tensor(0., device=device), c))))
        transform[0, 3] = x
        transforms.append(transform)
    return transforms


def render_reference(gaussians: dict, transform: torch.Tensor, height: int, width: int, focal_px: float,
                     background: float = 0.5) -> torch.Tensor:
    """Low-memory CPU reference renderer using depth-sorted bilinear splats.

    It deliberately uses a one-pixel footprint. This makes it appropriate for
    reproducible CPU profiling and viewpoint validation; opacity/scale-aware
    elliptical 3DGS rasterization belongs in the production native renderer.
    """
    xyz = gaussians["xyz"]
    xyz_h = torch.cat((xyz, torch.ones((xyz.shape[0], 1), dtype=xyz.dtype)), dim=1)
    points = (transform @ xyz_h.T).T[:, :3]
    z = points[:, 2]
    valid = z > 1e-4
    points, z = points[valid], z[valid]
    opacity = gaussians["opacity"][valid]
    color = gaussians["color"][valid]
    u = focal_px * points[:, 0] / z + (width - 1) * 0.5
    v = focal_px * points[:, 1] / z + (height - 1) * 0.5
    x0, y0 = torch.floor(u).long(), torch.floor(v).long()
    base_weight = opacity / torch.clamp(z, min=1e-3)
    accum_rgb = torch.zeros((3, height * width), dtype=torch.float32)
    accum_weight = torch.zeros(height * width, dtype=torch.float32)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xx, yy = x0 + dx, y0 + dy
        inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        if not torch.any(inside):
            continue
        bilinear = (1.0 - torch.abs(u[inside] - xx[inside].float())) * (1.0 - torch.abs(v[inside] - yy[inside].float()))
        weights = base_weight[inside] * bilinear
        flat = yy[inside] * width + xx[inside]
        accum_weight.scatter_add_(0, flat, weights)
        for channel in range(3):
            accum_rgb[channel].scatter_add_(0, flat, weights * color[inside, channel])
    rgb = accum_rgb / torch.clamp(accum_weight, min=1e-6)
    rgb = torch.where(accum_weight.unsqueeze(0) > 0, rgb, torch.full_like(rgb, background))
    return rgb.reshape(3, height, width).clamp(0, 1)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    TVF.to_pil_image(tensor.cpu()).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint file or directory containing model_*.pth")
    parser.add_argument("--output", type=Path, default=Path("outputs/cpu_benchmark"))
    parser.add_argument("--config", type=Path, default=Path("configs"))
    parser.add_argument("--experiment", default="layered_re10k", help="Config name in configs/experiment without .yaml")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--views", type=int, default=9)
    parser.add_argument("--render-height", type=int, default=192)
    parser.add_argument("--render-width", type=int, default=288)
    parser.add_argument("--translation", type=float, default=0.12, help="Left/right translation in source-depth units")
    parser.add_argument("--yaw", type=float, default=8.0, help="Maximum yaw in degrees")
    parser.add_argument("--keep-ratio", type=float, default=0.35, help="Keep the highest-opacity fraction (0, 1]")
    parser.add_argument("--focal-scale", type=float, default=1.0)
    return parser.parse_args()


def load_cfg(args: argparse.Namespace) -> DictConfig:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(args.config.resolve())):
        cfg = hydra.compose(config_name="config", overrides=[f"+experiment={args.experiment}"])
    cfg.data_loader.batch_size = 1
    cfg.dataset.height = int(cfg.dataset.height)
    cfg.dataset.width = int(cfg.dataset.width)
    cfg.model.gaussian_rendering = False  # native project renderer is CUDA-only
    cfg.model.randomise_bg_colour = False
    # The RE10K checkpoint restores this backbone completely.  Avoid a needless
    # 98 MB ImageNet download when running an offline benchmark.
    cfg.model.backbone.weights_init = "scratch"
    return cfg


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not (0 < args.keep_ratio <= 1):
        raise ValueError("--keep-ratio must be in (0, 1]")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    device = torch.device("cpu")
    cfg = load_cfg(args)
    args.output.mkdir(parents=True, exist_ok=True)

    model = GaussianPredictor(cfg).to(device)
    model.load_model(args.checkpoint, device="cpu")
    model.set_eval()
    inputs = make_input(args.image, cfg.dataset.height, cfg.dataset.width, cfg.dataset.pad_border_aug, device)
    process = psutil.Process(os.getpid())
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(inputs)
        tracemalloc.start()
        rss_before = process.memory_info().rss
        times = []
        outputs = None
        for _ in range(args.runs):
            start = time.perf_counter()
            outputs = model(inputs)
            times.append(time.perf_counter() - start)
        _, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = process.memory_info().rss
        gaussians = collect_gaussians(outputs, cfg.model.gaussians_per_pixel, args.keep_ratio)
        focal = args.focal_scale * 0.5 * (args.render_width + args.render_height)
        trajectory = make_trajectory(args.views, args.translation, args.yaw, device)
        point_cloud_metadata = {
            "format_version": 1,
            "coordinate_system": "Flash3D source-camera coordinates: x right, y down, z forward",
            "input_image": str(args.image.resolve()),
            "input_size_hw": [cfg.dataset.height, cfg.dataset.width],
            "padded_size_hw": [
                cfg.dataset.height + 2 * cfg.dataset.pad_border_aug,
                cfg.dataset.width + 2 * cfg.dataset.pad_border_aug,
            ],
            "gaussians_per_pixel": int(cfg.model.gaussians_per_pixel),
            "max_sh_degree": int(cfg.model.max_sh_degree),
            "checkpoint": str(args.checkpoint.resolve()),
        }
        save_gaussian_data(outputs, args.output, cfg.model.gaussians_per_pixel, point_cloud_metadata)
        torch.save(torch.stack(trajectory).cpu(), args.output / "render_poses.pt")
        with (args.output / "render_poses.json").open("w", encoding="utf-8") as file:
            json.dump([pose.cpu().tolist() for pose in trajectory], file, indent=2)
        render_times = []
        for index, transform in enumerate(trajectory):
            start = time.perf_counter()
            frame = render_reference(gaussians, transform, args.render_height, args.render_width, focal)
            render_times.append(time.perf_counter() - start)
            save_image(frame, args.output / "views" / f"view_{index:03d}.png")

    summary = {
        "device": "cpu",
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": args.threads,
        "input": str(args.image.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "input_size": [cfg.dataset.height, cfg.dataset.width],
        "render_size": [args.render_height, args.render_width],
        "inference_seconds": {"runs": times, "mean": float(np.mean(times)), "p50": float(np.median(times))},
        "render_seconds": {"views": render_times, "mean": float(np.mean(render_times)), "total": float(np.sum(render_times))},
        "gaussians": {"raw": int(outputs["gauss_means"].shape[0] * outputs["gauss_means"].shape[-1]), "kept": int(gaussians["xyz"].shape[0]), "keep_ratio": args.keep_ratio},
        "memory_mb": {"rss_delta": (rss_after - rss_before) / 1024**2, "rss_final": rss_after / 1024**2, "python_tracemalloc_peak": traced_peak / 1024**2},
        "notes": "Rendering uses a CPU reference bilinear splatter (not the CUDA 3DGS rasterizer); use it for reproducibility and functional validation only.",
    }
    with (args.output / "benchmark.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with (args.output / "timings.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["stage", "index", "seconds"])
        writer.writerows(("inference", i, value) for i, value in enumerate(times))
        writer.writerows(("render", i, value) for i, value in enumerate(render_times))
    print(json.dumps(summary, indent=2))
    print(f"Saved raw splats, rendered views and benchmark report to: {args.output.resolve()}")


if __name__ == "__main__":
    main()

