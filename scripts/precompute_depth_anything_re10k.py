#!/usr/bin/env python3
"""Precompute frozen Depth Anything V2 depth maps in Flash3D's RE10K format.

Output files are ``OUTPUT/{train,test}/SEQUENCE/TIMESTAMP.png`` and carry the
``min_value``/``max_value`` PNG metadata consumed by ``datasets/re10k.py``.
Use them with ``dataset.preload_depths=true dataset.depth_path=OUTPUT`` to
remove the frozen ViT-B encoder from the NPU training step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, PngImagePlugin
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.re10k import load_seq_data
from models.encoder.depth_anything_encoder import DEPTH_ANYTHING_CONFIGS


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-path", type=Path, required=True, help="RealEstate10K root containing train/test metadata")
    parser.add_argument("--output", type=Path, required=True, help="Depth cache root")
    parser.add_argument("--source-dir", type=Path, default=Path("third_party/Depth-Anything-V2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/depth-anything-v2/depth_anything_v2_metric_vkitti_vitb.pth"))
    parser.add_argument("--encoder", choices=tuple(DEPTH_ANYTHING_CONFIGS), default="vitb")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-depth", type=float, default=80.0, help="Metric checkpoint maximum depth; 80 for VKITTI, 20 for Hypersim")
    parser.add_argument("--device", choices=("auto", "npu", "cuda", "cpu"), default="auto")
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=("train", "test"))
    parser.add_argument("--limit", type=int, default=0, help="Maximum total images; 0 means all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "npu" or name == "auto":
        try:
            import torch_npu  # noqa: F401
            if torch.npu.is_available():
                return torch.device("npu:0")
        except ImportError:
            if name == "npu":
                raise
    if name == "cuda" or name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if name == "cuda":
            raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cpu")


def load_model(source_dir: Path, checkpoint: Path, encoder: str, max_depth: float, device: torch.device) -> torch.nn.Module:
    source_dir = source_dir / "metric_depth"
    module_file = source_dir / "depth_anything_v2" / "dpt.py"
    if not module_file.is_file():
        raise FileNotFoundError(f"Missing Depth Anything V2 source: {module_file}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing Depth Anything V2 checkpoint: {checkpoint}")
    sys.path.insert(0, str(source_dir))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    finally:
        sys.path.pop(0)
    model = DepthAnythingV2(encoder=encoder, **DEPTH_ANYTHING_CONFIGS[encoder], max_depth=max_depth)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state), strict=True)
    return model.to(device).eval()


def infer(model: torch.nn.Module, image: torch.Tensor, input_size: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    ratio = input_size / max(height, width)
    resized_height = max(14, round(height * ratio / 14) * 14)
    resized_width = max(14, round(width * ratio / 14) * 14)
    interpolation_kwargs = {"mode": "bilinear", "align_corners": False}
    if image.device.type != "npu":
        interpolation_kwargs["antialias"] = True
    image = F.interpolate(image, (resized_height, resized_width), **interpolation_kwargs)
    mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    depth = model((image - mean) / std).unsqueeze(1)
    return F.interpolate(depth, (height, width), **interpolation_kwargs).clamp_min(1e-4)


def save_depth(depth: torch.Tensor, path: Path) -> None:
    value = depth.squeeze().detach().float().cpu().numpy()
    low, high = float(value.min()), float(value.max())
    span = max(high - low, 1e-6)
    encoded = np.round((value - low) / span * (2**16 - 1)).clip(0, 2**16 - 1).astype(np.uint16)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("min_value", repr(low))
    metadata.add_text("max_value", repr(high))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded, mode="I;16").save(path, pnginfo=metadata)


def main() -> None:
    args = arguments()
    device = select_device(args.device)
    model = load_model(args.source_dir, args.checkpoint, args.encoder, args.max_depth, device)
    processed = 0
    for split in args.splits:
        sequences = load_seq_data(args.data_path, split)
        work = ((key, timestamp) for key, data in sequences.items() for timestamp in data["timestamps"])
        for sequence, timestamp in tqdm(work, desc=f"Depth Anything {split}"):
            image_path = args.data_path / split / sequence / f"{timestamp}.jpg"
            output_path = args.output / split / sequence / f"{timestamp}.png"
            if not image_path.is_file():
                continue
            if output_path.is_file() and not args.overwrite:
                continue
            with Image.open(image_path) as image:
                image = image.convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
                tensor = to_tensor(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                depth = infer(model, tensor, args.input_size)
            save_depth(depth, output_path)
            processed += 1
            if args.limit and processed >= args.limit:
                print(f"Stopped after {processed} images as requested.")
                return
    print(f"Saved {processed} Depth Anything V2 maps under {args.output.resolve()}")


if __name__ == "__main__":
    main()
