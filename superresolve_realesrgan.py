#!/usr/bin/env python3
"""Batch Real-ESRGAN super-resolution for Flash3D rendered views.

Examples
--------
Upscale all rendered RGB views by 2x on CPU::

    python superresolve_realesrgan.py \
      --input outputs/cpu_multiview_physical_ten_test/rgb \
      --output outputs/cpu_multiview_physical_ten_test/rgb_x2 \
      --outscale 2 --tile 128

The default official RealESRGAN_x4plus checkpoint is downloaded only when it
is missing.  Use ``--model-path`` to point to an already downloaded checkpoint
for offline inference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


# The x4plus checkpoint is hosted in the upstream project's v0.1.0 release.
# Later releases contain the compact video/general models but not this file.
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="One image or a directory of rendered RGB views")
    parser.add_argument("--output", type=Path, required=True, help="Output image directory (or file for a single input)")
    parser.add_argument("--model-path", type=Path, default=Path("weights/realesrgan/RealESRGAN_x4plus.pth"))
    parser.add_argument("--outscale", type=float, default=2.0, help="Output scaling, up to the native 4x model scale")
    parser.add_argument("--tile", type=int, default=128, help="Tile width/height; reduce to lower peak memory, 0 disables tiling")
    parser.add_argument("--tile-pad", type=int, default=10)
    parser.add_argument("--suffix", default="realesrgan", help="Suffix for single-file output names")
    parser.add_argument("--recursive", action="store_true", help="Process nested image folders and preserve their layout")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True, help="Download the official checkpoint if --model-path is absent")
    return parser.parse_args()


def collect_images(input_path: Path, recursive: bool) -> tuple[list[Path], Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path], input_path.parent
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images found in {input_path}")
    return images, input_path


def ensure_checkpoint(model_path: Path, should_download: bool) -> None:
    if model_path.is_file():
        return
    if not should_download:
        raise FileNotFoundError(f"Real-ESRGAN checkpoint is missing: {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(model_path.suffix + ".part")
    print(f"Downloading RealESRGAN_x4plus checkpoint to {model_path}")
    try:
        urllib.request.urlretrieve(MODEL_URL, temporary_path)
        temporary_path.replace(model_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def build_upsampler(model_path: Path, tile: int, tile_pad: int):
    try:
        import torch
        # BasicSR 1.4 imports the torchvision module under its pre-0.15 name.
        # Flash3D's supported newer torchvision exposes the same functions as
        # ``_functional_tensor`` instead, so register a compatibility alias.
        try:
            import torchvision.transforms.functional_tensor  # noqa: F401
        except ModuleNotFoundError:
            from torchvision.transforms import _functional_tensor
            sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError as error:
        raise RuntimeError(
            "Real-ESRGAN is not installed. Run: python -m pip install -r requirements-realesrgan.txt"
        ) from error
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=0,
        half=False,  # CPU does not reliably support the half-precision Real-ESRGAN path.
        device=torch.device("cpu"),
    )


def output_path_for(image_path: Path, input_root: Path, output: Path, is_single: bool, suffix: str) -> Path:
    if is_single and output.suffix:
        return output
    relative = image_path.name if is_single else image_path.relative_to(input_root)
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.with_name(f"{target.stem}_{suffix}.png") if is_single else target.with_suffix(".png")


def main() -> None:
    args = parse_args()
    if not 1.0 <= args.outscale <= 4.0:
        raise ValueError("--outscale must be between 1 and 4 for RealESRGAN_x4plus")
    if args.tile < 0:
        raise ValueError("--tile must be >= 0")
    images, input_root = collect_images(args.input, args.recursive)
    ensure_checkpoint(args.model_path, args.download)
    upsampler = build_upsampler(args.model_path, args.tile, args.tile_pad)
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required. Run: python -m pip install -r requirements-realesrgan.txt") from error

    args.output.mkdir(parents=True, exist_ok=True) if not (len(images) == 1 and args.output.suffix) else args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Unable to read image: {image_path}")
        if image.ndim != 3 or image.shape[2] not in {3, 4}:
            raise ValueError(f"Only RGB/RGBA images are supported: {image_path}")
        start = time.perf_counter()
        enhanced, _ = upsampler.enhance(image, outscale=args.outscale)
        output_path = output_path_for(image_path, input_root, args.output, len(images) == 1, args.suffix)
        if not cv2.imwrite(str(output_path), enhanced):
            raise RuntimeError(f"Unable to write image: {output_path}")
        elapsed = time.perf_counter() - start
        results.append({"input": str(image_path.resolve()), "output": str(output_path.resolve()), "input_size_hw": list(image.shape[:2]), "output_size_hw": list(enhanced.shape[:2]), "seconds": elapsed})
        print(f"[{index}/{len(images)}] {image_path.name} -> {output_path.name}: {elapsed:.2f} s")

    report_path = (args.output if args.output.is_dir() or not args.output.suffix else args.output.parent) / "realesrgan_report.json"
    report_path.write_text(json.dumps({"model": str(args.model_path.resolve()), "outscale": args.outscale, "tile": args.tile, "device": "cpu", "images": results}, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} super-resolved images and {report_path}")


if __name__ == "__main__":
    main()
