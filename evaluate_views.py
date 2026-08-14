#!/usr/bin/env python3
"""Evaluate rendered novel views against ground-truth images.

Predicted and ground-truth images are paired by relative path. If no relative
matches exist, unique file stems are used as a fallback. Metrics are computed
in RGB [0,1]: PSNR, SSIM and LPIPS. An optional alpha mask excludes unobserved
pixels from every metric.

Example:
  python evaluate_views.py --pred outputs/cpu_benchmark/views \
      --gt data/eval_scene/gt --output outputs/cpu_benchmark/evaluation
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
)
from torchvision.transforms import functional as TVF


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pred", required=True, type=Path, help="Directory of rendered images")
    parser.add_argument("--gt", required=True, type=Path, help="Directory of ground-truth images")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--device", choices=("cpu", "cuda", "npu", "auto"), default="cpu")
    parser.add_argument("--metrics", nargs="+", choices=("psnr", "ssim", "lpips"), default=("psnr", "ssim", "lpips"))
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--crop-border", type=float, default=0.05, help="Fraction cropped from every image edge")
    parser.add_argument("--resize-gt", action="store_true", help="Resize GT to prediction size when shapes differ")
    parser.add_argument("--alpha", type=Path, default=None, help="Optional alpha/mask image directory paired to --pred")
    parser.add_argument("--mask-threshold", type=float, default=0.01)
    parser.add_argument("--require-all-pairs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def list_images(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def pair_images(pred_root: Path, gt_root: Path, recursive: bool, require_all: bool) -> list[tuple[Path, Path, str]]:
    pred_files = list_images(pred_root, recursive)
    gt_files = list_images(gt_root, recursive)
    if not pred_files:
        raise RuntimeError(f"No prediction images found in {pred_root}")
    if not gt_files:
        raise RuntimeError(f"No ground-truth images found in {gt_root}")

    gt_by_relative = {path.relative_to(gt_root).with_suffix("").as_posix(): path for path in gt_files}
    pairs = []
    for pred in pred_files:
        key = pred.relative_to(pred_root).with_suffix("").as_posix()
        if key in gt_by_relative:
            pairs.append((pred, gt_by_relative[key], key))
    if pairs:
        if require_all and len(pairs) != len(pred_files):
            paired = {pred for pred, _, _ in pairs}
            missing = [str(path.relative_to(pred_root)) for path in pred_files if path not in paired]
            raise RuntimeError(f"Missing GT pairs for {len(missing)} prediction image(s): {missing[:5]}")
        return pairs

    gt_by_stem: dict[str, list[Path]] = {}
    for path in gt_files:
        gt_by_stem.setdefault(path.stem, []).append(path)
    for pred in pred_files:
        matches = gt_by_stem.get(pred.stem, [])
        if len(matches) == 1:
            pairs.append((pred, matches[0], pred.stem))
    if not pairs:
        raise RuntimeError(
            "No image pairs found. Use matching relative paths or matching unique file names "
            "such as pred/view_000.png and gt/view_000.png."
        )
    if require_all and len(pairs) != len(pred_files):
        paired = {pred for pred, _, _ in pairs}
        missing = [str(path.relative_to(pred_root)) for path in pred_files if path not in paired]
        raise RuntimeError(f"Missing GT pairs for {len(missing)} prediction image(s): {missing[:5]}")
    return pairs


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return TVF.to_tensor(image.convert("RGB")).unsqueeze(0)


def crop_pair(pred: torch.Tensor, gt: torch.Tensor, fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    if fraction == 0:
        return pred, gt
    if not 0 <= fraction < 0.5:
        raise ValueError("--crop-border must be in [0, 0.5)")
    height, width = pred.shape[-2:]
    y0, y1 = math.ceil(fraction * height), math.floor((1 - fraction) * height)
    x0, x1 = math.ceil(fraction * width), math.floor((1 - fraction) * width)
    return pred[..., y0:y1, x0:x1], gt[..., y0:y1, x0:x1]


def resolve_mask(alpha_root: Path | None, pred_root: Path, pred_path: Path, key: str) -> Path | None:
    if alpha_root is None:
        return None
    candidate = alpha_root / pred_path.relative_to(pred_root)
    if candidate.is_file():
        return candidate
    matches = list(alpha_root.rglob(f"{Path(key).name}.*"))
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"No unique alpha mask for prediction {pred_path}")


def main() -> None:
    args = parse_args()
    if not args.pred.is_dir() or not args.gt.is_dir():
        raise FileNotFoundError("--pred and --gt must both be existing directories")
    if args.alpha is not None and not args.alpha.is_dir():
        raise FileNotFoundError("--alpha must be an existing directory")
    if not 0 <= args.mask_threshold <= 1:
        raise ValueError("--mask-threshold must be in [0, 1]")
    if args.device == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError as error:
            raise RuntimeError("NPU evaluation requires torch_npu.") from error
    if args.device == "auto":
        if hasattr(torch, "npu") and torch.npu.is_available():
            device_name = "npu"
        elif torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"
    else:
        device_name = args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device_name == "npu" and (not hasattr(torch, "npu") or not torch.npu.is_available()):
        raise RuntimeError("NPU was requested but is not available")
    device = torch.device(device_name)

    pairs = pair_images(args.pred, args.gt, args.recursive, args.require_all_pairs)
    requested = tuple(dict.fromkeys(args.metrics))
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval() if "ssim" in requested else None
    lpips = LearnedPerceptualImagePatchSimilarity(net_type=args.lpips_net).to(device).eval() if "lpips" in requested else None
    rows = []
    with torch.inference_mode():
        for pred_path, gt_path, key in pairs:
            pred, gt = load_rgb(pred_path), load_rgb(gt_path)
            mask_path = resolve_mask(args.alpha, args.pred, pred_path, key)
            mask = load_rgb(mask_path).mean(dim=1, keepdim=True) if mask_path else None
            if pred.shape[-2:] != gt.shape[-2:]:
                if not args.resize_gt:
                    raise RuntimeError(
                        f"Image size mismatch for {key}: pred={tuple(pred.shape[-2:])}, "
                        f"gt={tuple(gt.shape[-2:])}. Pass --resize-gt to allow resizing."
                    )
                gt = F.interpolate(gt, size=pred.shape[-2:], mode="bilinear", align_corners=False, antialias=True)
            if mask is not None and mask.shape[-2:] != pred.shape[-2:]:
                mask = F.interpolate(mask, size=pred.shape[-2:], mode="bilinear", align_corners=False, antialias=True)
            pred, gt = crop_pair(pred.to(device), gt.to(device), args.crop_border)
            if mask is not None:
                mask, _ = crop_pair(mask.to(device), gt, args.crop_border)
                mask = mask >= args.mask_threshold
                coverage = float(mask.float().mean().cpu())
                if not mask.any():
                    raise RuntimeError(f"Alpha mask has no valid pixels for {key}")
                mse = ((pred - gt).square() * mask).sum() / (mask.sum() * pred.shape[1])
                pred_metric = pred * mask + 0.5 * (~mask)
                gt_metric = gt * mask + 0.5 * (~mask)
            else:
                coverage = 1.0
                mse = F.mse_loss(pred, gt)
                pred_metric, gt_metric = pred, gt
            psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-12))
            row = {
                "name": key,
                "pred": str(pred_path.resolve()),
                "gt": str(gt_path.resolve()),
                "mask": str(mask_path.resolve()) if mask_path else None,
                "valid_fraction": coverage,
            }
            if "psnr" in requested:
                row["psnr"] = float(psnr.cpu())
            if ssim is not None:
                row["ssim"] = float(ssim(pred_metric, gt_metric).cpu())
            if lpips is not None:
                row["lpips"] = float(lpips(pred_metric * 2 - 1, gt_metric * 2 - 1).cpu())
            rows.append(row)
            if ssim is not None:
                ssim.reset()
            if lpips is not None:
                lpips.reset()

    metric_names = requested
    summary = {
        "count": len(rows),
        "device": str(device),
        "lpips_net": args.lpips_net,
        "crop_border": args.crop_border,
        "alpha_directory": str(args.alpha.resolve()) if args.alpha else None,
        "mask_threshold": args.mask_threshold if args.alpha else None,
        "mean": {name: float(np.mean([row[name] for row in rows])) for name in metric_names},
        "median": {name: float(np.median([row[name] for row in rows])) for name in metric_names},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"summary": summary, "frames": rows}, file, indent=2, ensure_ascii=False)
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("name", "pred", "gt", "mask", "valid_fraction", *metric_names))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved evaluation reports to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
