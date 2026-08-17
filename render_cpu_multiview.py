#!/usr/bin/env python3
"""CPU multi-camera rendering for a Flash3D Gaussian scene.

Unlike ``render_cpu_alpha.py``'s compact interpolation trajectory, this entry
point renders a named virtual camera rig.  The default ``cross5`` rig creates
five distinct nearby camera positions (centre, left, right, up and down).

For calibrated/custom rigs, pass a JSON file with this schema::

    {"cameras": [
      {"name": "front_left", "translation_xyz": [-0.02, 0, 0],
       "yaw_deg": -2.0, "pitch_deg": 0.0, "roll_deg": 0.0}
    ]}

Transforms are source-camera to target-camera transforms.  Keep baselines
small for a single-image reconstruction: unseen geometry cannot be created by
Gaussian splatting alone.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import psutil
import torch

from render_cpu_alpha import (
    crop_and_downsample,
    load_gaussians,
    normalise_depth,
    project_gaussians,
    rasterize,
    save_gray,
    save_rgb,
    unsharp_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gaussians", type=Path, required=True, help="gaussians.pt exported by benchmark_cpu.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rig", choices=("cross5", "arc5", "grid9"), default="cross5")
    parser.add_argument("--camera-file", type=Path, default=None, help="Custom rig JSON; overrides --rig")
    parser.add_argument("--baseline", type=float, default=0.02, help="Horizontal source-to-target translation in scene units")
    parser.add_argument("--vertical-baseline", type=float, default=None, help="Vertical translation; defaults to 0.7 * baseline")
    parser.add_argument("--yaw", type=float, default=2.0, help="Horizontal view rotation for preset rigs, degrees")
    parser.add_argument("--pitch", type=float, default=1.5, help="Vertical view rotation for preset rigs, degrees")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--fx", type=float, default=390.0)
    parser.add_argument("--fy", type=float, default=390.0)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--crop-margin", type=int, default=20)
    parser.add_argument("--sharpen", type=float, default=0.35)
    parser.add_argument("--keep-ratio", type=float, default=1.0)
    parser.add_argument("--crop-padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-opacity", type=float, default=0.005)
    parser.add_argument("--scale-modifier", type=float, default=0.55)
    parser.add_argument("--sigma-cutoff", type=float, default=2.5)
    parser.add_argument("--min-variance", type=float, default=0.2)
    parser.add_argument("--max-radius", type=float, default=96.0)
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=1000.0)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--background", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def preset_cameras(rig: str, baseline: float, vertical_baseline: float, yaw: float, pitch: float) -> list[dict]:
    """Return conservative, named local camera transforms for a single-image scene."""
    if rig == "cross5":
        return [
            {"name": "center", "translation_xyz": [0.0, 0.0, 0.0], "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
            {"name": "left", "translation_xyz": [-baseline, 0.0, 0.0], "yaw_deg": -yaw, "pitch_deg": 0.0, "roll_deg": 0.0},
            {"name": "right", "translation_xyz": [baseline, 0.0, 0.0], "yaw_deg": yaw, "pitch_deg": 0.0, "roll_deg": 0.0},
            {"name": "up", "translation_xyz": [0.0, -vertical_baseline, 0.0], "yaw_deg": 0.0, "pitch_deg": -pitch, "roll_deg": 0.0},
            {"name": "down", "translation_xyz": [0.0, vertical_baseline, 0.0], "yaw_deg": 0.0, "pitch_deg": pitch, "roll_deg": 0.0},
        ]
    if rig == "arc5":
        return [
            {"name": f"arc_{index:02d}", "translation_xyz": [offset, 0.0, 0.0], "yaw_deg": angle, "pitch_deg": 0.0, "roll_deg": 0.0}
            for index, (offset, angle) in enumerate(
                zip(torch.linspace(-baseline, baseline, 5).tolist(), torch.linspace(-yaw, yaw, 5).tolist())
            )
        ]
    if rig == "grid9":
        cameras = []
        for row, (dy, pitch_factor) in enumerate(((vertical_baseline, pitch), (0.0, 0.0), (-vertical_baseline, -pitch))):
            for column, (dx, yaw_factor) in enumerate(((-baseline, -yaw), (0.0, 0.0), (baseline, yaw))):
                cameras.append({"name": f"r{row}_c{column}", "translation_xyz": [dx, dy, 0.0], "yaw_deg": yaw_factor, "pitch_deg": pitch_factor, "roll_deg": 0.0})
        return cameras
    raise ValueError(f"Unsupported preset rig: {rig}")


def read_cameras(args: argparse.Namespace) -> list[dict]:
    if args.camera_file is None:
        vertical = args.vertical_baseline if args.vertical_baseline is not None else args.baseline * 0.7
        cameras = preset_cameras(args.rig, args.baseline, vertical, args.yaw, args.pitch)
    else:
        payload = json.loads(args.camera_file.read_text(encoding="utf-8"))
        cameras = payload["cameras"] if isinstance(payload, dict) else payload
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("Camera JSON must contain a non-empty 'cameras' list")
    names = set()
    for index, camera in enumerate(cameras):
        name = str(camera.get("name", f"camera_{index:03d}"))
        if name in names:
            raise ValueError(f"Duplicate camera name: {name}")
        names.add(name)
        translation = camera.get("translation_xyz", (0.0, 0.0, 0.0))
        if len(translation) != 3:
            raise ValueError(f"Camera {name}: translation_xyz must have 3 values")
        camera["name"] = name
        camera["translation_xyz"] = [float(value) for value in translation]
        for key in ("yaw_deg", "pitch_deg", "roll_deg"):
            camera[key] = float(camera.get(key, 0.0))
    return cameras


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> torch.Tensor:
    """Local yaw(Y), pitch(X), roll(Z) rotation matching render_cpu_alpha yaw."""
    yaw, pitch, roll = (math.radians(value) for value in (yaw_deg, pitch_deg, roll_deg))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(roll), math.sin(roll)
    yaw_matrix = torch.tensor(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)), dtype=torch.float32)
    pitch_matrix = torch.tensor(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)), dtype=torch.float32)
    roll_matrix = torch.tensor(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)), dtype=torch.float32)
    return roll_matrix @ pitch_matrix @ yaw_matrix


def camera_transform(camera: dict) -> torch.Tensor:
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = rotation_matrix(camera["yaw_deg"], camera["pitch_deg"], camera["roll_deg"])
    transform[:3, 3] = torch.tensor(camera["translation_xyz"], dtype=torch.float32)
    return transform


def main() -> None:
    args = parse_args()
    if not 0 < args.keep_ratio <= 1:
        raise ValueError("--keep-ratio must be in (0, 1]")
    if args.supersample < 1:
        raise ValueError("--supersample must be >= 1")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cameras = read_cameras(args)
    gaussians, _ = load_gaussians(args.gaussians, args.keep_ratio, args.min_opacity, args.crop_padding)
    args.output.mkdir(parents=True, exist_ok=True)
    rgb_dir, alpha_dir, depth_dir = (args.output / "rgb", args.output / "alpha", args.output / "depth")
    for directory in (rgb_dir, alpha_dir, depth_dir):
        directory.mkdir(parents=True, exist_ok=True)

    render_height, render_width = args.height * args.supersample, args.width * args.supersample
    fx, fy = args.fx * args.supersample, args.fy * args.supersample
    cx = ((args.cx if args.cx is not None else (args.width - 1) * 0.5) + 0.5) * args.supersample - 0.5
    cy = ((args.cy if args.cy is not None else (args.height - 1) * 0.5) + 0.5) * args.supersample - 0.5
    background = torch.tensor(args.background, dtype=torch.float32)
    report_cameras = []

    for index, camera in enumerate(cameras):
        start = time.perf_counter()
        transform = camera_transform(camera)
        projected = project_gaussians(gaussians, transform, fx, fy, cx, cy, args.near, args.far, args.min_variance, args.sigma_cutoff, args.max_radius, args.scale_modifier, render_width, render_height)
        rgb, alpha, depth = rasterize(projected, render_height, render_width, background, args.tile_size, args.chunk_size, args.sigma_cutoff)
        elapsed = time.perf_counter() - start
        rgb = unsharp_mask(crop_and_downsample(rgb, args.height, args.width, args.supersample, args.crop_margin), args.sharpen)
        alpha = crop_and_downsample(alpha, args.height, args.width, args.supersample, args.crop_margin).clamp(0, 1)
        depth = crop_and_downsample(depth, args.height, args.width, args.supersample, args.crop_margin)
        stem = f"{index:02d}_{camera['name']}"
        save_rgb(rgb, rgb_dir / f"{stem}.png")
        save_gray(alpha, alpha_dir / f"{stem}.png")
        save_gray(normalise_depth(depth, alpha), depth_dir / f"{stem}.png")
        report_cameras.append({**camera, "transform": transform.tolist(), "visible_gaussians": projected["z"].numel(), "mean_alpha": alpha.mean().item(), "seconds": elapsed, "rgb": str((rgb_dir / f"{stem}.png").resolve())})
        print(f"{stem}: {projected['z'].numel()} visible Gaussians, {elapsed:.3f} s, mean alpha={alpha.mean().item():.4f}")

    report = {
        "device": "cpu",
        "renderer": "PyTorch anisotropic 3D Gaussian + front-to-back alpha blending",
        "gaussians_input": str(args.gaussians.resolve()),
        "gaussians_after_filter": gaussians["xyz"].shape[0],
        "image_size_hw": [args.height, args.width],
        "intrinsics_at_output_resolution": {"fx": args.fx, "fy": args.fy, "cx": (args.width - 1) * 0.5 if args.cx is None else args.cx, "cy": (args.height - 1) * 0.5 if args.cy is None else args.cy},
        "preset_rig": None if args.camera_file else args.rig,
        "camera_file": None if args.camera_file is None else str(args.camera_file.resolve()),
        "cameras": report_cameras,
        "mean_seconds_per_view": sum(camera["seconds"] for camera in report_cameras) / len(report_cameras),
        "rss_final_mb": psutil.Process().memory_info().rss / 1024**2,
    }
    (args.output / "multiview_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "camera_rig.json").write_text(json.dumps({"cameras": cameras}, indent=2), encoding="utf-8")
    print(f"Saved {len(cameras)} camera views, rig manifest, and report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
