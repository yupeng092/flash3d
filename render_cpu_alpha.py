#!/usr/bin/env python3
"""Pure PyTorch/CPU 3D Gaussian rasterizer with front-to-back alpha blending.

This consumes ``gaussians.pt`` and ``render_poses.pt`` produced by
``benchmark_cpu.py``.  It projects every anisotropic 3D Gaussian to a 2D
ellipse, evaluates that ellipse in screen-space tiles, and composites samples
from near to far using

    C = sum_i(T_i * alpha_i * color_i),  T_i = product_{j<i}(1-alpha_j).

It is an inference/reference implementation.  It does not require CUDA,
Triton, xFormers, or diff-gaussian-rasterization.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    """Encode a linear-RGB image for ordinary sRGB PNG display."""
    image = image.clamp(0.0, 1.0)
    return torch.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * image.pow(1.0 / 2.4) - 0.055,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gaussians", type=Path, required=True)
    parser.add_argument(
        "--poses", type=Path, default=None,
        help="Optional PT camera trajectory. If omitted, generate a small-baseline trajectory",
    )
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--translation", type=float, default=0.03)
    parser.add_argument("--yaw", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument(
        "--sharpen", type=float, default=0.0,
        help="Unsharp-mask amount applied after rendering (0 disables it)",
    )
    parser.add_argument(
        "--linear-to-srgb", action=argparse.BooleanOptionalAction, default=False,
        help="Encode rendered linear RGB to sRGB before PNG output; useful for UniSHARP exports",
    )
    parser.add_argument(
        "--crop-margin", type=int, default=0,
        help="Uniform safety crop in final-output pixels, resized back to the requested size",
    )
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--keep-ratio", type=float, default=1.0)
    parser.add_argument(
        "--crop-padding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Discard Gaussians predicted from Flash3D's zero-padded image border",
    )
    parser.add_argument("--min-opacity", type=float, default=0.005)
    parser.add_argument("--scale-modifier", type=float, default=1.0)
    parser.add_argument("--sigma-cutoff", type=float, default=3.0)
    parser.add_argument("--min-variance", type=float, default=0.30)
    parser.add_argument("--max-radius", type=float, default=96.0)
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=1000.0)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--background", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--pose-index", type=int, nargs="*", default=None)
    return parser.parse_args()


def quaternion_to_rotation(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert Flash3D/3DGS scalar-first quaternions (w, x, y, z) to matrices."""
    quaternion = torch.nn.functional.normalize(quaternion.float(), dim=-1)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def make_small_baseline_trajectory(views: int, translation: float, yaw_deg: float) -> torch.Tensor:
    """Create source-to-target transforms over a conservative horizontal arc."""
    if views < 1:
        raise ValueError("--views must be >= 1")
    angles = torch.linspace(-yaw_deg, yaw_deg, views) * math.pi / 180.0
    positions = torch.linspace(-translation, translation, views)
    transforms = []
    for angle, position in zip(angles, positions):
        cosine, sine = torch.cos(angle), torch.sin(angle)
        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, :3] = torch.tensor(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=torch.float32,
        )
        transform[0, 3] = position
        transforms.append(transform)
    return torch.stack(transforms)


def _read_binary_ply_vertices(path: Path) -> dict[str, torch.Tensor]:
    """Read the vertex element of a binary-little-endian 3DGS PLY without extra dependencies."""
    ply_types = {
        "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        "uchar": "u1", "uint8": "u1", "int": "<i4", "int32": "<i4",
        "uint": "<u4", "uint32": "<u4",
    }
    with path.open("rb") as file:
        header_lines = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError(f"Only binary_little_endian 3DGS PLY is supported: {path}")
        vertex_count = None
        fields: list[tuple[str, str]] = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
            elif parts and parts[0] == "element" and in_vertex:
                in_vertex = False
            elif in_vertex and parts and parts[0] == "property":
                if len(parts) != 3 or parts[1] not in ply_types:
                    raise ValueError(f"Unsupported PLY vertex property: {line}")
                fields.append((parts[2], ply_types[parts[1]]))
        if vertex_count is None or not fields:
            raise ValueError(f"PLY has no vertex element: {path}")
        vertices = np.fromfile(file, dtype=np.dtype(fields), count=vertex_count)
    return {name: torch.from_numpy(vertices[name].copy()).float() for name, _ in fields}


def _load_standard_3dgs_ply(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    fields = _read_binary_ply_vertices(path)
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    missing = required.difference(fields)
    if missing:
        raise KeyError(f"PLY lacks required 3DGS fields: {sorted(missing)}")
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(fields):
        features_dc = torch.stack((fields["f_dc_0"], fields["f_dc_1"], fields["f_dc_2"]), dim=-1)
    elif {"red", "green", "blue"}.issubset(fields):
        rgb = torch.stack((fields["red"], fields["green"], fields["blue"]), dim=-1) / 255.0
        features_dc = (rgb - 0.5) / SH_C0
    else:
        raise KeyError("PLY requires f_dc_0..2 or red/green/blue colour fields")
    opacity = fields["opacity"]
    # Official 3DGS PLY stores opacity and scaling before activation.
    if opacity.min() < 0 or opacity.max() > 1:
        opacity = opacity.sigmoid()
    scaling = torch.stack((fields["scale_0"], fields["scale_1"], fields["scale_2"]), dim=-1)
    if scaling.min() <= 0:
        scaling = scaling.exp()
    raw = {
        "xyz": torch.stack((fields["x"], fields["y"], fields["z"]), dim=-1),
        "opacity": opacity,
        "scaling": scaling,
        "rotation": torch.stack((fields["rot_0"], fields["rot_1"], fields["rot_2"], fields["rot_3"]), dim=-1),
        "features_dc": features_dc,
    }
    rest_names = sorted(
        (name for name in fields if name.startswith("f_rest_")),
        key=lambda name: int(name.removeprefix("f_rest_")),
    )
    if rest_names:
        if len(rest_names) % 3:
            raise ValueError(f"Invalid f_rest field count ({len(rest_names)}) in {path}")
        # 3DGS serialises coefficients in coefficient-major, RGB-minor order.
        raw["features_rest"] = torch.stack([fields[name] for name in rest_names], dim=-1).reshape(-1, len(rest_names) // 3, 3)
    return raw, {"source_format": "standard_3dgs_ply"}


def _load_unisharp_gaussians(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "unisharp_gaussians":
        raise ValueError(f"Not a UniSHARP Gaussian export: {path}")
    gaussians = payload["gaussians"]
    colours = gaussians["colors"].reshape(-1, 3).float()
    raw = {
        "xyz": gaussians["mean_vectors"].reshape(-1, 3).float(),
        "opacity": gaussians["opacities"].reshape(-1).float(),
        "scaling": gaussians["singular_values"].reshape(-1, 3).float(),
        "rotation": gaussians["quaternions"].reshape(-1, 4).float(),
        "features_dc": (colours - 0.5) / SH_C0,
    }
    image = payload.get("image", {})
    return raw, {
        "source_format": "unisharp_gaussians",
        "input_size_hw": (image.get("height", 0), image.get("width", 0)),
        "camera": payload.get("camera", {}),
    }


def _load_sharp_gaussians(path: Path, payload: dict) -> tuple[dict[str, torch.Tensor], dict]:
    """Load SHARP's compact PT export (one Gaussian tensor per image pixel)."""
    colours = payload["colors"].reshape(-1, 3).float()
    image_shape = tuple(int(value) for value in payload.get("image_shape", (0, 0)))
    if len(image_shape) != 2:
        image_shape = (0, 0)
    height, width = image_shape
    focal = float(payload.get("f_px", 0.0))
    raw = {
        "xyz": payload["mean_vectors"].reshape(-1, 3).float(),
        "opacity": payload["opacities"].reshape(-1).float(),
        "scaling": payload["singular_values"].reshape(-1, 3).float(),
        "rotation": payload["quaternions"].reshape(-1, 4).float(),
        "features_dc": (colours - 0.5) / SH_C0,
    }
    return raw, {
        "source_format": "sharp_gaussians",
        "input_size_hw": image_shape,
        "camera": {
            "intrinsics": [focal, focal, (width - 1) * 0.5, (height - 1) * 0.5]
        } if focal > 0 and height > 0 and width > 0 else {},
        "source": str(payload.get("source", "")),
    }


def load_gaussians(
    path: Path, keep_ratio: float, min_opacity: float, crop_padding: bool
) -> tuple[dict[str, torch.Tensor], dict]:
    if path.suffix.lower() == ".ply":
        raw, metadata = _load_standard_3dgs_ply(path)
    else:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and payload.get("format") == "unisharp_gaussians":
            raw, metadata = _load_unisharp_gaussians(path)
        elif isinstance(payload, dict) and {"mean_vectors", "singular_values", "quaternions", "colors", "opacities"}.issubset(payload):
            raw, metadata = _load_sharp_gaussians(path, payload)
        else:
            raw = payload["gaussians"] if "gaussians" in payload else payload
            metadata = payload.get("metadata", {})
    # Render dumps produced by render_cpu_multiview.py keep a ready-to-use
    # colour rather than the original DC coefficients.  Accepting them makes
    # those dumps useful as CPU-renderable Flash3D test inputs as well.
    if "features_dc" not in raw and "color" in raw:
        raw = dict(raw)
        raw["features_dc"] = (raw["color"].detach().float() - 0.5) / SH_C0
    required = {"xyz", "opacity", "scaling", "rotation", "features_dc"}
    missing = required.difference(raw)
    if missing:
        raise KeyError(f"Missing Gaussian fields: {sorted(missing)}")

    gaussians = {key: raw[key].detach().float().cpu() for key in required}
    if "features_rest" in raw:
        features_rest = raw["features_rest"].detach().float().cpu()
        # Flash3D writes [N, 3 * number_of_SH_coefficients], while standard
        # 3DGS PLY loading already yields [N, number_of_coefficients, 3].
        # Canonicalise both forms before view-dependent SH evaluation.
        if features_rest.ndim == 2:
            if features_rest.shape[1] % 3:
                raise ValueError("features_rest channel count must be divisible by RGB channels")
            features_rest = features_rest.reshape(features_rest.shape[0], -1, 3)
        elif features_rest.ndim != 3 or features_rest.shape[-1] != 3:
            raise ValueError("features_rest must have shape [N, coefficients, 3]")
        gaussians["features_rest"] = features_rest
    gaussians["opacity"] = gaussians["opacity"].reshape(-1).clamp(0.0, 1.0)
    # Keep the DC coefficients as well as the convenient RGB view.  The
    # PyTorch renderer uses ``color``; Flash3D's native CUDA rasterizer needs
    # ``features_dc`` (and optionally ``features_rest``) to evaluate SH in
    # exactly the same way as the original model code.
    gaussians["color"] = (0.5 + SH_C0 * gaussians["features_dc"]).clamp(0.0, 1.0)

    valid = (
        torch.isfinite(gaussians["xyz"]).all(dim=-1)
        & torch.isfinite(gaussians["scaling"]).all(dim=-1)
        & (gaussians["opacity"] >= min_opacity)
    )
    if crop_padding and metadata:
        input_h, input_w = metadata.get("input_size_hw", (0, 0))
        padded_h, padded_w = metadata.get("padded_size_hw", (input_h, input_w))
        layers = int(metadata.get("gaussians_per_pixel", 1))
        expected = layers * padded_h * padded_w
        if input_h and input_w and padded_h and padded_w and len(valid) == expected:
            pad_y = (padded_h - input_h) // 2
            pad_x = (padded_w - input_w) // 2
            spatial = torch.zeros((padded_h, padded_w), dtype=torch.bool)
            spatial[pad_y : pad_y + input_h, pad_x : pad_x + input_w] = True
            valid &= spatial.reshape(-1).repeat(layers)
    if keep_ratio < 1.0:
        keep_count = max(1, math.ceil(valid.sum().item() * keep_ratio))
        valid_indices = torch.where(valid)[0]
        chosen = torch.topk(gaussians["opacity"][valid_indices], keep_count, sorted=False).indices
        mask = torch.zeros_like(valid)
        mask[valid_indices[chosen]] = True
        valid = mask
    return ({key: value[valid].contiguous() for key, value in gaussians.items()}, metadata)


def evaluate_view_dependent_color(
    gaussians: dict[str, torch.Tensor], camera_center: torch.Tensor
) -> torch.Tensor:
    """Evaluate 3DGS real SH colours for one camera centre.

    Flash3D and UniSHARP exports provide only RGB/DC colour and therefore
    correctly fall back to a fixed colour.  Standard 3DGS PLY exports often
    include ``f_rest``; evaluating it here prevents a view-dependent material
    from being rendered as a fixed, incorrect texture during camera motion.
    """
    coefficients = gaussians.get("features_rest")
    if coefficients is None or coefficients.numel() == 0:
        return gaussians["color"]
    directions = torch.nn.functional.normalize(
        gaussians["xyz"] - camera_center[None], dim=-1, eps=1e-8)
    x, y, z = directions.unbind(dim=-1)
    # Start from the un-clipped DC coefficient.  Clipping is part of the
    # final 3DGS/gsplat RGB activation; doing it earlier changes higher-order
    # SH colours for saturated Gaussians.
    result = SH_C0 * gaussians["features_dc"]
    degree = min(3, int(math.sqrt(coefficients.shape[1] + 1)) - 1)
    if degree >= 1:
        result = result + (
            -SH_C1 * y[:, None] * coefficients[:, 0]
            + SH_C1 * z[:, None] * coefficients[:, 1]
            - SH_C1 * x[:, None] * coefficients[:, 2]
        )
    if degree >= 2:
        xx, yy, zz = x.square(), y.square(), z.square()
        xy, yz, xz = x * y, y * z, x * z
        result = result + (
            SH_C2[0] * xy[:, None] * coefficients[:, 3]
            + SH_C2[1] * yz[:, None] * coefficients[:, 4]
            + SH_C2[2] * (2.0 * zz - xx - yy)[:, None] * coefficients[:, 5]
            + SH_C2[3] * xz[:, None] * coefficients[:, 6]
            + SH_C2[4] * (xx - yy)[:, None] * coefficients[:, 7]
        )
    if degree >= 3:
        result = result + (
            SH_C3[0] * (y * (3.0 * xx - yy))[:, None] * coefficients[:, 8]
            + SH_C3[1] * (x * y * z)[:, None] * coefficients[:, 9]
            + SH_C3[2] * (y * (4.0 * zz - xx - yy))[:, None] * coefficients[:, 10]
            + SH_C3[3] * (z * (2.0 * zz - 3.0 * xx - 3.0 * yy))[:, None] * coefficients[:, 11]
            + SH_C3[4] * (x * (4.0 * zz - xx - yy))[:, None] * coefficients[:, 12]
            + SH_C3[5] * (z * (xx - yy))[:, None] * coefficients[:, 13]
            + SH_C3[6] * (x * (xx - 3.0 * yy))[:, None] * coefficients[:, 14]
        )
    return (result + 0.5).clamp(0.0, 1.0)


def project_gaussians(
    gaussians: dict[str, torch.Tensor],
    world_to_camera: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    near: float,
    far: float,
    min_variance: float,
    sigma_cutoff: float,
    max_radius: float,
    scale_modifier: float,
    width: int,
    height: int,
    camera_center: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    xyz = gaussians["xyz"]
    camera_rotation = world_to_camera[:3, :3].float()
    camera_translation = world_to_camera[:3, 3].float()
    camera_xyz = xyz @ camera_rotation.T + camera_translation
    x, y, z = camera_xyz.unbind(dim=-1)

    rotation = quaternion_to_rotation(gaussians["rotation"])
    scales = gaussians["scaling"].clamp_min(1e-7) * scale_modifier
    basis = rotation * scales[:, None, :]
    covariance_world = basis @ basis.transpose(1, 2)
    covariance_camera = camera_rotation[None] @ covariance_world @ camera_rotation.T[None]

    inverse_z = z.clamp_min(near).reciprocal()
    jacobian = torch.zeros((xyz.shape[0], 2, 3), dtype=torch.float32)
    jacobian[:, 0, 0] = fx * inverse_z
    jacobian[:, 0, 2] = -fx * x * inverse_z.square()
    jacobian[:, 1, 1] = fy * inverse_z
    jacobian[:, 1, 2] = -fy * y * inverse_z.square()
    covariance_2d = jacobian @ covariance_camera @ jacobian.transpose(1, 2)
    covariance_2d[:, 0, 0] += min_variance
    covariance_2d[:, 1, 1] += min_variance

    a = covariance_2d[:, 0, 0]
    b = covariance_2d[:, 0, 1]
    c = covariance_2d[:, 1, 1]
    determinant = (a * c - b.square()).clamp_min(1e-10)
    inverse = torch.stack((c / determinant, -b / determinant, a / determinant), dim=-1)
    largest_eigenvalue = 0.5 * (
        a + c + torch.sqrt(((a - c).square() + 4 * b.square()).clamp_min(0.0))
    )
    radius = (sigma_cutoff * torch.sqrt(largest_eigenvalue.clamp_min(0.0))).clamp(max=max_radius)
    u = fx * x * inverse_z + cx
    v = fy * y * inverse_z + cy

    visible = (
        (z > near)
        & (z < far)
        & torch.isfinite(inverse).all(dim=-1)
        & (u + radius >= 0)
        & (u - radius < width)
        & (v + radius >= 0)
        & (v - radius < height)
    )
    order = torch.argsort(z[visible])
    colours = gaussians["color"] if camera_center is None else evaluate_view_dependent_color(gaussians, camera_center)
    return {
        "u": u[visible][order],
        "v": v[visible][order],
        "z": z[visible][order],
        "radius": radius[visible][order],
        "inverse": inverse[visible][order],
        "opacity": gaussians["opacity"][visible][order],
        "color": colours[visible][order],
    }


@torch.inference_mode()
def rasterize(
    projected: dict[str, torch.Tensor],
    height: int,
    width: int,
    background: torch.Tensor,
    tile_size: int,
    chunk_size: int,
    sigma_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = torch.empty((height, width, 3), dtype=torch.float32)
    alpha_image = torch.empty((height, width), dtype=torch.float32)
    depth_image = torch.empty((height, width), dtype=torch.float32)
    cutoff_squared = sigma_cutoff * sigma_cutoff

    u, v, radius = projected["u"], projected["v"], projected["radius"]
    for y0 in range(0, height, tile_size):
        y1 = min(y0 + tile_size, height)
        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            overlaps = (
                (u + radius >= x0)
                & (u - radius < x1)
                & (v + radius >= y0)
                & (v - radius < y1)
            )
            indices = torch.where(overlaps)[0]
            yy, xx = torch.meshgrid(
                torch.arange(y0, y1, dtype=torch.float32),
                torch.arange(x0, x1, dtype=torch.float32),
                indexing="ij",
            )
            pixels_x, pixels_y = xx.reshape(-1), yy.reshape(-1)
            transmittance = torch.ones(pixels_x.numel(), dtype=torch.float32)
            tile_rgb = torch.zeros((pixels_x.numel(), 3), dtype=torch.float32)
            tile_depth = torch.zeros(pixels_x.numel(), dtype=torch.float32)

            for start in range(0, indices.numel(), chunk_size):
                selected = indices[start : start + chunk_size]
                dx = pixels_x[None] - projected["u"][selected, None]
                dy = pixels_y[None] - projected["v"][selected, None]
                inverse = projected["inverse"][selected]
                mahalanobis = (
                    inverse[:, 0, None] * dx.square()
                    + 2 * inverse[:, 1, None] * dx * dy
                    + inverse[:, 2, None] * dy.square()
                )
                alpha = projected["opacity"][selected, None] * torch.exp(-0.5 * mahalanobis)
                alpha.masked_fill_(mahalanobis > cutoff_squared, 0.0)
                alpha.clamp_(0.0, 0.999)

                one_minus_alpha = 1.0 - alpha
                exclusive = torch.cumprod(
                    torch.cat((torch.ones_like(one_minus_alpha[:1]), one_minus_alpha[:-1]), dim=0),
                    dim=0,
                )
                weights = alpha * exclusive * transmittance[None]
                tile_rgb += weights.T @ projected["color"][selected]
                tile_depth += weights.T @ projected["z"][selected]
                transmittance *= one_minus_alpha.prod(dim=0)

            tile_alpha = 1.0 - transmittance
            tile_rgb += transmittance[:, None] * background[None]
            tile_depth = torch.where(
                tile_alpha > 1e-6, tile_depth / tile_alpha.clamp_min(1e-6), torch.zeros_like(tile_depth)
            )
            rgb[y0:y1, x0:x1] = tile_rgb.reshape(y1 - y0, x1 - x0, 3)
            alpha_image[y0:y1, x0:x1] = tile_alpha.reshape(y1 - y0, x1 - x0)
            depth_image[y0:y1, x0:x1] = tile_depth.reshape(y1 - y0, x1 - x0)
    return rgb.clamp(0.0, 1.0), alpha_image.clamp(0.0, 1.0), depth_image


def save_rgb(image: torch.Tensor, path: Path) -> None:
    array = (image * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).numpy()
    Image.fromarray(array, mode="RGB").save(path)


def save_gray(image: torch.Tensor, path: Path) -> None:
    array = (image.numpy() * 255.0 + 0.5).clip(0, 255).astype("uint8")
    Image.fromarray(array, mode="L").save(path)


def normalise_depth(depth: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    valid = (alpha > 0.01) & torch.isfinite(depth) & (depth > 0)
    output = torch.zeros_like(depth)
    if valid.any():
        low, high = torch.quantile(depth[valid], torch.tensor([0.02, 0.98]))
        output[valid] = 1.0 - ((depth[valid] - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    return output


def crop_and_downsample(
    image: torch.Tensor,
    output_height: int,
    output_width: int,
    supersample: int,
    crop_margin: int,
) -> torch.Tensor:
    margin = crop_margin * supersample
    if margin:
        if 2 * margin >= min(image.shape[0], image.shape[1]):
            raise ValueError("--crop-margin is too large for the output size")
        image = image[margin:-margin, margin:-margin]
    if image.shape[:2] == (output_height, output_width):
        return image
    channels_last = image.ndim == 3
    tensor = image.permute(2, 0, 1)[None] if channels_last else image[None, None]
    tensor = F.interpolate(
        tensor,
        size=(output_height, output_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return tensor[0].permute(1, 2, 0) if channels_last else tensor[0, 0]


def unsharp_mask(image: torch.Tensor, amount: float) -> torch.Tensor:
    if amount <= 0:
        return image
    tensor = image.permute(2, 0, 1)[None]
    # Separable [1, 4, 6, 4, 1] / 16 Gaussian blur.
    kernel_1d = torch.tensor([1, 4, 6, 4, 1], dtype=tensor.dtype) / 16.0
    kernel_x = kernel_1d.reshape(1, 1, 1, 5).repeat(3, 1, 1, 1)
    kernel_y = kernel_1d.reshape(1, 1, 5, 1).repeat(3, 1, 1, 1)
    padded = F.pad(tensor, (2, 2, 2, 2), mode="reflect")
    blurred = F.conv2d(padded, kernel_x, groups=3)
    blurred = F.conv2d(blurred, kernel_y, groups=3)
    return (tensor + amount * (tensor - blurred))[0].permute(1, 2, 0).clamp(0, 1)


def main() -> None:
    args = parse_args()
    if not 0 < args.keep_ratio <= 1:
        raise ValueError("--keep-ratio must be in (0, 1]")
    if args.supersample < 1:
        raise ValueError("--supersample must be >= 1")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    rgb_dir, alpha_dir, depth_dir = (
        args.output / "rgb",
        args.output / "alpha",
        args.output / "depth",
    )
    for directory in (rgb_dir, alpha_dir, depth_dir):
        directory.mkdir(parents=True, exist_ok=True)

    gaussians, metadata = load_gaussians(
        args.gaussians, args.keep_ratio, args.min_opacity, args.crop_padding
    )
    if args.poses is not None:
        poses = torch.load(args.poses, map_location="cpu", weights_only=True).float()
        poses_source = str(args.poses.resolve())
    else:
        poses = make_small_baseline_trajectory(args.views, args.translation, args.yaw)
        poses_source = "generated small-baseline trajectory"
        torch.save(poses, args.output / "render_poses.pt")
        with (args.output / "render_poses.json").open("w", encoding="utf-8") as file:
            json.dump(poses.tolist(), file, indent=2)
    pose_indices = args.pose_index if args.pose_index is not None else list(range(len(poses)))
    render_height = args.height * args.supersample
    render_width = args.width * args.supersample
    base_fx = args.fx if args.fx is not None else args.focal_scale * 0.5 * (args.width + args.height)
    base_fy = args.fy if args.fy is not None else base_fx
    base_cx = args.cx if args.cx is not None else (args.width - 1) * 0.5
    base_cy = args.cy if args.cy is not None else (args.height - 1) * 0.5
    fx, fy = base_fx * args.supersample, base_fy * args.supersample
    cx = (base_cx + 0.5) * args.supersample - 0.5
    cy = (base_cy + 0.5) * args.supersample - 0.5
    background = torch.tensor(args.background, dtype=torch.float32)

    timings = []
    visible_counts = []
    for pose_index in pose_indices:
        start = time.perf_counter()
        projected = project_gaussians(
            gaussians,
            poses[pose_index],
            fx,
            fy,
            cx,
            cy,
            args.near,
            args.far,
            args.min_variance,
            args.sigma_cutoff,
            args.max_radius,
            args.scale_modifier,
            render_width,
            render_height,
        )
        rgb, alpha, depth = rasterize(
            projected,
            render_height,
            render_width,
            background,
            args.tile_size,
            args.chunk_size,
            args.sigma_cutoff,
        )
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        visible_counts.append(projected["z"].numel())
        stem = f"view_{pose_index:03d}"
        rgb = crop_and_downsample(rgb, args.height, args.width, args.supersample, args.crop_margin)
        alpha = crop_and_downsample(alpha, args.height, args.width, args.supersample, args.crop_margin).clamp(0, 1)
        depth = crop_and_downsample(depth, args.height, args.width, args.supersample, args.crop_margin)
        rgb = unsharp_mask(rgb, args.sharpen)
        if args.linear_to_srgb:
            rgb = linear_to_srgb(rgb)
        save_rgb(rgb, rgb_dir / f"{stem}.png")
        save_gray(alpha, alpha_dir / f"{stem}.png")
        save_gray(normalise_depth(depth, alpha), depth_dir / f"{stem}.png")
        print(
            f"{stem}: {projected['z'].numel()} visible Gaussians, "
            f"{elapsed:.3f} s, mean alpha={alpha.mean().item():.4f}"
        )

    report = {
        "device": "cpu",
        "renderer": "PyTorch anisotropic 3D Gaussian + front-to-back alpha blending",
        "gaussians_input": str(args.gaussians.resolve()),
        "poses_input": poses_source,
        "gaussians_after_filter": gaussians["xyz"].shape[0],
        "visible_gaussians": visible_counts,
        "image_size_hw": [args.height, args.width],
        "intrinsics_at_output_resolution": {
            "fx": base_fx, "fy": base_fy, "cx": base_cx, "cy": base_cy
        },
        "seconds_per_view": timings,
        "mean_seconds_per_view": sum(timings) / len(timings),
        "rss_final_mb": psutil.Process().memory_info().rss / 1024**2,
        "settings": {
            "keep_ratio": args.keep_ratio,
            "crop_padding": args.crop_padding,
            "min_opacity": args.min_opacity,
            "scale_modifier": args.scale_modifier,
            "sigma_cutoff": args.sigma_cutoff,
            "min_variance": args.min_variance,
            "max_radius": args.max_radius,
            "tile_size": args.tile_size,
            "chunk_size": args.chunk_size,
            "background": list(args.background),
            "supersample": args.supersample,
            "crop_margin": args.crop_margin,
            "sharpen": args.sharpen,
            "linear_to_srgb": args.linear_to_srgb,
        },
    }
    with (args.output / "render_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Saved RGB, alpha, depth, and report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
