#!/usr/bin/env python3
"""CPU multi-camera rendering for a Flash3D Gaussian scene.

Unlike ``render_cpu_alpha.py``'s compact interpolation trajectory, this entry
point renders a named physical-camera rig. The default ``cross5`` rig creates
five distinct camera centres (centre, left, right, up and down), and every
camera is aimed at a target in the reconstructed scene.

For calibrated/custom rigs, pass a JSON file with this schema::

    {"cameras": [
      {"name": "front_left", "position_xyz": [-0.5, 0, 0],
       "look_at_xyz": [0, 0, 7.2], "roll_deg": 0.0}
    ]}

``position_xyz`` and ``look_at_xyz`` are in the input/source-camera coordinate
system. Keep baselines conservative for a single-image reconstruction: unseen
geometry cannot be created by Gaussian splatting alone.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import torch
from PIL import Image, ImageDraw, ImageFont

from render_cpu_alpha import (
    crop_and_downsample,
    evaluate_view_dependent_color,
    linear_to_srgb,
    load_gaussians,
    normalise_depth,
    project_gaussians,
    quaternion_to_rotation,
    rasterize,
    save_gray,
    save_rgb,
    unsharp_mask,
)


def _native_renderer_dependencies() -> tuple[object, object, object]:
    """Load the exact CUDA rasterizer used by Flash3D's gauss_util.py.

    Kept lazy so that importing/running the CPU reference renderer never
    requires a CUDA build of diff-gaussian-rasterization.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--backend native requires a CUDA-enabled PyTorch build and GPU. "
            "Use --backend cpu on this machine."
        )
    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        from models.decoder.gauss_util import getProjectionMatrix
    except ImportError as exc:
        raise RuntimeError(
            "The native Flash3D renderer needs the project's CUDA extension "
            "diff_gaussian_rasterization. Build/install the original Flash3D "
            "CUDA environment, then retry --backend native."
        ) from exc
    return GaussianRasterizationSettings, GaussianRasterizer, getProjectionMatrix


def _gsplat_renderer_dependency() -> object:
    """Load gsplat only for its CUDA backend, keeping CPU usage dependency-free."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--backend gsplat requires a CUDA-enabled PyTorch build and GPU. "
            "Use --backend torch on this machine."
        )
    try:
        # Current gsplat releases export this at package level.  The fallback
        # keeps the adapter usable with older package layouts.
        from gsplat import rasterization
    except ImportError:
        try:
            from gsplat.rendering import rasterization
        except ImportError as exc:
            raise RuntimeError(
                "--backend gsplat requires the CUDA gsplat package. Install it "
                "in the CUDA environment with: python -m pip install gsplat"
            ) from exc
    return rasterization


def _native_sh_degree(gaussians: dict[str, torch.Tensor]) -> int:
    """Infer the supported real-SH degree from canonical [N, K, 3] terms."""
    rest = gaussians.get("features_rest")
    if rest is None or rest.numel() == 0:
        return 0
    coefficients = rest.shape[1] + 1  # include DC
    degree = int(round(math.sqrt(coefficients) - 1))
    if (degree + 1) ** 2 != coefficients:
        raise ValueError(f"Invalid SH coefficient count: {coefficients}")
    return degree


@torch.inference_mode()
def rasterize_gsplat(
    gaussians: dict[str, torch.Tensor],
    world_to_camera: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    background: torch.Tensor,
    near: float,
    far: float,
    scale_modifier: float,
    eps2d: float,
    radius_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Render a Flash3D Gaussian scene through gsplat's CUDA rasterizer.

    gsplat uses conventional (non-transposed) world-to-camera matrices and
    scalar-first ``wxyz`` quaternions, which is the same quaternion layout
    exported by Flash3D.  In contrast, Flash3D's own CUDA extension receives
    transposed matrices; do not transpose ``world_to_camera`` here.
    """
    rasterization = _gsplat_renderer_dependency()
    device = torch.device("cuda")
    means = gaussians["xyz"].to(device, non_blocking=True).contiguous()
    quats = gaussians["rotation"].to(device, non_blocking=True).contiguous()
    scales = (gaussians["scaling"] * scale_modifier).to(device, non_blocking=True).contiguous()
    opacities = gaussians["opacity"].to(device, non_blocking=True).reshape(-1).contiguous()
    features_dc = gaussians["features_dc"].to(device, non_blocking=True).reshape(-1, 1, 3).contiguous()
    features_rest = gaussians.get("features_rest")
    if features_rest is not None and features_rest.numel() > 0:
        colors = torch.cat((features_dc, features_rest.to(device, non_blocking=True)), dim=1).contiguous()
        sh_degree = _native_sh_degree(gaussians)
    else:
        # Degree-zero SH avoids an activation/convention mismatch with RGB
        # colours while retaining exactly the Flash3D DC coefficients.
        colors = features_dc
        sh_degree = 0
    intrinsic = torch.tensor(
        ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=torch.float32, device=device
    )
    render, alpha, metadata = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=world_to_camera.to(device, non_blocking=True).float().unsqueeze(0),
        Ks=intrinsic.unsqueeze(0),
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        radius_clip=radius_clip,
        eps2d=eps2d,
        sh_degree=sh_degree,
        packed=False,
        tile_size=16,
        backgrounds=background.to(device, non_blocking=True).float().unsqueeze(0),
        render_mode="RGB+ED",
        rasterize_mode="classic",
    )
    # gsplat returns [C, H, W, RGB + expected-depth] and [C, H, W, 1].
    rgb = render[0, ..., :3].detach().float().cpu()
    depth = render[0, ..., 3].detach().float().cpu()
    alpha = alpha[0, ..., 0].detach().float().cpu()
    radii = metadata.get("radii")
    visible = int((radii[0] > 0).sum().item()) if radii is not None and radii.ndim == 2 else int(means.shape[0])
    return rgb, alpha, depth, visible


@torch.inference_mode()
def rasterize_torch_flash3d(
    gaussians: dict[str, torch.Tensor],
    world_to_camera: torch.Tensor,
    camera_center: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    background: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Run Flash3D's portable Torch rasterizer on CPU for one camera.

    ``models/decoder/npu_differentiable_renderer.py`` is the project's
    reimplementation of the native diff-gaussian-rasterization algorithm with
    normal PyTorch operations.  It works unchanged on CPU, and this adapter
    supplies the calibrated camera matrix needed by a saved Gaussian scene.
    """
    from models.decoder.npu_differentiable_renderer import render_predicted_torch

    # Keep all selected Gaussians and all per-tile candidates: unlike the
    # training-debug profile, offline multiview evaluation must not discard
    # splats merely to cap CPU/NPU training memory.
    max_tile_reach = math.ceil(args.max_radius / args.tile_size)
    renderer_cfg = SimpleNamespace(
        dataset=SimpleNamespace(znear=args.near, zfar=args.far),
        model=SimpleNamespace(
            npu_renderer_min_variance=args.min_variance,
            npu_renderer_sigma_cutoff=args.sigma_cutoff,
            npu_renderer_max_radius=args.max_radius,
            npu_renderer_max_gaussians=0,
            npu_renderer_tile_size=args.tile_size,
            # The original CUDA renderer assigns every overlapping tile.  The
            # portable binner uses a finite square around the centre, so size
            # it from the maximum retained radius instead of silently clipping
            # a large ellipse at neighbouring tiles.
            npu_renderer_tile_span=max(5, 2 * max_tile_reach + 3),
            npu_renderer_max_gaussians_per_tile=0,
        ),
    )
    fov_x = 2.0 * math.atan(width / (2.0 * fx))
    fov_y = 2.0 * math.atan(height / (2.0 * fy))
    # render_predicted_torch expects the transposed matrix convention used by
    # Flash3D model.py before it hands matrices to the CUDA rasterizer.
    viewmatrix = world_to_camera.T.contiguous()
    result = render_predicted_torch(
        renderer_cfg,
        gaussians,
        viewmatrix,
        viewmatrix,  # API-compatible unused full projection on Torch path
        viewmatrix,  # API-compatible unused raw projection on Torch path
        camera_center,
        (fov_x, fov_y),
        (height, width),
        background,
        _native_sh_degree(gaussians),
        args.scale_modifier,
        principal_point=(cx, cy),
    )
    rgb = result["render"].permute(1, 2, 0).detach().float().cpu()
    depth = result["depth"].detach().float().cpu()
    alpha = result["alpha"].detach().float().cpu()
    return rgb, alpha, depth, int(result["visibility_filter"].sum().item())


@torch.inference_mode()
def rasterize_gsplat_cpu_reference(
    gaussians: dict[str, torch.Tensor],
    world_to_camera: torch.Tensor,
    camera_center: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    background: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """CPU reference for gsplat's 3DGS ``classic`` rasterization path.

    This follows the public gsplat reference equations: conventional
    world-to-camera transform, clamped Jacobian for perspective covariance,
    ``eps2d`` diagonal regularization, 3.33-sigma axis-aligned tile bounds,
    pixel-centre Gaussian evaluation, and front-to-back alpha compositing.
    It intentionally trades CUDA throughput for inspectable CPU semantics.
    """
    if args.tile_size != 16:
        raise ValueError("gsplat-compatible CPU reference requires --tile-size 16")
    xyz = gaussians["xyz"].float()
    rotation = world_to_camera[:3, :3].float()
    translation = world_to_camera[:3, 3].float()
    camera_xyz = xyz @ rotation.T + translation
    x, y, z = camera_xyz.unbind(dim=-1)

    # Matches gsplat.cuda._torch_impl._persp_proj: the mean uses the ordinary
    # perspective projection while the Jacobian is bounded near the FOV edge.
    inverse_z = z.clamp_min(args.near).reciprocal()
    tan_fov_x, tan_fov_y = 0.5 * width / fx, 0.5 * height / fy
    x_for_jacobian = z * torch.clamp(
        x * inverse_z,
        min=-(cx / fx + 0.3 * tan_fov_x),
        max=((width - cx) / fx + 0.3 * tan_fov_x),
    )
    y_for_jacobian = z * torch.clamp(
        y * inverse_z,
        min=-(cy / fy + 0.3 * tan_fov_y),
        max=((height - cy) / fy + 0.3 * tan_fov_y),
    )
    means2d = torch.stack((fx * x * inverse_z + cx, fy * y * inverse_z + cy), dim=-1)

    # gsplat builds the 3D covariance directly from the supplied scales.  Do
    # not pre-clamp here: its only opacity cap is applied per pixel below.
    scales = gaussians["scaling"].float() * args.scale_modifier
    basis = quaternion_to_rotation(gaussians["rotation"].float()) * scales[:, None, :]
    covariance_world = basis @ basis.transpose(1, 2)
    covariance_camera = rotation[None] @ covariance_world @ rotation.T[None]
    jacobian = torch.zeros((len(xyz), 2, 3), dtype=torch.float32)
    jacobian[:, 0, 0] = fx * inverse_z
    jacobian[:, 0, 2] = -fx * x_for_jacobian * inverse_z.square()
    jacobian[:, 1, 1] = fy * inverse_z
    jacobian[:, 1, 2] = -fy * y_for_jacobian * inverse_z.square()
    covariance_2d = jacobian @ covariance_camera @ jacobian.transpose(1, 2)
    covariance_2d[:, 0, 0] += args.gsplat_eps2d
    covariance_2d[:, 1, 1] += args.gsplat_eps2d
    a, b, c = covariance_2d[:, 0, 0], covariance_2d[:, 0, 1], covariance_2d[:, 1, 1]
    determinant = (a * c - b * b).clamp_min(1e-10)
    conics = torch.stack((c / determinant, -b / determinant, a / determinant), dim=-1)
    radii = torch.stack((torch.ceil(3.33 * a.clamp_min(0).sqrt()), torch.ceil(3.33 * c.clamp_min(0).sqrt())), dim=-1)
    visible = (
        (z > args.near) & (z < args.far) & torch.isfinite(conics).all(dim=-1)
        & (means2d[:, 0] + radii[:, 0] > 0) & (means2d[:, 0] - radii[:, 0] < width)
        & (means2d[:, 1] + radii[:, 1] > 0) & (means2d[:, 1] - radii[:, 1] < height)
        & (radii.amax(dim=-1) > args.gsplat_radius_clip)
    )
    opacity = gaussians["opacity"].float()
    colours = evaluate_view_dependent_color(gaussians, camera_center).float()
    rgb = torch.empty((height, width, 3), dtype=torch.float32)
    alpha_image = torch.empty((height, width), dtype=torch.float32)
    depth_image = torch.empty((height, width), dtype=torch.float32)
    tile_size = args.tile_size
    for y0 in range(0, height, tile_size):
        y1 = min(y0 + tile_size, height)
        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            selected = torch.where(
                visible
                & (means2d[:, 0] + radii[:, 0] > x0)
                & (means2d[:, 0] - radii[:, 0] < x1)
                & (means2d[:, 1] + radii[:, 1] > y0)
                & (means2d[:, 1] - radii[:, 1] < y1)
            )[0]
            if selected.numel() == 0:
                rgb[y0:y1, x0:x1] = background
                alpha_image[y0:y1, x0:x1] = 0
                depth_image[y0:y1, x0:x1] = 0
                continue
            # gsplat's intersection key orders each tile by float32 camera z.
            selected = selected[torch.argsort(z[selected].float(), stable=True)]
            yy, xx = torch.meshgrid(
                torch.arange(y0, y1, dtype=torch.float32) + 0.5,
                torch.arange(x0, x1, dtype=torch.float32) + 0.5,
                indexing="ij",
            )
            dx, dy = xx.reshape(1, -1) - means2d[selected, 0, None], yy.reshape(1, -1) - means2d[selected, 1, None]
            conic = conics[selected]
            sigma = 0.5 * (conic[:, 0, None] * dx.square() + conic[:, 2, None] * dy.square()) + conic[:, 1, None] * dx * dy
            alpha = (opacity[selected, None] * torch.exp(-sigma)).clamp(max=0.999)
            transmittance = torch.cumprod(
                torch.cat((torch.ones_like(alpha[:1]), 1.0 - alpha[:-1]), dim=0), dim=0
            )
            weights = alpha * transmittance
            final_transmittance = (1.0 - alpha).prod(dim=0)
            tile_alpha = 1.0 - final_transmittance
            tile_rgb = weights.T @ colours[selected] + final_transmittance[:, None] * background[None]
            accumulated_depth = (weights * z[selected, None]).sum(dim=0)
            tile_depth = torch.where(tile_alpha > 1e-6, accumulated_depth / tile_alpha, torch.zeros_like(tile_alpha))
            rgb[y0:y1, x0:x1] = tile_rgb.reshape(y1 - y0, x1 - x0, 3)
            alpha_image[y0:y1, x0:x1] = tile_alpha.reshape(y1 - y0, x1 - x0)
            depth_image[y0:y1, x0:x1] = tile_depth.reshape(y1 - y0, x1 - x0)
    return rgb, alpha_image, depth_image, int(visible.sum().item())


@torch.inference_mode()
def rasterize_native_flash3d(
    gaussians: dict[str, torch.Tensor],
    world_to_camera: torch.Tensor,
    camera_center: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    background: torch.Tensor,
    near: float,
    far: float,
    scale_modifier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Render one physical camera with Flash3D's original CUDA path.

    Matrix construction, principal-point conversion, SH layout, scales and
    quaternions intentionally match ``models/decoder/gauss_util.py``.  This
    is not a CPU approximation: it calls its ``GaussianRasterizer`` directly.
    """
    GaussianRasterizationSettings, GaussianRasterizer, getProjectionMatrix = _native_renderer_dependencies()
    device = torch.device("cuda")
    xyz = gaussians["xyz"].to(device, non_blocking=True).contiguous()
    scaling = gaussians["scaling"].to(device, non_blocking=True).contiguous()
    rotation = gaussians["rotation"].to(device, non_blocking=True).contiguous()
    opacity = gaussians["opacity"].to(device, non_blocking=True).reshape(-1, 1).contiguous()
    features_dc = gaussians["features_dc"].to(device, non_blocking=True).reshape(-1, 1, 3).contiguous()
    features_rest = gaussians.get("features_rest")
    if features_rest is not None:
        features_rest = features_rest.to(device, non_blocking=True).contiguous()
        shs = torch.cat((features_dc, features_rest), dim=1).contiguous()
    else:
        shs = features_dc

    # Same focal-to-FOV and K_to_NDC_pp conventions used by Flash3D model.py.
    fov_x = 2.0 * math.atan(width / (2.0 * fx))
    fov_y = 2.0 * math.atan(height / (2.0 * fy))
    principal_x = 2.0 * cx / width - 1.0
    principal_y = 2.0 * cy / height - 1.0
    viewmatrix = world_to_camera.to(device, non_blocking=True).T.contiguous()
    projmatrix_raw = getProjectionMatrix(near, far, fov_x, fov_y, principal_x, principal_y).to(device).T.contiguous()
    full_proj_transform = (viewmatrix @ projmatrix_raw).contiguous()
    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=math.tan(fov_x * 0.5),
        tanfovy=math.tan(fov_y * 0.5),
        bg=background.to(device, non_blocking=True).contiguous(),
        scale_modifier=scale_modifier,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_transform,
        # Flash3D enables renderer_w_pose for RE10K, therefore preserve its
        # raw projection matrix extension argument.
        projmatrix_raw=projmatrix_raw,
        sh_degree=_native_sh_degree(gaussians),
        campos=camera_center.to(device, non_blocking=True).contiguous(),
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # Gradients are irrelevant at inference, unlike render_predicted()'s
    # training-only retained screen-space points.
    means2d = torch.zeros_like(xyz)
    outputs = rasterizer(
        means3D=xyz,
        means2D=means2d,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scaling,
        rotations=rotation,
        cov3D_precomp=None,
    )
    rendered, radii = outputs[:2]
    rgb = rendered.permute(1, 2, 0).detach().float().cpu()
    # The Flash3D fork returns depth and alpha.  Retain portable image output
    # for an older upstream extension which returns only image/radii.
    if len(outputs) >= 4:
        depth = outputs[2].detach().float().squeeze().cpu()
        alpha = outputs[3].detach().float().squeeze().cpu()
    else:
        depth = torch.zeros((height, width), dtype=torch.float32)
        alpha = torch.ones((height, width), dtype=torch.float32)
    return rgb, alpha, depth, int((radii > 0).sum().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gaussians", type=Path, required=True, help="gaussians.pt exported by benchmark_cpu.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("torch", "cpu", "flash3d_torch", "legacy", "native", "gsplat"), default="torch",
        help=(
            "torch/default and cpu: gsplat-classic CPU reference; "
            "flash3d_torch: prior Flash3D portable PyTorch renderer; legacy is the earlier lightweight "
            "CPU approximation; native uses Flash3D CUDA "
            "diff-gaussian-rasterization; gsplat uses gsplat CUDA"
        ),
    )
    parser.add_argument("--rig", choices=("cross5", "arc5", "grid9"), default="cross5")
    parser.add_argument("--camera-file", type=Path, default=None, help="Custom rig JSON; overrides --rig")
    parser.add_argument("--baseline", type=float, default=0.5, help="Physical horizontal camera-centre offset in scene units")
    parser.add_argument("--vertical-baseline", type=float, default=None, help="Physical vertical camera-centre offset; defaults to 0.7 * baseline")
    parser.add_argument("--position-scale", type=float, default=1.0, help="Scale every camera centre from --camera-file/preset; use this to preserve the same angular motion across scenes with different depth scales")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--fx", type=float, default=390.0)
    parser.add_argument("--fy", type=float, default=390.0)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument(
        "--use-source-intrinsics", action=argparse.BooleanOptionalAction, default=False,
        help="Use calibrated intrinsics embedded in a compatible Gaussian export (currently UniSHARP); falls back to CLI intrinsics when unavailable",
    )
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--crop-margin", type=int, default=20)
    parser.add_argument("--sharpen", type=float, default=0.35)
    parser.add_argument(
        "--linear-to-srgb", action=argparse.BooleanOptionalAction, default=False,
        help="Encode rendered linear RGB to sRGB before PNG output; use for UniSHARP colour exports",
    )
    parser.add_argument("--keep-ratio", type=float, default=1.0)
    parser.add_argument("--crop-padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-opacity", type=float, default=0.005)
    parser.add_argument("--scale-modifier", type=float, default=0.55)
    parser.add_argument("--sigma-cutoff", type=float, default=2.5)
    parser.add_argument("--min-variance", type=float, default=0.2)
    parser.add_argument("--max-radius", type=float, default=96.0)
    parser.add_argument(
        "--gsplat-eps2d", type=float, default=0.3,
        help="gsplat only: projected-covariance regularizer in pixel-squared units",
    )
    parser.add_argument(
        "--gsplat-radius-clip", type=float, default=0.0,
        help="gsplat-compatible CPU/CUDA backends: skip splats whose projected radius is at or below this value",
    )
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=1000.0)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--background", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument(
        "--prune-source-depth-outliers", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Before novel-view rendering, remove Gaussians that project behind a nearer "
            "source-view surface. This reduces floating/occluded-layer speckles."
        ),
    )
    parser.add_argument(
        "--source-prune-grid-scale", type=float, default=2.0,
        help="Depth-pruning grid resolution relative to --width/--height",
    )
    parser.add_argument(
        "--source-prune-relative-tolerance", type=float, default=0.06,
        help="Keep source-view points within this relative depth distance of the front layer",
    )
    parser.add_argument(
        "--source-prune-absolute-tolerance", type=float, default=0.08,
        help="Keep source-view points within this absolute depth distance of the front layer",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--save-gaussians", action=argparse.BooleanOptionalAction, default=True, help="Save the filtered Gaussian point cloud as PT and coloured PLY")
    parser.add_argument("--contact-sheet-columns", type=int, default=5, help="Number of columns in comparison_grid.png")
    parser.add_argument("--contact-sheet-title", type=str, default=None, help="Optional title drawn above comparison_grid.png")
    parser.add_argument("--contact-sheet-title-size", type=int, default=36, help="Font size for the centred contact-sheet title")
    return parser.parse_args()


def preset_cameras(rig: str, baseline: float, vertical_baseline: float) -> list[dict]:
    """Return named physical camera centres in source-camera/world coordinates."""
    if rig == "cross5":
        return [
            {"name": "center", "position_xyz": [0.0, 0.0, 0.0], "source_camera": True},
            {"name": "left", "position_xyz": [-baseline, 0.0, 0.0]},
            {"name": "right", "position_xyz": [baseline, 0.0, 0.0]},
            {"name": "up", "position_xyz": [0.0, -vertical_baseline, 0.0]},
            {"name": "down", "position_xyz": [0.0, vertical_baseline, 0.0]},
        ]
    if rig == "arc5":
        return [
            {"name": f"arc_{index:02d}", "position_xyz": [offset, 0.0, 0.0]}
            for index, offset in enumerate(torch.linspace(-baseline, baseline, 5).tolist())
        ]
    if rig == "grid9":
        cameras = []
        for row, dy in enumerate((vertical_baseline, 0.0, -vertical_baseline)):
            for column, dx in enumerate((-baseline, 0.0, baseline)):
                cameras.append({"name": f"r{row}_c{column}", "position_xyz": [dx, dy, 0.0]})
        return cameras
    raise ValueError(f"Unsupported preset rig: {rig}")


def read_cameras(args: argparse.Namespace) -> list[dict]:
    if args.camera_file is None:
        vertical = args.vertical_baseline if args.vertical_baseline is not None else args.baseline * 0.7
        cameras = preset_cameras(args.rig, args.baseline, vertical)
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
        if "translation_xyz" in camera:
            raise ValueError(
                f"Camera {name}: translation_xyz is an old world-to-camera extrinsic field. "
                "Use physical position_xyz instead."
            )
        position = camera.get("position_xyz", (0.0, 0.0, 0.0))
        if len(position) != 3:
            raise ValueError(f"Camera {name}: position_xyz must have 3 values")
        look_at = camera.get("look_at_xyz")
        if look_at is not None and len(look_at) != 3:
            raise ValueError(f"Camera {name}: look_at_xyz must have 3 values")
        camera["name"] = name
        camera["position_xyz"] = [float(value) for value in position]
        camera["look_at_xyz"] = None if look_at is None else [float(value) for value in look_at]
        camera["roll_deg"] = float(camera.get("roll_deg", 0.0))
        camera["source_camera"] = bool(camera.get("source_camera", False))
    return cameras


def camera_transform(camera: dict, default_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a world-to-camera transform from a physical centre and look-at point.

    The source image's camera coordinate system is used as world space: x is
    right, y is down, and z is forward.  Therefore the original camera is at
    [0, 0, 0], and its centre view has an identity world-to-camera transform.
    """
    camera_center = torch.tensor(camera["position_xyz"], dtype=torch.float32)
    if camera.get("source_camera", False):
        if torch.linalg.vector_norm(camera_center) > 1e-6:
            raise ValueError(f"Camera {camera['name']}: source_camera requires position_xyz=[0, 0, 0]")
        if camera["roll_deg"] != 0.0 or camera["look_at_xyz"] is not None:
            raise ValueError(f"Camera {camera['name']}: source_camera cannot set roll_deg or look_at_xyz")
        return torch.eye(4, dtype=torch.float32), default_target
    target = default_target if camera["look_at_xyz"] is None else torch.tensor(camera["look_at_xyz"], dtype=torch.float32)
    forward = target - camera_center
    if torch.linalg.vector_norm(forward) < 1e-6:
        raise ValueError(f"Camera {camera['name']}: position_xyz must differ from look_at_xyz")
    forward = torch.nn.functional.normalize(forward, dim=0)
    # In the source image coordinate system, physical up is negative y.
    reference_up = torch.tensor((0.0, -1.0, 0.0), dtype=torch.float32)
    if torch.abs(torch.dot(forward, reference_up)) > 0.98:
        reference_up = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32)
    right = torch.nn.functional.normalize(torch.linalg.cross(forward, reference_up), dim=0)
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack((right, down, forward))
    roll = math.radians(camera["roll_deg"])
    if roll:
        cosine, sine = math.cos(roll), math.sin(roll)
        in_camera_roll = torch.tensor(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)), dtype=torch.float32)
        rotation = in_camera_roll @ rotation
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ camera_center
    return transform, target


def save_gaussian_point_cloud(gaussians: dict[str, torch.Tensor], output: Path, source: Path) -> dict[str, str]:
    """Save a reusable tensor dump and a viewer-friendly coloured point cloud."""
    tensor_path = output / "gaussians_filtered.pt"
    ply_path = output / "gaussians_pointcloud.ply"
    torch.save({"gaussians": gaussians, "source_gaussians": str(source.resolve())}, tensor_path)
    xyz = gaussians["xyz"].numpy()
    colour = (gaussians["color"].clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    opacity = gaussians["opacity"].numpy()
    scale = gaussians["scaling"].numpy()
    with ply_path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(xyz)}\n")
        for property_name in ("x", "y", "z", "red", "green", "blue", "opacity", "scale_x", "scale_y", "scale_z"):
            property_type = "uchar" if property_name in {"red", "green", "blue"} else "float"
            file.write(f"property {property_type} {property_name}\n")
        file.write("end_header\n")
        for point, rgb, alpha, scaling in zip(xyz, colour, opacity, scale):
            file.write(f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} {rgb[0]} {rgb[1]} {rgb[2]} {alpha:.7g} {scaling[0]:.7g} {scaling[1]:.7g} {scaling[2]:.7g}\n")
    return {"tensor": str(tensor_path.resolve()), "ply": str(ply_path.resolve())}


def save_contact_sheet(entries: list[tuple[str, Path]], output: Path, columns: int, title: str | None = None, title_size: int = 36) -> Path:
    """Create a labelled RGB comparison grid without altering individual renders."""
    if columns < 1:
        raise ValueError("--contact-sheet-columns must be >= 1")
    images = [(name, Image.open(path).convert("RGB")) for name, path in entries]
    width, height = images[0][1].size
    label_height, padding = 24, 4
    if title_size < 1:
        raise ValueError("--contact-sheet-title-size must be >= 1")
    try:
        title_font = ImageFont.truetype("arial.ttf", title_size)
    except OSError:
        title_font = ImageFont.load_default()
    title_height = title_size + 18 if title else 0
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * (width + padding) + padding, title_height + rows * (height + label_height + padding) + padding), "white")
    draw = ImageDraw.Draw(sheet)
    if title:
        box = draw.textbbox((0, 0), title, font=title_font)
        draw.text(((sheet.width - (box[2] - box[0])) * 0.5, 7), title, fill="black", font=title_font)
    for index, (name, image) in enumerate(images):
        row, column = divmod(index, columns)
        x = padding + column * (width + padding)
        y = title_height + padding + row * (height + label_height + padding)
        draw.text((x + 2, y + 4), name, fill="black")
        sheet.paste(image, (x, y + label_height))
    sheet_path = output / "comparison_grid.png"
    sheet.save(sheet_path)
    return sheet_path


def source_intrinsics_at_output_resolution(
    metadata: dict, height: int, width: int
) -> tuple[float, float, float, float] | None:
    """Read and resize the [fx, fy, cx, cy] camera record embedded by UniSHARP."""
    camera = metadata.get("camera", {}) if isinstance(metadata, dict) else {}
    intrinsics = camera.get("intrinsics") if isinstance(camera, dict) else None
    if intrinsics is None:
        return None
    # Benchmark exports JSON, so a 3x3 matrix is commonly a nested Python
    # list rather than a Tensor.  Canonicalise both forms before extracting
    # fx, fy, cx and cy.
    matrix = torch.as_tensor(intrinsics, dtype=torch.float32).detach().cpu()
    if matrix.numel() >= 9 and tuple(matrix.shape[-2:]) == (3, 3):
        matrix = matrix.reshape(-1, 3, 3)[0]
        values = [matrix[0, 0].item(), matrix[1, 1].item(), matrix[0, 2].item(), matrix[1, 2].item()]
    else:
        values = matrix.reshape(-1).tolist()
    if len(values) < 4:
        return None
    input_h, input_w = metadata.get("input_size_hw", (height, width))
    if not input_h or not input_w:
        return None
    sx, sy = width / float(input_w), height / float(input_h)
    fx, fy, cx, cy = (float(value) for value in values[:4])
    return fx * sx, fy * sy, cx * sx, cy * sy


@torch.inference_mode()
def prune_source_depth_outliers(
    gaussians: dict[str, torch.Tensor],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near: float,
    grid_scale: float,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[dict[str, torch.Tensor], dict[str, int | float]]:
    """Cull points hidden behind the nearest source-view surface per grid cell.

    A single-image Gaussian prediction frequently contains several candidate
    depth layers along a source camera ray.  The layer behind the visible
    surface is useful neither for reproducing the source image nor for stable
    small-baseline novel views; when it leaks around an occlusion edge it
    appears as a dark floating speckle.  This conservative filter retains the
    front layer and any Gaussian close enough to it to cover a continuous
    surface, accounting for its 3D scale.
    """
    if grid_scale <= 0:
        raise ValueError("--source-prune-grid-scale must be > 0")
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("source depth-pruning tolerances must be >= 0")

    grid_width = max(1, round(width * grid_scale))
    grid_height = max(1, round(height * grid_scale))
    xyz = gaussians["xyz"]
    z = xyz[:, 2]
    # Resample camera coordinates using pixel-centre convention.
    u = (fx * xyz[:, 0] / z.clamp_min(near) + cx + 0.5) * grid_scale - 0.5
    v = (fy * xyz[:, 1] / z.clamp_min(near) + cy + 0.5) * grid_scale - 0.5
    u_int, v_int = torch.floor(u).long(), torch.floor(v).long()
    inside = (
        (z > near)
        & torch.isfinite(u)
        & torch.isfinite(v)
        & (u_int >= 0)
        & (u_int < grid_width)
        & (v_int >= 0)
        & (v_int < grid_height)
    )
    cells = v_int[inside] * grid_width + u_int[inside]
    front_depth = torch.full((grid_width * grid_height,), float("inf"), dtype=torch.float32)
    front_depth.scatter_reduce_(0, cells, z[inside].float(), reduce="amin", include_self=True)
    reference_depth = torch.full_like(z, float("inf"))
    reference_depth[inside] = front_depth[cells]
    scale_slack = 2.0 * gaussians["scaling"].abs().amax(dim=-1)
    tolerated_depth = reference_depth * (1.0 + relative_tolerance) + absolute_tolerance + scale_slack
    occluded = inside & (z > tolerated_depth)
    keep = ~occluded
    filtered = {key: value[keep].contiguous() for key, value in gaussians.items()}
    return filtered, {
        "grid_width": grid_width,
        "grid_height": grid_height,
        "input_gaussians": int(len(z)),
        "projected_gaussians": int(inside.sum().item()),
        "removed_occluded_gaussians": int(occluded.sum().item()),
        "kept_gaussians": int(keep.sum().item()),
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.keep_ratio <= 1:
        raise ValueError("--keep-ratio must be in (0, 1]")
    if args.supersample < 1:
        raise ValueError("--supersample must be >= 1")
    if args.gsplat_eps2d < 0:
        raise ValueError("--gsplat-eps2d must be >= 0")
    if args.gsplat_radius_clip < 0:
        raise ValueError("--gsplat-radius-clip must be >= 0")
    if args.backend == "native":
        # Fail before creating partial output directories or loading a large
        # point cloud when this machine is intentionally CPU-only.
        _native_renderer_dependencies()
    elif args.backend == "gsplat":
        _gsplat_renderer_dependency()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    gaussians, metadata = load_gaussians(args.gaussians, args.keep_ratio, args.min_opacity, args.crop_padding)
    source_intrinsics = source_intrinsics_at_output_resolution(metadata, args.height, args.width)
    if args.use_source_intrinsics and source_intrinsics is not None:
        output_fx, output_fy, output_cx, output_cy = source_intrinsics
        intrinsics_source = "embedded_source_camera"
    else:
        output_fx, output_fy = args.fx, args.fy
        output_cx = (args.width - 1) * 0.5 if args.cx is None else args.cx
        output_cy = (args.height - 1) * 0.5 if args.cy is None else args.cy
        intrinsics_source = "cli" if source_intrinsics is None else "cli_override"
    depth_pruning = None
    if args.prune_source_depth_outliers:
        gaussians, depth_pruning = prune_source_depth_outliers(
            gaussians,
            output_fx,
            output_fy,
            output_cx,
            output_cy,
            args.width,
            args.height,
            args.near,
            args.source_prune_grid_scale,
            args.source_prune_relative_tolerance,
            args.source_prune_absolute_tolerance,
        )
        print(
            "source-depth pruning: "
            f"removed {depth_pruning['removed_occluded_gaussians']:,} / "
            f"{depth_pruning['input_gaussians']:,} Gaussians"
        )
    cameras = read_cameras(args)
    if args.position_scale <= 0:
        raise ValueError("--position-scale must be > 0")
    if args.position_scale != 1.0:
        for camera in cameras:
            camera["position_xyz"] = [value * args.position_scale for value in camera["position_xyz"]]
    # The opacity-filtered median gives a stable, visible scene point to aim at.
    # It preserves the identity pose for the original camera in typical scenes.
    default_target = torch.quantile(gaussians["xyz"], 0.5, dim=0)
    args.output.mkdir(parents=True, exist_ok=True)
    rgb_dir, alpha_dir, depth_dir = (args.output / "rgb", args.output / "alpha", args.output / "depth")
    for directory in (rgb_dir, alpha_dir, depth_dir):
        directory.mkdir(parents=True, exist_ok=True)

    render_height, render_width = args.height * args.supersample, args.width * args.supersample
    fx, fy = output_fx * args.supersample, output_fy * args.supersample
    cx = (output_cx + 0.5) * args.supersample - 0.5
    cy = (output_cy + 0.5) * args.supersample - 0.5
    background = torch.tensor(args.background, dtype=torch.float32)
    report_cameras = []
    rgb_entries = []

    for index, camera in enumerate(cameras):
        if args.backend in {"native", "gsplat"}:
            torch.cuda.synchronize()
        start = time.perf_counter()
        transform, target = camera_transform(camera, default_target)
        camera_center = torch.tensor(camera["position_xyz"], dtype=torch.float32)
        if args.backend == "native":
            rgb, alpha, depth, visible_gaussians = rasterize_native_flash3d(
                gaussians, transform, camera_center, fx, fy, cx, cy,
                render_width, render_height, background, args.near, args.far,
                args.scale_modifier,
            )
        elif args.backend == "gsplat":
            rgb, alpha, depth, visible_gaussians = rasterize_gsplat(
                gaussians, transform, fx, fy, cx, cy,
                render_width, render_height, background, args.near, args.far,
                args.scale_modifier, args.gsplat_eps2d, args.gsplat_radius_clip,
            )
        elif args.backend in {"torch", "cpu"}:
            rgb, alpha, depth, visible_gaussians = rasterize_gsplat_cpu_reference(
                gaussians, transform, camera_center, fx, fy, cx, cy,
                render_width, render_height, background, args,
            )
        elif args.backend == "flash3d_torch":
            rgb, alpha, depth, visible_gaussians = rasterize_torch_flash3d(
                gaussians, transform, camera_center, fx, fy, cx, cy,
                render_width, render_height, background, args,
            )
        else:
            projected = project_gaussians(
                gaussians, transform, fx, fy, cx, cy, args.near, args.far,
                args.min_variance, args.sigma_cutoff, args.max_radius,
                args.scale_modifier, render_width, render_height, camera_center,
            )
            rgb, alpha, depth = rasterize(
                projected, render_height, render_width, background,
                args.tile_size, args.chunk_size, args.sigma_cutoff,
            )
            visible_gaussians = projected["z"].numel()
        if args.backend in {"native", "gsplat"}:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rgb = unsharp_mask(crop_and_downsample(rgb, args.height, args.width, args.supersample, args.crop_margin), args.sharpen)
        if args.linear_to_srgb:
            rgb = linear_to_srgb(rgb)
        alpha = crop_and_downsample(alpha, args.height, args.width, args.supersample, args.crop_margin).clamp(0, 1)
        depth = crop_and_downsample(depth, args.height, args.width, args.supersample, args.crop_margin)
        stem = f"{index:02d}_{camera['name']}"
        save_rgb(rgb, rgb_dir / f"{stem}.png")
        save_gray(alpha, alpha_dir / f"{stem}.png")
        save_gray(normalise_depth(depth, alpha), depth_dir / f"{stem}.png")
        rgb_path = rgb_dir / f"{stem}.png"
        rgb_entries.append((camera["name"], rgb_path))
        report_cameras.append({**camera, "look_at_xyz": target.tolist(), "world_to_camera": transform.tolist(), "visible_gaussians": visible_gaussians, "mean_alpha": alpha.mean().item(), "seconds": elapsed, "rgb": str(rgb_path.resolve())})
        print(f"{stem}: {visible_gaussians} visible Gaussians, {elapsed:.3f} s, mean alpha={alpha.mean().item():.4f}")

    comparison_grid = save_contact_sheet(rgb_entries, args.output, args.contact_sheet_columns, args.contact_sheet_title, args.contact_sheet_title_size)
    point_cloud = save_gaussian_point_cloud(gaussians, args.output, args.gaussians) if args.save_gaussians else None
    report = {
        "device": "cuda" if args.backend in {"native", "gsplat"} else "cpu",
        "renderer": (
            "Flash3D native diff-gaussian-rasterization CUDA backend"
            if args.backend == "native"
            else (
                "gsplat CUDA 3D Gaussian rasterizer"
                if args.backend == "gsplat"
                else (
                    "gsplat classic-semantics CPU reference renderer"
                    if args.backend in {"torch", "cpu"}
                    else (
                        "Flash3D portable PyTorch tile renderer (CPU)"
                        if args.backend == "flash3d_torch"
                        else "Legacy PyTorch anisotropic 3D Gaussian + front-to-back alpha blending"
                    )
                )
            )
        ),
        "backend": "torch" if args.backend == "cpu" else args.backend,
        "gaussians_input": str(args.gaussians.resolve()),
        "gaussians_after_filter": gaussians["xyz"].shape[0],
        "image_size_hw": [args.height, args.width],
        "intrinsics_at_output_resolution": {"fx": output_fx, "fy": output_fy, "cx": output_cx, "cy": output_cy, "source": intrinsics_source},
        "preset_rig": None if args.camera_file else args.rig,
        "camera_file": None if args.camera_file is None else str(args.camera_file.resolve()),
        "position_scale": args.position_scale,
        "source_depth_pruning": depth_pruning,
        "linear_to_srgb": args.linear_to_srgb,
        "default_look_at_xyz": default_target.tolist(),
        "comparison_grid": str(comparison_grid.resolve()),
        "contact_sheet_title": args.contact_sheet_title,
        "contact_sheet_title_size": args.contact_sheet_title_size,
        "point_cloud": point_cloud,
        "cameras": report_cameras,
        "mean_seconds_per_view": sum(camera["seconds"] for camera in report_cameras) / len(report_cameras),
        "rss_final_mb": psutil.Process().memory_info().rss / 1024**2,
    }
    (args.output / "multiview_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "camera_rig.json").write_text(json.dumps({"cameras": cameras}, indent=2), encoding="utf-8")
    print(f"Saved {len(cameras)} camera views, rig manifest, and report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
