"""Portable differentiable 3D Gaussian renderer for NPU pre-training.

This is intentionally implemented with standard PyTorch tensor operators so
that ``torch_npu`` can lower it to Ascend kernels.  It is not the endpoint
renderer: unlike an AscendC fused kernel it keeps autograd state and is meant
only for Flash3D's reconstruction loss during pre-training.

The implementation bins Gaussians into screen-space tiles, depth-sorts the
Gaussians in each tile, and composites them front-to-back.  Tile assignment and
depth ordering are discrete, just as in the original 3DGS rasterizer; gradients
are propagated through projection, covariance, opacity and colour for the
Gaussians selected by those operations.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199


def _model_option(cfg: Any, name: str, default: Any) -> Any:
    """Read an optional renderer setting from Hydra/OmegaConf config."""
    return getattr(cfg.model, name, default)


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert Flash3D's scalar-first quaternion to a differentiable matrix."""
    quaternion = F.normalize(quaternion.float(), dim=-1, eps=1e-8)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def _evaluate_colour(
    pc: dict[str, torch.Tensor], camera_center: torch.Tensor, max_sh_degree: int
) -> torch.Tensor:
    """Evaluate DC/degree-1 SH colours in the same convention as 3DGS."""
    if "features_dc" not in pc:
        raise KeyError("pc must contain features_dc")
    dc = pc["features_dc"].float()
    if dc.ndim == 3:
        dc = dc[:, 0]
    if dc.shape[-1] != 3:
        raise ValueError("features_dc must have three colour channels")

    colour = _SH_C0 * dc
    if max_sh_degree > 0 and "features_rest" in pc:
        rest = pc["features_rest"].float()
        # Flash3D's default is degree 1 (three non-DC terms).  Higher-degree
        # terms are deliberately ignored in this pre-training renderer.
        if rest.ndim == 3 and rest.shape[1] >= 3:
            direction = F.normalize(
                pc["xyz"].float() - camera_center.float().reshape(1, 3),
                dim=-1,
                eps=1e-8,
            )
            x, y, z = direction.unbind(dim=-1)
            colour = (
                colour
                - _SH_C1 * y[:, None] * rest[:, 0]
                + _SH_C1 * z[:, None] * rest[:, 1]
                - _SH_C1 * x[:, None] * rest[:, 2]
            )
    # This matches 3DGS's SH-to-RGB offset.  Do not clamp the upper end during
    # training: saturation there would eliminate useful colour gradients.
    return (colour + 0.5).clamp_min(0.0)


def _project_gaussians(
    cfg: Any,
    pc: dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,
    camera_center: torch.Tensor,
    fov: tuple[float, float],
    image_size: tuple[int, int],
    max_sh_degree: int,
    scale_modifier: float,
    override_color: torch.Tensor | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Project 3D Gaussians and return visible, depth-sorted candidates."""
    height, width = image_size
    fov_x, fov_y = fov
    device = pc["xyz"].device
    xyz = pc["xyz"].float()
    n_gaussians = xyz.shape[0]
    near = float(getattr(cfg.dataset, "znear", 0.01))
    far = float(getattr(cfg.dataset, "zfar", 100.0))
    min_variance = float(_model_option(cfg, "npu_renderer_min_variance", 0.30))
    sigma_cutoff = float(_model_option(cfg, "npu_renderer_sigma_cutoff", 3.0))
    max_radius = float(_model_option(cfg, "npu_renderer_max_radius", 32.0))

    # Flash3D passes matrices transposed for the CUDA renderer.  With row-wise
    # points the equivalent camera transform is therefore ``points @ matrix``.
    homogeneous = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=-1)
    camera_xyz = (homogeneous @ world_view_transform.float())[:, :3]
    x, y, z = camera_xyz.unbind(dim=-1)
    focal_x = width / (2.0 * math.tan(fov_x * 0.5))
    focal_y = height / (2.0 * math.tan(fov_y * 0.5))
    inverse_z = z.clamp_min(near).reciprocal()
    u = focal_x * x * inverse_z + (width - 1.0) * 0.5
    v = focal_y * y * inverse_z + (height - 1.0) * 0.5

    scales = pc["scaling"].float().clamp_min(1e-7) * scale_modifier
    rotation = _quaternion_to_matrix(pc["rotation"])
    basis = rotation * scales[:, None, :]
    covariance_world = basis @ basis.transpose(-1, -2)
    camera_rotation = world_view_transform[:3, :3].float().transpose(0, 1)
    covariance_camera = (
        camera_rotation[None]
        @ covariance_world
        @ camera_rotation.transpose(0, 1)[None]
    )

    jacobian = torch.zeros((n_gaussians, 2, 3), device=device, dtype=torch.float32)
    jacobian[:, 0, 0] = focal_x * inverse_z
    jacobian[:, 0, 2] = -focal_x * x * inverse_z.square()
    jacobian[:, 1, 1] = focal_y * inverse_z
    jacobian[:, 1, 2] = -focal_y * y * inverse_z.square()
    covariance_2d = jacobian @ covariance_camera @ jacobian.transpose(-1, -2)
    covariance_2d[:, 0, 0] = covariance_2d[:, 0, 0] + min_variance
    covariance_2d[:, 1, 1] = covariance_2d[:, 1, 1] + min_variance
    a, b, c = covariance_2d[:, 0, 0], covariance_2d[:, 0, 1], covariance_2d[:, 1, 1]
    determinant = (a * c - b.square()).clamp_min(1e-10)
    inverse_covariance = torch.stack((c / determinant, -b / determinant, a / determinant), dim=-1)
    largest_eigenvalue = 0.5 * (a + c + torch.sqrt(((a - c).square() + 4 * b.square()).clamp_min(0.0)))
    radius = (sigma_cutoff * torch.sqrt(largest_eigenvalue.clamp_min(0.0))).clamp(max=max_radius)

    visible = (
        (z > near)
        & (z < far)
        & torch.isfinite(inverse_covariance).all(dim=-1)
        & (u + radius >= 0)
        & (u - radius < width)
        & (v + radius >= 0)
        & (v - radius < height)
    )
    visible_indices = torch.where(visible)[0]
    opacity = pc["opacity"].float().reshape(-1).clamp(0.0, 0.999)
    max_gaussians = int(_model_option(cfg, "npu_renderer_max_gaussians", 0))
    if max_gaussians > 0 and visible_indices.numel() > max_gaussians:
        # A curriculum safeguard for the first NPU implementation.  It keeps
        # the most opaque splats, while the final AscendC inference renderer
        # should use all Gaussians.
        keep = torch.topk(opacity[visible_indices], k=max_gaussians, sorted=False).indices
        visible_indices = visible_indices[keep]

    radii = torch.zeros(n_gaussians, device=device, dtype=torch.float32)
    radii = radii.scatter(0, torch.where(visible)[0], radius[visible])
    colours = override_color.float() if override_color is not None else _evaluate_colour(pc, camera_center, max_sh_degree)
    projected = {
        "u": u[visible_indices],
        "v": v[visible_indices],
        "z": z[visible_indices],
        "radius": radius[visible_indices],
        "inverse": inverse_covariance[visible_indices],
        "opacity": opacity[visible_indices],
        "colour": colours[visible_indices],
    }
    return projected, radii, visible


def _bin_to_tiles(
    projected: dict[str, torch.Tensor], height: int, width: int, tile_size: int, tile_span: int
) -> tuple[torch.Tensor, list[int], list[int], int]:
    """Return Gaussian indices grouped by tile without a point-by-pixel tensor."""
    n_tiles_x = (width + tile_size - 1) // tile_size
    n_tiles_y = (height + tile_size - 1) // tile_size
    n_tiles = n_tiles_x * n_tiles_y
    n = projected["u"].numel()
    if n == 0:
        return torch.empty(0, device=projected["u"].device, dtype=torch.long), [0] * n_tiles, [0] * n_tiles, n_tiles_x
    if tile_span < 1 or tile_span % 2 == 0:
        raise ValueError("npu_renderer_tile_span must be a positive odd number")

    tile_x_min = torch.div((projected["u"] - projected["radius"]).floor().long(), tile_size, rounding_mode="floor").clamp(0, n_tiles_x - 1)
    tile_x_max = torch.div((projected["u"] + projected["radius"]).floor().long(), tile_size, rounding_mode="floor").clamp(0, n_tiles_x - 1)
    tile_y_min = torch.div((projected["v"] - projected["radius"]).floor().long(), tile_size, rounding_mode="floor").clamp(0, n_tiles_y - 1)
    tile_y_max = torch.div((projected["v"] + projected["radius"]).floor().long(), tile_size, rounding_mode="floor").clamp(0, n_tiles_y - 1)
    half_span = tile_span // 2
    offsets = torch.arange(-half_span, half_span + 1, device=projected["u"].device)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    offset_x, offset_y = offset_x.reshape(1, -1), offset_y.reshape(1, -1)
    centre_x = torch.div(projected["u"].floor().long(), tile_size, rounding_mode="floor")[:, None]
    centre_y = torch.div(projected["v"].floor().long(), tile_size, rounding_mode="floor")[:, None]
    candidate_x, candidate_y = centre_x + offset_x, centre_y + offset_y
    inside = (
        (candidate_x >= tile_x_min[:, None])
        & (candidate_x <= tile_x_max[:, None])
        & (candidate_y >= tile_y_min[:, None])
        & (candidate_y <= tile_y_max[:, None])
        & (candidate_x >= 0)
        & (candidate_x < n_tiles_x)
        & (candidate_y >= 0)
        & (candidate_y < n_tiles_y)
    )
    gaussian_indices = torch.arange(n, device=projected["u"].device)[:, None].expand_as(candidate_x)[inside]
    tile_ids = (candidate_y * n_tiles_x + candidate_x)[inside]
    order = torch.argsort(tile_ids)
    counts = torch.bincount(tile_ids, minlength=n_tiles).detach().cpu().tolist()
    starts: list[int] = []
    cursor = 0
    for count in counts:
        starts.append(cursor)
        cursor += count
    return gaussian_indices[order], starts, counts, n_tiles_x


def _rasterize_tiles(
    cfg: Any,
    projected: dict[str, torch.Tensor],
    height: int,
    width: int,
    background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiably composite a binned projected Gaussian set."""
    tile_size = int(_model_option(cfg, "npu_renderer_tile_size", 16))
    tile_span = int(_model_option(cfg, "npu_renderer_tile_span", 5))
    max_per_tile = int(_model_option(cfg, "npu_renderer_max_gaussians_per_tile", 128))
    sigma_cutoff = float(_model_option(cfg, "npu_renderer_sigma_cutoff", 3.0))
    if tile_size < 1 or max_per_tile < 1:
        raise ValueError("NPU renderer tile_size and max_gaussians_per_tile must be positive")
    device = projected["u"].device
    background = background.to(device=device, dtype=torch.float32).reshape(3, 1, 1)
    rgb = background.expand(3, height, width).clone()
    alpha_image = torch.zeros((height, width), device=device, dtype=torch.float32)
    depth_image = torch.zeros((height, width), device=device, dtype=torch.float32)
    sorted_indices, starts, counts, tiles_x = _bin_to_tiles(projected, height, width, tile_size, tile_span)
    cutoff_squared = sigma_cutoff * sigma_cutoff

    for tile_id, (start, count) in enumerate(zip(starts, counts)):
        if count == 0:
            continue
        selected = sorted_indices[start : start + count]
        if selected.numel() > max_per_tile:
            keep = torch.topk(projected["opacity"][selected], k=max_per_tile, sorted=False).indices
            selected = selected[keep]
        selected = selected[torch.argsort(projected["z"][selected])]
        tile_y, tile_x = divmod(tile_id, tiles_x)
        x0, y0 = tile_x * tile_size, tile_y * tile_size
        x1, y1 = min(x0 + tile_size, width), min(y0 + tile_size, height)
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, device=device, dtype=torch.float32),
            torch.arange(x0, x1, device=device, dtype=torch.float32),
            indexing="ij",
        )
        dx = xx.reshape(1, -1) - projected["u"][selected, None]
        dy = yy.reshape(1, -1) - projected["v"][selected, None]
        inverse = projected["inverse"][selected]
        mahalanobis = (
            inverse[:, 0, None] * dx.square()
            + 2.0 * inverse[:, 1, None] * dx * dy
            + inverse[:, 2, None] * dy.square()
        )
        alpha = projected["opacity"][selected, None] * torch.exp(-0.5 * mahalanobis)
        alpha = torch.where(mahalanobis <= cutoff_squared, alpha, torch.zeros_like(alpha)).clamp(0.0, 0.999)
        one_minus_alpha = 1.0 - alpha
        exclusive_transmittance = torch.cumprod(
            torch.cat((torch.ones_like(one_minus_alpha[:1]), one_minus_alpha[:-1]), dim=0), dim=0
        )
        weights = alpha * exclusive_transmittance
        final_transmittance = one_minus_alpha.prod(dim=0)
        tile_rgb = weights.transpose(0, 1) @ projected["colour"][selected]
        tile_rgb = tile_rgb + final_transmittance[:, None] * background[:, 0, 0][None]
        tile_alpha = 1.0 - final_transmittance
        tile_depth = (weights * projected["z"][selected, None]).sum(dim=0)
        tile_depth = torch.where(tile_alpha > 1e-6, tile_depth / tile_alpha.clamp_min(1e-6), torch.zeros_like(tile_depth))
        rgb[:, y0:y1, x0:x1] = tile_rgb.transpose(0, 1).reshape(3, y1 - y0, x1 - x0)
        alpha_image[y0:y1, x0:x1] = tile_alpha.reshape(y1 - y0, x1 - x0)
        depth_image[y0:y1, x0:x1] = tile_depth.reshape(y1 - y0, x1 - x0)
    return rgb, depth_image, alpha_image


def render_predicted_torch(
    cfg: Any,
    pc: dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,
    full_proj_transform: torch.Tensor,
    proj_mtrx: torch.Tensor,
    camera_center: torch.Tensor,
    fov: tuple[float, float],
    img_size: tuple[int, int],
    bg_color: torch.Tensor,
    max_sh_degree: int,
    scaling_modifier: float = 1.0,
    override_color: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Render one Flash3D sample using differentiable standard Torch/NPU ops.

    ``full_proj_transform`` and ``proj_mtrx`` are retained for drop-in API
    compatibility with ``render_predicted``.  Projection is expressed from the
    view matrix and FOV to keep it portable across torch_npu releases.
    """
    del full_proj_transform, proj_mtrx
    projected, radii, visibility = _project_gaussians(
        cfg, pc, world_view_transform, camera_center, fov, img_size,
        max_sh_degree, scaling_modifier, override_color,
    )
    rendered_image, rendered_depth, rendered_alpha = _rasterize_tiles(
        cfg, projected, img_size[0], img_size[1], bg_color
    )
    # A tensor rather than None keeps Flash3D's logging/splitting API stable.
    viewspace_points = torch.stack(
        (projected["u"], projected["v"], projected["z"]), dim=-1
    )
    return {
        "render": rendered_image,
        "depth": rendered_depth,
        "alpha": rendered_alpha,
        "opacity": pc["opacity"],
        "viewspace_points": viewspace_points,
        "visibility_filter": visibility,
        "radii": radii,
    }
