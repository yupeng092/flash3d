#!/usr/bin/env python3
"""Compare the portable Flash3D renderer on CPU and Ascend NPU.

The test checks output agreement and verifies gradients for every Gaussian
attribute trained by Flash3D. It is deliberately small and completes before a
costly RE10K run is started.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.decoder.npu_differentiable_renderer import render_predicted_torch


def config() -> SimpleNamespace:
    return SimpleNamespace(
        dataset=SimpleNamespace(znear=0.01, zfar=20.0),
        model=SimpleNamespace(
            npu_renderer_min_variance=0.30,
            npu_renderer_sigma_cutoff=3.0,
            npu_renderer_max_radius=24.0,
            npu_renderer_max_gaussians=0,
            npu_renderer_tile_size=8,
            npu_renderer_tile_span=7,
            npu_renderer_max_gaussians_per_tile=64,
        ),
    )


def make_point_cloud(device: torch.device) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    n = 24
    return {
        "xyz": torch.cat((torch.randn(n, 2) * 0.3, torch.rand(n, 1) * 2 + 2), dim=1).to(device).requires_grad_(),
        "opacity": (torch.rand(n, 1) * 0.5 + 0.3).to(device).requires_grad_(),
        "scaling": (torch.rand(n, 3) * 0.08 + 0.03).to(device).requires_grad_(),
        "rotation": torch.cat((torch.ones(n, 1), torch.randn(n, 3) * 0.05), dim=1).to(device).requires_grad_(),
        "features_dc": torch.randn(n, 1, 3).to(device).requires_grad_(),
    }


def run(device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    pc = make_point_cloud(device)
    matrix = torch.eye(4, device=device)
    result = render_predicted_torch(
        config(), pc, matrix, matrix, matrix, torch.zeros(3, device=device),
        (1.0, 1.0), (32, 40), torch.tensor([0.5, 0.5, 0.5], device=device), 0,
    )
    (result["render"].mean() + 0.01 * result["depth"].mean()).backward()
    for name, value in pc.items():
        if value.grad is None or not torch.isfinite(value.grad).all():
            raise AssertionError(f"Missing or non-finite gradient for {name}")
    return result, pc


def main() -> None:
    try:
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise SystemExit("torch_npu is required for this test.") from error
    if not torch.npu.is_available():
        raise SystemExit("No Ascend NPU is available.")
    cpu_result, _ = run(torch.device("cpu"))
    npu_result, _ = run(torch.device("npu:0"))
    torch.npu.synchronize()
    rgb_error = (cpu_result["render"] - npu_result["render"].cpu()).abs().max().item()
    alpha_error = (cpu_result["alpha"] - npu_result["alpha"].cpu()).abs().max().item()
    if rgb_error > 5e-3 or alpha_error > 5e-3:
        raise AssertionError(f"CPU/NPU mismatch: max RGB={rgb_error:.6f}, alpha={alpha_error:.6f}")
    print(f"renderer test passed: max_rgb_error={rgb_error:.6f}, max_alpha_error={alpha_error:.6f}")


if __name__ == "__main__":
    main()
