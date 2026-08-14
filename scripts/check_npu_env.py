#!/usr/bin/env python3
"""Fail-fast Ascend NPU environment check for Flash3D single-card training."""

from __future__ import annotations

import json
import os
import platform

import torch


def main() -> None:
    try:
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise SystemExit("torch_npu is unavailable; install the CANN-matched wheel first.") from error
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("torch_npu imported, but no usable NPU is visible. Check driver/CANN and ASCEND_RT_VISIBLE_DEVICES.")

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.npu.empty_cache()
    if hasattr(torch.npu, "reset_peak_memory_stats"):
        torch.npu.reset_peak_memory_stats(device)
    x = torch.randn(1024, 1024, device=device, dtype=torch.float16, requires_grad=True)
    y = (x @ x).float().mean()
    y.backward()
    torch.npu.synchronize(device)
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "npu_count": torch.npu.device_count(),
        "current_device": torch.npu.current_device(),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "all"),
        "ascend_home": os.environ.get("ASCEND_HOME_PATH"),
        "ascend_opp": os.environ.get("ASCEND_OPP_PATH"),
        "matmul_loss": float(y.detach().cpu()),
    }
    if hasattr(torch.npu, "max_memory_allocated"):
        report["peak_allocated_mib"] = round(torch.npu.max_memory_allocated(device) / 2**20, 2)
    print(json.dumps(report, indent=2))
    print("NPU forward/backward smoke test passed.")


if __name__ == "__main__":
    main()
