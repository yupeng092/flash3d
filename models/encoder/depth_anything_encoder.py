"""Depth Anything V2 adapter for Flash3D's layered Gaussian decoder.

The adapter has the same output contract as :class:`UniDepthExtended`: it
produces a base depth, a source-camera intrinsic matrix, and then uses the
existing Flash3D ResNet/Gaussian heads.  Consequently the datasets, losses,
renderer and ``train.py`` do not need to be rewritten.

Install the official Depth-Anything-V2 source under ``third_party`` and place a
matching checkpoint in ``weights`` as configured in
``configs/model/depth/depth_anything_v2.yaml``.  The recommended first model is
the *metric* ViT-S checkpoint: Flash3D benefits from an absolute scale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.decoder.resnet_decoder import ResnetDepthDecoder, ResnetDecoder
from models.encoder.resnet_encoder import ResnetEncoder


DEPTH_ANYTHING_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
}


class IntrinsicsHead(nn.Module):
    """Small fallback camera head for single-image inference without K.

    It is intentionally optional.  RE10K/KITTI training batches already carry
    calibrated ``K_src`` and should use that exact geometry.  To train this
    head, set ``intrinsics_source: learned`` so render losses flow through it.
    """

    def __init__(self, initial_focal_ratio: float = 0.9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.predict = nn.Linear(32, 4)
        nn.init.zeros_(self.predict.weight)
        # log focal multipliers and principal-point offsets in normalized units.
        nn.init.constant_(self.predict.bias, 0.0)
        self.initial_focal_ratio = initial_focal_ratio

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = image.shape
        parameters = self.predict(self.features(image).flatten(1))
        base = max(height, width) * self.initial_focal_ratio
        fx = base * torch.exp(parameters[:, 0].clamp(-1.0, 1.0))
        fy = base * torch.exp(parameters[:, 1].clamp(-1.0, 1.0))
        cx = (width - 1) * 0.5 + parameters[:, 2].tanh() * width * 0.15
        cy = (height - 1) * 0.5 + parameters[:, 3].tanh() * height * 0.15
        intrinsics = torch.zeros((batch, 3, 3), dtype=image.dtype, device=image.device)
        intrinsics[:, 0, 0], intrinsics[:, 1, 1] = fx, fy
        intrinsics[:, 0, 2], intrinsics[:, 1, 2] = cx, cy
        intrinsics[:, 2, 2] = 1.0
        return intrinsics


class DepthAnythingV2Extended(nn.Module):
    """Drop-in lightweight-depth alternative to ``UniDepthExtended``."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        depth_cfg = cfg.model.depth
        encoder_name = depth_cfg.encoder
        if encoder_name not in DEPTH_ANYTHING_CONFIGS:
            raise ValueError(f"Unsupported Depth Anything encoder: {encoder_name}")

        project_root = Path(__file__).resolve().parents[2]
        source_dir = project_root / Path(depth_cfg.source_dir)
        checkpoint = project_root / Path(depth_cfg.checkpoint)
        if not source_dir.joinpath("depth_anything_v2", "dpt.py").is_file():
            raise FileNotFoundError(
                f"Depth Anything V2 source is missing at {source_dir}. "
                "Clone https://github.com/DepthAnything/Depth-Anything-V2 there."
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Depth Anything V2 checkpoint is missing: {checkpoint}"
            )
        sys.path.insert(0, str(source_dir))
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        finally:
            sys.path.pop(0)

        self.depth_model = DepthAnythingV2(encoder=encoder_name, **DEPTH_ANYTHING_CONFIGS[encoder_name])
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "model" in checkpoint_data:
            checkpoint_data = checkpoint_data["model"]
        info = self.depth_model.load_state_dict(checkpoint_data, strict=True)
        if info.missing_keys or info.unexpected_keys:
            raise RuntimeError(f"Depth Anything checkpoint mismatch: {info}")
        self.depth_model.requires_grad_(not depth_cfg.freeze)

        self.intrinsics_head = IntrinsicsHead(depth_cfg.initial_focal_ratio)
        self.encoder = ResnetEncoder(
            num_layers=cfg.model.backbone.num_layers,
            pretrained=cfg.model.backbone.weights_init == "pretrained",
            bn_order=cfg.model.backbone.resnet_bn_order,
        )
        if cfg.model.backbone.depth_cond:
            self.encoder.encoder.conv1 = nn.Conv2d(
                4,
                self.encoder.encoder.conv1.out_channels,
                kernel_size=self.encoder.encoder.conv1.kernel_size,
                padding=self.encoder.encoder.conv1.padding,
                stride=self.encoder.encoder.conv1.stride,
            )

        models = {}
        if cfg.model.gaussians_per_pixel > 1:
            models["depth"] = ResnetDepthDecoder(cfg=cfg, num_ch_enc=self.encoder.num_ch_enc)
        for index in range(cfg.model.gaussians_per_pixel):
            models[f"gauss_decoder_{index}"] = ResnetDecoder(cfg=cfg, num_ch_enc=self.encoder.num_ch_enc)
            if cfg.model.one_gauss_decoder:
                break
        self.models = nn.ModuleDict(models)

        self.parameters_to_train = [
            {"params": self.encoder.parameters()},
            {"params": self.models.parameters()},
        ]
        if not depth_cfg.freeze:
            self.parameters_to_train.append({"params": self.depth_model.parameters(), "lr": depth_cfg.finetune_lr})
        if depth_cfg.intrinsics_source == "learned":
            self.parameters_to_train.append({"params": self.intrinsics_head.parameters()})

    def get_parameter_groups(self):
        return self.parameters_to_train

    def _infer_depth(self, image: torch.Tensor) -> torch.Tensor:
        """ImageNet normalization and aspect-ratio-preserving multiple-of-14 input."""
        height, width = image.shape[-2:]
        long_side = int(self.cfg.model.depth.input_size)
        scale = long_side / max(height, width)
        resized_h = max(14, round(height * scale / 14) * 14)
        resized_w = max(14, round(width * scale / 14) * 14)
        resized = F.interpolate(image, (resized_h, resized_w), mode="bilinear", align_corners=False, antialias=True)
        normalised = (resized - image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None])
        normalised = normalised / image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        context = torch.enable_grad() if not self.cfg.model.depth.freeze else torch.no_grad()
        with context:
            depth = self.depth_model(normalised).unsqueeze(1)
        return F.interpolate(depth, (height, width), mode="bilinear", align_corners=False, antialias=True).clamp_min(
            self.cfg.model.depth.min_depth
        )

    def _source_intrinsics(self, inputs: dict, image: torch.Tensor) -> torch.Tensor:
        use_provided = self.cfg.model.depth.intrinsics_source == "provided"
        if use_provided and ("K_src", 0) in inputs:
            return inputs[("K_src", 0)].to(dtype=image.dtype)
        return self.intrinsics_head(image)

    def forward(self, inputs):
        image = inputs["color_aug", 0, 0]
        # Re10K already supports the historical ``unidepth`` key for an
        # externally precomputed frozen depth prior.  Reuse that transport key
        # so NPU pre-training can omit the frozen ViT-B forward pass entirely.
        cached_depth = inputs.get(("unidepth", 0, 0))
        if cached_depth is None:
            depth = self._infer_depth(image)
        else:
            depth = cached_depth.to(device=image.device, dtype=image.dtype).clamp_min(
                self.cfg.model.depth.min_depth
            )
        source_intrinsics = self._source_intrinsics(inputs, image)
        outputs_gauss = {
            ("K_src", 0): source_intrinsics,
            ("inv_K_src", 0): torch.linalg.inv(source_intrinsics),
        }

        conditioned_image = torch.cat((image, depth / self.cfg.model.depth.depth_normalizer), dim=1) if self.cfg.model.backbone.depth_cond else image
        encoded_features = self.encoder(conditioned_image)
        if self.cfg.model.gaussians_per_pixel > 1:
            depth_offsets = self.models["depth"](encoded_features)
            depth_offsets[("depth", 0)] = rearrange(
                depth_offsets[("depth", 0)], "(b n) ... -> b n ...", n=self.cfg.model.gaussians_per_pixel - 1
            )
            layered_depth = torch.cumsum(
                torch.cat((depth[:, None], depth_offsets[("depth", 0)]), dim=1), dim=1
            )
            outputs_gauss[("depth", 0)] = rearrange(
                layered_depth, "b n c h w -> (b n) c h w", n=self.cfg.model.gaussians_per_pixel
            )
        else:
            outputs_gauss[("depth", 0)] = depth

        gaussian_outputs = {}
        for index in range(self.cfg.model.gaussians_per_pixel):
            prediction = self.models[f"gauss_decoder_{index}"](encoded_features)
            if self.cfg.model.one_gauss_decoder:
                gaussian_outputs |= prediction
                break
            for key, value in prediction.items():
                gaussian_outputs[key] = value if index == 0 else torch.cat((gaussian_outputs[key], value), dim=1)
        for key, value in gaussian_outputs.items():
            gaussian_outputs[key] = rearrange(value, "b n ... -> (b n) ...")
        return outputs_gauss | gaussian_outputs
