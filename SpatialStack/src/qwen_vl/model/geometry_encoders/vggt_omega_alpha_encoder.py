"""VGGT-Omega alpha encoder that exposes camera + register tokens only."""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional

import torch

from .base import BaseGeometryEncoder, GeometryEncoderConfig
from .vggt_omega_encoder import (
    _extract_model_state_dict,
    _import_vggt_omega_model,
    _resolve_vggt_omega_checkpoint,
)


_SUPPORTED_LAYER_INDICES = {4, 11, 17, 23}


class VGGTOmegaAlphaEncoder(BaseGeometryEncoder):
    """Frozen VGGT-Omega wrapper that returns per-frame camera + scene tokens."""

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)

        vggt_omega_cls = _import_vggt_omega_model()
        self.vggt_omega = vggt_omega_cls(
            enable_camera=False,
            enable_depth=False,
            enable_alignment=False,
        )
        if self.freeze_encoder:
            for param in self.vggt_omega.parameters():
                param.requires_grad = False

        self.patch_size = 16
        self.num_special_tokens = 17

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_layers(images, layer_indices=[23])[0]

    def encode_layers(
        self,
        images: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        spatial_merge_size: Optional[int] = None,
        include_camera_token: bool = False,
    ):
        del spatial_merge_size
        del include_camera_token

        self.vggt_omega.eval()
        images = self._apply_reference_frame_transform(images)
        dtype, autocast_context = self._autocast_context(images)

        with torch.no_grad():
            with autocast_context:
                aggregated_tokens_list, patch_token_start = self.vggt_omega.aggregator(images[None])

        if layer_indices is None:
            layer_indices = [23]

        tensor_features = []
        for layer_idx in layer_indices:
            if layer_idx not in _SUPPORTED_LAYER_INDICES:
                raise ValueError(
                    f"Unsupported VGGT-Omega alpha layer index {layer_idx}. "
                    f"Supported cached layers: {sorted(_SUPPORTED_LAYER_INDICES)}."
                )

            tokens = aggregated_tokens_list[layer_idx]
            if tokens is None:
                raise ValueError(
                    f"VGGT-Omega aggregator did not cache layer {layer_idx}. "
                    f"Supported cached layers: {sorted(_SUPPORTED_LAYER_INDICES)}."
                )

            tokens = self._apply_inverse_reference_frame_transform(tokens[0])
            special_tokens = tokens[:, :patch_token_start]
            if special_tokens.shape[1] != self.num_special_tokens:
                raise ValueError(
                    "VGGT-Omega alpha path expected 17 special tokens per frame, "
                    f"but got {special_tokens.shape[1]}."
                )
            tensor_features.append(special_tokens.to(dtype).contiguous())

        return tensor_features

    def _autocast_context(self, images: torch.Tensor):
        if not torch.cuda.is_available():
            return images.dtype, nullcontext()

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return dtype, torch.amp.autocast("cuda", dtype=dtype)

    def get_feature_dim(self) -> int:
        return 2048

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)

    def _apply_reference_frame_transform(self, images: torch.Tensor) -> torch.Tensor:
        if self.reference_frame != "first":
            return torch.flip(images, dims=(0,))
        return images

    def _apply_inverse_reference_frame_transform(self, features: torch.Tensor) -> torch.Tensor:
        if self.reference_frame != "first":
            return torch.flip(features, dims=(0,))
        return features

    def load_model(self, model_path: str) -> None:
        checkpoint_path = _resolve_vggt_omega_checkpoint(model_path)
        raw_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = _extract_model_state_dict(raw_state_dict)

        expected_state_dict = self.vggt_omega.state_dict()
        filtered_state_dict = {
            key: value for key, value in state_dict.items() if key in expected_state_dict
        }
        missing_keys = sorted(set(expected_state_dict.keys()) - set(filtered_state_dict.keys()))
        if missing_keys:
            raise RuntimeError(
                "VGGT-Omega checkpoint is missing weights required by the SpatialStack alpha adapter. "
                f"Missing keys include: {missing_keys[:8]}"
            )

        self.vggt_omega.load_state_dict(filtered_state_dict, strict=True)
        if self.freeze_encoder:
            for param in self.vggt_omega.parameters():
                param.requires_grad = False
