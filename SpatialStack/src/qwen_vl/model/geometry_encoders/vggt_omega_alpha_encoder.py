"""VGGT-Omega alpha geometry encoder implementation."""

from __future__ import annotations

from typing import Optional

import torch

from .base import GeometryEncoderConfig
from .vggt_omega_encoder import (
    VGGTOmegaEncoder,
    _SUPPORTED_LAYER_INDICES,
)


class VGGTOmegaAlphaEncoder(VGGTOmegaEncoder):
    """Frozen VGGT-Omega wrapper that exposes camera plus register tokens only."""

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_camera_and_register_tokens(images)

    def encode_camera_and_register_tokens(
        self,
        images: torch.Tensor,
        layer_index: Optional[int] = 23,
    ) -> torch.Tensor:
        self.vggt_omega.eval()
        images = self._apply_reference_frame_transform(images)

        dtype, autocast_context = self._autocast_context(images)

        with torch.no_grad():
            with autocast_context:
                aggregated_tokens_list, patch_token_start = self.vggt_omega.aggregator(images[None])

        if layer_index is None:
            layer_index = 23
        if layer_index not in _SUPPORTED_LAYER_INDICES:
            raise ValueError(
                f"Unsupported VGGT-Omega alpha layer index {layer_index}. "
                f"Supported cached layers: {sorted(_SUPPORTED_LAYER_INDICES)}."
            )

        tokens = aggregated_tokens_list[layer_index]
        if tokens is None:
            raise ValueError(
                f"VGGT-Omega aggregator did not cache layer {layer_index}. "
                f"Supported cached layers: {sorted(_SUPPORTED_LAYER_INDICES)}."
            )

        tokens = self._apply_inverse_reference_frame_transform(tokens[0])
        camera_and_register_tokens = tokens[:, :patch_token_start]
        return camera_and_register_tokens.to(dtype).contiguous()

