"""VGGT-Omega geometry encoder implementation."""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import torch
from huggingface_hub import snapshot_download

from .base import BaseGeometryEncoder, GeometryEncoderConfig


_DEFAULT_VGGT_OMEGA_CHECKPOINT = "vggt_omega_1b_512.pt"
_SUPPORTED_LAYER_INDICES = {4, 11, 17, 23}


def _resolve_vggt_omega_repo_root() -> Path:
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        candidate = parent / "vggt-omega" / "vggt_omega"
        if candidate.is_dir():
            return candidate.parent
    raise RuntimeError(
        "Unable to locate the sibling `vggt-omega/` repository. "
        "Expected it to exist next to the SpatialStack workspace root."
    )


def _import_vggt_omega_model():
    repo_root = _resolve_vggt_omega_repo_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    try:
        from vggt_omega.models import VGGTOmega
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import `vggt_omega`. Make sure the sibling repository "
            "`vggt-omega/` is present and its Python package can be imported."
        ) from exc

    return VGGTOmega


def _resolve_vggt_omega_checkpoint(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    if path.is_file():
        return path

    if path.is_dir():
        preferred = path / _DEFAULT_VGGT_OMEGA_CHECKPOINT
        if preferred.is_file():
            return preferred
        pt_files = sorted(path.glob("*.pt"))
        if len(pt_files) == 1:
            return pt_files[0]
        if not pt_files:
            raise FileNotFoundError(
                f"No `.pt` checkpoint found under VGGT-Omega directory: {model_path}"
            )
        raise FileNotFoundError(
            f"Multiple `.pt` checkpoints found under {model_path}; "
            f"please pass the exact file path or include {_DEFAULT_VGGT_OMEGA_CHECKPOINT}."
        )

    cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HUB_CACHE")
    snapshot_dir = snapshot_download(
        repo_id=model_path,
        repo_type="model",
        cache_dir=cache_dir,
        allow_patterns=[_DEFAULT_VGGT_OMEGA_CHECKPOINT],
    )
    checkpoint_path = Path(snapshot_dir) / _DEFAULT_VGGT_OMEGA_CHECKPOINT
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Unable to resolve `{_DEFAULT_VGGT_OMEGA_CHECKPOINT}` from HF repo `{model_path}`."
        )
    return checkpoint_path


class VGGTOmegaEncoder(BaseGeometryEncoder):
    """VGGT-Omega geometry encoder wrapper."""

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

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images and return the final cached patch tokens."""
        return self.encode_layers(images, layer_indices=[23])[0]

    def encode_layers(
        self,
        images: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        spatial_merge_size: int = 1,
        include_camera_token: bool = False,
    ):
        """Encode images and return features from specific Omega aggregator layers."""
        del spatial_merge_size

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
                    f"Unsupported VGGT-Omega layer index {layer_idx}. "
                    f"Supported cached layers: {sorted(_SUPPORTED_LAYER_INDICES)}."
                )

            tokens = aggregated_tokens_list[layer_idx]
            if tokens is None:
                raise ValueError(
                    f"VGGT-Omega aggregator did not cache layer {layer_idx}. "
                    f"Supported cached layers: {sorted(_SUPPORTED_LAYER_INDICES)}."
                )

            tokens = self._apply_inverse_reference_frame_transform(tokens[0])
            patch_tokens = tokens[:, patch_token_start:]
            if include_camera_token:
                camera_token = tokens[:, 0:1]
                patch_tokens = torch.cat([camera_token, patch_tokens], dim=1)
            tensor_features.append(patch_tokens.to(dtype).contiguous())

        return tensor_features

    def _autocast_context(self, images: torch.Tensor):
        if not torch.cuda.is_available():
            return images.dtype, nullcontext()

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return dtype, torch.amp.autocast("cuda", dtype=dtype)

    def get_feature_dim(self) -> int:
        """Get VGGT-Omega feature dimension."""
        return 2048

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for compatibility."""
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
        """Load pretrained VGGT-Omega checkpoint from local path or HF repo id."""
        checkpoint_path = _resolve_vggt_omega_checkpoint(model_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.vggt_omega.load_state_dict(state_dict, strict=True)

        if self.freeze_encoder:
            for param in self.vggt_omega.parameters():
                param.requires_grad = False
