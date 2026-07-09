"""Qwen3.5 adapters for the Qwen3-VL feature extraction pipeline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import torch

try:
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
except ImportError as exc:
    raise ImportError(
        "Qwen3.5 feature extraction requires transformers==5.3.0; "
        "install the version used by SpatialStack."
    ) from exc

from ..qwen3vl.qwen3vl_extractor import Qwen3VLExtractor


def _ensure_spatialstack_imports() -> None:
    workspace_root = Path(__file__).resolve().parents[3]
    for path in (workspace_root / "SpatialStack" / "src", workspace_root / "vggt-omega"):
        path_str = str(path)
        if path.is_dir() and path_str not in sys.path:
            sys.path.insert(0, path_str)


class Qwen35Extractor(Qwen3VLExtractor):
    """Extract Qwen3.5 language-layer states at visual token positions."""

    model_class = Qwen3_5ForConditionalGeneration
    model_family = "Qwen3.5"

    def __init__(
        self,
        model_path: str,
        select_layers: List[int],
        question: str = "",
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(model_path)
        if getattr(config, "use_geometry_encoder", False):
            raise ValueError(
                "Geometry-enabled SpatialStack checkpoints require a dedicated "
                "extractor that supplies geometry_encoder_inputs; loading one as "
                "plain Qwen3.5 would not measure SpatialStack features."
            )
        super().__init__(model_path, select_layers, question, **kwargs)


class SpatialStackQwen35Extractor(Qwen3VLExtractor):
    """Extract Qwen3.5 LLM hidden states with SpatialStack geometry injection active."""

    model_family = "SpatialStack-Qwen3.5"

    def __init__(
        self,
        model_path: str,
        select_layers: List[int],
        question: str = "",
        geometry_encoder_path: Optional[str] = None,
        geometry_encoder_type: Optional[str] = None,
        **kwargs,
    ):
        _ensure_spatialstack_imports()
        config = AutoConfig.from_pretrained(model_path)
        if not getattr(config, "use_geometry_encoder", False):
            raise ValueError(
                "spatialstack-qwen35 requires a geometry-enabled SpatialStack checkpoint "
                "with config.use_geometry_encoder=True."
            )
        if geometry_encoder_path is not None:
            setattr(config, "geometry_encoder_path", geometry_encoder_path)
        if geometry_encoder_type is not None:
            setattr(config, "geometry_encoder_type", geometry_encoder_type)
        if getattr(config, "geometry_encoder_type", "vggt") == "vggt_omega_alpha":
            raise ValueError(
                "spatialstack-qwen35 extraction currently supports the Phase-1 "
                "VGGT/VGGT-Omega geometry path, not vggt_omega_alpha token expansion."
            )

        from qwen_vl.model.modeling_qwen3_5 import (
            Qwen3_5ForConditionalGenerationWithGeometry,
        )

        self.model_class = Qwen3_5ForConditionalGenerationWithGeometry
        self._spatialstack_config = config
        self.geometry_encoder_type = getattr(config, "geometry_encoder_type", "vggt")
        super().__init__(model_path, select_layers, question, **kwargs)

    def _load_model(self, model_path: str):
        return self.model_class.from_pretrained(
            model_path,
            config=self._spatialstack_config,
            torch_dtype=self.torch_dtype,
            attn_implementation=self.attn_implementation,
            device_map=None,
            low_cpu_mem_usage=False,
        )

    def _forward_with_spatialstack_hidden_states(
        self,
        inputs_on_device: Dict[str, torch.Tensor],
        visual_mask_2d: torch.Tensor,
        geometry_encoder_inputs: List[torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        layer_outputs: Dict[int, torch.Tensor] = {}
        handles = []
        language_model = self.model.model.language_model
        if getattr(self.model.model, "language_feature_fusion", None) is None:
            self.model.model.initialize_geometry_modules()

        def make_layer_hook(layer_number: int):
            def hook(_module, _args, output):
                layer_outputs[layer_number] = output[0] if isinstance(output, tuple) else output

            return hook

        for idx, layer in enumerate(language_model.layers, start=1):
            handles.append(layer.register_forward_hook(make_layer_hook(idx)))

        def fusion_hook(_module, args, output):
            if len(args) < 3:
                return
            layer_number = int(args[2]) + 1
            base = layer_outputs.get(layer_number)
            if base is None:
                return
            patched = base.clone()
            patched[visual_mask_2d] = output
            layer_outputs[layer_number] = patched

        fusion_module = getattr(self.model.model, "language_feature_fusion", None)
        if fusion_module is not None:
            handles.append(fusion_module.register_forward_hook(fusion_hook))

        def norm_hook(_module, _args, output):
            layer_outputs[self.num_layers] = output

        handles.append(language_model.norm.register_forward_hook(norm_hook))
        try:
            self.model.model(
                **inputs_on_device,
                geometry_encoder_inputs=geometry_encoder_inputs,
            )
        finally:
            for handle in handles:
                handle.remove()

        missing = [layer for layer in self.select_layers if layer not in layer_outputs]
        if missing:
            raise ValueError(f"SpatialStack did not expose hidden states for layers: {missing}")
        return {layer: layer_outputs[layer] for layer in self.select_layers}

    @torch.no_grad()
    def forward_with_hidden_states(
        self,
        images: List["Image.Image"],
        video_parity: bool = True,
    ):
        from qwen_vl.data.utils import build_qwen3_5_geometry_inputs

        messages = self.build_messages(images)
        extra_kwargs = {}
        if video_parity and len(images) > 1:
            per_min, per_max = self._compute_per_image_pixels(len(images))
            extra_kwargs["min_pixels"] = per_min
            extra_kwargs["max_pixels"] = per_max

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **extra_kwargs,
        )
        inputs_on_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        image_grid_thw = inputs_on_device.get("image_grid_thw")
        if image_grid_thw is None:
            raise ValueError(
                "spatialstack-qwen35 extraction requires image_grid_thw "
                f"from the processor, got keys: {list(inputs.keys())}"
            )

        geometry_inputs = build_qwen3_5_geometry_inputs(
            images,
            inputs["image_grid_thw"],
            geometry_encoder_type=self.geometry_encoder_type,
        )
        geometry_encoder_inputs = [
            torch.stack(geometry_inputs).to(self.device, non_blocking=True)
        ]

        input_ids = inputs_on_device["input_ids"]
        visual_mask = (input_ids == self.image_token_id) | (input_ids == self.video_token_id)
        hidden_states = self._forward_with_spatialstack_hidden_states(
            inputs_on_device,
            visual_mask,
            geometry_encoder_inputs,
        )
        return hidden_states, input_ids, visual_mask.reshape(-1), image_grid_thw


class Qwen35VisionExtractor(Qwen35Extractor):
    """Extract Qwen3.5 visual-encoder layer states before language decoding."""

    def __init__(
        self,
        model_path: str,
        select_layers: List[int],
        question: str = "",
        **kwargs,
    ):
        super().__init__(model_path, select_layers, question, **kwargs)
        visual = getattr(getattr(self.model, "model", None), "visual", None)
        if visual is None:
            raise AttributeError("Qwen3.5 model has no model.visual module")
        self.visual = visual
        self.visual_config = getattr(visual, "config", None)
        self.hidden_size = getattr(self.visual_config, "hidden_size", self.hidden_size)
        self.num_layers = (
            getattr(self.visual_config, "num_hidden_layers", None)
            or getattr(self.visual_config, "depth", None)
            or self.num_layers
        )

    def _resolve_vision_layer_idx(self, layer: int, num_hidden_states: int) -> int:
        if layer < 0:
            layer_idx = num_hidden_states + layer
        elif num_hidden_states == self.num_layers:
            layer_idx = layer - 1 if layer > 0 else 0
        else:
            layer_idx = layer
        if layer_idx < 0 or layer_idx >= num_hidden_states:
            raise ValueError(
                f"Vision layer {layer} resolved to index {layer_idx}, but "
                f"visual encoder returned {num_hidden_states} hidden states."
            )
        return layer_idx

    def _reshape_visual_layer(
        self,
        layer_feat: "torch.Tensor",
        image_grid_thw: "torch.Tensor",
    ) -> "torch.Tensor":
        if layer_feat.ndim == 3:
            if layer_feat.shape[0] != 1:
                raise ValueError(
                    "Expected batch size 1 for visual hidden states, "
                    f"got {tuple(layer_feat.shape)}"
                )
            layer_feat = layer_feat[0]
        if layer_feat.ndim != 2:
            raise ValueError(
                f"Expected visual hidden states [tokens,C], got {tuple(layer_feat.shape)}"
            )

        split_sizes = image_grid_thw.prod(-1).tolist()
        pieces = []
        for feat, grid in zip(torch.split(layer_feat, split_sizes, dim=0), image_grid_thw):
            t, h, w = [int(x) for x in grid.tolist()]
            pieces.append(feat.reshape(t, h, w, feat.shape[-1]))
        return torch.cat(pieces, dim=0)

    @torch.no_grad()
    def extract(
        self,
        frame_dir: str,
        num_frames: int,
        frame_ext: str = "png",
        start_idx: int = 0,
        gt_num_frames: int = None,
        video_parity: bool = True,
        use_query_frame_indices: bool = False,
        context_len: int = 76,
        query_idx_divisor: int = 4,
    ) -> Dict[int, "torch.Tensor"]:
        images = self.load_frames(
            frame_dir,
            num_frames,
            frame_ext,
            start_idx=start_idx,
            gt_num_frames=gt_num_frames,
            use_query_frame_indices=use_query_frame_indices,
            context_len=context_len,
            query_idx_divisor=query_idx_divisor,
        )

        messages = self.build_messages(images)
        extra_kwargs = {}
        if video_parity and len(images) > 1:
            per_min, per_max = self._compute_per_image_pixels(len(images))
            extra_kwargs["min_pixels"] = per_min
            extra_kwargs["max_pixels"] = per_max

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **extra_kwargs,
        )
        inputs_on_device = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }
        pixel_values = inputs_on_device.get("pixel_values")
        image_grid_thw = inputs_on_device.get("image_grid_thw")
        if pixel_values is None or image_grid_thw is None:
            raise ValueError(
                "qwen35-visual extraction requires pixel_values and image_grid_thw "
                f"from the processor, got keys: {list(inputs.keys())}"
            )

        image_outputs = self.model.model.get_image_features(
            pixel_values,
            image_grid_thw,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden_states = getattr(image_outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Qwen3.5 visual encoder did not return hidden_states")

        features = {}
        for layer in self.select_layers:
            layer_idx = self._resolve_vision_layer_idx(layer, len(hidden_states))
            features[layer] = self._reshape_visual_layer(
                hidden_states[layer_idx],
                image_grid_thw,
            )
        return features


@lru_cache(maxsize=4)
def get_qwen35_extractor(
    model_path: str,
    model_type: str = "qwen35",
    select_layers: Tuple[int, ...] = (8, 16, 24, 32),
    question: str = "",
    device: str = "cuda:0",
    target_size: Optional[Tuple[int, int]] = (960, 540),
    attn_implementation: str = "sdpa",
    geometry_encoder_path: Optional[str] = None,
    geometry_encoder_type: Optional[str] = None,
) -> Qwen35Extractor:
    if model_type == "qwen35":
        extractor_cls = Qwen35Extractor
    elif model_type == "qwen35-visual":
        extractor_cls = Qwen35VisionExtractor
    elif model_type == "spatialstack-qwen35":
        extractor_cls = SpatialStackQwen35Extractor
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    kwargs = dict(
        model_path=model_path,
        select_layers=list(select_layers),
        question=question,
        device=device,
        target_size=target_size,
        attn_implementation=attn_implementation,
    )
    if extractor_cls is SpatialStackQwen35Extractor:
        kwargs["geometry_encoder_path"] = geometry_encoder_path
        kwargs["geometry_encoder_type"] = geometry_encoder_type
    return extractor_cls(**kwargs)
