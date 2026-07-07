"""Qwen3.5 adapters for the Qwen3-VL feature extraction pipeline."""

from __future__ import annotations

from functools import lru_cache
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
) -> Qwen35Extractor:
    if model_type == "qwen35":
        extractor_cls = Qwen35Extractor
    elif model_type == "qwen35-visual":
        extractor_cls = Qwen35VisionExtractor
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return extractor_cls(
        model_path=model_path,
        select_layers=list(select_layers),
        question=question,
        device=device,
        target_size=target_size,
        attn_implementation=attn_implementation,
    )
