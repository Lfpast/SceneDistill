"""Qwen3.5 adapter for the Qwen3-VL feature extraction pipeline."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Tuple

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
    if model_type != "qwen35":
        raise ValueError(f"Unknown model type: {model_type}")
    return Qwen35Extractor(
        model_path=model_path,
        select_layers=list(select_layers),
        question=question,
        device=device,
        target_size=target_size,
        attn_implementation=attn_implementation,
    )
