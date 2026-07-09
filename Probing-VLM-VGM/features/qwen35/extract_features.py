#!/usr/bin/env python3
"""Extract Qwen3.5 hidden states for probe training.

Usage:
    python -m features.qwen35.extract_features \
        --scene-dir data/DL3DV/DL3DV-ALL-960P/1K/<hash>/images_4 \
        --data-sft data/DL3DV/DL3DV-processed/1K/<hash>.sft \
        --out-dir data/DL3DV/FEAT/qwen3.5-4b/1K/<hash> \
        --model-path Qwen/Qwen3.5-4B \
        --model-type qwen35 \
        --use-query-frame-indices \
        --output-layers 20
"""

from __future__ import annotations

from typing import List

import torch

from ..qwen3vl.extract_features import main as run_qwen_extraction
from .qwen35_extractor import get_qwen35_extractor


def main(argv: List[str] | None = None) -> None:
    run_qwen_extraction(
        argv,
        extractor_factory=get_qwen35_extractor,
        model_types=("qwen35", "qwen35-visual", "spatialstack-qwen35"),
        default_model_type="qwen35",
        model_family="Qwen3.5",
        default_layers=(8, 16, 24, 32),
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
