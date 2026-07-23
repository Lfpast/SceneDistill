from typing import Optional, Union

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.qwen3_5 import Qwen3_5, parse_qwen3_5_layer_indices


@register_model("qwen3_5_vgllm_add")
class Qwen3_5VGLLMAdd(Qwen3_5):
    """Compatibility adapter for ddz16/qwen35-4B-vgllm-add eval only."""

    def __init__(
        self,
        geometry_encoder_layers: Optional[Union[str, int]] = None,
        **kwargs,
    ) -> None:
        self._vgllm_add_geometry_encoder_layers = parse_qwen3_5_layer_indices(
            geometry_encoder_layers,
            "geometry_encoder_layers",
        )
        super().__init__(**kwargs)

    def _prepare_config_for_eval(self, config, geometry_encoder_path):
        architectures = getattr(config, "architectures", None) or []
        encoder_type = getattr(config, "geometry_encoder_type", None)
        fusion_method = getattr(config, "feature_fusion_method", None)

        if "Qwen3_5ForConditionalGenerationWithGeometry" not in architectures:
            raise ValueError(
                "qwen3_5_vgllm_add only supports checkpoints with architecture "
                "Qwen3_5ForConditionalGenerationWithGeometry."
            )
        if encoder_type != "vggt":
            raise ValueError(
                "qwen3_5_vgllm_add only supports geometry_encoder_type='vggt'. "
                f"Got {encoder_type!r}."
            )
        if fusion_method != "add":
            raise ValueError(
                "qwen3_5_vgllm_add only supports feature_fusion_method='add'. "
                f"Got {fusion_method!r}."
            )

        if self._vgllm_add_geometry_encoder_layers is not None:
            setattr(config, "geometry_encoder_layers", self._vgllm_add_geometry_encoder_layers)
        if not getattr(config, "geometry_encoder_layers", None):
            raise ValueError(
                "qwen3_5_vgllm_add requires geometry_encoder_layers. "
                "Use geometry_encoder_layers=23 if the checkpoint config does not provide it."
            )

        setattr(config, "use_geometry_encoder", True)
        geometry_encoder_path = geometry_encoder_path or getattr(config, "geometry_encoder_path", None)
        eval_logger.warning(
            "Using qwen3_5_vgllm_add compatibility adapter: "
            f"geometry_encoder_type={encoder_type!r}, feature_fusion_method={fusion_method!r}, "
            f"geometry_encoder_layers={getattr(config, 'geometry_encoder_layers', None)}."
        )
        return config, geometry_encoder_path
