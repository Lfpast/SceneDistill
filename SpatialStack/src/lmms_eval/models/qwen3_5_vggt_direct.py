from typing import Optional, Union

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.qwen3_5 import Qwen3_5, parse_qwen3_5_layer_indices


@register_model("qwen3_5_vggt_direct")
class Qwen3_5VGGTDIRECT(Qwen3_5):
    """Compatibility adapter for ddz16/qwen35-4B-vggt-direct eval only."""

    def __init__(
        self,
        geometry_encoder_layers: Optional[Union[str, int]] = None,
        geometry_fusion_layers: Optional[Union[str, int]] = None,
        geometry_direct_token_mode: Optional[str] = None,
        geometry_token_insert_position: Optional[str] = None,
        compatibility_geometry_encoder_type: str = "vggt_omega_direct",
        **kwargs,
    ) -> None:
        self._direct_geometry_encoder_layers = parse_qwen3_5_layer_indices(
            geometry_encoder_layers,
            "geometry_encoder_layers",
        )
        self._direct_geometry_fusion_layers = parse_qwen3_5_layer_indices(
            geometry_fusion_layers,
            "geometry_fusion_layers",
        )
        self._direct_token_mode = geometry_direct_token_mode
        self._direct_token_insert_position = geometry_token_insert_position
        self._compatibility_geometry_encoder_type = compatibility_geometry_encoder_type
        super().__init__(**kwargs)

    def _prepare_config_for_eval(self, config, geometry_encoder_path):
        architectures = getattr(config, "architectures", None) or []
        original_encoder_type = getattr(config, "geometry_encoder_type", None)
        is_direct_checkpoint = (
            original_encoder_type == "vggt_omega_direct"
            or "Qwen3_5ForConditionalGenerationWithVGGTOmegaDirect" in architectures
        )
        if not is_direct_checkpoint:
            raise ValueError(
                "qwen3_5_vggt_direct only supports checkpoints with "
                "geometry_encoder_type='vggt_omega_direct' or architecture "
                "Qwen3_5ForConditionalGenerationWithVGGTOmegaDirect."
            )
        if self._compatibility_geometry_encoder_type != "vggt_omega_direct":
            raise ValueError(
                "qwen3_5_vggt_direct requires compatibility_geometry_encoder_type='vggt_omega_direct' "
                "so eval uses the direct-injection model class."
            )

        setattr(config, "use_geometry_encoder", True)
        setattr(config, "geometry_encoder_type", original_encoder_type or self._compatibility_geometry_encoder_type)
        if self._direct_geometry_encoder_layers is not None:
            setattr(config, "geometry_encoder_layers", self._direct_geometry_encoder_layers)
        if self._direct_geometry_fusion_layers is not None:
            setattr(config, "geometry_fusion_layers", self._direct_geometry_fusion_layers)
        if self._direct_token_mode is not None:
            setattr(config, "geometry_direct_token_mode", self._direct_token_mode)
        if self._direct_token_insert_position is not None:
            setattr(config, "geometry_token_insert_position", self._direct_token_insert_position)

        geometry_encoder_path = geometry_encoder_path or getattr(config, "geometry_encoder_path", None)
        eval_logger.warning(
            "Using qwen3_5_vggt_direct compatibility adapter: "
            f"geometry_encoder_type={getattr(config, 'geometry_encoder_type', None)!r}, "
            f"geometry_direct_token_mode={getattr(config, 'geometry_direct_token_mode', None)!r}, "
            f"geometry_token_insert_position={getattr(config, 'geometry_token_insert_position', None)!r}."
        )
        return config, geometry_encoder_path
