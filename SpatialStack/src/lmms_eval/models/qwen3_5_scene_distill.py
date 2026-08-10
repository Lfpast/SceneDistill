from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.qwen3_5 import Qwen3_5


@register_model("qwen3_5_scene_distill")
class Qwen3_5SceneDistill(Qwen3_5):
    """LMMS-Eval adapter for SceneDistill checkpoints."""

    def __init__(self, compatibility_geometry_encoder_type: str = "scene_distill", **kwargs) -> None:
        self._compatibility_geometry_encoder_type = compatibility_geometry_encoder_type
        super().__init__(**kwargs)

    def _prepare_config_for_eval(self, config, geometry_encoder_path):
        architectures = getattr(config, "architectures", None) or []
        original_encoder_type = getattr(config, "geometry_encoder_type", None)
        is_scene_distill_checkpoint = (
            original_encoder_type == "scene_distill"
            or "Qwen3_5ForConditionalGenerationWithSceneDistill" in architectures
        )
        if not is_scene_distill_checkpoint:
            raise ValueError(
                "qwen3_5_scene_distill only supports checkpoints with "
                "geometry_encoder_type='scene_distill' or architecture "
                "Qwen3_5ForConditionalGenerationWithSceneDistill."
            )
        if self._compatibility_geometry_encoder_type != "scene_distill":
            raise ValueError(
                "qwen3_5_scene_distill requires compatibility_geometry_encoder_type='scene_distill'."
            )

        setattr(config, "use_geometry_encoder", True)
        setattr(config, "geometry_encoder_type", "scene_distill")
        setattr(config, "geometry_encoder_freeze", True)
        setattr(config, "geometry_direct_token_mode", "special17")
        setattr(config, "geometry_token_insert_position", "front")
        setattr(config, "reference_frame", "first")

        geometry_encoder_path = geometry_encoder_path or getattr(config, "geometry_encoder_path", None)
        eval_logger.warning(
            "Using qwen3_5_scene_distill adapter with frozen VGGT-Omega teacher metadata; "
            "generation runs only the SceneDistill student path."
        )
        return config, geometry_encoder_path
