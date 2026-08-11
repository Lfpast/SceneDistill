from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.qwen3_5 import Qwen3_5


@register_model("qwen3_5_scene_distill")
class Qwen3_5SceneDistill(Qwen3_5):
    """LMMS-Eval adapter for SceneDistill checkpoints."""

    def __init__(
        self,
        compatibility_geometry_encoder_type: str = "scene_distill",
        scene_distill_stage1_compatibility: bool = False,
        **kwargs,
    ) -> None:
        self._compatibility_geometry_encoder_type = compatibility_geometry_encoder_type
        self._stage1_compatibility = bool(scene_distill_stage1_compatibility)
        kwargs.pop("geometry_encoder_path", None)
        super().__init__(**kwargs)

    def _prepare_config_for_eval(self, config, _geometry_encoder_path):
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
        setattr(config, "geometry_encoder_path", None)
        if self._stage1_compatibility:
            setattr(config, "pre_distill_weight", float(getattr(config, "distill_weight", 0.05)))
            setattr(config, "post_distill_weight", 0.0)
            setattr(config, "scene_distill_stage1_compatibility", True)

        eval_logger.warning(
            "Using qwen3_5_scene_distill student-only evaluation; VGGT-Omega teacher weights "
            "will not be constructed or loaded."
        )
        if self._stage1_compatibility:
            eval_logger.warning(
                "Stage 1 compatibility is enabled: legacy scene_distill_module weights will be "
                "mapped to the current scene_distill_pre_module for evaluation."
            )
        return config, None
