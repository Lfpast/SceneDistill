"""Qwen3.5 wrapper for online VGGT-Omega camera-scene token distillation."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache

from .geometry_encoders import create_geometry_encoder
from .modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGenerationWithGeometry,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5ModelWithGeometry,
    Qwen3_5PreTrainedModel,
    align_qwen3_5_geometry_modules,
)
from .scene_distill_module import (
    LLM_BLOCK_INDICES,
    NUM_SPECIAL_TOKENS,
    POST_DISTILL_DEPTH,
    POST_DISTILL_WEIGHT,
    PRE_DISTILL_WEIGHT,
    SceneDistillPostModule,
    SceneDistillPreModule,
    remove_teacher_weights,
    scene_distillation_loss,
    select_pre_vision_layer_outputs,
)
from .vggt_omega_direct_packing import (
    build_direct_only_mask,
    compute_mrope_position_deltas,
    expand_image_embeds_with_direct_tokens,
    expand_visual_placeholders,
)


SCENE_DISTILL_ENCODER_TYPE = "scene_distill"


def is_scene_distill_geometry_encoder(geometry_encoder_type: str) -> bool:
    return str(geometry_encoder_type or "").strip().lower() == SCENE_DISTILL_ENCODER_TYPE


def align_qwen3_5_scene_distill_modules(model):
    model = align_qwen3_5_geometry_modules(model)
    inner_model = getattr(model, "model", None)
    if inner_model is None:
        return model

    reference_tensor = getattr(getattr(model, "lm_head", None), "weight", None)
    if reference_tensor is not None and reference_tensor.device.type != "meta":
        scene_distill = getattr(getattr(inner_model, "language_model", None), "scene_distill", None)
        if scene_distill is not None:
            scene_distill.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
    return model


class Qwen3_5ModelWithSceneDistill(Qwen3_5ModelWithGeometry):
    def __init__(self, config):
        if hasattr(config, "distill_weight"):
            delattr(config, "distill_weight")
        if not hasattr(config, "pre_distill_weight"):
            config.pre_distill_weight = PRE_DISTILL_WEIGHT
        if not hasattr(config, "post_distill_weight"):
            config.post_distill_weight = POST_DISTILL_WEIGHT
        self._last_pre_distill_loss = None
        self._last_post_distill_loss = None
        super().__init__(config)

    def _is_scene_distill(self) -> bool:
        return is_scene_distill_geometry_encoder(getattr(self.config, "geometry_encoder_type", "vggt"))

    def _validate_geometry_config(self, config):
        if not is_scene_distill_geometry_encoder(getattr(config, "geometry_encoder_type", "vggt")):
            super()._validate_geometry_config(config)
            return

        if getattr(config, "geometry_token_insert_position", "front") != "front":
            raise ValueError("SceneDistill requires geometry_token_insert_position='front'.")
        if getattr(config, "geometry_direct_token_mode", "special17") != "special17":
            raise ValueError("SceneDistill always uses one camera plus 16 scene tokens (special17).")
        if getattr(config, "reference_frame", "first") != "first":
            raise ValueError("SceneDistill requires reference_frame='first'.")
        if not getattr(config, "geometry_encoder_freeze", True):
            raise ValueError("SceneDistill requires a frozen VGGT-Omega teacher.")
        if float(getattr(config, "pre_distill_weight", PRE_DISTILL_WEIGHT)) < 0:
            raise ValueError("SceneDistill pre_distill_weight must be non-negative.")
        if float(getattr(config, "post_distill_weight", POST_DISTILL_WEIGHT)) < 0:
            raise ValueError("SceneDistill post_distill_weight must be non-negative.")
        if int(config.text_config.num_hidden_layers) <= max(LLM_BLOCK_INDICES):
            raise ValueError(
                "SceneDistill Post requires text_config.num_hidden_layers > "
                f"{max(LLM_BLOCK_INDICES)}, got {config.text_config.num_hidden_layers}."
            )

    def initialize_geometry_modules(self):
        if self._geometry_modules_initialized:
            return
        if not self._is_scene_distill():
            super().initialize_geometry_modules()
            return

        config = self.config
        self._validate_geometry_config(config)
        geometry_encoder_path = getattr(config, "geometry_encoder_path", None)
        if geometry_encoder_path:
            self.geometry_encoder = create_geometry_encoder(
                encoder_type="vggt_omega_direct",
                model_path=geometry_encoder_path,
                reference_frame="first",
                freeze_encoder=True,
                direct_token_mode="special17",
            )
        self.language_model.scene_distill = nn.ModuleDict(
            {
                "pre": SceneDistillPreModule(
                    visual_dim=int(config.vision_config.hidden_size),
                    text_hidden_dim=int(config.text_config.hidden_size),
                ),
                "post": SceneDistillPostModule(
                    special_dim=1024,
                    llm_hidden_dim=int(config.text_config.hidden_size),
                    num_heads=16,
                    depth=POST_DISTILL_DEPTH,
                ),
            }
        )
        self._geometry_modules_initialized = True

    def align_geometry_modules(self, reference_tensor: Optional[torch.Tensor] = None):
        super().align_geometry_modules(reference_tensor=reference_tensor)
        if reference_tensor is None:
            try:
                reference_tensor = next(self.language_model.parameters())
            except StopIteration:
                return
        if reference_tensor.device.type == "meta":
            return
        scene_distill = getattr(self.language_model, "scene_distill", None)
        if scene_distill is not None:
            scene_distill.to(device=reference_tensor.device, dtype=reference_tensor.dtype)

    @staticmethod
    def _video_sizes(video_grid_thw: torch.Tensor) -> List[int]:
        video_sizes = [int(grid[0]) for grid in video_grid_thw.tolist()]
        if not video_sizes or any(size <= 0 for size in video_sizes):
            raise ValueError(f"SceneDistill requires at least one temporal group per video, got {video_sizes}.")
        return video_sizes

    @staticmethod
    def _raw_frame_sizes(video_grid_thw: torch.Tensor) -> List[int]:
        frame_sizes = []
        for t, h, w in video_grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            frame_sizes.extend([h * w] * t)
        return frame_sizes

    @staticmethod
    def _merged_frame_sizes(video_grid_thw: torch.Tensor, spatial_merge_size: int) -> List[int]:
        frame_sizes = []
        for t, h, w in video_grid_thw.tolist():
            merged_size = (int(h) * int(w)) // (spatial_merge_size**2)
            frame_sizes.extend([merged_size] * int(t))
        return frame_sizes

    @staticmethod
    def _validate_expanded_video_spans(
        video_mask_2d: torch.Tensor,
        direct_only_mask: torch.Tensor,
        llm_frame_sizes: List[int],
    ) -> None:
        expected_total = sum(llm_frame_sizes)
        actual_total = int(video_mask_2d.sum().item())
        if actual_total != expected_total:
            raise ValueError(
                f"SceneDistill expanded video mask contains {actual_total} tokens, expected {expected_total}."
            )
        if int(direct_only_mask.sum().item()) != len(llm_frame_sizes) * NUM_SPECIAL_TOKENS:
            raise ValueError(
                "SceneDistill direct-only mask must contain exactly "
                f"{NUM_SPECIAL_TOKENS} positions per frame."
            )

        direct_flags = direct_only_mask[video_mask_2d]
        for frame_index, frame_flags in enumerate(torch.split(direct_flags, llm_frame_sizes)):
            expected_flags = torch.arange(
                frame_flags.shape[0],
                device=frame_flags.device,
            ) < NUM_SPECIAL_TOKENS
            if not torch.equal(frame_flags, expected_flags):
                raise ValueError(
                    f"SceneDistill temporal group {frame_index} video span must start with exactly "
                    f"{NUM_SPECIAL_TOKENS} direct positions."
                )

    def _collect_teacher_features(
        self,
        geometry_encoder_inputs: List[torch.Tensor],
        video_grid_thw: torch.Tensor,
        target_device: torch.device,
    ) -> torch.Tensor:
        if self.geometry_encoder is None:
            raise RuntimeError("SceneDistill training requires a VGGT-Omega teacher.")
        if len(geometry_encoder_inputs) != video_grid_thw.shape[0]:
            raise ValueError(
                "SceneDistill teacher/video mismatch: "
                f"geometry_inputs={len(geometry_encoder_inputs)}, video_grids={video_grid_thw.shape[0]}."
            )
        teacher_features = []
        for video_index, (sample_inputs, grid) in enumerate(
            zip(geometry_encoder_inputs, video_grid_thw.tolist())
        ):
            target_frames = int(grid[0])
            teacher = self.geometry_encoder.encode(sample_inputs).to(
                device=target_device,
                dtype=torch.float32,
            )
            if teacher.ndim != 3 or teacher.shape[1:] != (NUM_SPECIAL_TOKENS, 2048):
                raise ValueError(
                    f"VGGT-Omega video {video_index} must return [S,{NUM_SPECIAL_TOKENS},2048], "
                    f"got {tuple(teacher.shape)}."
                )
            source_frames = int(teacher.shape[0])
            if source_frames == target_frames:
                aligned = teacher
            elif source_frames == 2 * target_frames:
                aligned = teacher.reshape(
                    target_frames,
                    2,
                    NUM_SPECIAL_TOKENS,
                    2048,
                ).mean(dim=1)
            elif source_frames > target_frames:
                channels = teacher.permute(1, 2, 0).reshape(
                    1,
                    NUM_SPECIAL_TOKENS * 2048,
                    source_frames,
                )
                aligned = F.adaptive_avg_pool1d(channels, target_frames).reshape(
                    NUM_SPECIAL_TOKENS,
                    2048,
                    target_frames,
                ).permute(2, 0, 1)
            else:
                raise ValueError(
                    f"SceneDistill video {video_index} has only {source_frames} teacher frames for "
                    f"{target_frames} Qwen temporal groups; the two branches did not consume the same frames."
                )
            teacher_features.append(aligned)
        return torch.cat(teacher_features, dim=0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        geometry_encoder_inputs: Optional[List[torch.Tensor]] = None,
        compute_pre_distill_loss: bool = False,
        compute_post_distill_loss: bool = False,
        **kwargs,
    ) -> Qwen3_5ModelOutputWithPast:
        self._last_pre_distill_loss = None
        self._last_post_distill_loss = None
        if self._is_scene_distill() and (pixel_values is not None or image_grid_thw is not None):
            raise ValueError("SceneDistill accepts visual inputs only through the native video fields.")
        scene_distill_active = (
            self._is_scene_distill()
            and pixel_values_videos is not None
            and video_grid_thw is not None
            and (cache_position is None or (isinstance(cache_position, torch.Tensor) and cache_position[0] == 0))
        )
        if not scene_distill_active:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                cache_position=cache_position,
                geometry_encoder_inputs=geometry_encoder_inputs,
                **kwargs,
            )

        if input_ids is None:
            raise ValueError("SceneDistill requires input_ids so visual placeholders can be expanded.")
        if inputs_embeds is not None:
            raise ValueError("SceneDistill does not support precomputed inputs_embeds on the first visual step.")

        inputs_embeds = self.get_input_embeddings()(input_ids)
        scene_distill = getattr(self.language_model, "scene_distill", None)
        if scene_distill is None:
            self.initialize_geometry_modules()
            scene_distill = self.language_model.scene_distill
        self.align_geometry_modules(inputs_embeds)

        video_sizes = self._video_sizes(video_grid_thw)
        frame_sizes = self._raw_frame_sizes(video_grid_thw)

        video_outputs = self.get_video_features(
            pixel_values_videos,
            video_grid_thw,
            return_dict=True,
            output_hidden_states=True,
        )
        vision_hidden_states = getattr(video_outputs, "hidden_states", None)
        visual_layer_outputs = select_pre_vision_layer_outputs(vision_hidden_states)

        pre_embeds, pre_features, pre_global_tokens = scene_distill["pre"](
            visual_layer_outputs,
            frame_sizes=frame_sizes,
            video_sizes=video_sizes,
        )
        teacher_features = None
        if compute_pre_distill_loss or compute_post_distill_loss:
            if geometry_encoder_inputs is None:
                raise ValueError("SceneDistill training losses require geometry_encoder_inputs.")
            teacher_features = self._collect_teacher_features(
                geometry_encoder_inputs,
                video_grid_thw,
                target_device=pre_features.device,
            )
        if compute_pre_distill_loss:
            self._last_pre_distill_loss = scene_distillation_loss(pre_features, teacher_features)

        pooler_output = video_outputs.pooler_output
        video_embeds = (
            pooler_output
            if isinstance(pooler_output, torch.Tensor)
            else torch.cat(list(pooler_output), dim=0)
        )
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        pre_embeds = pre_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        merged_frame_sizes = self._merged_frame_sizes(
            video_grid_thw,
            spatial_merge_size=int(getattr(self.config.vision_config, "spatial_merge_size", 2)),
        )
        video_embeds_expanded = expand_image_embeds_with_direct_tokens(
            video_embeds,
            pre_embeds,
            merged_frame_sizes,
            insert_position="front",
        )

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=None,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        new_input_ids, _, new_attention_mask, new_position_ids = expand_visual_placeholders(
            input_ids=input_ids,
            labels=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            placeholder_token_id=int(self.config.video_token_id),
            num_extra_per_frame=NUM_SPECIAL_TOKENS,
            insert_position="front",
        )
        expanded_inputs_embeds = self.get_input_embeddings()(new_input_ids).to(
            inputs_embeds.device,
            inputs_embeds.dtype,
        )
        if cache_position is not None and cache_position.shape[0] != expanded_inputs_embeds.shape[1]:
            start_position = int(cache_position[0].item())
            cache_position = torch.arange(
                start_position,
                start_position + expanded_inputs_embeds.shape[1],
                device=cache_position.device,
                dtype=cache_position.dtype,
            )

        _, video_mask = self.get_placeholder_mask(
            new_input_ids,
            inputs_embeds=expanded_inputs_embeds,
            video_features=video_embeds_expanded,
        )
        expanded_inputs_embeds = expanded_inputs_embeds.masked_scatter(video_mask, video_embeds_expanded)
        video_mask_2d = video_mask[..., 0]
        self._direct_only_mask = build_direct_only_mask(
            video_mask_2d,
            NUM_SPECIAL_TOKENS,
            insert_position="front",
        )
        llm_frame_sizes = [NUM_SPECIAL_TOKENS + frame_size for frame_size in merged_frame_sizes]
        self._validate_expanded_video_spans(
            video_mask_2d,
            self._direct_only_mask,
            llm_frame_sizes,
        )
        self.rope_deltas = compute_mrope_position_deltas(
            new_position_ids,
            new_input_ids,
            new_attention_mask,
        )

        outputs = self.language_model(
            input_ids=None,
            position_ids=new_position_ids,
            attention_mask=new_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=expanded_inputs_embeds,
            cache_position=cache_position,
            scene_distill_post_tokens=pre_global_tokens,
            scene_distill_image_mask=video_mask_2d,
            scene_distill_special_mask=self._direct_only_mask,
            scene_distill_frame_sizes=llm_frame_sizes,
            scene_distill_video_sizes=video_sizes,
            return_scene_distill_post_features=compute_post_distill_loss,
            **kwargs,
        )
        if compute_post_distill_loss:
            final_post_features = outputs.hidden_states
            if final_post_features is None:
                raise RuntimeError(
                    f"SceneDistill Post did not complete layers {LLM_BLOCK_INDICES}."
                )
            post_features = torch.cat(final_post_features, dim=-1)
            self._last_post_distill_loss = scene_distillation_loss(post_features, teacher_features)

        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


class Qwen3_5ForConditionalGenerationWithSceneDistill(Qwen3_5ForConditionalGenerationWithGeometry):
    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = Qwen3_5ModelWithSceneDistill(config)
        self.geometry_encoder = self.model.geometry_encoder
        self.language_feature_fusion = self.model.language_feature_fusion
        self.feature_fusion = self.model.feature_fusion
        self.geometry_merger = self.model.geometry_merger
        self.geometry_merger_list = self.model.geometry_merger_list
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()
        scene_distill = getattr(self.model.language_model, "scene_distill", None)
        if scene_distill is not None:
            for module in scene_distill.values():
                module.reset_parameters()

    def _init_weights(self, module):
        super()._init_weights(module)
        inner_model = getattr(self, "model", None)
        scene_distill = getattr(getattr(inner_model, "language_model", None), "scene_distill", None)
        post_module = None if scene_distill is None else scene_distill["post"]
        if post_module is not None and any(
            module is projection
            for projection in post_module.inject
        ):
            nn.init.zeros_(module.weight)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return align_qwen3_5_scene_distill_modules(model)

    def state_dict(self, *args, **kwargs):
        return remove_teacher_weights(super().state_dict(*args, **kwargs))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        geometry_encoder_inputs: Optional[List[torch.Tensor]] = None,
        tag: Optional[str] = None,
        **kwargs,
    ) -> Qwen3_5CausalLMOutputWithPast:
        scene_distill_active = (
            is_scene_distill_geometry_encoder(getattr(self.config, "geometry_encoder_type", "vggt"))
            and pixel_values_videos is not None
            and video_grid_thw is not None
            and (cache_position is None or (isinstance(cache_position, torch.Tensor) and cache_position[0] == 0))
        )

        expanded_labels = labels
        if scene_distill_active and labels is not None and input_ids is not None:
            _, expanded_labels, _, _ = expand_visual_placeholders(
                input_ids=input_ids,
                labels=labels,
                attention_mask=None,
                position_ids=None,
                placeholder_token_id=int(self.config.video_token_id),
                num_extra_per_frame=NUM_SPECIAL_TOKENS,
                insert_position="front",
            )

        pre_distill_weight = float(
            getattr(self.config, "pre_distill_weight", PRE_DISTILL_WEIGHT)
        )
        post_distill_weight = float(
            getattr(self.config, "post_distill_weight", POST_DISTILL_WEIGHT)
        )
        compute_distillation = bool(self.training and labels is not None and scene_distill_active)
        compute_pre_distill_loss = compute_distillation and pre_distill_weight > 0
        compute_post_distill_loss = compute_distillation and post_distill_weight > 0
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            cache_position=cache_position,
            geometry_encoder_inputs=geometry_encoder_inputs,
            compute_pre_distill_loss=compute_pre_distill_loss,
            compute_post_distill_loss=compute_post_distill_loss,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if expanded_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=expanded_labels,
                vocab_size=self.config.text_config.vocab_size,
            )

        pre_distill_loss = self.model._last_pre_distill_loss
        post_distill_loss = self.model._last_post_distill_loss
        self.model._last_pre_distill_loss = None
        self.model._last_post_distill_loss = None
        if compute_pre_distill_loss:
            if loss is None or pre_distill_loss is None:
                raise RuntimeError("SceneDistill Pre training requires both SFT and pre distillation losses.")
            loss = loss + pre_distill_weight * pre_distill_loss
        if compute_post_distill_loss:
            if loss is None or post_distill_loss is None:
                raise RuntimeError("SceneDistill Post training requires both SFT and post distillation losses.")
            loss = loss + post_distill_weight * post_distill_loss

        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


__all__ = [
    "Qwen3_5ForConditionalGenerationWithSceneDistill",
    "Qwen3_5ModelWithSceneDistill",
    "SCENE_DISTILL_ENCODER_TYPE",
    "align_qwen3_5_scene_distill_modules",
    "is_scene_distill_geometry_encoder",
]
