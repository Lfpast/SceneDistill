"""Qwen3.5 wrapper for online VGGT-Omega camera-scene token distillation."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from transformers.cache_utils import Cache

from .geometry_encoders import GeometryEncoderConfig, create_geometry_encoder
from .modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGenerationWithGeometry,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5ModelWithGeometry,
    Qwen3_5PreTrainedModel,
    _load_qwen3_5_geometry_submodules,
    _resolve_qwen3_5_checkpoint_root,
    align_qwen3_5_geometry_modules,
)
from .scene_distill_module import (
    DISTILL_WEIGHT,
    NUM_SPECIAL_TOKENS,
    SceneDistillModule,
    remove_teacher_weights,
    scene_distillation_loss,
    select_vision_layer_outputs,
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
        scene_distill_module = getattr(inner_model, "scene_distill_module", None)
        if scene_distill_module is not None:
            scene_distill_module.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
    return model


class Qwen3_5ModelWithSceneDistill(Qwen3_5ModelWithGeometry):
    def __init__(self, config):
        self.scene_distill_module = None
        self._last_scene_distill_loss = None
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

    def initialize_geometry_modules(self):
        if self._geometry_modules_initialized:
            return
        if not self._is_scene_distill():
            super().initialize_geometry_modules()
            return

        config = self.config
        self._validate_geometry_config(config)
        encoder_config = GeometryEncoderConfig(
            encoder_type="vggt_omega_direct",
            model_path=getattr(config, "geometry_encoder_path", None),
            reference_frame="first",
            freeze_encoder=True,
            encoder_kwargs={"direct_token_mode": "special17"},
        )
        self.geometry_encoder = create_geometry_encoder(
            encoder_type=encoder_config.encoder_type,
            model_path=encoder_config.model_path,
            reference_frame=encoder_config.reference_frame,
            freeze_encoder=encoder_config.freeze_encoder,
            **encoder_config.encoder_kwargs,
        )
        self.scene_distill_module = SceneDistillModule(
            visual_dim=int(config.vision_config.hidden_size),
            text_hidden_dim=int(config.text_config.hidden_size),
        )
        self._geometry_modules_initialized = True

    def align_geometry_modules(self, reference_tensor: Optional[torch.Tensor] = None):
        super().align_geometry_modules(reference_tensor=reference_tensor)
        if reference_tensor is None:
            try:
                reference_tensor = next(self.language_model.parameters())
            except StopIteration:
                return
        if reference_tensor.device.type == "meta" or self.scene_distill_module is None:
            return
        self.scene_distill_module.to(device=reference_tensor.device, dtype=reference_tensor.dtype)

    @staticmethod
    def _video_sizes(geometry_encoder_inputs: List[torch.Tensor]) -> List[int]:
        video_sizes = [int(sample_inputs.shape[0]) for sample_inputs in geometry_encoder_inputs]
        if not video_sizes or any(size <= 0 for size in video_sizes):
            raise ValueError(f"SceneDistill requires at least one frame per video, got {video_sizes}.")
        return video_sizes

    @staticmethod
    def _raw_frame_sizes(image_grid_thw: torch.Tensor) -> List[int]:
        frame_sizes = []
        for frame_index, (t, h, w) in enumerate(image_grid_thw.tolist()):
            t, h, w = int(t), int(h), int(w)
            if t != 1:
                raise NotImplementedError(
                    "SceneDistill requires the multi-image frame path with image_grid_thw[:, 0] == 1; "
                    f"frame {frame_index} has t={t}."
                )
            frame_sizes.append(t * h * w)
        return frame_sizes

    @staticmethod
    def _merged_frame_sizes(image_grid_thw: torch.Tensor, spatial_merge_size: int) -> List[int]:
        return [
            (int(t) * int(h) * int(w)) // (spatial_merge_size ** 2)
            for t, h, w in image_grid_thw.tolist()
        ]

    def _collect_teacher_features(
        self,
        geometry_encoder_inputs: List[torch.Tensor],
        target_device: torch.device,
    ) -> torch.Tensor:
        teacher_features = []
        for sample_inputs in geometry_encoder_inputs:
            teacher_features.append(self.geometry_encoder.encode(sample_inputs).to(device=target_device))
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
        compute_scene_distill_loss: bool = False,
        **kwargs,
    ) -> Qwen3_5ModelOutputWithPast:
        self._last_scene_distill_loss = None
        scene_distill_active = (
            self._is_scene_distill()
            and pixel_values is not None
            and image_grid_thw is not None
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

        if pixel_values_videos is not None:
            raise NotImplementedError(
                "SceneDistill only supports videos represented as ordered multi-image frames."
            )
        if input_ids is None:
            raise ValueError("SceneDistill requires input_ids so visual placeholders can be expanded.")
        if inputs_embeds is not None:
            raise ValueError("SceneDistill does not support precomputed inputs_embeds on the first visual step.")
        if geometry_encoder_inputs is None:
            raise ValueError("SceneDistill requires geometry_encoder_inputs for frame grouping and teacher targets.")

        inputs_embeds = self.get_input_embeddings()(input_ids)
        if self.geometry_encoder is None or self.scene_distill_module is None:
            self.initialize_geometry_modules()
        self.align_geometry_modules(inputs_embeds)

        video_sizes = self._video_sizes(geometry_encoder_inputs)
        frame_sizes = self._raw_frame_sizes(image_grid_thw)
        if sum(video_sizes) != len(frame_sizes):
            raise ValueError(
                f"SceneDistill frame order mismatch: geometry inputs describe {sum(video_sizes)} frames "
                f"but image_grid_thw contains {len(frame_sizes)} frames."
            )

        image_outputs = self.get_image_features(
            pixel_values,
            image_grid_thw,
            return_dict=True,
            output_hidden_states=True,
        )
        vision_hidden_states = getattr(image_outputs, "hidden_states", None)
        visual_layer_outputs = select_vision_layer_outputs(vision_hidden_states)

        student_embeds, student_features = self.scene_distill_module(
            visual_layer_outputs,
            frame_sizes=frame_sizes,
            video_sizes=video_sizes,
        )
        if compute_scene_distill_loss:
            teacher_features = self._collect_teacher_features(
                geometry_encoder_inputs,
                target_device=student_features.device,
            )
            self._last_scene_distill_loss = scene_distillation_loss(student_features, teacher_features)

        pooler_output = image_outputs.pooler_output
        image_embeds = (
            pooler_output
            if isinstance(pooler_output, torch.Tensor)
            else torch.cat(list(pooler_output), dim=0)
        )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        student_embeds = student_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        merged_frame_sizes = self._merged_frame_sizes(
            image_grid_thw,
            spatial_merge_size=int(getattr(self.config.vision_config, "spatial_merge_size", 2)),
        )
        image_embeds_expanded = expand_image_embeds_with_direct_tokens(
            image_embeds,
            student_embeds,
            merged_frame_sizes,
            insert_position="front",
        )

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
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
            placeholder_token_id=int(self.config.image_token_id),
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

        image_mask, _ = self.get_placeholder_mask(
            new_input_ids,
            inputs_embeds=expanded_inputs_embeds,
            image_features=image_embeds_expanded,
        )
        expanded_inputs_embeds = expanded_inputs_embeds.masked_scatter(image_mask, image_embeds_expanded)
        self._direct_only_mask = build_direct_only_mask(
            image_mask[..., 0],
            NUM_SPECIAL_TOKENS,
            insert_position="front",
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
            **kwargs,
        )
        return Qwen3_5ModelOutputWithPast(**outputs, rope_deltas=self.rope_deltas)


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
        if self.model.scene_distill_module is not None:
            self.model.scene_distill_module.reset_parameters()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        architectures = getattr(model.config, "architectures", None) or []
        if cls.__name__ in architectures:
            checkpoint_root = _resolve_qwen3_5_checkpoint_root(pretrained_model_name_or_path)
            loaded_keys = _load_qwen3_5_geometry_submodules(model, checkpoint_root)
            if loaded_keys == 0:
                raise RuntimeError(
                    "SceneDistill checkpoint declares the SceneDistill architecture but contains no "
                    "scene_distill_module weights."
                )
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
            and pixel_values is not None
            and image_grid_thw is not None
            and (cache_position is None or (isinstance(cache_position, torch.Tensor) and cache_position[0] == 0))
        )

        expanded_labels = labels
        if scene_distill_active and labels is not None and input_ids is not None:
            _, expanded_labels, _, _ = expand_visual_placeholders(
                input_ids=input_ids,
                labels=labels,
                attention_mask=None,
                position_ids=None,
                placeholder_token_id=int(self.config.image_token_id),
                num_extra_per_frame=NUM_SPECIAL_TOKENS,
                insert_position="front",
            )

        compute_scene_distill_loss = bool(self.training and labels is not None and scene_distill_active)
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
            compute_scene_distill_loss=compute_scene_distill_loss,
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

        distill_loss = self.model._last_scene_distill_loss
        self.model._last_scene_distill_loss = None
        if compute_scene_distill_loss:
            if loss is None or distill_loss is None:
                raise RuntimeError("SceneDistill training requires both SFT and distillation losses.")
            loss = loss + DISTILL_WEIGHT * distill_loss

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
