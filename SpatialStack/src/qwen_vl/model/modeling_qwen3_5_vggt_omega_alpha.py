"""Qwen3.5 wrapper for the VGGT-Omega alpha input-side injection path."""

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
    align_qwen3_5_geometry_modules,
)
from .vggt_omega_alpha_packing import (
    build_alpha_only_mask,
    expand_image_embeds_with_alpha_tokens,
    expand_visual_placeholders,
)
from .vggt_omega_alpha_projector import VGGTOmegaAlphaProjector, resolve_progressive_hidden_dim


def align_qwen3_5_alpha_modules(model):
    model = align_qwen3_5_geometry_modules(model)
    inner_model = getattr(model, "model", None)
    if inner_model is None:
        return model

    reference_tensor = getattr(getattr(model, "lm_head", None), "weight", None)
    if reference_tensor is not None and reference_tensor.device.type != "meta":
        alpha_projector = getattr(inner_model, "alpha_projector", None)
        if alpha_projector is not None and hasattr(alpha_projector, "to"):
            alpha_projector.to(device=reference_tensor.device, dtype=reference_tensor.dtype)

    model.alpha_projector = getattr(inner_model, "alpha_projector", None)
    return model


class Qwen3_5ModelWithVGGTOmegaAlpha(Qwen3_5ModelWithGeometry):
    def __init__(self, config):
        self.alpha_projector = None
        super().__init__(config)
        self.alpha_projector = getattr(self, "alpha_projector", None)

    def _is_alpha_geometry(self) -> bool:
        return getattr(self.config, "geometry_encoder_type", "vggt") == "vggt_omega_alpha"

    def _validate_geometry_config(self, config):
        if getattr(config, "geometry_encoder_type", "vggt") == "vggt_omega_alpha":
            return
        super()._validate_geometry_config(config)

    def initialize_geometry_modules(self):
        if self._geometry_modules_initialized:
            return

        if not self._is_alpha_geometry():
            super().initialize_geometry_modules()
            return

        config = self.config
        encoder_config = GeometryEncoderConfig(
            encoder_type=getattr(config, "geometry_encoder_type", "vggt_omega_alpha"),
            model_path=getattr(config, "geometry_encoder_path", None),
            reference_frame=getattr(config, "reference_frame", "first"),
            freeze_encoder=getattr(config, "geometry_encoder_freeze", True),
        )
        self.geometry_encoder = create_geometry_encoder(
            encoder_type=encoder_config.encoder_type,
            model_path=encoder_config.model_path,
            reference_frame=encoder_config.reference_frame,
            freeze_encoder=encoder_config.freeze_encoder,
        )
        projector_input_dim = self.geometry_encoder.get_feature_dim()
        projector_output_dim = config.text_config.hidden_size
        projector_hidden_dim = resolve_progressive_hidden_dim(
            projector_input_dim,
            projector_output_dim,
            projector_output_dim,
        )
        self.alpha_projector = VGGTOmegaAlphaProjector(
            input_dim=projector_input_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
        )
        self.alpha_projector.apply(self._init_weights)
        self._geometry_modules_initialized = True

    def align_geometry_modules(self, reference_tensor: Optional[torch.Tensor] = None):
        super().align_geometry_modules(reference_tensor=reference_tensor)
        if reference_tensor is None:
            try:
                reference_tensor = next(self.language_model.parameters())
            except StopIteration:
                reference_tensor = None
        if reference_tensor is None or getattr(reference_tensor, "device", None) is None:
            return
        if getattr(reference_tensor, "device").type == "meta":
            return
        if self.alpha_projector is not None:
            self.alpha_projector.to(
                device=reference_tensor.device,
                dtype=reference_tensor.dtype,
            )

    def _collect_alpha_features(
        self,
        geometry_encoder_inputs: List[torch.Tensor],
        target_device: torch.device,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        per_sample_features = []
        for sample_inputs in geometry_encoder_inputs:
            if sample_inputs.numel() == 0:
                continue
            features = self.geometry_encoder.encode(sample_inputs)
            per_sample_features.append(features.to(device=target_device, dtype=target_dtype))
        if not per_sample_features:
            return torch.zeros(
                0,
                getattr(self.geometry_encoder, "num_special_tokens", 17),
                self.geometry_encoder.get_feature_dim(),
                device=target_device,
                dtype=target_dtype,
            )
        return torch.cat(per_sample_features, dim=0)

    @staticmethod
    def _per_frame_visual_sizes(image_grid_thw: torch.Tensor, spatial_merge_size: int) -> List[int]:
        sizes = []
        for t, h, w in image_grid_thw.tolist():
            sizes.append((int(t) * int(h) * int(w)) // (spatial_merge_size ** 2))
        return sizes

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
        **kwargs,
    ) -> Qwen3_5ModelOutputWithPast:
        alpha_active = (
            self._is_alpha_geometry()
            and pixel_values is not None
            and image_grid_thw is not None
            and (cache_position is None or (isinstance(cache_position, torch.Tensor) and cache_position[0] == 0))
        )
        if not alpha_active:
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
                "vggt_omega_alpha only supports the multi-image path; native Qwen video inputs are unsupported."
            )
        if input_ids is None:
            raise ValueError("vggt_omega_alpha requires input_ids so visual placeholders can be expanded.")
        if geometry_encoder_inputs is None:
            raise ValueError("vggt_omega_alpha requires geometry_encoder_inputs.")
        if inputs_embeds is not None:
            raise ValueError("vggt_omega_alpha does not support precomputed inputs_embeds on the first visual step.")

        inputs_embeds = self.get_input_embeddings()(input_ids)

        if self.geometry_encoder is None or self.alpha_projector is None:
            self.initialize_geometry_modules()
        self.align_geometry_modules(inputs_embeds)

        image_outputs = self.get_image_features(
            pixel_values,
            image_grid_thw,
            return_dict=True,
            output_hidden_states=False,
        )
        pooler_output = image_outputs.pooler_output
        if isinstance(pooler_output, torch.Tensor):
            image_embeds = pooler_output
        else:
            image_embeds = torch.cat(list(pooler_output), dim=0)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        alpha_features = self._collect_alpha_features(
            geometry_encoder_inputs,
            target_device=inputs_embeds.device,
            target_dtype=inputs_embeds.dtype,
        )
        alpha_embeds = self.alpha_projector(alpha_features)
        alpha_tokens_per_frame = alpha_embeds.shape[1]
        patches_per_frame = self._per_frame_visual_sizes(
            image_grid_thw,
            spatial_merge_size=getattr(self.config.vision_config, "spatial_merge_size", 2),
        )
        image_embeds_expanded = expand_image_embeds_with_alpha_tokens(
            image_embeds,
            alpha_embeds,
            patches_per_frame,
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
            num_extra_per_frame=alpha_tokens_per_frame,
        )

        expanded_inputs_embeds = self.get_input_embeddings()(new_input_ids).to(
            inputs_embeds.device,
            inputs_embeds.dtype,
        )
        image_mask, _ = self.get_placeholder_mask(
            new_input_ids,
            inputs_embeds=expanded_inputs_embeds,
            image_features=image_embeds_expanded,
        )
        expanded_inputs_embeds = expanded_inputs_embeds.masked_scatter(image_mask, image_embeds_expanded)

        alpha_only_mask = build_alpha_only_mask(image_mask[..., 0], alpha_tokens_per_frame)
        self._alpha_only_mask = alpha_only_mask

        outputs = self.language_model(
            input_ids=None,
            position_ids=new_position_ids,
            attention_mask=new_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=expanded_inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return Qwen3_5ModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )


class Qwen3_5ForConditionalGenerationWithVGGTOmegaAlpha(Qwen3_5ForConditionalGenerationWithGeometry):
    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = Qwen3_5ModelWithVGGTOmegaAlpha(config)
        self.geometry_encoder = self.model.geometry_encoder
        self.alpha_projector = self.model.alpha_projector
        self.language_feature_fusion = self.model.language_feature_fusion
        self.feature_fusion = self.model.feature_fusion
        self.geometry_merger = self.model.geometry_merger
        self.geometry_merger_list = self.model.geometry_merger_list
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return align_qwen3_5_alpha_modules(model)

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
        alpha_active = (
            getattr(self.config, "geometry_encoder_type", "vggt") == "vggt_omega_alpha"
            and pixel_values is not None
            and image_grid_thw is not None
            and (cache_position is None or (isinstance(cache_position, torch.Tensor) and cache_position[0] == 0))
        )

        expanded_labels = labels
        if alpha_active and labels is not None and input_ids is not None:
            expanded_input_ids, expanded_labels, _, _ = expand_visual_placeholders(
                input_ids=input_ids,
                labels=labels,
                attention_mask=None,
                position_ids=None,
                placeholder_token_id=int(self.config.image_token_id),
                num_extra_per_frame=17,
            )
            del expanded_input_ids

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

        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
