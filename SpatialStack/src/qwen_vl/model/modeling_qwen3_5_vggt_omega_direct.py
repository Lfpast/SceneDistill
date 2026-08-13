"""Qwen3.5 wrapper for the shared VGGT-Omega direct-injection path."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache

from .geometry_encoders import GeometryEncoderConfig, create_geometry_encoder
from .modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGenerationWithGeometry,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5ModelWithGeometry,
    Qwen3_5PreTrainedModel,
    align_qwen3_5_geometry_modules,
    remove_geometry_encoder_weights,
)
from .vggt_omega_direct_config import (
    get_vggt_omega_direct_num_extra_tokens,
    is_vggt_omega_direct_geometry_encoder,
    resolve_vggt_omega_direct_token_mode,
)
from .vggt_omega_direct_packing import (
    compute_mrope_position_deltas,
    expand_image_embeds_with_direct_tokens,
    expand_visual_placeholders,
)
from .vggt_omega_direct_projector import (
    VGGTOmegaDirectProjector,
    resolve_progressive_hidden_dim,
)


def align_qwen3_5_direct_modules(model):
    model = align_qwen3_5_geometry_modules(model)
    inner_model = getattr(model, "model", None)
    if inner_model is None:
        return model

    reference_tensor = getattr(getattr(model, "lm_head", None), "weight", None)
    if reference_tensor is not None and reference_tensor.device.type != "meta":
        direct_projector = getattr(inner_model, "direct_projector", None)
        if direct_projector is not None and hasattr(direct_projector, "to"):
            direct_projector.to(device=reference_tensor.device, dtype=reference_tensor.dtype)

    return model


class Qwen3_5ModelWithVGGTOmegaDirect(Qwen3_5ModelWithGeometry):
    def __init__(self, config):
        self.direct_projector = None
        super().__init__(config)
        self.direct_projector = getattr(self, "direct_projector", None)

    def _direct_token_insert_position(self) -> str:
        return getattr(self.config, "geometry_token_insert_position", "front")

    def _direct_token_mode(self) -> str:
        return resolve_vggt_omega_direct_token_mode(
            getattr(self.config, "geometry_encoder_type", "vggt"),
            getattr(self.config, "geometry_direct_token_mode", None),
        )

    def _is_direct_geometry(self) -> bool:
        return is_vggt_omega_direct_geometry_encoder(
            getattr(self.config, "geometry_encoder_type", "vggt")
        )

    def _validate_geometry_config(self, config):
        if is_vggt_omega_direct_geometry_encoder(getattr(config, "geometry_encoder_type", "vggt")):
            return
        super()._validate_geometry_config(config)

    def initialize_geometry_modules(self):
        if self._geometry_modules_initialized:
            return

        if not self._is_direct_geometry():
            super().initialize_geometry_modules()
            return

        config = self.config
        encoder_config = GeometryEncoderConfig(
            encoder_type=getattr(config, "geometry_encoder_type", "vggt_omega_direct"),
            model_path=getattr(config, "geometry_encoder_path", None),
            reference_frame=getattr(config, "reference_frame", "first"),
            freeze_encoder=getattr(config, "geometry_encoder_freeze", True),
            encoder_kwargs={"direct_token_mode": self._direct_token_mode()},
        )
        self.geometry_encoder = create_geometry_encoder(
            encoder_type=encoder_config.encoder_type,
            model_path=encoder_config.model_path,
            reference_frame=encoder_config.reference_frame,
            freeze_encoder=encoder_config.freeze_encoder,
            **encoder_config.encoder_kwargs,
        )
        projector_input_dim = self.geometry_encoder.get_feature_dim()
        projector_output_dim = config.text_config.hidden_size
        projector_hidden_dim = resolve_progressive_hidden_dim(
            projector_input_dim,
            projector_output_dim,
            projector_output_dim,
        )
        self.direct_projector = VGGTOmegaDirectProjector(
            input_dim=projector_input_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
        )
        self.direct_projector.apply(self._init_weights)
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
        if self.direct_projector is not None:
            self.direct_projector.to(
                device=reference_tensor.device,
                dtype=reference_tensor.dtype,
            )

    def _collect_direct_features(
        self,
        geometry_encoder_inputs: List[torch.Tensor],
        video_grid_thw: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(geometry_encoder_inputs) != video_grid_thw.shape[0]:
            raise ValueError(
                "VGGT-Omega direct video mismatch: "
                f"geometry_inputs={len(geometry_encoder_inputs)}, video_grids={video_grid_thw.shape[0]}."
            )
        per_sample_features = []
        for video_index, (sample_inputs, grid) in enumerate(
            zip(geometry_encoder_inputs, video_grid_thw.tolist())
        ):
            target_frames = int(grid[0])
            features = self.geometry_encoder.encode(sample_inputs).to(
                device=target_device,
                dtype=torch.float32,
            )
            source_frames, num_tokens, feature_dim = features.shape
            if source_frames == target_frames:
                aligned = features
            elif source_frames == 2 * target_frames:
                aligned = features.reshape(
                    target_frames,
                    2,
                    num_tokens,
                    feature_dim,
                ).mean(dim=1)
            elif source_frames > target_frames:
                channels = features.permute(1, 2, 0).reshape(
                    1,
                    num_tokens * feature_dim,
                    source_frames,
                )
                aligned = F.adaptive_avg_pool1d(channels, target_frames).reshape(
                    num_tokens,
                    feature_dim,
                    target_frames,
                ).permute(2, 0, 1)
            else:
                raise ValueError(
                    f"VGGT-Omega direct video {video_index} has only {source_frames} frames for "
                    f"{target_frames} Qwen temporal groups; the two branches did not consume the same frames."
                )
            per_sample_features.append(aligned.to(dtype=target_dtype))
        return torch.cat(per_sample_features, dim=0)

    @staticmethod
    def _merged_frame_sizes(video_grid_thw: torch.Tensor, spatial_merge_size: int) -> List[int]:
        sizes = []
        for t, h, w in video_grid_thw.tolist():
            sizes.extend(
                [int(h) * int(w) // (spatial_merge_size ** 2)] * int(t)
            )
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
        if self._is_direct_geometry() and (pixel_values is not None or image_grid_thw is not None):
            raise ValueError("VGGT-Omega direct accepts visual inputs only through the native video fields.")
        direct_active = (
            self._is_direct_geometry()
            and pixel_values_videos is not None
            and video_grid_thw is not None
            and (cache_position is None or (isinstance(cache_position, torch.Tensor) and cache_position[0] == 0))
        )
        if not direct_active:
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
            raise ValueError("VGGT-Omega direct injection requires input_ids so visual placeholders can be expanded.")
        if geometry_encoder_inputs is None:
            raise ValueError("VGGT-Omega direct injection requires geometry_encoder_inputs.")
        if inputs_embeds is not None:
            raise ValueError("VGGT-Omega direct injection does not support precomputed inputs_embeds on the first visual step.")

        inputs_embeds = self.get_input_embeddings()(input_ids)

        if self.geometry_encoder is None or self.direct_projector is None:
            self.initialize_geometry_modules()
        self.align_geometry_modules(inputs_embeds)

        video_outputs = self.get_video_features(
            pixel_values_videos,
            video_grid_thw,
            return_dict=True,
            output_hidden_states=False,
        )
        pooler_output = video_outputs.pooler_output
        if isinstance(pooler_output, torch.Tensor):
            video_embeds = pooler_output
        else:
            video_embeds = torch.cat(list(pooler_output), dim=0)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        direct_features = self._collect_direct_features(
            geometry_encoder_inputs,
            video_grid_thw,
            target_device=inputs_embeds.device,
            target_dtype=inputs_embeds.dtype,
        )
        direct_embeds = self.direct_projector(direct_features)
        direct_tokens_per_frame = direct_embeds.shape[1]
        insert_position = self._direct_token_insert_position()
        patches_per_frame = self._merged_frame_sizes(
            video_grid_thw,
            spatial_merge_size=getattr(self.config.vision_config, "spatial_merge_size", 2),
        )
        video_embeds_expanded = expand_image_embeds_with_direct_tokens(
            video_embeds,
            direct_embeds,
            patches_per_frame,
            insert_position=insert_position,
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
            num_extra_per_frame=direct_tokens_per_frame,
            insert_position=insert_position,
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

        return Qwen3_5ModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )


class Qwen3_5ForConditionalGenerationWithVGGTOmegaDirect(Qwen3_5ForConditionalGenerationWithGeometry):
    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = Qwen3_5ModelWithVGGTOmegaDirect(config)
        self.geometry_encoder = self.model.geometry_encoder
        self.language_feature_fusion = self.model.language_feature_fusion
        self.feature_fusion = self.model.feature_fusion
        self.geometry_merger = self.model.geometry_merger
        self.geometry_merger_list = self.model.geometry_merger_list
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return align_qwen3_5_direct_modules(model)

    def state_dict(self, *args, **kwargs):
        return remove_geometry_encoder_weights(super().state_dict(*args, **kwargs))

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
        direct_active = (
            is_vggt_omega_direct_geometry_encoder(getattr(self.config, "geometry_encoder_type", "vggt"))
            and pixel_values_videos is not None
            and video_grid_thw is not None
            and (cache_position is None or (isinstance(cache_position, torch.Tensor) and cache_position[0] == 0))
        )

        expanded_labels = labels
        if direct_active and labels is not None and input_ids is not None:
            num_extra_per_frame = get_vggt_omega_direct_num_extra_tokens(
                getattr(self.config, "geometry_encoder_type", "vggt"),
                getattr(self.config, "geometry_direct_token_mode", None),
            )
            expanded_input_ids, expanded_labels, _, _ = expand_visual_placeholders(
                input_ids=input_ids,
                labels=labels,
                attention_mask=None,
                position_ids=None,
                placeholder_token_id=int(self.config.video_token_id),
                num_extra_per_frame=num_extra_per_frame,
                insert_position=getattr(self.config, "geometry_token_insert_position", "front"),
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


__all__ = [
    "Qwen3_5ForConditionalGenerationWithVGGTOmegaDirect",
    "Qwen3_5ModelWithVGGTOmegaDirect",
    "align_qwen3_5_direct_modules",
]
