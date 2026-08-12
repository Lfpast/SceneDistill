"""SceneDistill special-token GCTE and distillation loss."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_SCENE_TOKENS = 16
NUM_SPECIAL_TOKENS = 1 + NUM_SCENE_TOKENS
# Transformers records the input embedding at hidden_states[0], followed by
# each vision block output. These tuple indices therefore select the outputs of
# zero-based vision blocks 0, 4, 8, and 12.
PRE_VISION_BLOCK_INDICES = (1, 5, 9, 13)
LLM_BLOCK_INDICES = (4, 8, 12, 16, 20, 24)
STREAM_DIM = 1024
FEATURE_DIM = 2 * STREAM_DIM
NUM_HEADS = 16
PRE_DISTILL_DEPTH = len(PRE_VISION_BLOCK_INDICES)
POST_DISTILL_DEPTH = len(LLM_BLOCK_INDICES)
PRE_DISTILL_WEIGHT = 0.05
POST_DISTILL_WEIGHT = 0.05


def remove_teacher_weights(state_dict):
    """Remove the frozen online teacher from a SceneDistill checkpoint state dict."""
    filtered = OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if not key.startswith("geometry_encoder.") and not key.startswith("model.geometry_encoder.")
    )
    if hasattr(state_dict, "_metadata"):
        filtered._metadata = state_dict._metadata
    return filtered


def select_pre_vision_layer_outputs(hidden_states: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Select the 1st, 5th, 9th, and 13th vision block outputs."""
    if hidden_states is None or max(PRE_VISION_BLOCK_INDICES) >= len(hidden_states):
        num_layers = 0 if hidden_states is None else len(hidden_states)
        raise ValueError(
            f"Qwen3.5 returned {num_layers} vision layers; "
            f"SceneDistill Pre requires indices {PRE_VISION_BLOCK_INDICES}."
        )
    selected = [hidden_states[index] for index in PRE_VISION_BLOCK_INDICES]
    if any(features is None for features in selected):
        raise ValueError(
            f"Qwen3.5 did not capture every required SceneDistill Pre vision layer {PRE_VISION_BLOCK_INDICES}."
        )
    return selected


class FrameCrossAttentionLayer(nn.Module):
    """Special tokens attend to the visual tokens from their own frame."""

    def __init__(self, special_dim: int, visual_dim: int, num_heads: int):
        super().__init__()
        if special_dim % num_heads != 0:
            raise ValueError(f"special_dim={special_dim} must be divisible by num_heads={num_heads}.")

        self.num_heads = num_heads
        self.head_dim = special_dim // num_heads
        self.special_dim = special_dim
        self.visual_dim = visual_dim

        self.q_proj = nn.Linear(special_dim, special_dim)
        self.k_proj = nn.Linear(visual_dim, special_dim)
        self.v_proj = nn.Linear(visual_dim, special_dim)
        self.out_proj = nn.Linear(special_dim, special_dim)

        self.norm_q = nn.LayerNorm(special_dim)
        self.norm_kv = nn.LayerNorm(visual_dim)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-5)

        self.norm_ffn = nn.LayerNorm(special_dim)
        self.ffn = nn.Sequential(
            nn.Linear(special_dim, special_dim * 4),
            nn.GELU(),
            nn.Linear(special_dim * 4, special_dim),
        )
        self.ls_attn = nn.Parameter(torch.ones(special_dim) * 0.01)
        self.ls_ffn = nn.Parameter(torch.ones(special_dim) * 0.01)

    def forward(
        self,
        special_tokens: torch.Tensor,
        visual_features: torch.Tensor,
        frame_sizes: Sequence[int],
    ) -> torch.Tensor:
        if special_tokens.numel() == 0:
            return special_tokens
        if special_tokens.ndim != 3 or special_tokens.shape[-1] != self.special_dim:
            raise ValueError(
                f"special_tokens must have shape (frames, tokens, {self.special_dim}), got {special_tokens.shape}."
            )
        if visual_features.ndim != 2 or visual_features.shape[-1] != self.visual_dim:
            raise ValueError(
                f"visual_features must have shape (tokens, {self.visual_dim}), got {visual_features.shape}."
            )
        if len(frame_sizes) != special_tokens.shape[0]:
            raise ValueError(
                f"frame count mismatch: {len(frame_sizes)} frame sizes for {special_tokens.shape[0]} frames."
            )
        if sum(int(size) for size in frame_sizes) != visual_features.shape[0]:
            raise ValueError(
                "visual token count mismatch: "
                f"sum(frame_sizes)={sum(int(size) for size in frame_sizes)} but got {visual_features.shape[0]}."
            )

        frame_size_to_indices: dict[int, list[int]] = {}
        for frame_idx, frame_size in enumerate(frame_sizes):
            frame_size = int(frame_size)
            if frame_size <= 0:
                raise ValueError(f"frame_sizes[{frame_idx}] must be positive, got {frame_size}.")
            frame_size_to_indices.setdefault(frame_size, []).append(frame_idx)

        visual_splits = torch.split(visual_features, [int(size) for size in frame_sizes], dim=0)
        attention_output = torch.zeros_like(special_tokens)
        num_special_tokens = special_tokens.shape[1]

        for frame_size, indices in frame_size_to_indices.items():
            group_size = len(indices)
            q = self.q_proj(self.norm_q(special_tokens[indices]))
            q = q.view(group_size, num_special_tokens, self.num_heads, self.head_dim)
            q = self.q_norm(q.permute(0, 2, 1, 3))

            visual_group = torch.stack([visual_splits[index] for index in indices], dim=0)
            visual_group = self.norm_kv(visual_group)
            k = self.k_proj(visual_group).view(group_size, frame_size, self.num_heads, self.head_dim)
            v = self.v_proj(visual_group).view(group_size, frame_size, self.num_heads, self.head_dim)
            k = self.k_norm(k.permute(0, 2, 1, 3))
            v = v.permute(0, 2, 1, 3)

            output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            output = output.permute(0, 2, 1, 3).reshape(group_size, num_special_tokens, self.special_dim)
            attention_output[indices] = self.out_proj(output)

        special_tokens = special_tokens + self.ls_attn * attention_output
        return special_tokens + self.ls_ffn * self.ffn(self.norm_ffn(special_tokens))


class GlobalSelfAttentionLayer(nn.Module):
    """Run self-attention over all 17 special tokens across one video at a time."""

    def __init__(self, special_dim: int, num_heads: int):
        super().__init__()
        if special_dim % num_heads != 0:
            raise ValueError(f"special_dim={special_dim} must be divisible by num_heads={num_heads}.")

        self.num_heads = num_heads
        self.head_dim = special_dim // num_heads
        self.special_dim = special_dim
        self.qkv = nn.Linear(special_dim, special_dim * 3)
        self.out_proj = nn.Linear(special_dim, special_dim)
        self.norm = nn.LayerNorm(special_dim)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
        self.norm_ffn = nn.LayerNorm(special_dim)
        self.ffn = nn.Sequential(
            nn.Linear(special_dim, special_dim * 4),
            nn.GELU(),
            nn.Linear(special_dim * 4, special_dim),
        )
        self.ls_attn = nn.Parameter(torch.ones(special_dim) * 0.01)
        self.ls_ffn = nn.Parameter(torch.ones(special_dim) * 0.01)

    def forward(self, special_tokens: torch.Tensor, video_sizes: Sequence[int]) -> torch.Tensor:
        if special_tokens.numel() == 0:
            return special_tokens
        if special_tokens.ndim != 3 or special_tokens.shape[-1] != self.special_dim:
            raise ValueError(
                f"special_tokens must have shape (frames, tokens, {self.special_dim}), got {special_tokens.shape}."
            )
        video_sizes = [int(size) for size in video_sizes]
        if any(size <= 0 for size in video_sizes):
            raise ValueError(f"video_sizes must contain positive frame counts, got {video_sizes}.")
        if sum(video_sizes) != special_tokens.shape[0]:
            raise ValueError(
                f"video frame count mismatch: sum(video_sizes)={sum(video_sizes)} "
                f"but got {special_tokens.shape[0]} frames."
            )

        num_special_tokens = special_tokens.shape[1]
        outputs = []
        for video_tokens in torch.split(special_tokens, video_sizes, dim=0):
            sequence = video_tokens.reshape(-1, self.special_dim)
            sequence_length = sequence.shape[0]
            qkv = self.qkv(self.norm(sequence)).reshape(
                sequence_length,
                3,
                self.num_heads,
                self.head_dim,
            )
            q, k, v = qkv.permute(1, 2, 0, 3).unbind(0)
            q = self.q_norm(q)
            k = self.k_norm(k)
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            output = self.out_proj(output.transpose(0, 1).reshape(sequence_length, self.special_dim))
            sequence = sequence + self.ls_attn * output
            sequence = sequence + self.ls_ffn * self.ffn(self.norm_ffn(sequence))
            outputs.append(sequence.reshape(video_tokens.shape[0], num_special_tokens, self.special_dim))

        return torch.cat(outputs, dim=0)


class SceneDistillPreProjector(nn.Module):
    """Project 2048-D distilled features into the Qwen text hidden space."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear_fc1 = nn.Linear(input_dim, input_dim)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.to(self.linear_fc1.weight.dtype)
        features = self.norm(features)
        return self.linear_fc2(self.act_fn(self.linear_fc1(features)))


class SceneDistillPreModule(nn.Module):
    """Four-stage pre-LLM camera-scene GCTE student and its LLM projector."""

    def __init__(
        self,
        visual_dim: int,
        text_hidden_dim: int,
        stream_dim: int = STREAM_DIM,
        num_heads: int = NUM_HEADS,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.stream_dim = stream_dim
        self.feature_dim = 2 * stream_dim

        self.pre_camera_token = nn.Parameter(torch.empty(1, 2, 1, stream_dim))
        self.pre_scene_token = nn.Parameter(torch.empty(1, 2, NUM_SCENE_TOKENS, stream_dim))
        self.pre_frame_layers = nn.ModuleList(
            FrameCrossAttentionLayer(stream_dim, visual_dim, num_heads)
            for _ in range(PRE_DISTILL_DEPTH)
        )
        self.pre_global_layers = nn.ModuleList(
            GlobalSelfAttentionLayer(stream_dim, num_heads)
            for _ in range(PRE_DISTILL_DEPTH)
        )
        self.pre_projector = SceneDistillPreProjector(self.feature_dim, text_hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pre_camera_token, std=1e-3)
        nn.init.normal_(self.pre_scene_token, std=1e-3)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def prepare_pre_special_tokens(self, video_sizes: Sequence[int]) -> torch.Tensor:
        video_sizes = [int(size) for size in video_sizes]
        if not video_sizes or any(size <= 0 for size in video_sizes):
            raise ValueError(f"video_sizes must contain positive frame counts, got {video_sizes}.")

        first_variant = torch.cat([self.pre_camera_token[:, 0], self.pre_scene_token[:, 0]], dim=1)
        other_variant = torch.cat([self.pre_camera_token[:, 1], self.pre_scene_token[:, 1]], dim=1)
        per_video_tokens = []
        for video_size in video_sizes:
            per_video_tokens.append(first_variant)
            if video_size > 1:
                per_video_tokens.append(other_variant.expand(video_size - 1, -1, -1))
        return torch.cat(per_video_tokens, dim=0)

    def forward(
        self,
        visual_layer_outputs: Sequence[torch.Tensor],
        frame_sizes: Sequence[int],
        video_sizes: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(visual_layer_outputs) != PRE_DISTILL_DEPTH:
            raise ValueError(
                f"SceneDistill Pre requires {PRE_DISTILL_DEPTH} visual layers, got {len(visual_layer_outputs)}."
            )
        if len(frame_sizes) != sum(int(size) for size in video_sizes):
            raise ValueError(
                f"frame_sizes has {len(frame_sizes)} entries but video_sizes describes "
                f"{sum(int(size) for size in video_sizes)} frames."
            )

        special_tokens = self.prepare_pre_special_tokens(video_sizes)
        target_device = self.pre_camera_token.device
        target_dtype = self.pre_camera_token.dtype
        special_tokens = special_tokens.to(device=target_device, dtype=target_dtype)

        special_after_frame = None
        for layer_index, visual_features in enumerate(visual_layer_outputs):
            visual_features = visual_features.detach().to(device=target_device, dtype=target_dtype)
            special_tokens = self.pre_frame_layers[layer_index](special_tokens, visual_features, frame_sizes)
            if layer_index == PRE_DISTILL_DEPTH - 1:
                special_after_frame = special_tokens
            special_tokens = self.pre_global_layers[layer_index](special_tokens, video_sizes)

        pre_features = torch.cat([special_after_frame, special_tokens], dim=-1)
        expected_shape = (sum(int(size) for size in video_sizes), NUM_SPECIAL_TOKENS, self.feature_dim)
        if pre_features.shape != expected_shape:
            raise ValueError(f"SceneDistill Pre features must have shape {expected_shape}, got {pre_features.shape}.")
        pre_embeds = self.pre_projector(pre_features)
        return pre_embeds, pre_features, special_tokens


class SceneDistillPostModule(nn.Module):
    """Six-stage post-LLM camera-scene GCTE with internal injection."""

    def __init__(
        self,
        llm_hidden_dim: int,
        special_dim: int = STREAM_DIM,
        num_heads: int = NUM_HEADS,
        depth: int = POST_DISTILL_DEPTH,
    ):
        super().__init__()
        if depth != POST_DISTILL_DEPTH:
            raise ValueError(
                f"SceneDistill Post depth is fixed at {POST_DISTILL_DEPTH}, got {depth}."
            )

        self.llm_hidden_dim = int(llm_hidden_dim)
        self.special_dim = int(special_dim)
        self.feature_dim = 2 * self.special_dim
        self.layer_indices = LLM_BLOCK_INDICES
        self.post_frame_layers = nn.ModuleList(
            FrameCrossAttentionLayer(self.special_dim, self.llm_hidden_dim, num_heads)
            for _ in range(POST_DISTILL_DEPTH)
        )
        self.post_global_layers = nn.ModuleList(
            GlobalSelfAttentionLayer(self.special_dim, num_heads)
            for _ in range(POST_DISTILL_DEPTH)
        )
        self.post_injection_projections = nn.ModuleList(
            nn.Linear(self.special_dim, self.llm_hidden_dim, bias=False)
            for _ in range(POST_DISTILL_DEPTH)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for projection in self.post_injection_projections:
            nn.init.zeros_(projection.weight)

    def forward(
        self,
        stage_index: int,
        post_tokens: torch.Tensor,
        llm_layer_features: torch.Tensor,
        frame_sizes: Sequence[int],
        video_sizes: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stage_index = int(stage_index)
        if stage_index < 0 or stage_index >= POST_DISTILL_DEPTH:
            raise ValueError(
                f"SceneDistill Post stage_index must be in [0, {POST_DISTILL_DEPTH - 1}], got {stage_index}."
            )

        video_sizes = [int(size) for size in video_sizes]
        frame_sizes = [int(size) for size in frame_sizes]
        num_frames = sum(video_sizes)
        expected_pre_shape = (num_frames, NUM_SPECIAL_TOKENS, self.special_dim)
        if post_tokens.shape != expected_pre_shape:
            raise ValueError(
                f"post_tokens must have shape {expected_pre_shape}, got {post_tokens.shape}."
            )
        if len(frame_sizes) != num_frames or any(size <= NUM_SPECIAL_TOKENS for size in frame_sizes):
            raise ValueError(
                "SceneDistill Post frame_sizes must contain one positive visual span plus "
                f"{NUM_SPECIAL_TOKENS} special tokens per frame, got {frame_sizes}."
            )

        expected_llm_tokens = sum(frame_sizes)
        expected_shape = (expected_llm_tokens, self.llm_hidden_dim)
        if llm_layer_features.shape != expected_shape:
            raise ValueError(
                f"LLM layer {LLM_BLOCK_INDICES[stage_index]} features must have shape "
                f"{expected_shape}, got {llm_layer_features.shape}."
            )

        reference_parameter = next(self.parameters())
        post_tokens = post_tokens.to(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        llm_layer_features = llm_layer_features.to(device=post_tokens.device, dtype=post_tokens.dtype)
        post_after_frame = self.post_frame_layers[stage_index](
            post_tokens,
            llm_layer_features,
            frame_sizes,
        )
        post_after_global = self.post_global_layers[stage_index](post_after_frame, video_sizes)
        injection_delta = self.post_injection_projections[stage_index](post_after_global)

        expected_special_shape = (num_frames, NUM_SPECIAL_TOKENS, self.special_dim)
        expected_injection_shape = (num_frames, NUM_SPECIAL_TOKENS, self.llm_hidden_dim)
        if post_after_frame.shape != expected_special_shape or post_after_global.shape != expected_special_shape:
            raise ValueError(
                "SceneDistill Post special states must both have shape "
                f"{expected_special_shape}, got {post_after_frame.shape} and {post_after_global.shape}."
            )
        if injection_delta.shape != expected_injection_shape:
            raise ValueError(
                f"SceneDistill injection delta must have shape {expected_injection_shape}, "
                f"got {injection_delta.shape}."
            )
        return post_after_frame, post_after_global, injection_delta


def scene_distillation_loss(student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
    """Index-aligned cosine loss: sum 17 tokens per frame, then mean frames."""
    if student_features.shape != teacher_features.shape:
        raise ValueError(
            f"student/teacher shape mismatch: student={student_features.shape}, teacher={teacher_features.shape}."
        )
    if student_features.ndim != 3 or student_features.shape[1:] != (NUM_SPECIAL_TOKENS, FEATURE_DIM):
        raise ValueError(
            "SceneDistill features must have shape "
            f"(frames, {NUM_SPECIAL_TOKENS}, {FEATURE_DIM}), got {student_features.shape}."
        )
    if student_features.shape[0] == 0:
        raise ValueError("SceneDistill requires at least one frame for distillation.")

    student = student_features.float()
    teacher = teacher_features.detach().to(device=student.device, dtype=torch.float32)
    if not torch.isfinite(student).all():
        raise FloatingPointError("SceneDistill student features contain non-finite values.")
    if not torch.isfinite(teacher).all():
        raise FloatingPointError("VGGT-Omega teacher features contain non-finite values.")

    per_token_loss = 1.0 - F.cosine_similarity(student, teacher, dim=-1)
    return per_token_loss.sum(dim=-1).mean()


__all__ = [
    "FEATURE_DIM",
    "FrameCrossAttentionLayer",
    "GlobalSelfAttentionLayer",
    "LLM_BLOCK_INDICES",
    "NUM_SCENE_TOKENS",
    "NUM_SPECIAL_TOKENS",
    "POST_DISTILL_DEPTH",
    "POST_DISTILL_WEIGHT",
    "PRE_DISTILL_DEPTH",
    "PRE_DISTILL_WEIGHT",
    "PRE_VISION_BLOCK_INDICES",
    "SceneDistillPostModule",
    "SceneDistillPreModule",
    "SceneDistillPreProjector",
    "STREAM_DIM",
    "remove_teacher_weights",
    "scene_distillation_loss",
    "select_pre_vision_layer_outputs",
]
