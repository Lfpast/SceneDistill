"""Packing helpers for the VGGT-Omega alpha Qwen3.5 input path."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


def _frame_run_starts(input_ids_row: torch.Tensor, placeholder_id: int) -> torch.Tensor:
    is_placeholder = input_ids_row == placeholder_id
    if not bool(is_placeholder.any()):
        return is_placeholder
    prev = torch.zeros_like(is_placeholder)
    prev[1:] = is_placeholder[:-1]
    return is_placeholder & ~prev


def expand_visual_placeholders(
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    placeholder_token_id: int,
    num_extra_per_frame: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Insert extra image placeholder tokens before each visual run.

    The inserted tokens inherit the same temporal coordinate as the frame and
    are assigned the center position of that frame's patch run in H/W space.
    """

    if num_extra_per_frame <= 0:
        return input_ids, labels, attention_mask, position_ids

    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    extra = num_extra_per_frame

    runs_per_row = torch.zeros(batch_size, dtype=torch.long, device=device)
    for row_idx in range(batch_size):
        runs_per_row[row_idx] = _frame_run_starts(input_ids[row_idx], placeholder_token_id).sum()
    max_runs = int(runs_per_row.max().item()) if batch_size > 0 else 0
    new_seq_len = seq_len + max_runs * extra

    new_input_ids = torch.full((batch_size, new_seq_len), 0, device=device, dtype=input_ids.dtype)
    new_labels = (
        torch.full((batch_size, new_seq_len), -100, device=device, dtype=labels.dtype)
        if labels is not None
        else None
    )
    new_attention_mask = (
        torch.zeros((batch_size, new_seq_len), device=device, dtype=attention_mask.dtype)
        if attention_mask is not None
        else None
    )

    if position_ids is None:
        new_position_ids = None
    elif position_ids.dim() == 3:
        new_position_ids = torch.zeros(
            (position_ids.shape[0], batch_size, new_seq_len),
            device=position_ids.device,
            dtype=position_ids.dtype,
        )
    elif position_ids.dim() == 2:
        new_position_ids = torch.zeros(
            (batch_size, new_seq_len),
            device=position_ids.device,
            dtype=position_ids.dtype,
        )
    else:
        raise ValueError(f"Unsupported position_ids shape {tuple(position_ids.shape)}")

    for row_idx in range(batch_size):
        ids_row = input_ids[row_idx]
        starts_mask = _frame_run_starts(ids_row, placeholder_token_id)
        offsets = starts_mask.long().cumsum(0) * extra
        src_idx = torch.arange(seq_len, device=device)
        dst_idx = src_idx + offsets

        new_input_ids[row_idx, dst_idx] = ids_row
        if new_labels is not None and labels is not None:
            new_labels[row_idx, dst_idx] = labels[row_idx]
        if new_attention_mask is not None and attention_mask is not None:
            new_attention_mask[row_idx, dst_idx] = attention_mask[row_idx]
        if new_position_ids is not None:
            if position_ids.dim() == 3:
                new_position_ids[:, row_idx, dst_idx] = position_ids[:, row_idx, :]
            else:
                new_position_ids[row_idx, dst_idx] = position_ids[row_idx, :]

        starts_dst = dst_idx[starts_mask]
        if starts_dst.numel() == 0:
            continue

        is_placeholder_orig = ids_row == placeholder_token_id
        run_lengths: List[int] = []
        current = 0
        for value in is_placeholder_orig.tolist():
            if value:
                current += 1
            elif current > 0:
                run_lengths.append(current)
                current = 0
        if current > 0:
            run_lengths.append(current)

        if len(run_lengths) != starts_dst.numel():
            run_lengths = [0] * starts_dst.numel()

        for run_idx in range(starts_dst.numel()):
            run_start = int(starts_dst[run_idx].item())
            run_length = run_lengths[run_idx]
            run_end = min(run_start + run_length, new_seq_len)

            if new_position_ids is not None and position_ids is not None:
                if position_ids.dim() == 3:
                    position_slice = new_position_ids[:, row_idx, run_start:run_end]
                    if position_slice.numel() == 0:
                        anchor_t = int(new_position_ids[0, row_idx, run_start].item())
                        anchor_h = int(new_position_ids[1, row_idx, run_start].item())
                        anchor_w = int(new_position_ids[2, row_idx, run_start].item())
                    else:
                        anchor_t = int(position_slice[0, 0].item())
                        anchor_h = int(position_slice[1].max().item()) // 2
                        anchor_w = int(position_slice[2].max().item()) // 2
                else:
                    anchor_t = int(new_position_ids[row_idx, run_start].item())
                    anchor_h = anchor_t
                    anchor_w = anchor_t

            for extra_idx in range(extra):
                insert_pos = run_start - extra + extra_idx
                new_input_ids[row_idx, insert_pos] = placeholder_token_id
                if new_attention_mask is not None:
                    new_attention_mask[row_idx, insert_pos] = 1

                if new_position_ids is not None and position_ids is not None:
                    if position_ids.dim() == 3:
                        new_position_ids[0, row_idx, insert_pos] = anchor_t
                        new_position_ids[1, row_idx, insert_pos] = anchor_h
                        new_position_ids[2, row_idx, insert_pos] = anchor_w
                    else:
                        new_position_ids[row_idx, insert_pos] = anchor_t

    return new_input_ids, new_labels, new_attention_mask, new_position_ids


def expand_image_embeds_with_alpha_tokens(
    image_embeds: torch.Tensor,
    alpha_embeds: torch.Tensor,
    patches_per_frame: Sequence[int],
) -> torch.Tensor:
    expected_tokens = sum(int(v) for v in patches_per_frame)
    if image_embeds.shape[0] != expected_tokens:
        raise ValueError(
            f"image_embeds has {image_embeds.shape[0]} tokens, expected {expected_tokens}."
        )
    if alpha_embeds.shape[0] != len(patches_per_frame):
        raise ValueError(
            f"alpha_embeds has {alpha_embeds.shape[0]} frames, expected {len(patches_per_frame)}."
        )

    chunks = []
    offset = 0
    for frame_idx, patch_count in enumerate(patches_per_frame):
        chunks.append(alpha_embeds[frame_idx].to(dtype=image_embeds.dtype, device=image_embeds.device))
        chunks.append(image_embeds[offset:offset + int(patch_count)])
        offset += int(patch_count)
    return torch.cat(chunks, dim=0)


def build_alpha_only_mask(expanded_image_mask_2d: torch.Tensor, num_extra_per_frame: int) -> torch.Tensor:
    mask = torch.zeros_like(expanded_image_mask_2d, dtype=torch.bool)
    if num_extra_per_frame <= 0:
        return mask

    batch_size, seq_len = expanded_image_mask_2d.shape
    for row_idx in range(batch_size):
        row_mask = expanded_image_mask_2d[row_idx]
        pos = 0
        while pos < seq_len:
            if not row_mask[pos]:
                pos += 1
                continue
            run_end = pos
            while run_end < seq_len and row_mask[run_end]:
                run_end += 1
            special_end = min(pos + num_extra_per_frame, run_end)
            mask[row_idx, pos:special_end] = True
            pos = run_end
    return mask


__all__ = [
    "build_alpha_only_mask",
    "expand_image_embeds_with_alpha_tokens",
    "expand_visual_placeholders",
]
