"""Helpers for direct visual-token injection into Qwen3.5 image spans."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


def _validate_insert_position(insert_position: str) -> str:
    if insert_position not in {"front", "back"}:
        raise ValueError(f"Unsupported insert_position={insert_position!r}; expected 'front' or 'back'.")
    return insert_position


def _find_placeholder_runs(input_ids_row: torch.Tensor, placeholder_id: int) -> List[Tuple[int, int]]:
    is_placeholder = input_ids_row == placeholder_id
    if not bool(is_placeholder.any()):
        return []

    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, is_run_token in enumerate(is_placeholder.tolist()):
        if is_run_token and start is None:
            start = idx
        elif not is_run_token and start is not None:
            runs.append((start, idx))
            start = None

    if start is not None:
        runs.append((start, input_ids_row.shape[0]))
    return runs


def _build_extra_position_ids(
    run_position_ids: Optional[torch.Tensor],
    num_extra_tokens: int,
) -> Optional[torch.Tensor]:
    if run_position_ids is None:
        return None

    if run_position_ids.dim() == 2:
        # Vision MRoPE path: [3, run_len]. Place the inserted token(s) at the
        # frame's top-left coordinate in H/W while keeping the same temporal
        # coordinate as the frame.
        temporal = int(run_position_ids[0, 0].item())

        height_ids = run_position_ids[1]
        width_ids = run_position_ids[2]
        height_min = int(height_ids.min().item())
        width_min = int(width_ids.min().item())

        return torch.tensor(
            [[temporal], [height_min], [width_min]],
            device=run_position_ids.device,
            dtype=run_position_ids.dtype,
        ).expand(-1, num_extra_tokens)

    if run_position_ids.dim() == 1:
        anchor = int(run_position_ids[0].item())
        return torch.full(
            (num_extra_tokens,),
            anchor,
            device=run_position_ids.device,
            dtype=run_position_ids.dtype,
        )

    raise ValueError(f"Unsupported per-row position_ids shape {tuple(run_position_ids.shape)}")


def _concat_position_parts(parts: List[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    if reference.dim() == 2:
        return torch.cat(parts, dim=1)
    return torch.cat(parts, dim=0)


def _expand_visual_row(
    input_ids_row: torch.Tensor,
    labels_row: Optional[torch.Tensor],
    attention_mask_row: Optional[torch.Tensor],
    position_ids_row: Optional[torch.Tensor],
    placeholder_token_id: int,
    num_extra_per_frame: int,
    insert_position: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    runs = _find_placeholder_runs(input_ids_row, placeholder_token_id)
    if not runs or num_extra_per_frame <= 0:
        return input_ids_row, labels_row, attention_mask_row, position_ids_row

    device = input_ids_row.device
    cursor = 0

    input_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []
    attention_parts: List[torch.Tensor] = []
    position_parts: List[torch.Tensor] = []

    for start, end in runs:
        if start > cursor:
            input_parts.append(input_ids_row[cursor:start])
            if labels_row is not None:
                label_parts.append(labels_row[cursor:start])
            if attention_mask_row is not None:
                attention_parts.append(attention_mask_row[cursor:start])
            if position_ids_row is not None:
                if position_ids_row.dim() == 2:
                    position_parts.append(position_ids_row[:, cursor:start])
                else:
                    position_parts.append(position_ids_row[cursor:start])

        run_input_ids = input_ids_row[start:end]
        run_labels = labels_row[start:end] if labels_row is not None else None
        run_attention = attention_mask_row[start:end] if attention_mask_row is not None else None
        if position_ids_row is not None:
            if position_ids_row.dim() == 2:
                run_positions = position_ids_row[:, start:end]
            else:
                run_positions = position_ids_row[start:end]
        else:
            run_positions = None

        extra_input_ids = torch.full(
            (num_extra_per_frame,),
            placeholder_token_id,
            device=device,
            dtype=input_ids_row.dtype,
        )
        extra_labels = (
            torch.full(
                (num_extra_per_frame,),
                -100,
                device=labels_row.device,
                dtype=labels_row.dtype,
            )
            if labels_row is not None
            else None
        )
        extra_attention = (
            torch.ones(
                (num_extra_per_frame,),
                device=attention_mask_row.device,
                dtype=attention_mask_row.dtype,
            )
            if attention_mask_row is not None
            else None
        )
        extra_positions = _build_extra_position_ids(run_positions, num_extra_per_frame)

        if insert_position == "front":
            input_parts.extend([extra_input_ids, run_input_ids])
            if labels_row is not None:
                label_parts.extend([extra_labels, run_labels])
            if attention_mask_row is not None:
                attention_parts.extend([extra_attention, run_attention])
            if position_ids_row is not None:
                position_parts.extend([extra_positions, run_positions])
        else:
            input_parts.extend([run_input_ids, extra_input_ids])
            if labels_row is not None:
                label_parts.extend([run_labels, extra_labels])
            if attention_mask_row is not None:
                attention_parts.extend([run_attention, extra_attention])
            if position_ids_row is not None:
                position_parts.extend([run_positions, extra_positions])

        cursor = end

    if cursor < input_ids_row.shape[0]:
        input_parts.append(input_ids_row[cursor:])
        if labels_row is not None:
            label_parts.append(labels_row[cursor:])
        if attention_mask_row is not None:
            attention_parts.append(attention_mask_row[cursor:])
        if position_ids_row is not None:
            if position_ids_row.dim() == 2:
                position_parts.append(position_ids_row[:, cursor:])
            else:
                position_parts.append(position_ids_row[cursor:])

    new_input_ids = torch.cat(input_parts, dim=0)
    new_labels = torch.cat(label_parts, dim=0) if labels_row is not None else None
    new_attention = torch.cat(attention_parts, dim=0) if attention_mask_row is not None else None
    new_positions = _concat_position_parts(position_parts, position_ids_row) if position_ids_row is not None else None
    return new_input_ids, new_labels, new_attention, new_positions


def expand_visual_placeholders(
    input_ids: torch.Tensor,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    placeholder_token_id: int,
    num_extra_per_frame: int,
    insert_position: str = "front",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Insert extra placeholder tokens into each visual span.

    Expand the visual span in `input_ids`, then place the injected token(s)
    at the frame-center MRoPE coordinate while leaving the downstream text
    positions unchanged.
    """

    insert_position = _validate_insert_position(insert_position)
    if num_extra_per_frame <= 0:
        return input_ids, labels, attention_mask, position_ids

    batch_size = input_ids.shape[0]

    expanded_rows: List[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]] = []
    max_seq_len = input_ids.shape[1]

    for row_idx in range(batch_size):
        row_position_ids = None
        if position_ids is not None:
            if position_ids.dim() == 3:
                row_position_ids = position_ids[:, row_idx, :]
            elif position_ids.dim() == 2:
                row_position_ids = position_ids[row_idx]
            else:
                raise ValueError(f"Unsupported position_ids shape {tuple(position_ids.shape)}")

        expanded_row = _expand_visual_row(
            input_ids_row=input_ids[row_idx],
            labels_row=(labels[row_idx] if labels is not None else None),
            attention_mask_row=(attention_mask[row_idx] if attention_mask is not None else None),
            position_ids_row=row_position_ids,
            placeholder_token_id=placeholder_token_id,
            num_extra_per_frame=num_extra_per_frame,
            insert_position=insert_position,
        )
        expanded_rows.append(expanded_row)
        max_seq_len = max(max_seq_len, expanded_row[0].shape[0])

    new_input_ids = torch.zeros(
        (batch_size, max_seq_len),
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    new_labels = (
        torch.full((batch_size, max_seq_len), -100, device=labels.device, dtype=labels.dtype)
        if labels is not None
        else None
    )
    new_attention_mask = (
        torch.zeros((batch_size, max_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
        if attention_mask is not None
        else None
    )

    if position_ids is None:
        new_position_ids = None
    elif position_ids.dim() == 3:
        new_position_ids = torch.zeros(
            (position_ids.shape[0], batch_size, max_seq_len),
            device=position_ids.device,
            dtype=position_ids.dtype,
        )
    else:
        new_position_ids = torch.zeros(
            (batch_size, max_seq_len),
            device=position_ids.device,
            dtype=position_ids.dtype,
        )

    for row_idx, (row_input_ids, row_labels, row_attention, row_positions) in enumerate(expanded_rows):
        row_len = row_input_ids.shape[0]
        new_input_ids[row_idx, :row_len] = row_input_ids
        if new_labels is not None and row_labels is not None:
            new_labels[row_idx, :row_len] = row_labels
        if new_attention_mask is not None and row_attention is not None:
            new_attention_mask[row_idx, :row_len] = row_attention
        if new_position_ids is not None and row_positions is not None:
            if new_position_ids.dim() == 3:
                new_position_ids[:, row_idx, :row_len] = row_positions
            else:
                new_position_ids[row_idx, :row_len] = row_positions

    return new_input_ids, new_labels, new_attention_mask, new_position_ids


def expand_image_embeds_with_direct_tokens(
    image_embeds: torch.Tensor,
    direct_embeds: torch.Tensor,
    patches_per_frame: Sequence[int],
    insert_position: str = "front",
) -> torch.Tensor:
    insert_position = _validate_insert_position(insert_position)

    expected_tokens = sum(int(v) for v in patches_per_frame)
    if image_embeds.shape[0] != expected_tokens:
        raise ValueError(
            f"image_embeds has {image_embeds.shape[0]} tokens, expected {expected_tokens}."
        )
    if direct_embeds.shape[0] != len(patches_per_frame):
        raise ValueError(
            f"direct_embeds has {direct_embeds.shape[0]} frames, expected {len(patches_per_frame)}."
        )

    chunks = []
    offset = 0
    for frame_idx, patch_count in enumerate(patches_per_frame):
        patch_tokens = image_embeds[offset:offset + int(patch_count)]
        direct_tokens = direct_embeds[frame_idx].to(dtype=image_embeds.dtype, device=image_embeds.device)
        if insert_position == "front":
            chunks.extend([direct_tokens, patch_tokens])
        else:
            chunks.extend([patch_tokens, direct_tokens])
        offset += int(patch_count)
    return torch.cat(chunks, dim=0)


def build_direct_only_mask(
    expanded_image_mask_2d: torch.Tensor,
    num_extra_per_frame: int,
    insert_position: str = "front",
) -> torch.Tensor:
    insert_position = _validate_insert_position(insert_position)
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
            if insert_position == "front":
                special_start = pos
                special_end = min(pos + num_extra_per_frame, run_end)
            else:
                special_start = max(run_end - num_extra_per_frame, pos)
                special_end = run_end
            mask[row_idx, special_start:special_end] = True
            pos = run_end
    return mask


def compute_mrope_position_deltas(
    position_ids: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if position_ids is None or position_ids.dim() != 3:
        return None

    deltas = []
    seq_len = input_ids.shape[1]
    for row_idx in range(input_ids.shape[0]):
        if attention_mask is not None:
            valid_positions = position_ids[:, row_idx, attention_mask[row_idx].bool()]
        else:
            valid_positions = position_ids[:, row_idx]

        if valid_positions.numel() == 0:
            deltas.append(0)
            continue

        deltas.append(int(valid_positions.max().item()) + 1 - seq_len)

    return torch.tensor(deltas, device=input_ids.device).unsqueeze(1)


__all__ = [
    "build_direct_only_mask",
    "compute_mrope_position_deltas",
    "expand_image_embeds_with_direct_tokens",
    "expand_visual_placeholders",
]
