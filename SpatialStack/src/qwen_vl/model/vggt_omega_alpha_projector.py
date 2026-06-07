"""Project VGGT-Omega alpha tokens into the Qwen3.5 text hidden space."""

from __future__ import annotations

import torch
import torch.nn as nn


def resolve_progressive_hidden_dim(input_dim: int, hidden_dim: int, output_dim: int) -> int:
    if hidden_dim != input_dim and hidden_dim != output_dim:
        return hidden_dim

    midpoint = (input_dim + output_dim) // 2
    if midpoint != input_dim and midpoint != output_dim:
        return midpoint

    step = max(abs(output_dim - input_dim) // 2, 1)
    candidate = min(input_dim, output_dim) + step
    if candidate != input_dim and candidate != output_dim:
        return candidate

    return max(input_dim, output_dim) + 1


class VGGTOmegaAlphaProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        hidden_dim = resolve_progressive_hidden_dim(input_dim, hidden_dim, output_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(dtype=self.fc1.weight.dtype)
        x = self.fc2(self.act(self.fc1(x)))
        return x.to(dtype=input_dtype)
