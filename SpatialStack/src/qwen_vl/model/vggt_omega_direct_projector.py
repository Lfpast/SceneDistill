"""Project VGGT-Omega direct tokens into the Qwen3.5 text hidden space."""

from __future__ import annotations

import torch
import torch.nn as nn


def resolve_progressive_hidden_dim(input_dim: int, hidden_dim: int, output_dim: int) -> int:
    """Pick a hidden dim that sits strictly between input_dim and output_dim
    when the caller passes hidden_dim==input_dim or hidden_dim==output_dim
    (which would degenerate the two-layer MLP into a single Linear).
    """
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


class VGGTOmegaDirectProjector(nn.Module):
    """VGGT-Omega direct tokens → LLM hidden dim.

    Structure: LayerNorm -> Linear -> GELU -> Linear (progressive hidden dim).
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 3072, output_dim: int = 4096):
        super().__init__()
        hidden_dim = resolve_progressive_hidden_dim(input_dim, hidden_dim, output_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.linear_fc1 = nn.Linear(input_dim, hidden_dim)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., input_dim) VGGT-Omega direct tokens (camera / register / special17)
        Returns:
            (..., output_dim) projected embeddings ready to inject into the LLM
        """
        x = x.to(self.linear_fc1.weight.dtype)
        x = self.norm(x)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


__all__ = ["VGGTOmegaDirectProjector", "resolve_progressive_hidden_dim"]
