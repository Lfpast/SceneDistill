"""Projector for VGGT-Omega alpha camera and scene tokens."""

import torch.nn as nn


class VGGTOmegaAlphaProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, tokens):
        return self.mlp(tokens)
