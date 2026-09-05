"""
Recurrent Latent-Reasoning Model for Maze Path Solving.
Operates via K iterative loops of a shared-weight Convolutional GRU core in latent tensor space.
Information propagates across grid cells without emitting intermediate language tokens,
producing empirical inference-time scaling and diminishing returns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRURecurrentCore(nn.Module):
    """
    Convolutional Gated Recurrent Unit (ConvGRU) shared core.
    Gated recurrence prevents latent magnitude explosion over large K,
    enabling stable iterative refinement and smooth plateauing curves.
    
    Spatial neighbor propagation (3x3 conv) is applied to recurrent state h_prev,
    while constant input context h_init is injected via 1x1 projection.
    """

    def __init__(self, hidden_dim: int = 48):
        super().__init__()
        # Spatial gating on recurrent state (receptive field expansion)
        self.gate_h = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.gate_x = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=1)

        # Spatial candidate state update
        self.cand_h = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.cand_x = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, h_prev: torch.Tensor, h_init: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_h(h_prev) + self.gate_x(h_init))
        r, z = torch.chunk(gates, 2, dim=1)
        cand = torch.tanh(self.cand_h(r * h_prev) + self.cand_x(h_init))
        h_next = (1.0 - z) * h_prev + z * cand
        return h_next


class RecurrentLatentMazeSolver(nn.Module):
    """
    Solves 2D maze pathfinding via iterative recurrent latent computation.
    
    Architecture:
    1. Input Encoder: 1x1 conv mapping (3, H, W) [walls, start, goal] into latent space (hidden_dim, H, W).
       Using 1x1 conv ensures spatial receptive field is initially 0, so solving MUST occur
       via the recurrent core loops.
    2. Shared Recurrent Core: Applied sequentially K times (effort parameter).
       Each step expands the effective receptive field by 2 grid cells.
    3. Readout Head: 1x1 conv mapping final latent state h_K to (1, H, W) path probability map.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 48):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection (point-wise 1x1 conv + normalization)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(num_groups=min(4, hidden_dim), num_channels=hidden_dim),
            nn.GELU(),
        )

        # Recurrent Core with shared weights across all reasoning steps
        self.recurrent_core = ConvGRURecurrentCore(hidden_dim=hidden_dim)

        # Output Readout Head (point-wise 1x1 conv + sigmoid)
        self.readout = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, k: int = 1) -> torch.Tensor:
        """
        Forward pass applying K loops of recurrent latent reasoning.
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            k: Number of recurrent reasoning iterations (effort level, k >= 1)
        Returns:
            Probability grid of shape (batch_size, 1, H, W)
        """
        if k < 1:
            raise ValueError(f"Effort step count K must be >= 1, got {k}")

        h_init = self.encoder(x)
        h_curr = h_init

        for _ in range(k):
            h_curr = self.recurrent_core(h_curr, h_init)

        out_prob = self.readout(h_curr)
        return out_prob


if __name__ == "__main__":
    model = RecurrentLatentMazeSolver(hidden_dim=64)
    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RecurrentLatentMazeSolver(hidden_dim=64) parameter count: {total_p:,}")
    dummy = torch.randn(2, 3, 10, 10)
    for k_val in [1, 5, 10]:
        y = model(dummy, k=k_val)
        print(f"K={k_val} output shape: {y.shape}")
