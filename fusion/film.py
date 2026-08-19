"""
Block 8: FiLM (Feature-wise Linear Modulation).

Turns each object's relevance score (block 7) into a per-channel scale/shift
that amplifies embeddings of objects the instruction actually cares about
and suppresses the rest, before threshold filtering (block 9) makes a hard
keep/drop decision.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, dim: int, cond_dim: int = 1):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim * 2),
        )
        # Zero-init the last layer so gamma/beta start at 0 -> the module
        # starts as an identity transform (object_embed * 1 + 0) and only
        # learns to modulate as training progresses.
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def forward(self, object_embed: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
        """
        Args:
            object_embed: [B, N, D]
            relevance: [B, N] (or [B, N, cond_dim] if cond_dim > 1)

        Returns:
            [B, N, D] modulated object embedding.
        """
        cond = relevance.unsqueeze(-1) if relevance.dim() == object_embed.dim() - 1 else relevance
        gamma, beta = self.generator(cond).chunk(2, dim=-1)
        return object_embed * (1.0 + gamma) + beta


if __name__ == "__main__":
    torch.manual_seed(0)
    dim = 256
    film = FiLM(dim=dim)

    B, N = 2, 5
    object_embed = torch.randn(B, N, dim, requires_grad=True)
    relevance = torch.randn(B, N, requires_grad=True)

    out = film(object_embed, relevance)
    print("output shape:", out.shape)
    assert out.shape == (B, N, dim)

    # Zero-init check: at init, FiLM should be a near-identity transform.
    identity_diff = (out - object_embed).abs().max().item()
    print(f"max |out - input| at init (should be ~0): {identity_diff:.6f}")
    assert identity_diff < 1e-5

    loss = out.sum()
    loss.backward()
    assert object_embed.grad is not None and relevance.grad is not None
    assert all(p.grad is not None for p in film.parameters())
    print("gradients flow through both inputs and all parameters: OK")
