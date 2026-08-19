"""
Block 6: Projection/Adapter.

Combines each object's raw features -- EEF-relative 3D position (block 5)
and a semantic embedding of the SAM3-matched label (e.g. the frozen
Florence2 token embedding of "bowl") -- into one common embedding space
that the rest of the Fusion pipeline (blocks 7-10) operates on.

The language embedding lookup itself lives with the X-VLA model (Florence2,
loaded only in the XVLA conda env) and is wired in during Phase 3 policy
integration. This block is pure tensor math and is unit-tested here with
synthetic tensors of the right shape (lang_embed_dim=1024 matches
Florence2's projection_dim, see X-VLA/models/configuration_xvla.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ObjectProjection(nn.Module):
    def __init__(self, pos_dim: int = 3, lang_embed_dim: int = 1024, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(pos_dim + lang_embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, relative_xyz: torch.Tensor, lang_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            relative_xyz: [..., pos_dim] EEF-relative position, block 5 output.
            lang_embed: [..., lang_embed_dim] embedding of the object's matched
                text label (mean-pooled over subword tokens if multi-token).

        Returns:
            [..., out_dim] projected object token.
        """
        x = torch.cat([relative_xyz, lang_embed], dim=-1)
        return self.net(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    proj = ObjectProjection(pos_dim=3, lang_embed_dim=1024, out_dim=256)

    # Synthetic batch: B=2 scenes, N=5 objects each (variable count is fine,
    # this module operates per-object independently).
    relative_xyz = torch.randn(2, 5, 3, requires_grad=True)
    lang_embed = torch.randn(2, 5, 1024, requires_grad=True)

    out = proj(relative_xyz, lang_embed)
    print("output shape:", out.shape)
    assert out.shape == (2, 5, 256)

    loss = out.sum()
    loss.backward()
    assert relative_xyz.grad is not None and lang_embed.grad is not None
    assert all(p.grad is not None for p in proj.parameters())
    print("gradients flow through both inputs and all parameters: OK")
