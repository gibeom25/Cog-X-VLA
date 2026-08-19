"""
Block 10: set-invariant encoder.

The number of detected objects varies per frame, but X-VLA's transformer
needs a fixed-length token stream (its positional embedding is a plain
indexed parameter, see project notes on the Phase 3 integration point).
This pools any N objects down to a fixed K learned "slot" tokens via
attention, K learnable queries cross-attending over the (variable-count)
object tokens.

No relevance-based filtering happens here (or anywhere in the Fusion
pipeline): block 8's FiLM already down-weights linguistically-irrelevant
objects, but every detected object -- including ones the instruction
doesn't mention -- still reaches this pooling step. This is intentional:
since the policy's only input is relative coordinates (no image fallback),
hard-dropping an object it isn't currently told to care about would make
it blind to that object as a physical obstacle (e.g. a blue cup sitting
in the path while the instruction only mentions a white cup and a yellow
bowl). Object *awareness* for obstacle avoidance and object *relevance*
for the task are kept as separate concerns.

Edge case handled explicitly: N == 0 (SAM3 found nothing at all) -- no
tokens to attend to, fall back to the raw slot queries themselves.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SetInvariantEncoder(nn.Module):
    def __init__(self, dim: int, num_slots: int = 8, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.slot_queries = nn.Parameter(torch.randn(num_slots, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, object_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            object_embed: [B, N, D] (N may be 0).

        Returns:
            [B, num_slots, D] fixed-length pooled tokens.
        """
        B, N = object_embed.shape[0], object_embed.shape[1]
        queries = self.slot_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        if N == 0:
            return self.norm(queries)

        attended, _ = self.attn(queries, object_embed, object_embed)
        return self.norm(attended)


if __name__ == "__main__":
    torch.manual_seed(0)
    dim, num_slots = 256, 8
    enc = SetInvariantEncoder(dim=dim, num_slots=num_slots, num_heads=4)

    # ---- normal case: variable N across calls, output shape stays fixed ----
    for N in (3, 7, 1):
        object_embed = torch.randn(2, N, dim, requires_grad=True)
        out = enc(object_embed)
        print(f"N={N}: output shape {out.shape}")
        assert out.shape == (2, num_slots, dim)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert object_embed.grad is not None

    # ---- edge case: N == 0 ----
    empty = torch.randn(2, 0, dim)
    out_empty = enc(empty)
    print("N=0 output shape:", out_empty.shape)
    assert out_empty.shape == (2, num_slots, dim)
    assert torch.isfinite(out_empty).all()

    print("all set-encoder edge cases OK")
