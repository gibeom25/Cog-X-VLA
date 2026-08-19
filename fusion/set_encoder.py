"""
Block 10: set-invariant encoder.

The number of surviving objects after threshold filtering (block 9) varies
per frame, but X-VLA's transformer needs a fixed-length token stream (its
positional embedding is a plain indexed parameter, see project notes on the
Phase 3 integration point). This pools any N objects down to a fixed K
learned "slot" tokens via attention, K learnable queries cross-attending
over the (variable-count) object tokens.

Edge cases handled explicitly:
  - N == 0 (SAM3 found nothing at all): no tokens to attend to, fall back
    to the raw slot queries themselves.
  - every object masked out by eval-mode hard threshold: masking all
    key positions would make attention softmax undefined (NaN), fall back
    to attending over everything unmasked for just those samples.
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

    def forward(self, object_embed: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            object_embed: [B, N, D] (N may be 0).
            gate: [B, N] in [0, 1]; only hard-masked (values < 0.5 excluded)
                when not in training mode -- during training the soft gate
                has already scaled object_embed via block 9, so re-masking
                here would zero out gradients for weakly-relevant objects.

        Returns:
            [B, num_slots, D] fixed-length pooled tokens.
        """
        B = object_embed.shape[0]
        N = object_embed.shape[1]
        queries = self.slot_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        if N == 0:
            return self.norm(queries)

        key_padding_mask = None
        if gate is not None and not self.training:
            key_padding_mask = gate < 0.5  # [B, N], True = ignore this position
            all_masked = key_padding_mask.all(dim=-1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked] = False  # fall back: attend to everything for these rows

        attended, _ = self.attn(queries, object_embed, object_embed, key_padding_mask=key_padding_mask)
        return self.norm(attended)


if __name__ == "__main__":
    torch.manual_seed(0)
    dim, num_slots = 256, 8
    enc = SetInvariantEncoder(dim=dim, num_slots=num_slots, num_heads=4)

    # ---- normal case: variable N across calls, output shape stays fixed ----
    for N in (3, 7, 1):
        object_embed = torch.randn(2, N, dim, requires_grad=True)
        gate = torch.rand(2, N)
        out = enc(object_embed, gate)
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

    # ---- edge case: eval mode, every object masked out for one sample ----
    enc.eval()
    object_embed = torch.randn(2, 4, dim)
    gate_all_zero_row0 = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                                        [1.0, 0.0, 1.0, 0.0]])
    with torch.no_grad():
        out_masked = enc(object_embed, gate_all_zero_row0)
    print("all-masked-row output shape:", out_masked.shape)
    assert out_masked.shape == (2, num_slots, dim)
    assert torch.isfinite(out_masked).all(), "expected fallback to avoid NaN when a whole row is masked"

    print("all set-encoder edge cases OK")
