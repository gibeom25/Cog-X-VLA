"""
Block 7: Cross-Attention between object tokens and language tokens.

Object tokens (block 6 output) act as queries; language tokens (frozen
Florence2 embeddings of the instruction, per project decision in §0) act as
keys/values. Produces both:
  - `attended`: each object's embedding re-weighted by which instruction
    words it's relevant to (feeds block 8's FiLM as content to modulate).
  - `relevance`: one scalar per object summarizing how strongly it relates
    to the instruction at all (feeds block 8's scale/shift generation and
    block 9's threshold filtering).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ObjectLanguageCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, object_tokens: torch.Tensor, lang_tokens: torch.Tensor,
                lang_mask: torch.Tensor | None = None):
        """
        Args:
            object_tokens: [B, N, D]
            lang_tokens: [B, T, D]
            lang_mask: [B, T] bool, True for valid (non-padding) language tokens.

        Returns:
            attended: [B, N, D]
            relevance: [B, N] -- higher means more relevant to the instruction.
        """
        B, N, D = object_tokens.shape
        T = lang_tokens.shape[1]

        q = self.q_proj(object_tokens).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,Dh]
        k = self.k_proj(lang_tokens).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)     # [B,H,T,Dh]
        v = self.v_proj(lang_tokens).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)     # [B,H,T,Dh]

        attn_logits = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,H,N,T]
        if lang_mask is not None:
            mask = lang_mask[:, None, None, :]  # [B,1,1,T]
            attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

        attn_weights = attn_logits.softmax(dim=-1)
        attended = attn_weights @ v                              # [B,H,N,Dh]
        attended = attended.transpose(1, 2).reshape(B, N, D)
        attended = self.out_proj(attended)

        # Per-object relevance: strongest association with any instruction
        # token, averaged across heads.
        relevance = attn_logits.mean(dim=1).amax(dim=-1)  # [B, N]

        return attended, relevance


if __name__ == "__main__":
    torch.manual_seed(0)
    dim = 256
    cross_attn = ObjectLanguageCrossAttention(dim=dim, num_heads=4)

    B, N, T = 2, 5, 8
    object_tokens = torch.randn(B, N, dim, requires_grad=True)
    lang_tokens = torch.randn(B, T, dim, requires_grad=True)
    lang_mask = torch.ones(B, T, dtype=torch.bool)
    lang_mask[0, 6:] = False  # simulate padding on sample 0

    attended, relevance = cross_attn(object_tokens, lang_tokens, lang_mask)
    print("attended shape:", attended.shape)
    print("relevance shape:", relevance.shape)
    print("relevance sample:", relevance[0])
    assert attended.shape == (B, N, dim)
    assert relevance.shape == (B, N)
    assert torch.isfinite(attended).all() and torch.isfinite(relevance).all()

    loss = attended.sum() + relevance.sum()
    loss.backward()
    assert object_tokens.grad is not None and lang_tokens.grad is not None
    assert all(p.grad is not None for p in cross_attn.parameters())
    print("gradients flow through both inputs and all parameters: OK")
