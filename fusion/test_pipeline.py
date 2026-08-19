"""
Integration test chaining blocks 6-8 + 10 end to end with synthetic tensors:
projection -> cross-attention -> FiLM -> set-invariant encoder.
(Block 9, threshold filtering, was removed -- see fusion/set_encoder.py's
docstring: hard-dropping linguistically-irrelevant objects would make the
policy blind to them as physical obstacles, since it has no image fallback.)
Confirms shapes are compatible across the whole Fusion phase and that a
variable object count always collapses to the same fixed-length output.
"""
from __future__ import annotations

import torch

from projection import ObjectProjection
from cross_attention import ObjectLanguageCrossAttention
from film import FiLM
from set_encoder import SetInvariantEncoder


def run_pipeline(num_objects: int, lang_embed_dim: int = 1024, fusion_dim: int = 256,
                  num_slots: int = 8, training: bool = True):
    B, T = 2, 6
    relative_xyz = torch.randn(B, num_objects, 3)
    lang_embed_per_obj = torch.randn(B, num_objects, lang_embed_dim)  # block6 needs per-object label embedding
    lang_tokens = torch.randn(B, T, fusion_dim)  # instruction tokens, already at fusion_dim for this test
    lang_mask = torch.ones(B, T, dtype=torch.bool)

    proj = ObjectProjection(pos_dim=3, lang_embed_dim=lang_embed_dim, out_dim=fusion_dim)
    cross_attn = ObjectLanguageCrossAttention(dim=fusion_dim, num_heads=4)
    film = FiLM(dim=fusion_dim)
    set_enc = SetInvariantEncoder(dim=fusion_dim, num_slots=num_slots, num_heads=4)

    modules = [proj, cross_attn, film, set_enc]
    for m in modules:
        m.train(training)

    object_tokens = proj(relative_xyz, lang_embed_per_obj)                     # [B, N, D]
    attended, relevance = cross_attn(object_tokens, lang_tokens, lang_mask)    # [B, N, D], [B, N]
    modulated = film(attended, relevance)                                      # [B, N, D] -- every object kept
    pooled = set_enc(modulated)                                                # [B, K, D]

    return pooled


if __name__ == "__main__":
    for N in (0, 1, 4, 12):
        for training in (True, False):
            out = run_pipeline(num_objects=N, training=training)
            mode = "train" if training else "eval"
            print(f"N={N:2d} [{mode}]: pooled shape {tuple(out.shape)}, "
                  f"finite={torch.isfinite(out).all().item()}")
            assert out.shape == (2, 8, 256)
            assert torch.isfinite(out).all()

    print("full Phase 2 pipeline (blocks 6->7->8->10) OK across all object counts and train/eval modes")
