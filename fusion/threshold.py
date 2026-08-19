"""
Block 9: threshold filtering on object relevance.

Training: soft sigmoid gate around a (learnable) threshold, so the decision
of "does this object matter" is fully differentiable and gradients can shape
what block 7's relevance scores mean.
Inference: hard cutoff -- objects below threshold are dropped (gate = 0),
matching the project's soft-train/hard-eval design decision.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ThresholdFilter(nn.Module):
    def __init__(self, init_threshold: float = 0.0, sharpness: float = 5.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(init_threshold)))
        self.sharpness = sharpness

    def forward(self, object_embed: torch.Tensor, relevance: torch.Tensor, hard_threshold: float | None = None):
        """
        Args:
            object_embed: [B, N, D]
            relevance: [B, N]
            hard_threshold: eval-mode override for the cutoff (defaults to
                the learned threshold if None).

        Returns:
            gated_embed: [B, N, D], object_embed scaled by the gate.
            gate: [B, N] in [0, 1] (soft) or {0, 1} (hard eval), usable as a
                keep-mask by block 10.
        """
        if self.training:
            gate = torch.sigmoid((relevance - self.threshold) * self.sharpness)
        else:
            cutoff = hard_threshold if hard_threshold is not None else self.threshold.item()
            gate = (relevance >= cutoff).float()

        gated_embed = object_embed * gate.unsqueeze(-1)
        return gated_embed, gate


if __name__ == "__main__":
    torch.manual_seed(0)
    dim = 256
    filt = ThresholdFilter(init_threshold=0.0, sharpness=5.0)

    B, N = 2, 6
    object_embed = torch.randn(B, N, dim, requires_grad=True)
    relevance = torch.tensor([[2.0, -2.0, 0.1, -0.1, 1.0, -1.0],
                               [0.5, -0.5, 3.0, -3.0, 0.0, 0.2]], requires_grad=True)

    # ---- training mode: soft gate, must be differentiable ----
    filt.train()
    gated, gate = filt(object_embed, relevance)
    print("train gate:", gate)
    assert gate.shape == (B, N)
    assert (gate >= 0).all() and (gate <= 1).all()
    assert ((gate > 0) & (gate < 1)).any(), "expected genuinely soft (non-binary) values in train mode"

    loss = gated.sum()
    loss.backward()
    assert object_embed.grad is not None and relevance.grad is not None
    assert filt.threshold.grad is not None
    print("gradients flow through embed, relevance, and learnable threshold: OK")

    # ---- eval mode: hard binary cutoff ----
    filt.eval()
    with torch.no_grad():
        _, gate_eval = filt(object_embed, relevance)
    print("eval gate:", gate_eval)
    assert set(gate_eval.unique().tolist()).issubset({0.0, 1.0}), "eval gate must be hard {0,1}"
    # sanity: positive-relevance objects should be kept, negative dropped
    assert (gate_eval[relevance > 0] == 1.0).all()
    assert (gate_eval[relevance < 0] == 0.0).all()
    print("eval-mode hard cutoff matches relevance sign: OK")
