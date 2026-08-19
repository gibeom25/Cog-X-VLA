"""
Block 11: X-VLA policy integration.

Vendor code in X-VLA/models/*.py is never edited directly -- this
subclasses XVLA and SoftPromptedTransformer to inject the Fusion
pipeline's pooled object tokens (blocks 6, 7, 8, 10) as a 3rd concat
stream, parallel to vlm_proj/aux_visual_proj. Integration points, per
docs/initial_plan.md Phase 3:
  - X-VLA/models/transformer.py:319-324 (vlm_proj/aux_visual_proj pattern)
    -> mirrored here as `object_proj`
  - X-VLA/models/transformer.py:326-327 (pos_emb) -> left untouched;
    `object_pos_emb` is a dedicated parameter instead, same pattern as
    `soft_prompt_hub`
  - X-VLA/models/transformer.py:376-383 (the concat itself) -> extended
    with the object stream
  - X-VLA/models/modeling_xvla.py forward_vlm() -> unchanged. Camera
    input composition (agentview replaced by object tokens, wrist image
    kept as-is) is a Phase 4/6 data-prep decision, not a model change --
    see docs/initial_plan.md §0 "카메라 입력 구성".

The whole Fusion pipeline runs at dim = Florence2's projection_dim
(X-VLA/models/configuration_xvla.py's hidden_size also happens to be
1024), so the language-embedding lookup used for cross-attention needs
no extra projection before entering fusion/cross_attention.py.
"""

from __future__ import annotations

import sys
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/fusion")
from projection import ObjectProjection
from cross_attention import ObjectLanguageCrossAttention
from film import FiLM
from set_encoder import SetInvariantEncoder

sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/X-VLA")
from models.modeling_xvla import XVLA
from models.transformer import SoftPromptedTransformer, timestep_embedding


# relative_xyz(3, block 5) + half_extents(3, block 4) -- see perception/depth_localize.py
OBJECT_RAW_DIM = 6


def _basic_reinit(module: nn.Module) -> None:
    """Generic, safe init for the module types used in fusion/*.py."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        nn.init.xavier_uniform_(module.in_proj_weight)
        nn.init.zeros_(module.in_proj_bias)
        nn.init.xavier_uniform_(module.out_proj.weight)
        nn.init.zeros_(module.out_proj.bias)


class ObjectAwareTransformer(SoftPromptedTransformer):
    """SoftPromptedTransformer plus a 3rd concat stream for pooled object tokens."""

    def __init__(self, *args, object_token_dim: int, num_object_slots: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_proj = nn.Linear(object_token_dim, self.hidden_size)
        self.object_pos_emb = nn.Parameter(torch.zeros(1, num_object_slots, self.hidden_size))
        nn.init.normal_(self.object_pos_emb, std=0.02)

    def forward(
        self,
        domain_id: torch.LongTensor,
        vlm_features: torch.Tensor,
        aux_visual_inputs: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        object_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Identical to SoftPromptedTransformer.forward except for the
        `object_tokens` [B, num_object_slots, D] stream, inserted after the
        vlm/aux streams (which share `pos_emb`) and before soft prompts."""
        B, num_actions = action_with_noise.shape[:2]

        time_emb = timestep_embedding(t, self.dim_time)
        time_tokens = time_emb.unsqueeze(1).expand(B, num_actions, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(B, num_actions, proprio.shape[-1])
        action_tokens = torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1)
        x = self.action_encoder(action_tokens, domain_id)

        if self.use_hetero_proj:
            x = torch.cat(
                [x, self.vlm_proj(vlm_features, domain_id), self.aux_visual_proj(aux_visual_inputs, domain_id)],
                dim=1,
            )
        else:
            x = torch.cat([x, self.vlm_proj(vlm_features), self.aux_visual_proj(aux_visual_inputs)], dim=1)

        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}.")
        x = x + self.pos_emb[:, :seq_len, :]

        if object_tokens is not None:
            obj = self.object_proj(object_tokens) + self.object_pos_emb
            x = torch.cat([x, obj], dim=1)

        if self.len_soft_prompts > 0:
            soft_prompts = self.soft_prompt_hub(domain_id).view(B, self.len_soft_prompts, self.hidden_size)
            x = torch.cat([x, soft_prompts], dim=1)

        for block in self.blocks:
            x = block(x)

        return self.action_decoder(self.norm(x[:, :num_actions]), domain_id)


class XVLAWithPerception(XVLA):
    """XVLA + object-centric perception fusion (blocks 6, 7, 8, 10, 11).

    Loadable via XVLAWithPerception.from_pretrained(<xvla checkpoint>):
    HF's from_pretrained() calls __init__ (which replaces self.transformer
    with ObjectAwareTransformer below) *before* loading the checkpoint, so
    weights are matched by name -- all the original pretrained parameters
    load normally, and the new object_proj/object_pos_emb/object_* fusion
    modules are simply left at their random init.
    """

    def __init__(self, config, num_object_slots: int = 8):
        super().__init__(config)

        fusion_dim = self.vlm.config.projection_dim  # Florence2's embedding space; no extra projection needed

        self.object_projection = ObjectProjection(pos_dim=OBJECT_RAW_DIM, lang_embed_dim=fusion_dim, out_dim=fusion_dim)
        self.object_cross_attn = ObjectLanguageCrossAttention(dim=fusion_dim, num_heads=4)
        self.object_film = FiLM(dim=fusion_dim)
        self.object_set_encoder = SetInvariantEncoder(dim=fusion_dim, num_slots=num_object_slots, num_heads=4)

        self.transformer = ObjectAwareTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=fusion_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=self.action_space.dim_action,
            dim_propio=getattr(self.action_space, "dim_proprio", self.action_space.dim_action),
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
            object_token_dim=fusion_dim,
            num_object_slots=num_object_slots,
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """HF's from_pretrained can construct checkpoint-absent submodules
        (all of our object_* modules) under a meta-device context for
        memory efficiency, which silently discards the nn.init calls each
        module's own __init__ makes and can leave real parameters as
        uninitialized memory (observed empirically: one Linear bias came
        back non-finite while its sibling parameters happened to look
        finite). Re-initialize explicitly, post-load, so this can't happen
        regardless of what the loading path did internally."""
        model = super().from_pretrained(*args, **kwargs)
        model._init_new_modules()
        return model

    def _init_new_modules(self) -> None:
        for module in (self.object_projection, self.object_cross_attn, self.object_film, self.object_set_encoder):
            module.apply(_basic_reinit)
        # FiLM's identity-at-init property (see fusion/film.py) needs its
        # last layer to start at exactly zero, which the generic reinit
        # above doesn't give it (xavier_uniform on the weight).
        nn.init.zeros_(self.object_film.generator[-1].weight)
        nn.init.zeros_(self.object_film.generator[-1].bias)
        nn.init.normal_(self.object_set_encoder.slot_queries, std=0.02)
        nn.init.normal_(self.transformer.object_pos_emb, std=0.02)
        _basic_reinit(self.transformer.object_proj)

    def compute_object_tokens(
        self,
        input_ids: torch.LongTensor,          # [B, L], same instruction ids passed to forward_vlm
        object_raw: torch.Tensor,             # [B, N, 6] relative_xyz + half_extents; N may be 0
        object_label_ids: torch.LongTensor,   # [B, N, L_label] tokenized matched-label text per object
        lang_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs blocks 6/7/8/10 to turn per-object detections into the fixed
        [B, num_object_slots, D] stream ObjectAwareTransformer expects."""
        embed = self.vlm.get_input_embeddings()  # frozen -- part of self.vlm

        lang_tokens = embed(input_ids)                       # [B, L, D]
        label_embeds = embed(object_label_ids).mean(dim=2)   # [B, N, D], mean-pool subword tokens

        object_tokens = self.object_projection(object_raw, label_embeds)   # [B, N, D]
        attended, relevance = self.object_cross_attn(object_tokens, lang_tokens, lang_mask)
        modulated = self.object_film(attended, relevance)   # every object kept, see fusion/set_encoder.py
        return self.object_set_encoder(modulated)            # [B, num_object_slots, D]

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        object_raw: Optional[torch.Tensor] = None,
        object_label_ids: Optional[torch.LongTensor] = None,
        lang_mask: Optional[torch.Tensor] = None,
    ):
        enc = self.forward_vlm(input_ids, image_input, image_mask)

        object_tokens = None
        if object_raw is not None:
            object_tokens = self.compute_object_tokens(input_ids, object_raw, object_label_ids, lang_mask)

        B = input_ids.shape[0]
        t = (torch.rand(1, device=input_ids.device)
             + torch.arange(B, device=input_ids.device) / B) % (1 - 1e-5)

        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            object_tokens=object_tokens,
            **enc,
        )
        return self.action_space.compute_loss(pred_action, action)

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        object_raw: Optional[torch.Tensor] = None,
        object_label_ids: Optional[torch.LongTensor] = None,
        lang_mask: Optional[torch.Tensor] = None,
        steps: int = 10,
    ) -> torch.Tensor:
        self.eval()
        enc = self.forward_vlm(input_ids, image_input, image_mask)

        object_tokens = None
        if object_raw is not None:
            object_tokens = self.compute_object_tokens(input_ids, object_raw, object_label_ids, lang_mask)

        B = input_ids.shape[0]
        D = self.action_space.dim_action
        x1 = torch.randn(B, self.num_actions, D, device=proprio.device, dtype=proprio.dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=proprio.device, dtype=proprio.dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                object_tokens=object_tokens,
                **enc,
            )
        return self.action_space.postprocess(action)
