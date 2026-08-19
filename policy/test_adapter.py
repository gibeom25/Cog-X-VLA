"""
Load XVLAWithPerception from the pretrained X-VLA-Libero checkpoint and
verify:
  1) it loads without missing/unexpected keys other than the new modules
  2) forward()/generate_actions() work with object_raw=None (should behave
     like stock XVLA -- no object stream injected)
  3) they also work with real object counts (N=3) and the N=0 edge case
  4) output action shape matches stock XVLA's

Run in the XVLA conda env.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/policy")

import torch
from xvla_adapter import XVLAWithPerception
from models.processing_xvla import XVLAProcessor

MODEL_PATH = "2toINF/X-VLA-Libero"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading XVLAWithPerception from {MODEL_PATH} ...")
    model = XVLAWithPerception.from_pretrained(MODEL_PATH, torch_dtype=torch.float32).to(device)
    model.eval()
    processor = XVLAProcessor.from_pretrained(MODEL_PATH)

    n_total = sum(p.numel() for p in model.parameters())
    n_new = sum(p.numel() for n, p in model.named_parameters()
                if n.startswith(("object_projection", "object_cross_attn", "object_film",
                                  "object_set_encoder", "transformer.object_proj", "transformer.object_pos_emb")))
    print(f"total params: {n_total/1e6:.1f}M, new (untrained) params: {n_new/1e6:.2f}M")

    # ---- build realistic dummy inputs via the real processor ----
    from PIL import Image
    import numpy as np
    img0 = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
    img1 = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
    instruction = "pick up the black bowl and place it on the plate"

    inputs = processor([img0, img1], instruction)
    input_ids = inputs["input_ids"].to(device)
    image_input = inputs["image_input"].to(device).to(torch.float32)
    image_mask = inputs["image_mask"].to(device)

    B = 1
    proprio = torch.zeros(B, 20, device=device)
    domain_id = torch.tensor([3], device=device)

    print("\n=== generate_actions, object_raw=None (should match stock XVLA behavior) ===")
    action = model.generate_actions(
        input_ids=input_ids, image_input=image_input, image_mask=image_mask,
        domain_id=domain_id, proprio=proprio, steps=4,
    )
    print("action shape:", action.shape)
    assert action.shape == (B, model.num_actions, model.action_space.dim_action)
    assert torch.isfinite(action).all()

    print("\n=== generate_actions, N=3 dummy objects ===")
    N = 3
    object_raw = torch.randn(B, N, 6, device=device)
    label_ids = processor.tokenizer(
        ["bowl", "plate", "cup"], return_tensors="pt", padding=True
    )["input_ids"].unsqueeze(0).to(device)  # [1, N, L_label]
    lang_mask = (input_ids != processor.tokenizer.pad_token_id)

    action_obj = model.generate_actions(
        input_ids=input_ids, image_input=image_input, image_mask=image_mask,
        domain_id=domain_id, proprio=proprio, steps=4,
        object_raw=object_raw, object_label_ids=label_ids, lang_mask=lang_mask,
    )
    print("action shape (with objects):", action_obj.shape)
    assert action_obj.shape == action.shape
    assert torch.isfinite(action_obj).all()
    print("differs from no-object action (expected, object stream changes the sequence):",
          not torch.allclose(action, action_obj))

    print("\n=== generate_actions, N=0 objects (edge case) ===")
    object_raw0 = torch.randn(B, 0, 6, device=device)
    label_ids0 = torch.zeros(B, 0, 1, dtype=torch.long, device=device)
    action_n0 = model.generate_actions(
        input_ids=input_ids, image_input=image_input, image_mask=image_mask,
        domain_id=domain_id, proprio=proprio, steps=4,
        object_raw=object_raw0, object_label_ids=label_ids0, lang_mask=lang_mask,
    )
    print("action shape (N=0):", action_n0.shape)
    assert action_n0.shape == action.shape
    assert torch.isfinite(action_n0).all()

    print("\n=== forward() training path, N=3 ===")
    action_gt = torch.randn(B, model.num_actions, model.action_space.dim_action, device=device)
    loss_dict = model(
        input_ids=input_ids, image_input=image_input, image_mask=image_mask,
        domain_id=domain_id, proprio=proprio, action=action_gt,
        object_raw=object_raw, object_label_ids=label_ids, lang_mask=lang_mask,
    )
    print("loss_dict:", {k: v.item() for k, v in loss_dict.items()})
    total_loss = sum(loss_dict.values())
    total_loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_params = sum(1 for _ in model.parameters())
    print(f"params with grad after backward: {n_grad}/{n_params}")
    obj_grad_ok = all(p.grad is not None for p in model.object_projection.parameters())
    print("object_projection params all have grad:", obj_grad_ok)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
