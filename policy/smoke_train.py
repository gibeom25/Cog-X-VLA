"""
Block 14 smoke test: confirm the training path works end to end on real
data (not just synthetic tensors, see policy/test_adapter.py) and that
loss actually decreases -- the standard "overfit a tiny batch" sanity
check for a new training pipeline.

Batch size 1 throughout (each sample has its own variable object count
N; padding a batch of different N together is real work the production
training loop -- not yet built -- will need, but isn't necessary just to
prove the pipeline is wired correctly).

Fine-tune scope matches docs/initial_plan.md Phase 3: Florence2 (self.vlm)
frozen entirely, only the first/last few SoftPromptedTransformer blocks
trainable, everything new (object_* fusion modules, object_proj/
object_pos_emb, action_encoder/decoder, vlm_proj/aux_visual_proj) trainable.

Run in the XVLA conda env.
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/policy")
sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/data")

import torch
from PIL import Image

from xvla_adapter import XVLAWithPerception
from models.processing_xvla import XVLAProcessor
from prepare_libero import load_demo, build_training_windows

MODEL_PATH = "2toINF/X-VLA-Libero"
REGEN_ROOT = "/home/gibeom_pilab/cog-xvla/data/regenerated"
N_UNFROZEN_BLOCKS = 2  # first + last N transformer blocks stay trainable


def freeze_for_finetune(model: XVLAWithPerception):
    for p in model.vlm.parameters():
        p.requires_grad = False
    n_blocks = len(model.transformer.blocks)
    for i, block in enumerate(model.transformer.blocks):
        trainable = i < N_UNFROZEN_BLOCKS or i >= n_blocks - N_UNFROZEN_BLOCKS
        for p in block.parameters():
            p.requires_grad = trainable


def build_sample(hdf5_path: str, demo_key: str, t: int, processor: XVLAProcessor, device):
    demo = load_demo(hdf5_path, demo_key)
    windows = build_training_windows(demo, num_actions=30)
    w = windows[t]

    objects_path = hdf5_path.replace(".hdf5", f"_{demo_key}_objects.json")
    with open(objects_path) as f:
        per_frame_objects = json.load(f)
    objects = per_frame_objects[t]

    lang = processor.encode_language(demo.instruction)
    input_ids = lang["input_ids"].to(device)

    wrist_img = Image.fromarray(w["robot0_eye_in_hand_rgb"])
    img_enc = processor.encode_image([wrist_img])
    image_input = img_enc["image_input"].to(device).to(torch.float32)
    image_mask = img_enc["image_mask"].to(device)

    if objects:
        labels = [o["label"] for o in objects]
        label_ids = processor.tokenizer(labels, return_tensors="pt", padding=True)["input_ids"]
        label_ids = label_ids.unsqueeze(0).to(device)  # [1, N, L]
        object_raw = torch.tensor(
            [o["relative_xyz"] + o["half_extents"] for o in objects], dtype=torch.float32
        ).unsqueeze(0).to(device)  # [1, N, 6]
    else:
        label_ids = torch.zeros(1, 0, 1, dtype=torch.long, device=device)
        object_raw = torch.zeros(1, 0, 6, device=device)

    lang_mask = (input_ids != processor.tokenizer.pad_token_id)

    proprio = torch.tensor(w["proprio20"], dtype=torch.float32, device=device).unsqueeze(0)
    action = torch.tensor(w["action20"], dtype=torch.float32, device=device).unsqueeze(0)
    domain_id = torch.tensor([3], device=device)

    return dict(
        input_ids=input_ids, image_input=image_input, image_mask=image_mask,
        domain_id=domain_id, proprio=proprio, action=action,
        object_raw=object_raw, object_label_ids=label_ids, lang_mask=lang_mask,
    ), len(objects)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading model from {MODEL_PATH} ...")
    model = XVLAWithPerception.from_pretrained(MODEL_PATH, torch_dtype=torch.float32).to(device)
    processor = XVLAProcessor.from_pretrained(MODEL_PATH)
    freeze_for_finetune(model)
    model.train()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M total")
    print(f"GPU mem after load: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, "
          f"{torch.cuda.memory_reserved()/1e9:.2f}GB reserved")

    # a few fixed windows from different demos, for the tiny-batch overfit check
    task_files = sorted(glob.glob(os.path.join(REGEN_ROOT, "*", "*.hdf5")))[:3]
    samples = []
    for path in task_files:
        sample, n_obj = build_sample(path, "demo_0", t=10, processor=processor, device=device)
        samples.append(sample)
        print(f"  loaded sample from {os.path.basename(path)}: {n_obj} objects")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    print("\n=== overfitting 3 fixed samples for 40 steps ===")
    losses = []
    for step in range(40):
        optim.zero_grad()
        total_loss = 0.0
        for sample in samples:
            loss_dict = model(**sample)
            loss = sum(loss_dict.values())
            loss.backward()
            total_loss += loss.item()
        optim.step()
        losses.append(total_loss / len(samples))
        if step % 5 == 0 or step == 39:
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"step {step:2d}: avg_loss={losses[-1]:.4f}  peak_mem={mem:.2f}GB")

    print(f"\nloss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f} "
          f"({'DECREASED' if losses[-1] < losses[0] else 'DID NOT DECREASE'})")
    assert all(torch.isfinite(torch.tensor(losses)))
    print("SMOKE TEST PASSED" if losses[-1] < losses[0] * 0.9 else "SMOKE TEST: loss did not drop enough, investigate")


if __name__ == "__main__":
    main()
