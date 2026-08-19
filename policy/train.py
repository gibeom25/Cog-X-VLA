"""
Block 14: real training loop over the full 40-demo (6898-window) subset.

Batch size 1 (see policy/smoke_train.py for why: variable object count N
per sample, no cross-sample padding built yet -- out of scope for this
prototype-scale run). Gradient accumulation over `accum_steps` samples
substitutes for a real batch dimension.

Run in the XVLA conda env.
"""
from __future__ import annotations

import glob
import os
import sys
import time

sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/policy")
sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/data")
sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/X-VLA")

import torch

from xvla_adapter import XVLAWithPerception
from models.processing_xvla import XVLAProcessor
from smoke_train import freeze_for_finetune
from dataset import LiberoDataset

MODEL_PATH = "2toINF/X-VLA-Libero"
REGEN_ROOT = "/home/gibeom_pilab/cog-xvla/data/regenerated"
CKPT_DIR = "/home/gibeom_pilab/cog-xvla/policy/checkpoints"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every_steps", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading model from {MODEL_PATH} ...")
    model = XVLAWithPerception.from_pretrained(MODEL_PATH, torch_dtype=torch.float32).to(device)
    processor = XVLAProcessor.from_pretrained(MODEL_PATH)
    freeze_for_finetune(model)
    model.train()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M total")

    task_files = sorted(glob.glob(os.path.join(REGEN_ROOT, "*", "*.hdf5")))
    ds = LiberoDataset(task_files, processor, device)
    print(f"dataset: {len(ds)} samples")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    os.makedirs(CKPT_DIR, exist_ok=True)

    step = 0
    t0 = time.time()
    running_loss = 0.0
    for epoch in range(args.epochs):
        optim.zero_grad()
        for i in range(len(ds)):
            sample = ds[i]
            loss_dict = model(**sample)
            loss = sum(loss_dict.values()) / args.accum_steps
            loss.backward()
            running_loss += loss.item() * args.accum_steps

            if (i + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                optim.zero_grad()
                step += 1

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    avg_loss = running_loss / (args.log_every * args.accum_steps)
                    running_loss = 0.0
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    print(f"epoch {epoch} step {step:5d} (sample {i+1}/{len(ds)}): "
                          f"loss={avg_loss:.4f} elapsed={elapsed/60:.1f}min peak_mem={mem:.2f}GB")

                if step % args.save_every_steps == 0:
                    ckpt_path = os.path.join(CKPT_DIR, f"step_{step}")
                    model.save_pretrained(ckpt_path, safe_serialization=True)
                    print(f"  saved checkpoint -> {ckpt_path}")

    final_ckpt = os.path.join(CKPT_DIR, "final")
    model.save_pretrained(final_ckpt, safe_serialization=True)
    print(f"\ndone. final checkpoint -> {final_ckpt}")


if __name__ == "__main__":
    main()
