"""
Measure real per-step timing (sample-build I/O + forward + backward) to
estimate full-dataset training time, instead of guessing.

Run in the XVLA conda env.
"""
from __future__ import annotations

import glob
import os
import sys
import time

sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/policy")
sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/data")

import torch

from xvla_adapter import XVLAWithPerception
from models.processing_xvla import XVLAProcessor
from prepare_libero import load_demo, build_training_windows
from smoke_train import build_sample, freeze_for_finetune

MODEL_PATH = "2toINF/X-VLA-Libero"
REGEN_ROOT = "/home/gibeom_pilab/cog-xvla/data/regenerated"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XVLAWithPerception.from_pretrained(MODEL_PATH, torch_dtype=torch.float32).to(device)
    processor = XVLAProcessor.from_pretrained(MODEL_PATH)
    freeze_for_finetune(model)
    model.train()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    task_files = sorted(glob.glob(os.path.join(REGEN_ROOT, "*", "*.hdf5")))

    # total window count across all 40 demos (dataset size for this subset)
    total_windows = 0
    for path in task_files:
        for demo_key in ["demo_0", "demo_1", "demo_2", "demo_3", "demo_4"]:
            demo = load_demo(path, demo_key)
            total_windows += len(build_training_windows(demo, num_actions=30))
    print(f"total training windows across 40 demos: {total_windows}")

    # ---- time sample building (I/O + tokenization) ----
    path = task_files[0]
    t0 = time.time()
    N_BUILD = 15
    for t in range(N_BUILD):
        sample, _ = build_sample(path, "demo_0", t, processor, device)
    build_time = (time.time() - t0) / N_BUILD
    print(f"avg sample-build time: {build_time*1000:.1f}ms")

    # ---- time forward+backward on freshly-built varying samples ----
    N_STEP = 15
    torch.cuda.synchronize()
    t0 = time.time()
    for t in range(N_STEP):
        sample, _ = build_sample(path, "demo_0", t, processor, device)
        optim.zero_grad()
        loss_dict = model(**sample)
        loss = sum(loss_dict.values())
        loss.backward()
        optim.step()
    torch.cuda.synchronize()
    step_time = (time.time() - t0) / N_STEP
    print(f"avg full step time (build+forward+backward+optim, B=1): {step_time*1000:.1f}ms")

    print(f"\nGPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f}GB peak")

    for n_epochs in (1, 3, 10):
        total_s = step_time * total_windows * n_epochs
        h = total_s / 3600
        print(f"estimated time for {n_epochs} epoch(s) over {total_windows} windows: {h:.2f}h ({total_s/60:.1f}min)")


if __name__ == "__main__":
    main()
