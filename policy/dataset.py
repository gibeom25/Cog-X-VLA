"""
Block 14: training dataset over data/regenerated/ + its cached object
detections.

Pre-loads each demo (HDF5 read + window construction + cached object
detections JSON parse) exactly once at construction time, instead of
re-reading per sample -- policy/time_training_step.py measured the naive
per-sample approach (policy/smoke_train.py's original build_sample,
which called load_demo/build_training_windows fresh every call) at
~580ms/sample of pure I/O, dwarfing the ~140ms of actual forward+
backward+optim compute. Sample-building itself (tokenization + image
encoding) still happens per __getitem__ -- that part isn't the
bottleneck.

Run in the XVLA conda env.
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import torch
from PIL import Image

sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/data")
from prepare_libero import load_demo, build_training_windows


def build_sample_from_window(instruction: str, window: dict, objects: list, processor, device):
    """The cheap per-sample part: tokenize + encode the already-extracted
    window/objects into the tensors XVLAWithPerception.forward() expects."""
    lang = processor.encode_language(instruction)
    input_ids = lang["input_ids"].to(device)

    wrist_img = Image.fromarray(window["robot0_eye_in_hand_rgb"])
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

    proprio = torch.tensor(window["proprio20"], dtype=torch.float32, device=device).unsqueeze(0)
    action = torch.tensor(window["action20"], dtype=torch.float32, device=device).unsqueeze(0)
    domain_id = torch.tensor([3], device=device)

    return dict(
        input_ids=input_ids, image_input=image_input, image_mask=image_mask,
        domain_id=domain_id, proprio=proprio, action=action,
        object_raw=object_raw, object_label_ids=label_ids, lang_mask=lang_mask,
    )


class LiberoDataset:
    def __init__(self, task_files: List[str], processor, device,
                 demo_keys: Optional[List[str]] = None, num_actions: int = 30):
        self.processor = processor
        self.device = device
        # (instruction, window_dict, objects_list) per training sample,
        # flattened across all demos -- built once here, not per __getitem__.
        self.index = []

        demo_keys = demo_keys or [f"demo_{i}" for i in range(5)]
        for path in task_files:
            for demo_key in demo_keys:
                objects_path = path.replace(".hdf5", f"_{demo_key}_objects.json")
                if not os.path.exists(objects_path):
                    continue
                demo = load_demo(path, demo_key)
                windows = build_training_windows(demo, num_actions=num_actions)
                with open(objects_path) as f:
                    per_frame_objects = json.load(f)
                for t, w in enumerate(windows):
                    self.index.append((demo.instruction, w, per_frame_objects[t]))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        instruction, window, objects = self.index[idx]
        return build_sample_from_window(instruction, window, objects, self.processor, self.device)


if __name__ == "__main__":
    import glob
    import time
    import sys
    sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/X-VLA")
    from models.processing_xvla import XVLAProcessor

    processor = XVLAProcessor.from_pretrained("2toINF/X-VLA-Libero")
    task_files = sorted(glob.glob("/home/gibeom_pilab/cog-xvla/data/regenerated/*/*.hdf5"))

    t0 = time.time()
    ds = LiberoDataset(task_files, processor, torch.device("cpu"))
    print(f"built dataset: {len(ds)} samples in {time.time()-t0:.1f}s (one-time cost)")

    t0 = time.time()
    N = 20
    for i in range(N):
        _ = ds[i]
    print(f"avg __getitem__ time after pre-load: {(time.time()-t0)/N*1000:.1f}ms")
