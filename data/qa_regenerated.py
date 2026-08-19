"""
QA check over regenerated LIBERO HDF5 files (RGB+depth).

Checks, per task file / per demo:
  - required keys present
  - shape consistency across obs/action arrays (same T)
  - RGB not degenerate (all-same-value -> failed render)
  - depth has no NaN/Inf and lies within a plausible range

Also saves a contact sheet (one agentview frame per task file) for a quick
visual sanity check across all suites at once.
"""
from __future__ import annotations

import glob
import os

import h5py
import numpy as np
from PIL import Image

ROOT = "/home/gibeom_pilab/cog-xvla/data/regenerated"
REQUIRED_OBS_KEYS = [
    "agentview_rgb", "agentview_depth",
    "robot0_eye_in_hand_rgb", "robot0_eye_in_hand_depth",
    "ee_ori", "ee_pos", "ee_states", "gripper_states", "joint_states",
]
DEPTH_MIN, DEPTH_MAX = 0.05, 10.0  # meters, plausible tabletop range


def check_file(path: str) -> list[str]:
    problems = []
    with h5py.File(path, "r") as f:
        demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")],
                            key=lambda k: int(k.split("_")[1]))
        if not demo_keys:
            problems.append("no demo_* groups found")
            return problems

        for dk in demo_keys:
            demo = f["data"][dk]
            for req in ("actions", "states", "dones", "rewards", "robot_states"):
                if req not in demo:
                    problems.append(f"{dk}: missing '{req}'")
            if "obs" not in demo:
                problems.append(f"{dk}: missing 'obs' group")
                continue
            obs = demo["obs"]
            T_ref = demo["actions"].shape[0] if "actions" in demo else None
            for k in REQUIRED_OBS_KEYS:
                if k not in obs:
                    problems.append(f"{dk}: missing obs/{k}")
                    continue
                v = obs[k]
                if T_ref is not None and v.shape[0] != T_ref:
                    problems.append(f"{dk}: obs/{k} T={v.shape[0]} != actions T={T_ref}")

            rgb = obs["agentview_rgb"][()]
            if rgb.shape[1:3] != (256, 256) or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                problems.append(f"{dk}: agentview_rgb bad shape/dtype {rgb.shape} {rgb.dtype}")
            if rgb.std() < 1.0:
                problems.append(f"{dk}: agentview_rgb looks degenerate (std={rgb.std():.3f}, likely failed render)")

            depth = obs["agentview_depth"][()]
            if not np.isfinite(depth).all():
                problems.append(f"{dk}: agentview_depth has NaN/Inf")
            dmin, dmax = float(depth.min()), float(depth.max())
            if dmin < DEPTH_MIN or dmax > DEPTH_MAX:
                problems.append(f"{dk}: agentview_depth out of plausible range [{dmin:.3f}, {dmax:.3f}]")

    return problems


def build_contact_sheet(task_files: list[str], out_path: str):
    thumbs = []
    labels = []
    for path in task_files:
        with h5py.File(path, "r") as f:
            demo = f["data"]["demo_0"]
            T = demo["obs"]["agentview_rgb"].shape[0]
            frame = demo["obs"]["agentview_rgb"][T // 2]
        thumbs.append(frame)
        labels.append(os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)[:30])

    n = len(thumbs)
    cols = 4
    rows = (n + cols - 1) // cols
    h, w = thumbs[0].shape[:2]
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = thumb
    Image.fromarray(sheet).save(out_path)
    print(f"contact sheet -> {out_path}")
    for i, lbl in enumerate(labels):
        print(f"  [{i}] {lbl}")


def main():
    task_files = sorted(glob.glob(os.path.join(ROOT, "*", "*.hdf5")))
    print(f"found {len(task_files)} task files")

    total_problems = 0
    for path in task_files:
        problems = check_file(path)
        status = "OK" if not problems else f"{len(problems)} ISSUE(S)"
        print(f"[{status}] {os.path.relpath(path, ROOT)}")
        for p in problems:
            print(f"    - {p}")
        total_problems += len(problems)

    print(f"\nTotal issues: {total_problems}")

    build_contact_sheet(task_files, "/tmp/claude-1001/-home-gibeom-pilab-X-VLA/2da855b6-dd5f-4276-9d8a-cfcde2098f35/scratchpad/depth_regen_test/contact_sheet.png")


if __name__ == "__main__":
    main()
