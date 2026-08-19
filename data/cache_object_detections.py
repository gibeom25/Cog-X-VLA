"""
Block 13 (part 2): cache per-frame object detections (blocks 1-5) for
every demo in data/regenerated/, per the offline-cache decision in
docs/initial_plan.md §0.

SAM3 re-detection only runs every `detect_every` frames (matches the
policy's action-chunk re-query cadence discussed for inference), not
every single frame -- expensive, and unnecessary since static objects'
world positions don't change between detections. Every frame still gets
its own EEF-relative transform (block 5) computed fresh from that
frame's own EEF pose against the most recently detected world positions,
per relative_pose.py's docstring on decoupling detection cadence from
per-step EEF updates.

Output: one JSON file per demo, alongside the source HDF5
(<task>_demo_<i>_objects.json), each entry a list of per-object dicts
{label, relative_xyz, half_extents} for that frame.

Run in the `sam3` env.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import sam3  # noqa: F401 -- must precede sys.path.insert, see perception/clustering.py
sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/fusion")
sys.path.insert(0, "/home/gibeom_pilab/cog-xvla")

import h5py
import numpy as np
from PIL import Image

from perception.np_extractor import extract_object_phrases
from perception.sam3_wrapper import Sam3Wrapper
from perception.clustering import representative_pixels
from perception.depth_localize import pixel_to_world, flipped_to_native_pixel, estimate_object_extent
from perception.relative_pose import world_to_eef_relative

ROOT = "/home/gibeom_pilab/cog-xvla/data/regenerated"
DETECT_EVERY = 10


def flip(img: np.ndarray) -> np.ndarray:
    return np.flip(np.flip(img, 0), 1)


def axisangle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """axis-angle -> 3x3 rotation matrix (Rodrigues' formula), without
    needing robosuite (not installed in the sam3 env)."""
    theta = np.linalg.norm(aa)
    if theta < 1e-8:
        return np.eye(3)
    axis = aa / theta
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def detect_frame(wrapper: Sam3Wrapper, rgb_native: np.ndarray, depth_native: np.ndarray,
                  K: np.ndarray, R: np.ndarray, instruction: str):
    """Run blocks 1-4 on one frame (native/unflipped RGB+depth as stored in
    the HDF5); returns world-frame detections: list of (label, world_xyz, half_extents)."""
    H, W = depth_native.shape
    rgb_flipped = flip(rgb_native)
    depth_flipped = flip(depth_native)

    candidates = extract_object_phrases(instruction)
    detections = wrapper.segment_all(rgb_flipped, candidates)

    results = []
    for cands, det in zip(candidates, detections):
        if det is None:
            continue
        for mask in det.masks:
            for p in representative_pixels(mask):
                d = float(depth_flipped[p.y, p.x])
                if d <= 0:
                    continue
                y_native, x_native = flipped_to_native_pixel(p.y, p.x, H, W)
                world_xyz = pixel_to_world(y_native, x_native, d, K, R)
                half_extents = estimate_object_extent(mask, depth_flipped, K, R)
                results.append((cands[0], world_xyz, half_extents))
    return results


def process_demo(wrapper: Sam3Wrapper, hdf5_path: str, demo_key: str):
    with h5py.File(hdf5_path, "r") as f:
        instruction = json.loads(f["data"].attrs["problem_info"])["language_instruction"]
        K = np.array(f["data"].attrs["camera_K"])
        R = np.array(f["data"].attrs["camera_R"])
        demo = f["data"][demo_key]
        agentview_rgb = demo["obs"]["agentview_rgb"][()]
        agentview_depth = demo["obs"]["agentview_depth"][()]
        ee_pos = demo["obs"]["ee_pos"][()]
        ee_ori = demo["obs"]["ee_ori"][()]

    T_frames = agentview_rgb.shape[0]
    last_world_objects = []  # list of (label, world_xyz, half_extents)
    per_frame = []

    for t in range(T_frames):
        if t % DETECT_EVERY == 0 or not last_world_objects:
            last_world_objects = detect_frame(wrapper, agentview_rgb[t], agentview_depth[t], K, R, instruction)

        eef_pos = ee_pos[t]
        eef_rot_mat = axisangle_to_rotmat(ee_ori[t])

        frame_objects = []
        for label, world_xyz, half_extents in last_world_objects:
            rel_xyz = world_to_eef_relative(world_xyz, eef_pos, eef_rot_mat)
            frame_objects.append({
                "label": label,
                "relative_xyz": rel_xyz.tolist(),
                "half_extents": half_extents.tolist(),
            })
        per_frame.append(frame_objects)

    return per_frame


def main():
    wrapper = Sam3Wrapper()
    task_files = sorted(glob.glob(os.path.join(ROOT, "*", "*.hdf5")))
    print(f"found {len(task_files)} task files")

    t0 = time.time()
    n_demos = 0
    for path in task_files:
        with h5py.File(path, "r") as f:
            demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")],
                                key=lambda k: int(k.split("_")[1]))
        for demo_key in demo_keys:
            out_path = path.replace(".hdf5", f"_{demo_key}_objects.json")
            per_frame = process_demo(wrapper, path, demo_key)
            with open(out_path, "w") as f:
                json.dump(per_frame, f)
            n_objs = sum(len(fo) for fo in per_frame)
            print(f"  {os.path.basename(out_path)}: {len(per_frame)} frames, {n_objs} object-detections total")
            n_demos += 1

    print(f"\ncached {n_demos} demos in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
