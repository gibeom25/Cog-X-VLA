"""
Block 5: world-frame object position -> EEF-relative (egocentric) coordinate.

Updated every timestep from the *current* EEF pose, independent of how often
the object's world position itself gets refreshed by perception (see project
notes on decoupling SAM3 re-detection cadence from EEF-pose updates -- static
objects stay correct every step even with a stale detection, since only the
EEF side of the subtraction changes).

Pure geometry: relative = R_eef^T @ (object_world - eef_world_pos), i.e. the
inverse of the EEF's own rigid transform, expressing the object in the EEF's
local frame.
"""

from __future__ import annotations

import numpy as np


def world_to_eef_relative(object_world_xyz: np.ndarray, eef_world_pos: np.ndarray,
                           eef_world_rot: np.ndarray) -> np.ndarray:
    """
    Args:
        object_world_xyz: [3] object position, world frame.
        eef_world_pos: [3] EEF position, world frame.
        eef_world_rot: [3,3] EEF orientation (rotation matrix), world frame.

    Returns:
        [3] object position expressed in the EEF's own (egocentric) frame.
    """
    delta = object_world_xyz - eef_world_pos
    return eef_world_rot.T @ delta


if __name__ == "__main__":
    import sys
    import argparse
    import sam3  # noqa: F401  -- see clustering.py for why this must come first
    sys.path.insert(0, "/home/gibeom_pilab/cog-xvla")
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from perception.np_extractor import extract_object_phrases
    from perception.sam3_wrapper import Sam3Wrapper
    from perception.clustering import representative_pixels
    from perception.depth_localize import pixel_to_world, flipped_to_native_pixel

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True, help="output dir from data/render_test_frame.py")
    parser.add_argument("--instruction", default=None, help="override instruction.txt")
    args = parser.parse_args()

    image = Image.open(f"{args.in_dir}/frame.png").convert("RGB")
    depth = np.load(f"{args.in_dir}/depth.npy")
    K = np.load(f"{args.in_dir}/K.npy")
    R = np.load(f"{args.in_dir}/R.npy")
    eef_pos = np.load(f"{args.in_dir}/eef_pos.npy")
    eef_rot = np.load(f"{args.in_dir}/eef_rot.npy")
    H, W = depth.shape

    if args.instruction:
        instruction = args.instruction
    else:
        with open(f"{args.in_dir}/instruction.txt") as f:
            instruction = f.read().strip()
    print(f"instruction: {instruction}")
    print(f"eef world pos: {np.round(eef_pos, 3)}")

    candidates = extract_object_phrases(instruction)
    wrapper = Sam3Wrapper()
    detections = wrapper.segment_all(image, candidates)

    results = []  # (label, world_xyz, relative_xyz)
    for cands, det in zip(candidates, detections):
        if det is None:
            print(f"{cands[0]!r}: no detection")
            continue
        for i, mask in enumerate(det.masks):
            for p in representative_pixels(mask):
                d = float(depth[p.y, p.x])
                y_native, x_native = flipped_to_native_pixel(p.y, p.x, H, W)
                world_xyz = pixel_to_world(y_native, x_native, d, K, R)
                rel_xyz = world_to_eef_relative(world_xyz, eef_pos, eef_rot)
                label = cands[0] if len(det.masks) == 1 else f"{cands[0]}#{i}"
                results.append((label, world_xyz, rel_xyz))
                dist = np.linalg.norm(rel_xyz)
                print(f"{label}: world={np.round(world_xyz, 3)} "
                      f"eef_relative={np.round(rel_xyz, 3)} dist={dist:.3f}m")

    # ---- Visualization: top-down (bird's-eye, world XY) + egocentric (EEF-relative XY) ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax = axes[0]
    ax.scatter([eef_pos[0]], [eef_pos[1]], c="red", marker="*", s=200, label="EEF (world)", zorder=5)
    for label, world_xyz, _ in results:
        ax.scatter([world_xyz[0]], [world_xyz[1]], s=80)
        ax.annotate(label, (world_xyz[0], world_xyz[1]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title("World frame (bird's-eye XY)")
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.scatter([0], [0], c="red", marker="*", s=200, label="EEF (origin)", zorder=5)
    for label, _, rel_xyz in results:
        ax.scatter([rel_xyz[0]], [rel_xyz[1]], s=80)
        ax.annotate(label, (rel_xyz[0], rel_xyz[1]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title("EEF-relative frame (egocentric XY)")
    ax.set_xlabel("relative X (m)")
    ax.set_ylabel("relative Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_path = f"{args.in_dir}/relative_pose_viz.png"
    plt.savefig(out_path, dpi=120)
    print(f"saved visualization -> {out_path}")
