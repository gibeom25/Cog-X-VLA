"""
Block 13: build training samples from the regenerated LIBERO HDF5 files
(data/regenerate_libero_frames.py output).

Converts each demo's raw per-frame proprio (`obs/ee_pos`, `obs/ee_ori`
axis-angle, and the original 7-dim delta action's gripper command) into
the absolute ee6d representation XVLAWithPerception expects: [pos3,
rot6d6, grip1] padded to 20 dims by zeroing the second 10-dim slot, the
same convention observed in evaluation/libero/libero_client.py's
proprio construction (that file is the only concrete evidence of X-VLA's
expected action layout available in this repo -- there's no access to
X-VLA's own original training data pipeline to confirm it against).

No robosuite dependency (Rodrigues' formula reimplemented locally, same
approach as data/cache_object_detections.py) so this also runs in the
XVLA conda env, which doesn't have robosuite installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import h5py
import numpy as np


def _axisangle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """axis-angle -> 3x3 rotation matrix (Rodrigues' formula)."""
    theta = np.linalg.norm(aa)
    if theta < 1e-8:
        return np.eye(3)
    axis = aa / theta
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def axisangle_to_rotate6d(aa: np.ndarray) -> np.ndarray:
    """[T,3] or [3] axis-angle -> [T,6] or [6] 6D rotation (Zhou et al.),
    matching evaluation/libero/libero_client.py's AxisAngle_to_Rotate6D
    (rotation matrix's first two columns)."""
    single = aa.ndim == 1
    if single:
        aa = aa[None]
    mats = np.stack([_axisangle_to_rotmat(a) for a in aa])  # [T,3,3]
    r6d = np.concatenate([mats[:, :3, 0], mats[:, :3, 1]], axis=-1)  # [T,6]
    return r6d[0] if single else r6d


def pad_to_20(pose10: np.ndarray) -> np.ndarray:
    """[..., 10] -> [..., 20], second slot zeroed -- see module docstring."""
    return np.concatenate([pose10, np.zeros_like(pose10)], axis=-1)


@dataclass
class DemoSamples:
    instruction: str
    abs_pose10: np.ndarray   # [T, 10] = pos3 + rot6d6 + grip1, one row per frame
    agentview_rgb: np.ndarray       # [T, H, W, 3] uint8
    agentview_depth: np.ndarray     # [T, H, W] float32
    robot0_eye_in_hand_rgb: np.ndarray     # [T, H, W, 3] uint8


def load_demo(hdf5_path: str, demo_key: str = "demo_0") -> DemoSamples:
    with h5py.File(hdf5_path, "r") as f:
        problem_info = json.loads(f["data"].attrs["problem_info"])
        instruction = problem_info["language_instruction"]

        demo = f["data"][demo_key]
        actions = demo["actions"][()]          # [T-ish, 7], one fewer than obs in some collections
        obs = demo["obs"]
        ee_pos = obs["ee_pos"][()]             # [T, 3]
        ee_ori = obs["ee_ori"][()]             # [T, 3] axis-angle
        agentview_rgb = obs["agentview_rgb"][()]
        agentview_depth = obs["agentview_depth"][()]
        robot0_eye_in_hand_rgb = obs["robot0_eye_in_hand_rgb"][()]

    T_obs = ee_pos.shape[0]
    rot6d = axisangle_to_rotate6d(ee_ori)  # [T, 6]

    # actions[k] is the command applied at frame k to reach frame k+1; its
    # gripper channel is the cleanest available signal for "gripper state at
    # frame k" (raw finger qpos is continuous and noisier to threshold).
    # LIBERO's convention is -1=open/open-going, +1=close -> map to {0,1}
    # for the BCE-style target EE6DActionSpace expects.
    grip_cmd = (actions[:, -1] + 1.0) / 2.0  # [T_obs - 1] typically
    if grip_cmd.shape[0] < T_obs:
        grip_cmd = np.concatenate([grip_cmd, grip_cmd[-1:]])  # repeat last for the final, action-less frame
    grip_cmd = grip_cmd[:T_obs]

    abs_pose10 = np.concatenate([ee_pos, rot6d, grip_cmd[:, None]], axis=-1)  # [T, 10]

    return DemoSamples(
        instruction=instruction,
        abs_pose10=abs_pose10.astype(np.float32),
        agentview_rgb=agentview_rgb,
        agentview_depth=agentview_depth,
        robot0_eye_in_hand_rgb=robot0_eye_in_hand_rgb,
    )


def build_training_windows(demo: DemoSamples, num_actions: int = 30) -> List[dict]:
    """One window per starting frame t: proprio = state at t, action target
    = the next `num_actions` absolute states (repeats the last frame past
    the episode end, standard padding for action chunking)."""
    T_obs = demo.abs_pose10.shape[0]
    windows = []
    for t in range(T_obs - 1):  # need at least 1 future frame
        end = min(t + 1 + num_actions, T_obs)
        chunk = demo.abs_pose10[t + 1:end]
        if chunk.shape[0] < num_actions:
            pad = np.repeat(chunk[-1:], num_actions - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        windows.append({
            "t": t,
            "proprio20": pad_to_20(demo.abs_pose10[t]),      # [20]
            "action20": pad_to_20(chunk),                    # [num_actions, 20]
            "robot0_eye_in_hand_rgb": demo.robot0_eye_in_hand_rgb[t],       # [H,W,3], per §0 "카메라 입력 구성"
            "agentview_rgb": demo.agentview_rgb[t],           # kept for perception (SAM3), not fed to the model
            "agentview_depth": demo.agentview_depth[t],
        })
    return windows


if __name__ == "__main__":
    path = "/home/gibeom_pilab/cog-xvla/data/regenerated/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
    demo = load_demo(path, "demo_0")
    print(f"instruction: {demo.instruction!r}")
    print(f"frames: {demo.abs_pose10.shape[0]}")
    print(f"abs_pose10[0]: {demo.abs_pose10[0]}")
    print(f"abs_pose10[-1]: {demo.abs_pose10[-1]}")

    step_deltas = np.linalg.norm(np.diff(demo.abs_pose10[:, :3], axis=0), axis=-1)
    print(f"per-step position delta: mean={step_deltas.mean():.4f}m max={step_deltas.max():.4f}m "
          f"(sanity: should be small/smooth, not jumpy)")

    grip = demo.abs_pose10[:, 9]
    print(f"gripper trace (0=open,1=close) unique values: {np.unique(np.round(grip, 2))}")
    n_close = int((grip > 0.5).sum())
    print(f"frames with gripper closed: {n_close}/{len(grip)}")

    windows = build_training_windows(demo, num_actions=30)
    print(f"\nbuilt {len(windows)} training windows")
    w = windows[0]
    print(f"window[0]: proprio20 shape={w['proprio20'].shape}, action20 shape={w['action20'].shape}, "
          f"robot0_eye_in_hand_rgb shape={w['robot0_eye_in_hand_rgb'].shape}")
