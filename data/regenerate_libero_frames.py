"""
Regenerate RGB + depth for LIBERO demos by replaying recorded mujoco states.

Each frame's `states[i]` fully determines the simulator state that produced
`obs/agentview_rgb[i]` in the original demo HDF5. Teleporting to `states[i]`
via `env.set_init_state(states[i])` and re-rendering therefore reproduces the
exact same frame, plus a depth channel that was never recorded originally.
This avoids any frame-alignment risk since RGB and depth come from the same
teleport-and-render call, index for index.

Run in the `libero_plus` conda env (needs `libero`/robosuite/mujoco).
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
from PIL import Image

from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.camera_utils as camera_utils


def regenerate_demo(hdf5_path: str, bddl_file: str, demo_key: str, camera_names, resolution: int):
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][demo_key]
        states = demo["states"][()]

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
        camera_names=camera_names,
        camera_depths=True,
    )
    env.seed(0)
    env.reset()

    rgb_out = {cam: [] for cam in camera_names}
    depth_out = {cam: [] for cam in camera_names}

    for i in range(states.shape[0]):
        obs = env.set_init_state(states[i])
        for cam in camera_names:
            rgb_out[cam].append(obs[f"{cam}_image"])
            raw_depth = obs[f"{cam}_depth"]  # normalized buffer, shape (H, W, 1)
            real_depth = camera_utils.get_real_depth_map(env.env.sim, raw_depth)
            depth_out[cam].append(real_depth[..., 0])  # meters, shape (H, W)

    env.close()
    for cam in camera_names:
        rgb_out[cam] = np.stack(rgb_out[cam])
        depth_out[cam] = np.stack(depth_out[cam])
    return rgb_out, depth_out


def regenerate_task_hdf5(src_path: str, out_path: str, num_demos: int, resolution: int,
                          camera_names=("agentview", "robot0_eye_in_hand")):
    """Regenerate the first `num_demos` demos of one task HDF5 into a new file
    with `resolution`x`resolution` RGB + a new real-depth channel per camera.
    All other original datasets (actions, states, dones, rewards, robot_states,
    proprio obs) are copied through unchanged.
    """
    from libero.libero import get_libero_path

    with h5py.File(src_path, "r") as f:
        bddl_rel = f["data"].attrs["bddl_file_name"]
        attrs = dict(f["data"].attrs)
        demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
        demo_keys = sorted(demo_keys, key=lambda k: int(k.split("_")[1]))[:num_demos]
    bddl_file = os.path.join(get_libero_path("bddl_files"), *bddl_rel.split("/")[-2:])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as out_f:
        grp = out_f.create_group("data")
        for k, v in attrs.items():
            grp.attrs[k] = v
        grp.attrs["depth_resolution"] = resolution
        grp.attrs["depth_camera_names"] = list(camera_names)

        for demo_key in demo_keys:
            print(f"  {os.path.basename(src_path)} :: {demo_key}")
            rgb_out, depth_out = regenerate_demo(src_path, bddl_file, demo_key, list(camera_names), resolution)

            demo_grp = out_f.create_group(f"data/{demo_key}")
            with h5py.File(src_path, "r") as f:
                src_demo = f["data"][demo_key]
                for passthrough in ("actions", "states", "dones", "rewards", "robot_states"):
                    if passthrough in src_demo:
                        demo_grp.create_dataset(passthrough, data=src_demo[passthrough][()])
                obs_grp = demo_grp.create_group("obs")
                for k in src_demo["obs"].keys():
                    if k.endswith("_rgb"):
                        continue  # replaced below at new resolution
                    obs_grp.create_dataset(k, data=src_demo["obs"][k][()])

            for cam in camera_names:
                obs_grp.create_dataset(f"{cam}_rgb", data=rgb_out[cam], compression="gzip", compression_opts=4)
                obs_grp.create_dataset(f"{cam}_depth", data=depth_out[cam].astype(np.float32),
                                        compression="gzip", compression_opts=4)
    print(f"wrote {out_path} ({len(demo_keys)} demos)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True, help="Path to a raw LIBERO demo HDF5 file")
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--out_dir", default="/tmp/regen_test")
    parser.add_argument("--num_frames", type=int, default=5, help="Only regenerate first N frames for a quick check")
    parser.add_argument("--resolution", type=int, default=128, help="Match original collection resolution for pixel-diff validation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.hdf5, "r") as f:
        bddl_rel = f["data"].attrs["bddl_file_name"]
        n_total = f["data"][args.demo]["states"].shape[0]
    from libero.libero import get_libero_path
    bddl_file = os.path.join(get_libero_path("bddl_files"), *bddl_rel.split("/")[-2:])

    n = min(args.num_frames, n_total)
    print(f"Regenerating {n}/{n_total} frames from {args.demo} @ {args.resolution}px")
    print(f"bddl: {bddl_file}")

    camera_names = ["agentview", "robot0_eye_in_hand"]
    rgb_out, depth_out = regenerate_demo(args.hdf5, bddl_file, args.demo, camera_names, args.resolution)

    # ---- Validation: compare regenerated agentview RGB against stored RGB ----
    with h5py.File(args.hdf5, "r") as f:
        stored_rgb = f["data"][args.demo]["obs"]["agentview_rgb"][()]

    for i in range(min(n, rgb_out["agentview"].shape[0], stored_rgb.shape[0])):
        regen = rgb_out["agentview"][i]
        orig = stored_rgb[i]
        if regen.shape != orig.shape:
            print(f"frame {i}: SHAPE MISMATCH regen={regen.shape} orig={orig.shape} (resolution differs, skip diff)")
            continue
        diff = np.abs(regen.astype(np.int16) - orig.astype(np.int16))
        print(f"frame {i}: mean_abs_diff={diff.mean():.3f} max_abs_diff={diff.max()}")

        Image.fromarray(regen).save(os.path.join(args.out_dir, f"regen_rgb_{i}.png"))
        Image.fromarray(orig).save(os.path.join(args.out_dir, f"orig_rgb_{i}.png"))

        d = depth_out["agentview"][i]
        d_norm = ((d - d.min()) / max(d.max() - d.min(), 1e-6) * 255).astype(np.uint8)
        Image.fromarray(d_norm).save(os.path.join(args.out_dir, f"depth_{i}.png"))
        print(f"  depth range: {d.min():.4f}m - {d.max():.4f}m")

    print(f"Saved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
