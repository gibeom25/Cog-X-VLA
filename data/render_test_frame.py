"""
Render one LIBERO frame + everything the perception pipeline needs to test
against it: RGB (display-flipped, what SAM3 expects), real depth (flipped to
match), camera intrinsic/extrinsic, and the EEF pose at that instant.

Run in the `libero_plus` conda env.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import robosuite.utils.camera_utils as camera_utils
import robosuite.utils.transform_utils as T
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--settle_steps", type=int, default=10,
                        help="Zero-action steps after reset, matches eval-client convention")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[args.suite]()
    task = suite.get_task(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"instruction: {task.language}")
    with open(os.path.join(args.out_dir, "instruction.txt"), "w") as f:
        f.write(task.language)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        camera_depths=True,
    )
    env.seed(0)
    env.reset()
    init_states = suite.get_task_init_states(args.task_id)
    obs = env.set_init_state(init_states[0])
    for _ in range(args.settle_steps):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))

    def flip(img):
        return np.flip(np.flip(img, 0), 1)

    rgb = flip(obs["agentview_image"])
    raw_depth = obs["agentview_depth"]
    real_depth = camera_utils.get_real_depth_map(env.env.sim, raw_depth)[..., 0]
    depth = flip(real_depth)

    K = camera_utils.get_camera_intrinsic_matrix(env.env.sim, "agentview", args.resolution, args.resolution)
    R = camera_utils.get_camera_extrinsic_matrix(env.env.sim, "agentview")  # camera-to-world, native orientation

    eef_pos = env.env.robots[0].controller.ee_pos
    eef_rot = env.env.robots[0].controller.ee_ori_mat  # 3x3, world frame

    Image.fromarray(rgb).save(os.path.join(args.out_dir, "frame.png"))
    np.save(os.path.join(args.out_dir, "depth.npy"), depth)
    np.save(os.path.join(args.out_dir, "K.npy"), K)
    np.save(os.path.join(args.out_dir, "R.npy"), R)
    np.save(os.path.join(args.out_dir, "eef_pos.npy"), eef_pos)
    np.save(os.path.join(args.out_dir, "eef_rot.npy"), eef_rot)

    print(f"eef_pos: {eef_pos}")
    print(f"saved all artifacts to {args.out_dir}")
    env.close()


if __name__ == "__main__":
    main()
