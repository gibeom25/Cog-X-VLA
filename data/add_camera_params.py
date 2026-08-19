"""
Add agentview camera intrinsic/extrinsic to already-regenerated task HDF5
files. agentview is a fixed external camera (unlike eye_in_hand, which
moves with the gripper), so K/R only need computing once per task, not
once per demo/frame -- reuses each file's stored bddl_file_name attr to
re-instantiate a throwaway env just for the camera params.

Run in the `libero_plus` conda env.
"""
from __future__ import annotations

import glob
import os

import h5py
import numpy as np
import robosuite.utils.camera_utils as camera_utils
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

ROOT = "/home/gibeom_pilab/cog-xvla/data/regenerated"


def add_params(hdf5_path: str):
    with h5py.File(hdf5_path, "r+") as f:
        if "camera_K" in f["data"].attrs and "camera_R" in f["data"].attrs:
            print(f"  skip (already has camera params): {os.path.basename(hdf5_path)}")
            return
        bddl_rel = f["data"].attrs["bddl_file_name"]
        resolution = int(f["data"].attrs.get("depth_resolution", 256))

    bddl_file = os.path.join(get_libero_path("bddl_files"), *bddl_rel.split("/")[-2:])
    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)
    env.reset()
    K = camera_utils.get_camera_intrinsic_matrix(env.env.sim, "agentview", resolution, resolution)
    R = camera_utils.get_camera_extrinsic_matrix(env.env.sim, "agentview")
    env.close()

    with h5py.File(hdf5_path, "r+") as f:
        f["data"].attrs["camera_K"] = K
        f["data"].attrs["camera_R"] = R
    print(f"  added camera params: {os.path.basename(hdf5_path)}")


def main():
    task_files = sorted(glob.glob(os.path.join(ROOT, "*", "*.hdf5")))
    print(f"found {len(task_files)} task files")
    for path in task_files:
        add_params(path)


if __name__ == "__main__":
    main()
