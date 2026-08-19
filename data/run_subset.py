"""
Prototype-scale depth regeneration over a small subset:
2 tasks per suite x 5 demos per task x 4 suites = 40 demos total.
"""
import glob
import os
import time

from regenerate_libero_frames import regenerate_task_hdf5

SRC_ROOT = "/home/gibeom_pilab/LIBERO/LIBERO-plus/libero/datasets"
OUT_ROOT = "/home/gibeom_pilab/cog-xvla/data/regenerated"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
TASKS_PER_SUITE = 2
DEMOS_PER_TASK = 5
RESOLUTION = 256

def main():
    t0 = time.time()
    total = 0
    for suite in SUITES:
        task_files = sorted(glob.glob(os.path.join(SRC_ROOT, suite, "*.hdf5")))[:TASKS_PER_SUITE]
        print(f"=== {suite}: {len(task_files)} tasks ===")
        for src_path in task_files:
            task_name = os.path.basename(src_path)
            out_path = os.path.join(OUT_ROOT, suite, task_name)
            regenerate_task_hdf5(src_path, out_path, num_demos=DEMOS_PER_TASK, resolution=RESOLUTION)
            total += DEMOS_PER_TASK
    print(f"Done: {total} demos regenerated in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
