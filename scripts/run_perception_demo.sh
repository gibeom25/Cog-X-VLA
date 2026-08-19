#!/usr/bin/env bash
# Phase 1 (perception, blocks 1-5) end-to-end demo + diagnostics.
#
# Usage:
#   scripts/run_perception_demo.sh [suite] [task_id] [out_dir]
#
# Examples:
#   scripts/run_perception_demo.sh                                   # libero_spatial task 0 -> outputs/my_test
#   scripts/run_perception_demo.sh libero_object 3 outputs/obj_task3
set -e

SUITE="${1:-libero_spatial}"
TASK_ID="${2:-0}"
OUT_DIR="${3:-outputs/my_test}"

cd "$(dirname "$0")/.."

echo "=== 1) render frame + depth/camera/EEF pose (libero_plus env) ==="
conda run -n libero_plus python data/render_test_frame.py \
  --suite "$SUITE" \
  --task_id "$TASK_ID" \
  --out_dir "$OUT_DIR"

echo ""
echo "=== 2) run blocks 1-5 (NP extract -> SAM3 -> cluster -> depth->3D -> EEF-relative) + viz (sam3 env) ==="
conda run -n sam3 python perception/relative_pose.py \
  --in_dir "$OUT_DIR"

echo ""
echo "=== 3) diagnostics: pixel/depth overlays + 3D coords labeled on the LIBERO frame (sam3 env) ==="
conda run -n sam3 python perception/debug_visualize.py \
  --in_dir "$OUT_DIR"

echo ""
echo "Done. Key output images in $OUT_DIR:"
echo "  relative_pose_viz.png   - world-frame + EEF-relative scatter plots"
echo "  debug_rgb_overlay.png   - mask outlines + points on the RGB frame"
echo "  debug_depth_overlay.png - same points on the colorized depth map"
echo "  debug_3d_on_image.png   - world/EEF-relative xyz labeled directly on the LIBERO frame"
