#!/usr/bin/env bash
# Visualize exactly what one training sample looks like once it reaches
# XVLAWithPerception: wrist image (actual model input), agentview frame
# (human reference only, not fed to the model), decoded object_raw/
# labels, decoded proprio, and the target action trajectory -- all in
# one saved PNG.
#
# Usage:
#   scripts/visualize_model_input.sh <hdf5_path> [demo_key] [frame_t]
#
# Example:
#   scripts/visualize_model_input.sh \
#     data/regenerated/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5 \
#     demo_0 10
set -e

HDF5="${1:?usage: $0 <hdf5_path> [demo_key] [frame_t]}"
DEMO="${2:-demo_0}"
T="${3:-10}"

cd "$(dirname "$0")/.."

conda run -n XVLA python policy/visualize_input.py \
  --hdf5 "$HDF5" \
  --demo "$DEMO" \
  --t "$T"
