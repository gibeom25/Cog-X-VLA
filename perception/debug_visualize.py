"""
Diagnostic tool: visualize exactly where each detected object's
representative pixel/mask lands (on the RGB frame AND on the depth map),
and independently sanity-check depth accuracy against the known table
surface -- rather than trusting the pixel->world->pixel round-trip alone,
which only proves the formula is self-consistent, not that the inputs are
right (see block 4's commit message for a real bug that check missed).

Run in the `sam3` env: python perception/debug_visualize.py --in_dir <dir>
"""
from __future__ import annotations

import argparse
import sys

import sam3  # noqa: F401  -- must be imported before cog-xvla goes on sys.path

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
from perception.relative_pose import world_to_eef_relative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--instruction", default=None)
    args = parser.parse_args()

    image = Image.open(f"{args.in_dir}/frame.png").convert("RGB")
    depth = np.load(f"{args.in_dir}/depth.npy")  # flipped, meters
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

    candidates = extract_object_phrases(instruction)
    wrapper = Sam3Wrapper()
    detections = wrapper.segment_all(image, candidates)

    rgb_overlay = np.array(image).copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 60, 255), (255, 200, 0), (255, 0, 255), (0, 255, 255)]
    points = []  # (label, color, y, x, depth, world, dist_from_camera)

    ci = 0
    for cands, det in zip(candidates, detections):
        if det is None:
            print(f"{cands[0]!r}: no detection")
            continue
        for i, mask in enumerate(det.masks):
            for p in representative_pixels(mask):
                color = colors[ci % len(colors)]
                ci += 1
                label = cands[0] if len(det.masks) == 1 else f"{cands[0]}#{i}"

                d = float(depth[p.y, p.x])
                y_native, x_native = flipped_to_native_pixel(p.y, p.x, H, W)
                world_xyz = pixel_to_world(y_native, x_native, d, K, R)

                # mask boundary, for context on how big/where the mask is
                edge = mask ^ np.roll(mask, 1, axis=0) | (mask ^ np.roll(mask, 1, axis=1))
                rgb_overlay[edge] = color
                rgb_overlay[max(p.y - 3, 0):p.y + 4, max(p.x - 3, 0):p.x + 4] = color

                rel_xyz = world_to_eef_relative(world_xyz, eef_pos, eef_rot)
                points.append((label, color, p.y, p.x, d, world_xyz, rel_xyz))
                print(f"{label}: flipped_px=({p.y},{p.x}) depth={d:.3f}m "
                      f"world={np.round(world_xyz, 3)} eef_relative={np.round(rel_xyz, 3)}")

    Image.fromarray(rgb_overlay).save(f"{args.in_dir}/debug_rgb_overlay.png")

    # ---- Depth map visualization with the same points marked ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im = axes[0].imshow(depth, cmap="viridis")
    axes[0].set_title("Depth map (m), flipped space")
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    for label, color, y, x, d, _, _ in points:
        c = tuple(v / 255 for v in color)
        axes[0].scatter([x], [y], c=[c], s=60, edgecolors="white")
        axes[0].annotate(f"{label}\n{d:.2f}m", (x, y), fontsize=7, color="white",
                          xytext=(4, 4), textcoords="offset points")

    axes[1].imshow(rgb_overlay)
    axes[1].set_title("RGB with mask outlines + representative points")
    for label, color, y, x, d, _, _ in points:
        c = tuple(v / 255 for v in color)
        axes[1].annotate(label, (x, y), fontsize=7, color="white",
                          xytext=(4, -10), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(f"{args.in_dir}/debug_depth_overlay.png", dpi=130)

    # ---- 3D coordinates labeled directly on the LIBERO frame ----
    fig2, ax2 = plt.subplots(figsize=(9, 9))
    ax2.imshow(image)
    ax2.set_title(f"3D coordinates on LIBERO frame\ninstruction: {instruction}", fontsize=9)
    for label, color, y, x, d, world_xyz, rel_xyz in points:
        c = tuple(v / 255 for v in color)
        ax2.scatter([x], [y], c=[c], s=120, edgecolors="white", linewidths=1.5, zorder=5)
        text = (f"{label}\n"
                f"world=({world_xyz[0]:.2f}, {world_xyz[1]:.2f}, {world_xyz[2]:.2f})\n"
                f"rel=({rel_xyz[0]:.2f}, {rel_xyz[1]:.2f}, {rel_xyz[2]:.2f})")
        ax2.annotate(text, (x, y), fontsize=7, color="black", fontweight="bold",
                     xytext=(6, 6), textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.25", fc=c, ec="black", alpha=0.85))
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig(f"{args.in_dir}/debug_3d_on_image.png", dpi=130)

    print(f"saved {args.in_dir}/debug_rgb_overlay.png, debug_depth_overlay.png, debug_3d_on_image.png")

    # ---- Independent depth-accuracy sanity check ----
    # Objects sit ON the table, so their world Z should cluster close to the
    # table surface height. We don't have a labeled "table" pixel, but we do
    # know the EEF's world Z is a real, independently-measured value (from
    # the sim's proprioception, nothing to do with our depth/camera math) --
    # so compare against that as an external reference instead of only
    # checking our own pixel_to_world/world_to_pixel round-trip.
    print(f"\nEEF world Z (independent reference, from sim proprioception): {eef_pos[2]:.3f}m")
    zs = [w[2] for _, _, _, _, _, w, _ in points]
    if zs:
        print(f"detected object world Z range: {min(zs):.3f}m - {max(zs):.3f}m "
              f"(mean {np.mean(zs):.3f}m)")
        print("-> objects should sit at/below the resting EEF height and within "
              "a small spread of each other if they're all on the same tabletop.")


if __name__ == "__main__":
    main()
