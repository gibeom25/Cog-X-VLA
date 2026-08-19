"""
Visualize exactly what one training sample looks like once it reaches
XVLAWithPerception.forward()/generate_actions() -- the wrist image after
the processor's resize+normalize, the decoded instruction, the decoded
per-object labels + relative_xyz + half_extents (object_raw as the model
actually receives it), decoded proprio, and the target action trajectory.
Saved as one combined figure so nothing needs cross-referencing separate
runs.

Run in the XVLA conda env.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/policy")
sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/data")
sys.path.insert(0, "/home/gibeom_pilab/cog-xvla/X-VLA")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.processing_xvla import XVLAProcessor
from smoke_train import build_sample
from prepare_libero import load_demo


def unnormalize_image(pixel_values: torch.Tensor, mean, std) -> np.ndarray:
    """[C,H,W] normalized tensor -> [H,W,C] uint8, inverse of the image
    processor's resize+normalize step."""
    img = pixel_values.detach().cpu().float().numpy()
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    img = (img * std + mean).clip(0, 1)
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True, help="path to a regenerated task HDF5")
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--t", type=int, default=10, help="window/frame index")
    parser.add_argument("--out", default=None, help="output PNG path (default: alongside --hdf5)")
    args = parser.parse_args()

    processor = XVLAProcessor.from_pretrained("2toINF/X-VLA-Libero")
    device = torch.device("cpu")  # pure input inspection, no model forward needed

    sample, n_obj = build_sample(args.hdf5, args.demo, args.t, processor, device)
    demo_for_ref = load_demo(args.hdf5, args.demo)  # for the agentview reference frame only -- not part of the model input

    # ---- decode everything back to human-readable form ----
    instruction = processor.tokenizer.decode(sample["input_ids"][0], skip_special_tokens=True)

    wrist_img = unnormalize_image(
        sample["image_input"][0, 0], processor.image_processor.image_mean, processor.image_processor.image_std
    )

    proprio = sample["proprio"][0].numpy()
    pos, rot6d, grip = proprio[0:3], proprio[3:9], proprio[9]

    object_raw = sample["object_raw"][0].numpy()          # [N, 6]
    label_ids = sample["object_label_ids"][0]              # [N, L]
    labels = [processor.tokenizer.decode(ids, skip_special_tokens=True) for ids in label_ids]

    action = sample["action"][0].numpy()                    # [30, 20]
    action_pos = action[:, 0:3]                              # position part of the target trajectory

    # ---- combined figure ----
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(wrist_img)
    ax_img.set_title("image_input[0,0] (wrist cam, un-normalized)\nthis is literally what the model sees")
    ax_img.axis("off")

    ax_ref = fig.add_subplot(gs[0, 1])
    ax_ref.imshow(demo_for_ref.agentview_rgb[args.t])
    ax_ref.set_title(f"agentview @ t={args.t} (human reference only)\nNOT fed to the model -- for sanity-checking object_raw against the real scene")
    ax_ref.axis("off")

    ax_obj = fig.add_subplot(gs[0, 2])
    ax_obj.scatter([0], [0], c="red", marker="*", s=220, label="EEF (origin)", zorder=5)
    for i in range(object_raw.shape[0]):
        x, y = object_raw[i, 0], object_raw[i, 1]
        r = max(object_raw[i, 3:5].max(), 0.01) * 4000
        ax_obj.scatter([x], [y], s=r, alpha=0.65)
        ax_obj.annotate(labels[i], (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax_obj.set_title(f"object_raw (block 6 input) -- N={object_raw.shape[0]}\nrelative_xyz(3)+half_extents(3), marker size ~ footprint")
    ax_obj.set_xlabel("relative X (m)")
    ax_obj.set_ylabel("relative Y (m)")
    ax_obj.axis("equal")
    ax_obj.grid(True, alpha=0.3)
    ax_obj.legend(fontsize=8)

    ax_txt = fig.add_subplot(gs[1, 0:2])
    ax_txt.axis("off")
    grip_state = "CLOSED" if grip > 0.5 else "OPEN"
    obj_lines = "\n".join(f"  [{i}] {labels[i]!r}: rel_xyz={np.round(object_raw[i,:3],3)} "
                           f"half_extents={np.round(object_raw[i,3:],3)}" for i in range(object_raw.shape[0])) or "  (none detected)"
    text = (
        f"instruction (decoded input_ids):\n  {instruction!r}\n\n"
        f"proprio (decoded, first 10 of 20 dims -- second half is zero-padding):\n"
        f"  pos={np.round(pos,3)}  rot6d={np.round(rot6d,3)}  gripper={grip:.2f} ({grip_state})\n\n"
        f"objects fed to block 6 (N={object_raw.shape[0]}):\n{obj_lines}\n\n"
        f"action target shape: {action.shape}  (30-step absolute pose chunk)\n"
        f"  action[0]  pos={np.round(action[0,:3],3)} grip={action[0,9]:.2f}\n"
        f"  action[-1] pos={np.round(action[-1,:3],3)} grip={action[-1,9]:.2f}"
    )
    ax_txt.text(0.0, 1.0, text, fontsize=10, family="monospace", va="top", ha="left", transform=ax_txt.transAxes)

    ax_traj = fig.add_subplot(gs[1, 2])
    ax_traj.plot(action_pos[:, 0], action_pos[:, 1], "-o", markersize=3, color="tab:blue")
    ax_traj.scatter([pos[0]], [pos[1]], c="red", marker="*", s=150, label="current EEF pos", zorder=5)
    ax_traj.scatter([action_pos[0, 0]], [action_pos[0, 1]], c="green", s=40, label="target t+1", zorder=4)
    ax_traj.scatter([action_pos[-1, 0]], [action_pos[-1, 1]], c="purple", s=40, label="target t+30", zorder=4)
    ax_traj.set_title("action target trajectory (world XY, absolute)")
    ax_traj.set_xlabel("world X (m)")
    ax_traj.set_ylabel("world Y (m)")
    ax_traj.axis("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(fontsize=8)

    plt.tight_layout()
    out_path = args.out or args.hdf5.replace(".hdf5", f"_{args.demo}_t{args.t}_model_input.png")
    plt.savefig(out_path, dpi=130)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
