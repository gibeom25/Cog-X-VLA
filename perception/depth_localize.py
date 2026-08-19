"""
Block 4: pixel + depth -> 3D world coordinate.

Pure geometry, no learned components. Uses the camera intrinsic (K) and
extrinsic (camera-to-world pose R) from robosuite's camera_utils. Depth
must already be the *real* depth (meters), i.e. already passed through
`robosuite.utils.camera_utils.get_real_depth_map` -- MuJoCo's raw depth
buffer is a normalized [0,1] value, not a distance.

Convention (matches robosuite's get_camera_extrinsic_matrix /
project_points_from_world_to_camera): camera frame is OpenCV-style
(x-right, y-down, z-forward into the scene), and R transforms points in
that frame directly into world coordinates.

IMPORTANT -- display flip: robosuite/LIBERO's raw render is upside down
relative to a natural photo (`np.flip(np.flip(img, 0), 1)` is the usual
correction, e.g. in evaluation/libero/libero_client.py). SAM3 was trained
on natural (right-side-up) images and its detection quality measurably
degrades on the raw orientation (confirmed empirically: "bowl" went from
3 confident detections on the flipped frame to 0 on the raw frame), so
the image handed to SAM3 must be flipped. K and R from robosuite describe
the *physical* camera and are computed for the raw/native orientation, so
a pixel coordinate coming from SAM3/clustering (which operated on the
flipped image) must be converted back to native coordinates -- via
`flipped_to_native_pixel` -- before calling `pixel_to_world`. Depth must
also be flipped the same way as the RGB frame before indexing it with a
flipped-space pixel coordinate.
"""

from __future__ import annotations

import numpy as np


def flipped_to_native_pixel(y: int, x: int, height: int, width: int):
    """Convert a pixel coord found on the display-flipped image (what SAM3
    sees) back to the raw/native orientation that K and R assume."""
    return height - 1 - y, width - 1 - x


def pixel_to_world(y: int, x: int, depth_value: float, K: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    """
    Args:
        y, x: pixel row, col (row = height axis, col = width axis).
        depth_value: real depth (meters) at that pixel.
        K: 3x3 camera intrinsic matrix.
        cam_to_world: 4x4 camera-to-world extrinsic matrix.

    Returns:
        [3] world-frame xyz coordinate (meters).
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (x - cx) * depth_value / fx
    y_cam = (y - cy) * depth_value / fy
    z_cam = depth_value

    p_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    p_world = cam_to_world @ p_cam
    return p_world[:3]


def estimate_object_extent(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, cam_to_world: np.ndarray,
                            subsample: int = 4) -> np.ndarray:
    """
    Object size estimate for obstacle clearance / rough grasp-affordance,
    without needing a full boundary/contour representation (see project
    notes: a single point+radius sphere has no notion of "how big", but a
    full 3D contour is viewpoint-partial, variable-size, and adds more
    complexity than it's worth at this stage -- this is the agreed middle
    ground). Unprojects every (subsampled) mask pixel to a 3D point using
    the same geometry as pixel_to_world, then returns the object's extent
    along its 3 principal axes via PCA, sorted largest to smallest.

    Deliberately drops orientation (which world/relative direction each
    axis points in) and keeps only the 3 magnitudes -- e.g. "this object
    has one long axis and two short ones" (a bottle) vs "roughly equal on
    all axes" (a ball) -- letting the policy's own pretrained shape/grasp
    priors (triggered by the semantic label embedding, block 6) do the
    rest, rather than hand-engineering a grasp direction.

    Args:
        mask: [H, W] bool, in the same flipped-image space as SAM3 output.
        depth: [H, W] real depth (meters), flipped to match mask.
        K, cam_to_world: as in pixel_to_world.
        subsample: use every Nth mask pixel (mask interiors are highly
            redundant for a PCA extent estimate; full density isn't needed).

    Returns:
        [3] half-extents (meters) along the object's 3 principal axes,
        sorted descending (major, mid, minor).
    """
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    ys, xs = ys[::subsample], xs[::subsample]
    if len(ys) < 3:
        return np.zeros(3)

    points = np.stack([
        pixel_to_world(*flipped_to_native_pixel(int(y), int(x), H, W), float(depth[y, x]), K, cam_to_world)
        for y, x in zip(ys, xs)
    ])  # [M, 3]

    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = (centered.T @ centered) / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    axes = eigvecs[:, ::-1]  # descending eigenvalue order

    projections = centered @ axes  # [M, 3]
    half_extents = (projections.max(axis=0) - projections.min(axis=0)) / 2.0
    return half_extents  # already descending, since axes are sorted by eigenvalue


def world_to_pixel(world_xyz: np.ndarray, K: np.ndarray, cam_to_world: np.ndarray):
    """Inverse of pixel_to_world, for round-trip validation. Mirrors
    robosuite.utils.camera_utils.project_points_from_world_to_camera exactly
    (same K_exp @ inv(R), perspective divide, height/width swap), reimplemented
    without a robosuite dependency so it also runs in the sam3 env."""
    world_to_cam = np.linalg.inv(cam_to_world)
    p_world = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0])
    p_cam = world_to_cam @ p_world  # [x_c, y_c, z_c, 1]
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    return round(v), round(u)  # (row, col)


if __name__ == "__main__":
    import sys
    import sam3  # noqa: F401  -- see clustering.py for why this must come first
    sys.path.insert(0, "/home/gibeom_pilab/cog-xvla")
    from PIL import Image
    from perception.np_extractor import extract_object_phrases
    from perception.sam3_wrapper import Sam3Wrapper
    from perception.clustering import representative_pixels

    OUT = "/tmp/claude-1001/-home-gibeom-pilab-X-VLA/2da855b6-dd5f-4276-9d8a-cfcde2098f35/scratchpad/sam3_test"
    image = Image.open(f"{OUT}/frame.png").convert("RGB")  # display-flipped, what SAM3 needs
    depth = np.load(f"{OUT}/depth.npy")  # flipped to match frame.png
    K = np.load(f"{OUT}/K.npy")
    R = np.load(f"{OUT}/R.npy")  # camera-to-world, native orientation
    H, W = depth.shape

    instruction = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    candidates = extract_object_phrases(instruction)
    wrapper = Sam3Wrapper()
    detections = wrapper.segment_all(image, candidates)

    for cands, det in zip(candidates, detections):
        if det is None:
            print(f"{cands[0]!r}: no detection")
            continue
        for i, mask in enumerate(det.masks):
            for p in representative_pixels(mask):
                d = float(depth[p.y, p.x])  # depth.npy is flipped, same space as p.y/p.x
                y_native, x_native = flipped_to_native_pixel(p.y, p.x, H, W)
                world_xyz = pixel_to_world(y_native, x_native, d, K, R)
                reproj_y, reproj_x = world_to_pixel(world_xyz, K, R)
                err = ((reproj_y - y_native) ** 2 + (reproj_x - x_native) ** 2) ** 0.5
                half_extents = estimate_object_extent(mask, depth, K, R)
                print(f"{cands[0]!r} inst{i}: flipped_px=({p.y},{p.x}) native_px=({y_native},{x_native}) "
                      f"depth={d:.3f}m -> world={np.round(world_xyz, 3)} err={err:.2f}px "
                      f"half_extents(m)={np.round(half_extents, 3)}")
