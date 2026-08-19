"""
Block 3: mask -> representative pixel coordinate(s).

SAM3 already separates instances (e.g. 3 distinct "bowl" masks for 3 bowls),
but each mask can still:
  (a) contain small disconnected noise fragments, and
  (b) be concave/ring-shaped (e.g. a bowl or drawer handle seen at an angle),
      where the naive centroid can land *outside* the actual object.

This block cleans each mask via connected components (keep components above
a minimum area -- this also transparently handles the case where a mask
genuinely contains multiple blobs) and picks, for each component, the point
of maximum distance from the mask boundary (the "most interior" point),
which is guaranteed to lie inside the mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import ndimage


@dataclass
class ObjectPoint:
    y: int
    x: int
    area_px: int


def representative_pixels(mask: np.ndarray, min_area: int = 20) -> List[ObjectPoint]:
    """
    Args:
        mask: [H, W] bool array (a single SAM3 instance mask).
        min_area: components smaller than this (pixels) are treated as noise
            and dropped.

    Returns:
        One ObjectPoint per surviving connected component, sorted by area
        descending (largest/most confident first).
    """
    labeled, n_components = ndimage.label(mask)
    points: List[ObjectPoint] = []

    for label_id in range(1, n_components + 1):
        component = labeled == label_id
        area = int(component.sum())
        if area < min_area:
            continue

        # Distance transform *within* the component: distance to the nearest
        # background pixel. Its argmax is the most-interior point, robust to
        # concave/ring shapes where the plain centroid can fall outside.
        dist = ndimage.distance_transform_edt(component)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        points.append(ObjectPoint(y=int(y), x=int(x), area_px=area))

    points.sort(key=lambda p: p.area_px, reverse=True)
    return points


if __name__ == "__main__":
    import sys
    import sam3  # noqa: F401  -- import before adding cog-xvla to sys.path: the
    # editable-installed "sam3" package and the "cog-xvla/sam3" submodule
    # directory (no __init__.py) share the name, and if "cog-xvla" is on
    # sys.path *first*, `import sam3` resolves to the submodule dir as a
    # broken namespace package instead of the real package.
    sys.path.insert(0, "/home/gibeom_pilab/cog-xvla")
    from PIL import Image
    from perception.np_extractor import extract_object_phrases
    from perception.sam3_wrapper import Sam3Wrapper

    instruction = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    image = Image.open(
        "/tmp/claude-1001/-home-gibeom-pilab-X-VLA/2da855b6-dd5f-4276-9d8a-cfcde2098f35/scratchpad/sam3_test/frame.png"
    ).convert("RGB")

    candidates = extract_object_phrases(instruction)
    wrapper = Sam3Wrapper()
    detections = wrapper.segment_all(image, candidates)

    overlay = np.array(image).copy()
    for cands, det in zip(candidates, detections):
        if det is None:
            print(f"{cands[0]!r}: no detection")
            continue
        for i, mask in enumerate(det.masks):
            pts = representative_pixels(mask)
            print(f"{cands[0]!r} instance {i}: {len(pts)} component(s)")
            for p in pts:
                print(f"    y={p.y} x={p.x} area={p.area_px}")
                overlay[max(p.y - 2, 0):p.y + 3, max(p.x - 2, 0):p.x + 3] = [255, 0, 255]

    Image.fromarray(overlay).save(
        "/tmp/claude-1001/-home-gibeom-pilab-X-VLA/2da855b6-dd5f-4276-9d8a-cfcde2098f35/scratchpad/sam3_test/cluster_points.png"
    )
    print("saved cluster_points.png")
