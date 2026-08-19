"""
Block 2: SAM3 inference wrapper.

Text prompt -> per-object instance masks/bboxes/scores, frozen & zero-shot.
Tries each candidate phrase from np_extractor's fallback list in order and
stops at the first one that yields at least one instance above threshold
(see cog-xvla/data QA notes on why the fallback exists: LIBERO catalog
names like "black bowl" sometimes don't match the rendered appearance).

Must run in an env with the official facebookresearch/sam3 package
installed (this repo's `sam3` conda env) and torch >= 2.13 (older cu128
builds hit a cuBLAS bf16 GEMM bug on Blackwell GPUs -- see project notes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


@dataclass
class Detection:
    query: str            # the noun phrase that was asked for, e.g. "black bowl"
    matched_prompt: str    # the candidate that actually produced a hit, e.g. "bowl"
    masks: np.ndarray      # [N, H, W] bool
    boxes: np.ndarray      # [N, 4] xyxy in pixel coords
    scores: np.ndarray     # [N]


class Sam3Wrapper:
    def __init__(self, score_threshold: float = 0.5):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        self.score_threshold = score_threshold

    def set_image(self, image: Image.Image | np.ndarray):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        self._state = self.processor.set_image(image)
        return self._state

    def _query_once(self, prompt: str):
        self.processor.reset_all_prompts(self._state)
        state = self.processor.set_text_prompt(state=self._state, prompt=prompt)
        scores = state["scores"]
        scores_np = scores.detach().float().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
        keep = scores_np >= self.score_threshold
        if keep.sum() == 0:
            return None
        masks = state["masks"]
        masks_np = masks.detach().float().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
        boxes = state["boxes"]
        boxes_np = boxes.detach().float().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)

        masks_np = masks_np[keep]
        if masks_np.ndim == 4:  # [N, 1, H, W] -> [N, H, W]
            masks_np = masks_np[:, 0]
        return Detection(
            query=prompt,
            matched_prompt=prompt,
            masks=(masks_np > 0.5),
            boxes=boxes_np[keep],
            scores=scores_np[keep],
        )

    def segment_candidates(self, candidates: Sequence[str]) -> Detection | None:
        """Try each candidate phrase in order, return the first successful hit."""
        for prompt in candidates:
            det = self._query_once(prompt)
            if det is not None:
                det.query = candidates[0]
                return det
        return None

    def segment_all(self, image, candidates_per_object: Sequence[Sequence[str]]) -> List[Detection | None]:
        """
        Args:
            image: RGB image (PIL or HxWx3 uint8 array)
            candidates_per_object: output of np_extractor.extract_object_phrases(),
                e.g. [["black bowl", "bowl"], ["plate"], ["ramekin"]]

        Returns:
            One Detection (or None if nothing matched) per object mention, in
            the same order as candidates_per_object.
        """
        self.set_image(image)
        return [self.segment_candidates(cands) for cands in candidates_per_object]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/gibeom_pilab/cog-xvla")
    from perception.np_extractor import extract_object_phrases

    instruction = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    image = Image.open(
        "/tmp/claude-1001/-home-gibeom-pilab-X-VLA/2da855b6-dd5f-4276-9d8a-cfcde2098f35/scratchpad/sam3_test/frame.png"
    ).convert("RGB")

    candidates = extract_object_phrases(instruction)
    print("candidates:", candidates)

    wrapper = Sam3Wrapper()
    detections = wrapper.segment_all(image, candidates)
    for cands, det in zip(candidates, detections):
        if det is None:
            print(f"{cands[0]!r}: NO MATCH (tried {cands})")
        else:
            print(f"{cands[0]!r}: matched via {det.matched_prompt!r}, "
                  f"n={len(det.scores)}, scores={np.round(det.scores, 3)}")
