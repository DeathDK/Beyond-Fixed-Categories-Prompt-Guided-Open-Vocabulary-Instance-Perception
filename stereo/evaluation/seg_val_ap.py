"""Helpers to compute known-class mask mAP / w-mAP during training validation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from openworld.known_decode import decode_known_instances
from stereo.evaluation.mask_ap import MaskAPCalculator


def _as_list(obj: Any) -> List[Any]:
    if obj is None:
        return []
    if isinstance(obj, dict):
        return [obj]
    return list(obj)


def polygon_to_mask(polygon: Sequence[float], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygon is None or len(polygon) < 6:
        return mask.astype(bool)
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool)


def unpad_polygon_right_top(
    polygon: Sequence[float],
    orig_hw: Tuple[int, int],
    pad_hw: Tuple[int, int],
) -> np.ndarray:
    """Map a RightTopPad polygon back to the original image canvas."""
    H0, W0 = orig_hw
    th, tw = pad_hw
    pad_top = max(0, th - H0)
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    pts[:, 1] -= pad_top
    return pts.reshape(-1)


def build_gt_entry_from_seg_labels(
    seg_labels: Any,
    image_id: str,
    orig_hw: Tuple[int, int],
    pad_hw: Tuple[int, int] = (384, 1248),
) -> Dict[str, Any]:
    H0, W0 = orig_hw
    classes: List[int] = []
    masks: List[np.ndarray] = []
    for lab in _as_list(seg_labels):
        if not isinstance(lab, dict):
            continue
        poly = lab.get("polygon")
        if poly is None:
            continue
        poly_np = unpad_polygon_right_top(poly, orig_hw, pad_hw)
        mask = polygon_to_mask(poly_np, H0, W0)
        if not mask.any():
            continue
        classes.append(int(lab["class_id"]))
        masks.append(mask)
    if masks:
        masks_arr = np.stack(masks, axis=0)
        classes_arr = np.asarray(classes, dtype=np.int64)
    else:
        masks_arr = np.zeros((0, H0, W0), dtype=bool)
        classes_arr = np.zeros((0,), dtype=np.int64)
    return {
        "image_id": image_id,
        "classes": classes_arr,
        "masks": masks_arr,
    }


def build_pred_entry_from_model(
    model_pred: dict,
    class_names: Sequence[str],
    image_id: str,
    orig_hw: Tuple[int, int],
    pad_hw: Tuple[int, int] = (384, 1248),
    conf_threshold: float = 0.001,
    batch_index: int = 0,
) -> Dict[str, Any]:
    """Decode one batch item into MaskAPCalculator prediction format."""
    # Slice batch dim for decode helpers that expect single-image tensors.
    sliced = {}
    for key, value in model_pred.items():
        if key == "seg_pred":
            if isinstance(value, (tuple, list)):
                sliced[key] = value  # shared across batch in current head design
            else:
                sliced[key] = value
        elif torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] > batch_index:
            sliced[key] = value[batch_index : batch_index + 1]
        else:
            sliced[key] = value

    # tensor2SEG expects full-batch seg tensors; prefer full model_pred seg + disp slice.
    decode_input = dict(model_pred)
    if torch.is_tensor(model_pred.get("disp_pred")):
        decode_input["disp_pred"] = model_pred["disp_pred"][batch_index : batch_index + 1]

    _, instances = decode_known_instances(
        decode_input,
        class_names=class_names,
        orig_hw=orig_hw,
        pad_hw=pad_hw,
        conf_threshold=conf_threshold,
    )
    if not instances:
        H0, W0 = orig_hw
        return {
            "image_id": image_id,
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "classes": np.zeros((0,), dtype=np.int64),
            "masks": np.zeros((0, H0, W0), dtype=bool),
        }

    name_to_id = {str(n).lower(): i for i, n in enumerate(class_names)}
    boxes, scores, classes, masks = [], [], [], []
    for inst in instances:
        cls_id = name_to_id.get(str(inst.class_name).lower())
        if cls_id is None:
            # Fallback: some decoders may already emit train ids via name==str(id)
            continue
        boxes.append(inst.bbox_xyxy)
        scores.append(float(inst.score))
        classes.append(int(cls_id))
        masks.append(np.asarray(inst.mask, dtype=bool))
    if masks:
        return {
            "image_id": image_id,
            "boxes": np.asarray(boxes, dtype=np.float32),
            "scores": np.asarray(scores, dtype=np.float32),
            "classes": np.asarray(classes, dtype=np.int64),
            "masks": np.stack(masks, axis=0),
        }
    H0, W0 = orig_hw
    return {
        "image_id": image_id,
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int64),
        "masks": np.zeros((0, H0, W0), dtype=bool),
    }


def compute_known_mask_ap(
    predictions: List[Dict[str, Any]],
    ground_truths: Dict[str, Dict[str, Any]],
    class_names: Sequence[str],
) -> Dict[str, float]:
    calc = MaskAPCalculator(class_names=class_names)
    return calc.evaluate(predictions, ground_truths, verbose=False)
