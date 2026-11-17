"""Instance-level point cloud helpers for known + unknown masks."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .instances import InstancePrediction


def disparity_to_points(
    disparity: np.ndarray,
    rgb: np.ndarray,
    mask: Optional[np.ndarray],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline: float,
    min_disp: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project masked pixels to 3D. Returns (Nx3 xyz, Nx3 rgb)."""
    if disparity.ndim == 3:
        disparity = disparity.squeeze()
    h, w = disparity.shape
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    else:
        if mask.shape != disparity.shape:
            import cv2

            mask = cv2.resize(
                mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = mask > 0.5
    valid = mask & np.isfinite(disparity) & (disparity > min_disp)
    ys, xs = np.where(valid)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    d = disparity[ys, xs]
    z = fx * baseline / d
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    if rgb.shape[0] == 3:  # CHW
        colors = rgb[:, ys, xs].T
    else:
        colors = rgb[ys, xs]
    if colors.max() > 1.5:
        colors = colors.astype(np.float32) / 255.0
    return xyz, colors.astype(np.float32)


def reconstruct_instances(
    disparity: np.ndarray,
    rgb: np.ndarray,
    instances: List[InstancePrediction],
    calib: Dict[str, float],
    min_disp: float = 1.0,
) -> Dict[int, Dict]:
    clouds = {}
    for inst in instances:
        xyz, colors = disparity_to_points(
            disparity,
            rgb,
            inst.mask,
            fx=float(calib.get("fx", 721.5377)),
            fy=float(calib.get("fy", 721.5377)),
            cx=float(calib.get("cx", 609.5593)),
            cy=float(calib.get("cy", 172.854)),
            baseline=float(calib.get("baseline", 0.54)),
            min_disp=min_disp,
        )
        clouds[inst.instance_id] = {
            "class_name": inst.class_name,
            "xyz": xyz,
            "rgb": colors,
            "num_points": int(xyz.shape[0]),
        }
    return clouds


def union_instance_point_count(
    disparity: np.ndarray,
    instances: List[InstancePrediction],
    min_disp: float = 1.0,
) -> int:
    if disparity.ndim == 3:
        disparity = disparity.squeeze()
    h, w = disparity.shape
    union = np.zeros((h, w), dtype=bool)
    for inst in instances:
        if inst.mask is None:
            continue
        m = inst.mask
        if m.shape != (h, w):
            import cv2

            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        union |= m > 0.5
    valid = union & np.isfinite(disparity) & (disparity > min_disp)
    return int(valid.sum())
