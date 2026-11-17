"""PG-GUIC geometry scoring and G0/G1 unknown mask refinement."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .instances import InstancePrediction, clone_instance

try:
    from stereo.modeling.common.edge_utils import disparity_to_edge_map
except Exception:  # pragma: no cover
    disparity_to_edge_map = None


def disparity_edge_numpy(
    disparity: np.ndarray,
    stop_grad: bool = True,
) -> np.ndarray:
    """Compute normalized disparity Sobel edge map [H,W] in [0,1]."""
    if disparity.ndim == 3:
        disparity = disparity.squeeze()
    disp = disparity.astype(np.float32)
    if stop_grad:
        disp = disp.copy()
    if disparity_to_edge_map is not None:
        t = torch.from_numpy(disp)[None, None]
        with torch.no_grad():
            edge = disparity_to_edge_map(t, normalize=True)[0, 0].cpu().numpy()
        return edge.astype(np.float32)
    gx = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    lo, hi = np.percentile(mag, [5, 95])
    if hi <= lo + 1e-6:
        return np.zeros_like(mag, dtype=np.float32)
    return np.clip((mag - lo) / (hi - lo), 0, 1).astype(np.float32)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0.5).astype(np.uint8)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
    return (mask_u8 - eroded).astype(np.float32)


def edge_align_score(mask: np.ndarray, edge: np.ndarray, band: int = 3) -> float:
    boundary = mask_boundary(mask)
    if boundary.sum() < 1:
        return 0.0
    edge_band = cv2.dilate(
        (edge > 0.2).astype(np.uint8),
        np.ones((band * 2 + 1, band * 2 + 1), np.uint8),
    )
    overlap = (boundary > 0) & (edge_band > 0)
    return float(overlap.sum() / (boundary.sum() + 1e-6))


def stereo_valid_score(mask: np.ndarray, disparity: np.ndarray, min_disp: float = 1.0) -> float:
    m = mask > 0.5
    if m.sum() < 1:
        return 0.0
    valid = (disparity > min_disp) & np.isfinite(disparity)
    return float((valid & m).sum() / (m.sum() + 1e-6))


def depth_coherence_score(mask: np.ndarray, disparity: np.ndarray) -> float:
    m = mask > 0.5
    if m.sum() < 8:
        return 0.0
    vals = disparity[m]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < 8:
        return 0.0
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-6
    # Higher when depths concentrate; map to (0,1).
    return float(np.clip(1.0 / (1.0 + mad / (abs(med) + 1e-3)), 0, 1))


def geometry_score(
    mask: np.ndarray,
    disparity: np.ndarray,
    edge: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    edge_band: int = 3,
) -> Dict[str, float]:
    ea = edge_align_score(mask, edge, band=edge_band)
    sv = stereo_valid_score(mask, disparity)
    dc = depth_coherence_score(mask, disparity)
    # Far-range: weaken edge term when stereo validity is low.
    edge_w = alpha * (0.3 + 0.7 * sv)
    g = edge_w * ea + beta * sv + gamma * dc
    return {
        "edge_align": ea,
        "stereo_valid": sv,
        "depth_coherence": dc,
        "geometry": float(np.clip(g, 0, 1)),
    }


def refine_mask_g0(
    mask: np.ndarray,
    edge: np.ndarray,
    band_pixels: int = 3,
) -> np.ndarray:
    """Keep interior; only adjust a narrow boundary band toward disparity edges."""
    mask_f = (mask > 0.5).astype(np.float32)
    boundary = mask_boundary(mask_f)
    if boundary.sum() < 1:
        return mask_f
    band = cv2.dilate(
        boundary.astype(np.uint8),
        np.ones((band_pixels * 2 + 1, band_pixels * 2 + 1), np.uint8),
    ).astype(np.float32)
    edge_bin = (edge > 0.25).astype(np.float32)
    refined = mask_f.copy()
    # Inside band: prefer edge-consistent foreground.
    want_fg = (band > 0) & (edge_bin > 0)
    want_bg = (band > 0) & (edge_bin == 0) & (boundary > 0)
    refined[want_fg] = 1.0
    refined[want_bg] = 0.0
    # Fill small holes / remove speckles.
    refined_u8 = (refined > 0.5).astype(np.uint8)
    refined_u8 = cv2.morphologyEx(refined_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return refined_u8.astype(np.float32)


def fuse_text_geometry_score(
    text_score: float,
    geom: float,
    mask_score: float = 1.0,
    a: float = 1.0,
    b: float = 0.8,
    c: float = 0.2,
    d: float = 0.0,
) -> float:
    def _logit(p: float) -> float:
        p = float(np.clip(p, 1e-4, 1 - 1e-4))
        return float(np.log(p / (1 - p)))

    z = a * _logit(text_score) + b * geom + c * mask_score + d
    return float(1.0 / (1.0 + np.exp(-z)))


def refine_unknown_instances_g0(
    instances: List[InstancePrediction],
    disparity: np.ndarray,
    edge_band_pixels: int = 3,
    stop_grad_disparity: bool = True,
) -> List[InstancePrediction]:
    if disparity.ndim == 3:
        disparity = disparity.squeeze()
    edge = disparity_edge_numpy(disparity, stop_grad=stop_grad_disparity)
    refined = []
    for inst in instances:
        if inst.mask is None:
            refined.append(inst)
            continue
        mask = inst.mask
        if mask.shape != disparity.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (disparity.shape[1], disparity.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        new_mask = refine_mask_g0(mask, edge, band_pixels=edge_band_pixels)
        scores = geometry_score(new_mask, disparity, edge, edge_band=edge_band_pixels)
        new_score = fuse_text_geometry_score(inst.score, scores["geometry"])
        refined.append(
            clone_instance(
                inst,
                mask=new_mask,
                score=new_score,
                geometry_score=scores["geometry"],
                extras={**inst.extras, **scores},
            )
        )
    return refined


@torch.no_grad()
def refine_unknown_instances_g1(
    instances: List[InstancePrediction],
    disparity: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    refiner: Optional[G1BoundaryRefiner] = None,
    device: str = "cpu",
    edge_band_pixels: int = 3,
) -> List[InstancePrediction]:
    """Apply trained G1 residual refiner; fall back to G0 if refiner is None."""
    if refiner is None:
        return refine_unknown_instances_g0(
            instances, disparity, edge_band_pixels=edge_band_pixels
        )
    if disparity.ndim == 3:
        disparity = disparity.squeeze()
    edge = disparity_edge_numpy(disparity, stop_grad=True)
    dmax = float(np.nanmax(disparity)) if np.isfinite(disparity).any() else 1.0
    dmax = max(dmax, 1.0)
    disp_norm = np.clip(disparity / dmax, 0, 1).astype(np.float32)
    if rgb is None:
        rgb_e = edge.copy()
    else:
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = rgb.astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        lo, hi = np.percentile(mag, [5, 95])
        rgb_e = (
            np.zeros_like(mag, dtype=np.float32)
            if hi <= lo + 1e-6
            else np.clip((mag - lo) / (hi - lo), 0, 1).astype(np.float32)
        )

    refiner = refiner.to(device).eval()
    refined = []
    for inst in instances:
        if inst.mask is None:
            refined.append(inst)
            continue
        mask = inst.mask.astype(np.float32)
        if mask.shape != disparity.shape:
            mask = cv2.resize(
                mask,
                (disparity.shape[1], disparity.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        t_mask = torch.from_numpy(mask[None, None]).float().to(device)
        t_edge = torch.from_numpy(edge[None, None]).float().to(device)
        t_disp = torch.from_numpy(disp_norm[None, None]).float().to(device)
        t_rgb = torch.from_numpy(rgb_e[None, None]).float().to(device)
        new_mask = refiner(t_mask, t_edge, t_disp, t_rgb)[0, 0].cpu().numpy()
        new_mask = (new_mask > 0.5).astype(np.float32)
        scores = geometry_score(new_mask, disparity, edge, edge_band=edge_band_pixels)
        new_score = fuse_text_geometry_score(inst.score, scores["geometry"])
        refined.append(
            clone_instance(
                inst,
                mask=new_mask,
                score=new_score,
                geometry_score=scores["geometry"],
                extras={**inst.extras, **scores, "geometry_mode": "g1"},
            )
        )
    return refined


class G1BoundaryRefiner(nn.Module):
    """Lightweight depthwise-separable residual refiner for unknown masks."""

    def __init__(self, in_ch: int = 4, mid: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        mask_prob: torch.Tensor,
        edge: torch.Tensor,
        disp_norm: torch.Tensor,
        rgb_edge: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([mask_prob, edge, disp_norm, rgb_edge], dim=1)
        delta = self.net(x)
        logit = torch.logit(mask_prob.clamp(1e-4, 1 - 1e-4))
        return torch.sigmoid(logit + delta)
