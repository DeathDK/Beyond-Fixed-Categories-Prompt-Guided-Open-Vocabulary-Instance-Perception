"""Known / unknown conflict resolution and public unknownj labeling."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .instances import InstancePrediction, clone_instance
from .label_mapping import ExperimentSplit
from .splits import get_split


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-6))


def _mask_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        import cv2

        b = cv2.resize(
            b.astype(np.uint8),
            (a.shape[1], a.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    aa = a > 0.5
    bb = b > 0.5
    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def nms_instances(
    instances: List[InstancePrediction],
    iou_thr: float = 0.5,
    use_mask: bool = True,
) -> List[InstancePrediction]:
    order = sorted(instances, key=lambda x: x.score, reverse=True)
    keep: List[InstancePrediction] = []
    for cand in order:
        ok = True
        for prev in keep:
            iou = (
                _mask_iou(cand.mask, prev.mask)
                if use_mask and cand.mask is not None and prev.mask is not None
                else _box_iou(cand.bbox_xyxy, prev.bbox_xyxy)
            )
            if iou >= iou_thr:
                ok = False
                break
        if ok:
            keep.append(cand)
    return keep


def assign_public_unknown_labels(
    unknown_instances: List[InstancePrediction],
    split: ExperimentSplit,
) -> List[InstancePrediction]:
    out = []
    for inst in unknown_instances:
        slot = inst.prompt_slot
        if slot is None and inst.prompt_text is not None:
            try:
                public = split.prompt_to_public(inst.prompt_text)
                slot = split.unknown_public_labels.index(public)
            except Exception:
                public = "unknown1"
                slot = 0
        else:
            slot = int(slot or 0)
            slot = max(0, min(slot, len(split.unknown_public_labels) - 1))
            public = split.unknown_public_labels[slot]
        out.append(
            clone_instance(
                inst,
                class_name=public,
                prompt_slot=slot,
                source="unknown",
            )
        )
    return out


def fuse_known_unknown(
    known_instances: List[InstancePrediction],
    unknown_instances: List[InstancePrediction],
    split: ExperimentSplit,
    mask_iou_conflict: float = 0.6,
    known_nms_iou: float = 0.5,
    unknown_nms_iou: float = 0.5,
    unknown_score_boost: float = 0.0,
    skip_conflict: bool = False,
) -> Tuple[List[InstancePrediction], dict]:
    """Fuse branches and emit public labels (Car/... + unknownj).

    Returns fused instances and a small stats dict for Known↔Unknown events.
    If ``skip_conflict`` is True, only per-branch NMS + unknown labeling are
    applied; known and unknown are concatenated with no Known↔Unknown arbitration.
    """
    boost = float(unknown_score_boost)

    stats = {
        "known_to_unknown": 0,
        "unknown_suppressed_by_known": 0,
        "unknown_kept": 0,
        "known_kept": 0,
        "mask_iou_conflict": float(mask_iou_conflict),
        "unknown_score_boost": boost,
        "skip_conflict": bool(skip_conflict),
    }

    fused: List[InstancePrediction] = []

    # No-fusion path: keep G0 unknowns as-is (no cross-branch arbitration, no
    # intra-unknown NMS). That is what eval compares as "pre_fusion".
    if skip_conflict:
        unknown = assign_public_unknown_labels(list(unknown_instances), split)
        for k in known_instances:
            fused.append(clone_instance(k, source="known"))
            stats["known_kept"] += 1
        for u in unknown:
            fused.append(u)
            stats["unknown_kept"] += 1
        for i, inst in enumerate(fused, start=1):
            inst.instance_id = i
        return fused, stats

    known = nms_instances(known_instances, iou_thr=known_nms_iou)
    unknown = nms_instances(unknown_instances, iou_thr=unknown_nms_iou)
    unknown = assign_public_unknown_labels(unknown, split)

    suppressed_known = set()
    suppressed_unknown = set()

    for ui, u in enumerate(unknown):
        conflicted = False
        for ki, k in enumerate(known):
            iou = (
                _mask_iou(u.mask, k.mask)
                if u.mask is not None and k.mask is not None
                else _box_iou(u.bbox_xyxy, k.bbox_xyxy)
            )
            if iou < mask_iou_conflict:
                continue
            conflicted = True
            # Prefer unknowns slightly so leave-out trucks are not eaten by Car.
            u_score = float(u.score) + boost
            if u_score >= k.score:
                suppressed_known.add(ki)
                stats["known_to_unknown"] += 1
            else:
                suppressed_unknown.add(ui)
                stats["unknown_suppressed_by_known"] += 1
        if not conflicted:
            pass

    for ki, k in enumerate(known):
        if ki in suppressed_known:
            continue
        fused.append(clone_instance(k, source="known"))
        stats["known_kept"] += 1

    for ui, u in enumerate(unknown):
        if ui in suppressed_unknown:
            continue
        fused.append(u)
        stats["unknown_kept"] += 1

    # Reassign contiguous instance IDs for publishing.
    for i, inst in enumerate(fused, start=1):
        inst.instance_id = i
    return fused, stats


def fuse_from_split_name(
    known_instances: List[InstancePrediction],
    unknown_instances: List[InstancePrediction],
    split_name: str,
    **kwargs,
):
    return fuse_known_unknown(
        known_instances,
        unknown_instances,
        get_split(split_name),
        **kwargs,
    )
