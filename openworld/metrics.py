"""Open-world evaluation metrics (known / unknown / full-5)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .instances import InstancePrediction
from .label_mapping import ExperimentSplit, normalize_canonical_name


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0.5
    bb = b > 0.5
    if aa.shape != bb.shape:
        import cv2

        bb = cv2.resize(
            bb.astype(np.uint8),
            (aa.shape[1], aa.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def greedy_match(
    preds: List[InstancePrediction],
    gts: List[InstancePrediction],
    iou_thr: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """One-to-one greedy matching by score then IoU."""
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            if p.mask is None or g.mask is None:
                continue
            iou = mask_iou(p.mask, g.mask)
            if iou >= iou_thr:
                pairs.append((pi, gi, iou, p.score))
    pairs.sort(key=lambda x: (x[3], x[2]), reverse=True)
    matched_p, matched_g = set(), set()
    matches = []
    for pi, gi, iou, _ in pairs:
        if pi in matched_p or gi in matched_g:
            continue
        matched_p.add(pi)
        matched_g.add(gi)
        matches.append((pi, gi, iou))
    unmatched_p = [i for i in range(len(preds)) if i not in matched_p]
    unmatched_g = [i for i in range(len(gts)) if i not in matched_g]
    return matches, unmatched_p, unmatched_g


def unknown_recall_and_miou(
    preds: List[InstancePrediction],
    gts: List[InstancePrediction],
    iou_thr: float = 0.5,
    require_prompt_slot_match: bool = True,
) -> Dict[str, float]:
    if require_prompt_slot_match:
        # Match within the same prompt_slot / public class_name.
        tp = 0
        ious = []
        total_gt = len(gts)
        used_pred = set()
        for gi, g in enumerate(gts):
            best = None
            for pi, p in enumerate(preds):
                if pi in used_pred:
                    continue
                if g.class_name and p.class_name and g.class_name != p.class_name:
                    # Also allow prompt_slot equality.
                    if g.prompt_slot is None or p.prompt_slot != g.prompt_slot:
                        continue
                if p.mask is None or g.mask is None:
                    continue
                iou = mask_iou(p.mask, g.mask)
                if iou >= iou_thr and (best is None or iou > best[1]):
                    best = (pi, iou)
            if best is not None:
                used_pred.add(best[0])
                tp += 1
                ious.append(best[1])
        recall = float(tp / total_gt) if total_gt > 0 else float("nan")
        matched_miou = float(np.mean(ious)) if ious else float("nan")
        return {
            "u_recall": recall,
            "u_matched_miou": matched_miou,
            "tp": float(tp),
            "n_gt": float(total_gt),
        }

    matches, _, unmatched_g = greedy_match(preds, gts, iou_thr)
    total_gt = len(gts)
    tp = len(matches)
    recall = float(tp / total_gt) if total_gt > 0 else float("nan")
    matched_miou = (
        float(np.mean([m[2] for m in matches])) if matches else float("nan")
    )
    return {
        "u_recall": recall,
        "u_matched_miou": matched_miou,
        "tp": float(tp),
        "n_gt": float(total_gt),
        "fn": float(len(unmatched_g)),
    }


def map_unknown_preds_to_canonical(
    preds: List[InstancePrediction],
    split: ExperimentSplit,
) -> List[InstancePrediction]:
    """Evaluator-only: unknownj -> Truck/Bus/... for Full-5 AP."""
    out = []
    for p in preds:
        if p.class_name.startswith("unknown"):
            try:
                canon = split.public_to_canonical(p.class_name)
            except KeyError:
                canon = p.class_name
            from .instances import clone_instance

            out.append(clone_instance(p, class_name=canon))
        else:
            out.append(p)
    return out


def average_precision_from_scores(
    scores: List[float],
    tps: List[bool],
    n_gt: int,
) -> float:
    if n_gt <= 0:
        return float("nan")
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tps_arr = np.asarray(tps, dtype=np.float32)[order]
    fps = 1.0 - tps_arr
    tp_cum = np.cumsum(tps_arr)
    fp_cum = np.cumsum(fps)
    recalls = tp_cum / n_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
    # COCO-style 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = precisions[recalls >= t]
        ap += float(p.max()) if p.size else 0.0
    return ap / 101.0


def classwise_mask_ap(
    preds_by_image: Dict[str, List[InstancePrediction]],
    gts_by_image: Dict[str, List[InstancePrediction]],
    class_names: Sequence[str],
    iou_thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    iou_thresholds = list(iou_thresholds or [x / 100 for x in range(50, 100, 5)])
    aps = {}
    class_set = {normalize_canonical_name(c) for c in class_names}
    for cls in class_names:
        ap_at_t = []
        for thr in iou_thresholds:
            scores, tps = [], []
            n_gt = 0
            for img_id, gts in gts_by_image.items():
                gt_cls = [g for g in gts if normalize_canonical_name(g.class_name) == normalize_canonical_name(cls)]
                pred_cls = [
                    p
                    for p in preds_by_image.get(img_id, [])
                    if normalize_canonical_name(p.class_name) == normalize_canonical_name(cls)
                ]
                n_gt += len(gt_cls)
                matches, unmatched_p, _ = greedy_match(pred_cls, gt_cls, thr)
                matched_pred = {m[0] for m in matches}
                for pi, p in enumerate(pred_cls):
                    scores.append(p.score)
                    tps.append(pi in matched_pred)
            ap_at_t.append(average_precision_from_scores(scores, tps, n_gt))
        vals = [a for a in ap_at_t if not np.isnan(a)]
        aps[cls] = float(np.mean(vals)) if vals else float("nan")
    valid = [v for v in aps.values() if not np.isnan(v)]
    aps["mAP50_95"] = float(np.mean(valid)) if valid else float("nan")

    # w_mAP / pooled: all listed-class instances jointly (same spirit as
    # MaskAPCalculator all_mAP@.50-.95 / training w_mAP).
    pooled_at_t = []
    for thr in iou_thresholds:
        scores, tps = [], []
        n_gt = 0
        for img_id, gts in gts_by_image.items():
            gt_sel = [
                g
                for g in gts
                if normalize_canonical_name(g.class_name) in class_set
            ]
            pred_sel = [
                p
                for p in preds_by_image.get(img_id, [])
                if normalize_canonical_name(p.class_name) in class_set
            ]
            n_gt += len(gt_sel)
            matches, _, _ = greedy_match(pred_sel, gt_sel, thr)
            matched_pred = {m[0] for m in matches}
            for pi, p in enumerate(pred_sel):
                scores.append(p.score)
                tps.append(pi in matched_pred)
        pooled_at_t.append(average_precision_from_scores(scores, tps, n_gt))
    pooled_vals = [a for a in pooled_at_t if not np.isnan(a)]
    w_map = float(np.mean(pooled_vals)) if pooled_vals else float("nan")
    aps["w_mAP"] = w_map
    aps["w_mAP50_95"] = w_map
    return aps
