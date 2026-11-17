"""Image-local instance-mask average precision evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Return intersection-over-union for two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union else 0.0


class MaskAPCalculator:
    """COCO-style 101-point mask AP with per-image one-to-one matching."""

    def __init__(
        self,
        class_names: Sequence[str],
        iou_thresholds: Optional[Iterable[float]] = None,
        ignored_class_names: Sequence[str] = ("none",),
    ) -> None:
        self.class_names = list(class_names)
        self.iou_thresholds = list(iou_thresholds or np.linspace(0.5, 0.95, 10))
        self.ignored_class_names = set(ignored_class_names)

    @staticmethod
    def _group_predictions(all_predictions: List[Dict]) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for pred_group in all_predictions:
            image_id = pred_group["image_id"]
            for index, score in enumerate(pred_group["scores"]):
                grouped[image_id].append({
                    "score": float(score),
                    "class_id": int(pred_group["classes"][index]),
                    "mask": pred_group["masks"][index],
                })
        return grouped

    @staticmethod
    def _group_ground_truths(all_gts: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for image_id, gt_group in all_gts.items():
            for index, class_id in enumerate(gt_group["classes"]):
                grouped[image_id].append({
                    "class_id": int(class_id),
                    "mask": gt_group["masks"][index],
                })
        return grouped

    @staticmethod
    def _interpolated_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
        values = []
        for recall_level in np.linspace(0.0, 1.0, 101):
            valid = precisions[recalls >= recall_level]
            values.append(float(valid.max()) if valid.size else 0.0)
        return float(np.mean(values))

    def calculate_ap_per_class(
        self,
        preds_by_image: Dict[str, List[Dict]],
        gts_by_image: Dict[str, List[Dict]],
        iou_thresh: float,
        class_id: Optional[int],
    ) -> Tuple[Optional[float], int, int]:
        """Calculate one class, or all classes jointly when class_id is None."""
        ground_truths: Dict[str, List[Dict]] = {}
        num_gt = 0
        for image_id, items in gts_by_image.items():
            selected = [g for g in items if class_id is None or g["class_id"] == class_id]
            ground_truths[image_id] = selected
            num_gt += len(selected)
        if num_gt == 0:
            return None, 0, 0

        predictions = []
        for image_id, items in preds_by_image.items():
            predictions.extend(
                (image_id, pred)
                for pred in items
                if class_id is None or pred["class_id"] == class_id
            )
        predictions.sort(key=lambda item: item[1]["score"], reverse=True)
        if not predictions:
            return 0.0, 0, num_gt

        matched = {
            image_id: np.zeros(len(items), dtype=bool)
            for image_id, items in ground_truths.items()
        }
        true_positives = np.zeros(len(predictions), dtype=np.float64)
        false_positives = np.zeros(len(predictions), dtype=np.float64)

        for pred_index, (image_id, pred) in enumerate(predictions):
            image_gts = ground_truths.get(image_id, [])
            image_matches = matched.get(image_id, np.zeros(0, dtype=bool))
            best_iou = 0.0
            best_gt_index = -1
            for gt_index, gt in enumerate(image_gts):
                if image_matches[gt_index]:
                    continue
                iou = calculate_mask_iou(pred["mask"], gt["mask"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index
            if best_gt_index >= 0 and best_iou >= iou_thresh:
                true_positives[pred_index] = 1.0
                image_matches[best_gt_index] = True
            else:
                false_positives[pred_index] = 1.0

        tp_cumsum = np.cumsum(true_positives)
        fp_cumsum = np.cumsum(false_positives)
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(float).eps)
        return self._interpolated_ap(recalls, precisions), int(tp_cumsum[-1]), num_gt

    def evaluate(
        self,
        all_predictions: List[Dict],
        all_gts: Dict[str, Dict],
        verbose: bool = True,
    ) -> Dict[str, float]:
        preds_by_image = self._group_predictions(all_predictions)
        gts_by_image = self._group_ground_truths(all_gts)
        class_ids = [
            index for index, name in enumerate(self.class_names)
            if name not in self.ignored_class_names
        ]
        aps_by_iou = {threshold: {} for threshold in self.iou_thresholds}
        all_class_aps = {}

        for threshold in self.iou_thresholds:
            if verbose:
                print(f"\n--- Calculating AP for IoU threshold: {threshold:.2f} ---")
            for class_id in class_ids:
                ap, total_tp, total_gt = self.calculate_ap_per_class(
                    preds_by_image, gts_by_image, threshold, class_id
                )
                aps_by_iou[threshold][class_id] = ap
                if ap is not None and verbose:
                    print(
                        f"AP for class '{self.class_names[class_id]}': {ap:.4f} "
                        f"(TPs: {total_tp}, GTs: {total_gt})"
                    )
            all_ap, total_tp, total_gt = self.calculate_ap_per_class(
                preds_by_image, gts_by_image, threshold, class_id=None
            )
            all_class_aps[threshold] = float(all_ap or 0.0)
            if verbose and abs(threshold - 0.50) < 1e-6:
                print(
                    f"AP for 'all' @0.50: {all_class_aps[threshold]:.4f} "
                    f"(TPs: {total_tp}, GTs: {total_gt})"
                )

        mean_aps = {}
        for threshold, class_aps in aps_by_iou.items():
            valid = [ap for ap in class_aps.values() if ap is not None]
            mean_aps[threshold] = float(np.mean(valid)) if valid else 0.0
        return {
            "mAP@.50": mean_aps.get(0.50, 0.0),
            "mAP@.75": mean_aps.get(0.75, 0.0),
            "mAP@.50-.95": float(np.mean(list(mean_aps.values()))) if mean_aps else 0.0,
            "all_mAP@.50": all_class_aps.get(0.50, 0.0),
            "all_mAP@.75": all_class_aps.get(0.75, 0.0),
            "all_mAP@.50-.95": float(np.mean(list(all_class_aps.values()))) if all_class_aps else 0.0,
            # Aliases matching the thesis / English paper naming.
            "mAP": float(np.mean(list(mean_aps.values()))) if mean_aps else 0.0,
            "w_mAP": float(np.mean(list(all_class_aps.values()))) if all_class_aps else 0.0,
        }
