"""Frozen YOLOE-26x-seg text-prompt unknown instance expert."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .instances import InstancePrediction


class YOLOEPromptExpert:
    """Wrap Ultralytics YOLOE text-prompt segmentation.

    Uses ``yoloe-26x-seg.pt`` (text prompt). Never load ``*-seg-pf.pt``.
    """

    def __init__(
        self,
        checkpoint: str = "yoloe-26x-seg.pt",
        conf_threshold: float = 0.05,
        iou_threshold: float = 0.5,
        device: Optional[str] = None,
        auto_download: bool = True,
    ):
        if checkpoint.endswith("-pf.pt"):
            raise ValueError(
                "Prompt-free YOLOE checkpoints are not supported for open-world "
                "text prompts. Use yoloe-26x-seg.pt"
            )
        self.checkpoint = checkpoint
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.device = device
        self._model = None
        self._current_prompts: Tuple[str, ...] = ()
        self._auto_download = auto_download
        # Optional runtime knobs (set by eval/diag).
        self.imgsz: Optional[int] = None
        self.multipass: bool = False

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise ImportError(
                "ultralytics>=8.4.0 is required for YOLOE-26. "
                "Install with: pip install 'ultralytics>=8.4.0'"
            ) from exc
        if self._auto_download or os.path.isfile(self.checkpoint):
            self._model = YOLOE(self.checkpoint)
        else:
            raise FileNotFoundError(f"YOLOE checkpoint not found: {self.checkpoint}")
        if self.device:
            self._model.to(self.device)

    def set_prompts(self, prompts: Sequence[str]) -> None:
        prompts = tuple(str(p).strip().lower() for p in prompts)
        self._ensure_model()
        if prompts == self._current_prompts:
            return
        if not prompts:
            self._current_prompts = ()
            return
        # First set_classes may download the text encoder.
        self._model.set_classes(list(prompts))
        self._current_prompts = prompts

    def predict_instances(
        self,
        image_bgr_or_path,
        prompts: Sequence[str],
        image_size_hw: Optional[Tuple[int, int]] = None,
        prompt_slots: Optional[Sequence[int]] = None,
        conf_threshold: Optional[float] = None,
    ) -> List[InstancePrediction]:
        """Run text-prompted instance segmentation on a left image.

        Args:
            image_bgr_or_path: filesystem path or HxWx3 uint8 BGR/RGB numpy array.
            prompts: text prompts for YOLOE set_classes.
            image_size_hw: optional (H, W) used when input is a path (for sanity).
            prompt_slots: optional per-prompt slot into unknownj (for synonyms).
            conf_threshold: override instance conf for this call only.
        """
        prompts = tuple(str(p).strip().lower() for p in prompts)
        if not prompts:
            return []
        if prompt_slots is not None and len(prompt_slots) != len(prompts):
            raise ValueError("prompt_slots length must match prompts")
        slots = (
            tuple(int(s) for s in prompt_slots)
            if prompt_slots is not None
            else tuple(range(len(prompts)))
        )
        conf = float(self.conf_threshold if conf_threshold is None else conf_threshold)
        predict_kw = {
            "conf": conf,
            "iou": self.iou_threshold,
            "verbose": False,
        }
        if self.imgsz is not None and int(self.imgsz) > 0:
            predict_kw["imgsz"] = int(self.imgsz)

        # Multipass: one set_classes+predict per prompt, then merge.
        # Avoids synonym competition inside a single YOLOE class set.
        if self.multipass and len(prompts) > 1:
            instances: List[InstancePrediction] = []
            for local_i, (prompt, slot) in enumerate(zip(prompts, slots)):
                self.set_prompts((prompt,))
                results = self._model.predict(source=image_bgr_or_path, **predict_kw)
                if not results:
                    continue
                result = results[0]
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                masks = None
                if result.masks is not None and result.masks.data is not None:
                    masks = result.masks.data.cpu().numpy()
                for i in range(len(boxes)):
                    mask = None
                    if masks is not None and i < len(masks):
                        mask = masks[i].astype(np.uint8)
                        if image_size_hw is not None and mask.shape != image_size_hw:
                            import cv2

                            mask = cv2.resize(
                                mask,
                                (image_size_hw[1], image_size_hw[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                    instances.append(
                        InstancePrediction(
                            instance_id=len(instances) + 1,
                            class_name=prompt,
                            score=float(scores[i]),
                            bbox_xyxy=[float(x) for x in boxes[i].tolist()],
                            mask=mask,
                            prompt_slot=int(slot),
                            prompt_text=prompt,
                            source="unknown",
                        )
                    )
            return instances

        self.set_prompts(prompts)
        results = self._model.predict(source=image_bgr_or_path, **predict_kw)
        if not results:
            return []
        result = results[0]
        instances = []
        if result.boxes is None or len(result.boxes) == 0:
            return instances

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(np.int64)
        masks = None
        if result.masks is not None and result.masks.data is not None:
            masks = result.masks.data.cpu().numpy()

        for i in range(len(boxes)):
            cls_idx = int(classes[i])
            if not 0 <= cls_idx < len(prompts):
                continue
            mask = None
            if masks is not None and i < len(masks):
                mask = masks[i].astype(np.uint8)
                if image_size_hw is not None and mask.shape != image_size_hw:
                    import cv2

                    mask = cv2.resize(
                        mask,
                        (image_size_hw[1], image_size_hw[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
            slot = slots[cls_idx]
            instances.append(
                InstancePrediction(
                    instance_id=i + 1,
                    class_name=prompts[cls_idx],  # temporary; fusion maps to unknownj
                    score=float(scores[i]),
                    bbox_xyxy=[float(x) for x in boxes[i].tolist()],
                    mask=mask,
                    prompt_slot=slot,
                    prompt_text=prompts[cls_idx],
                    source="unknown",
                )
            )
        return instances


def restore_right_top_pad_box(
    boxes_xyxy: np.ndarray,
    orig_hw: Tuple[int, int],
    pad_hw: Tuple[int, int] = (384, 1248),
) -> np.ndarray:
    """Map boxes from RightTopPad canvas back to original image coords."""
    H0, W0 = orig_hw
    th, tw = pad_hw
    pad_top = max(0, th - H0)
    out = np.asarray(boxes_xyxy, dtype=np.float32).copy()
    if out.size == 0:
        return out
    out[:, [1, 3]] -= pad_top
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, W0 - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, H0 - 1)
    return out


def restore_right_top_pad_mask(
    mask: np.ndarray,
    orig_hw: Tuple[int, int],
    pad_hw: Tuple[int, int] = (384, 1248),
) -> np.ndarray:
    H0, W0 = orig_hw
    th, tw = pad_hw
    pad_top = max(0, th - H0)
    if mask.shape == (H0, W0):
        return mask
    if mask.shape[0] >= pad_top + H0 and mask.shape[1] >= W0:
        return mask[pad_top : pad_top + H0, :W0]
    import cv2

    return cv2.resize(mask.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST)
