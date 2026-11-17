"""Frozen unknown expert: Grounding DINO (text->box) + SAM2 (box->mask).

Replaces YOLOE as the default open-vocabulary unknown instance proposer.
Weights stay frozen; no fine-tuning on leave-out KITTI classes.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .instances import InstancePrediction


def _to_rgb_uint8(image_bgr_or_rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_bgr_or_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _match_prompt_slot(
    label: str,
    prompts: Sequence[str],
    slots: Sequence[int],
) -> Optional[int]:
    """Map a Grounding-DINO text label back to a prompt slot."""
    lab = str(label).strip().lower()
    for i, p in enumerate(prompts):
        p_norm = str(p).strip().lower()
        if lab == p_norm or lab in p_norm or p_norm in lab:
            return int(slots[i])
    # Fallback: first slot when only one unknown class.
    if len(set(int(s) for s in slots)) == 1:
        return int(slots[0])
    return None


class GroundingSAM2Expert:
    """Text-prompted instance segmentation via Grounding DINO + SAM2 (HF)."""

    def __init__(
        self,
        gdino_model: str = "IDEA-Research/grounding-dino-base",
        sam2_model: str = "facebook/sam2.1-hiera-large",
        box_threshold: float = 0.10,
        text_threshold: float = 0.10,
        device: Optional[str] = None,
    ) -> None:
        self.gdino_model_id = gdino_model
        self.sam2_model_id = sam2_model
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.device = device or "cpu"
        self._gdino = None
        self._gdino_processor = None
        self._sam2 = None
        self._sam2_processor = None

    @property
    def conf_threshold(self) -> float:
        """Alias for box_threshold (compat with YOLOE expert call sites)."""
        return self.box_threshold

    @conf_threshold.setter
    def conf_threshold(self, value: float) -> None:
        self.box_threshold = float(value)

    def _ensure_models(self) -> None:
        if self._gdino is not None and self._sam2 is not None:
            return
        try:
            import torch
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
                Sam2Model,
                Sam2Processor,
            )
        except ImportError as exc:
            raise ImportError(
                "Grounding DINO + SAM2 requires transformers. "
                "Install with: pip install 'transformers>=4.45' accelerate"
            ) from exc

        self._torch = torch
        self._gdino_processor = AutoProcessor.from_pretrained(self.gdino_model_id)
        self._gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.gdino_model_id
        ).to(self.device)
        self._gdino.eval()

        self._sam2_processor = Sam2Processor.from_pretrained(self.sam2_model_id)
        self._sam2 = Sam2Model.from_pretrained(self.sam2_model_id).to(self.device)
        self._sam2.eval()

    def predict_instances(
        self,
        image_bgr_or_path,
        prompts: Sequence[str],
        image_size_hw: Optional[Tuple[int, int]] = None,
        prompt_slots: Optional[Sequence[int]] = None,
        conf_threshold: Optional[float] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[InstancePrediction]:
        """Run Grounding DINO + SAM2 and return InstancePrediction list.

        Args mirror YOLOEPromptExpert so call sites can swap experts.
        ``conf_threshold`` (if set) lowers/overrides box_threshold for recall sweeps.
        """
        prompts = tuple(str(p).strip().lower() for p in prompts if str(p).strip())
        if not prompts:
            return []
        slots = (
            tuple(int(s) for s in prompt_slots)
            if prompt_slots is not None
            else tuple(range(len(prompts)))
        )
        if len(slots) != len(prompts):
            raise ValueError("prompt_slots length must match prompts")

        self._ensure_models()
        from PIL import Image

        if isinstance(image_bgr_or_path, (str, bytes)):
            image = Image.open(image_bgr_or_path).convert("RGB")
            rgb = np.array(image)
        else:
            # Callers often pass BGR (OpenCV); convert to RGB for PIL/HF.
            arr = _to_rgb_uint8(image_bgr_or_path)
            # Heuristic: infer_openworld converts RGB->BGR before expert; treat as BGR.
            rgb = arr[:, :, ::-1].copy()
            image = Image.fromarray(rgb)

        if image_size_hw is not None and (rgb.shape[0], rgb.shape[1]) != image_size_hw:
            import cv2

            rgb = cv2.resize(
                rgb,
                (image_size_hw[1], image_size_hw[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            image = Image.fromarray(rgb)

        box_thr = float(
            conf_threshold
            if conf_threshold is not None
            else (box_threshold if box_threshold is not None else self.box_threshold)
        )
        text_thr = float(
            text_threshold if text_threshold is not None else self.text_threshold
        )

        # Grounding DINO: lowercase prompts joined with ". "
        text = ". ".join(prompts) + "."
        inputs = self._gdino_processor(
            images=image, text=text, return_tensors="pt"
        ).to(self.device)
        with self._torch.no_grad():
            outputs = self._gdino(**inputs)

        target_sizes = [(image.height, image.width)]
        post_kwargs = dict(
            outputs=outputs,
            input_ids=inputs.input_ids,
            target_sizes=target_sizes,
        )
        # transformers API: older box_threshold / newer threshold
        try:
            results = self._gdino_processor.post_process_grounded_object_detection(
                threshold=box_thr,
                text_threshold=text_thr,
                **post_kwargs,
            )
        except TypeError:
            results = self._gdino_processor.post_process_grounded_object_detection(
                box_threshold=box_thr,
                text_threshold=text_thr,
                **post_kwargs,
            )
        if not results:
            return []
        result = results[0]
        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("text_labels")
        if labels is None:
            labels = result.get("labels") or []
        if boxes is None or len(boxes) == 0:
            return []

        boxes_np = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
        scores_np = (
            scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
        )
        label_list = [str(x) for x in labels]

        # SAM2: input_boxes shape is [batch][num_boxes][4] in pixel xyxy.
        box_list = [[float(x) for x in box] for box in boxes_np]
        sam_inputs = self._sam2_processor(
            images=image,
            input_boxes=[box_list],
            return_tensors="pt",
        ).to(self.device)
        with self._torch.no_grad():
            sam_out = self._sam2(**sam_inputs, multimask_output=True)

        low_res = getattr(sam_out, "pred_masks", None)
        if low_res is None:
            return []
        # HF API: post_process_masks(masks, original_sizes) only — do NOT pass
        # reshaped sizes as a 3rd positional (that becomes mask_threshold).
        processed = self._sam2_processor.post_process_masks(
            low_res.cpu(),
            sam_inputs["original_sizes"],
        )
        # processed[0]: (num_boxes, num_masks, H, W)
        masks_t = processed[0]
        if masks_t.ndim == 4:
            if masks_t.shape[1] > 1 and getattr(sam_out, "iou_scores", None) is not None:
                iou = sam_out.iou_scores[0].detach().cpu()  # (N, M)
                best = iou.argmax(dim=-1)
                masks_sel = [masks_t[i, int(best[i])] for i in range(masks_t.shape[0])]
                masks_np = self._torch.stack(masks_sel, dim=0).detach().cpu().numpy()
            else:
                masks_np = masks_t[:, 0].detach().cpu().numpy()
        elif masks_t.ndim == 3:
            masks_np = masks_t.detach().cpu().numpy()
        else:
            return []

        instances: List[InstancePrediction] = []
        h, w = rgb.shape[:2]
        for i in range(len(box_list)):
            label = label_list[i] if i < len(label_list) else prompts[0]
            slot = _match_prompt_slot(label, prompts, slots)
            if slot is None:
                continue
            mask = masks_np[i]
            if mask.shape != (h, w):
                import cv2

                mask = cv2.resize(
                    mask.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask_u8 = (mask > 0.5).astype(np.uint8)
            if mask_u8.sum() < 8:
                continue
            # Prefer the original prompt text for this slot when possible.
            prompt_text = prompts[0]
            for p, s in zip(prompts, slots):
                if int(s) == int(slot):
                    prompt_text = p
                    break
            score = float(scores_np[i]) if i < len(scores_np) else 0.0
            instances.append(
                InstancePrediction(
                    instance_id=len(instances) + 1,
                    class_name=prompt_text,
                    score=score,
                    bbox_xyxy=[float(x) for x in box_list[i]],
                    mask=mask_u8,
                    prompt_slot=int(slot),
                    prompt_text=prompt_text,
                    source="unknown",
                )
            )
        return instances
