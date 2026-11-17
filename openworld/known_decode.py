"""Decode LightStereo-Instance segmentation head outputs into InstancePrediction."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

from stereo.modeling.models.lightstereo.infer_module import (
    crop_mask,
    remove_right_top_padding,
    tensor2SEG,
)

from .instances import InstancePrediction


def _process_mask_right_top_pad(protos, masks_in, bboxes, shape, th, tw):
    c, mh, mw = protos.shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = F.interpolate(
        masks.unsqueeze(1), (th, tw), mode="bilinear", align_corners=False
    ).squeeze(1)
    H0, W0 = shape
    masks = remove_right_top_padding(masks, (H0, W0), (th, tw))
    masks = crop_mask(masks, bboxes)
    return masks.gt_(0.0)


def _map_boxes_right_top_pad(boxes_xyxy, H0, W0, th=384, tw=1248):
    boxes = boxes_xyxy.clone().float() if torch.is_tensor(boxes_xyxy) else torch.as_tensor(
        boxes_xyxy, dtype=torch.float32
    )
    pad_top = max(0, th - H0)
    boxes[:, [1, 3]] -= pad_top
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp_(0, W0 - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp_(0, H0 - 1)
    return boxes


@torch.no_grad()
def decode_known_instances(
    model_pred: dict,
    class_names: Sequence[str],
    orig_hw: Tuple[int, int],
    pad_hw: Tuple[int, int] = (384, 1248),
    conf_threshold: float = 0.45,
    iou_threshold: float = 0.5,
    max_detections: int = 300,
) -> Tuple[np.ndarray, List[InstancePrediction]]:
    """Decode disparity + known instances from a model forward dict.

    Returns:
        disparity HxW float32, list of InstancePrediction with canonical known names.
    """
    H0, W0 = orig_hw
    th, tw = pad_hw
    disp = model_pred["disp_pred"]
    if torch.is_tensor(disp):
        disp_np = disp.detach().float().cpu().numpy()
    else:
        disp_np = np.asarray(disp, dtype=np.float32)
    if disp_np.ndim == 4:
        disp_np = disp_np[0, 0]
    elif disp_np.ndim == 3:
        disp_np = disp_np[0]
    pad_top = max(0, th - H0)
    if disp_np.shape == (th, tw):
        disp_np = disp_np[pad_top : pad_top + H0, :W0]
    elif disp_np.shape != (H0, W0):
        import cv2

        disp_np = cv2.resize(disp_np, (W0, H0), interpolation=cv2.INTER_LINEAR)

    seg_pred = model_pred["seg_pred"]
    # Training tuple vs infer (fused, raw)
    if isinstance(seg_pred, (tuple, list)) and len(seg_pred) == 2:
        seg_for_decode = seg_pred[1]
    else:
        seg_for_decode = seg_pred

    decoder = tensor2SEG()
    pred_classes, pred_confidences, pred_bboxes, pred_masks, proto = decoder(
        seg_for_decode
    )
    # batch size 1
    classes = pred_classes[0]
    conf = pred_confidences[0]
    boxes = pred_bboxes[0]
    mask_coeff = pred_masks[0]

    keep = conf >= conf_threshold
    classes, conf, boxes, mask_coeff = (
        classes[keep],
        conf[keep],
        boxes[keep],
        mask_coeff[keep],
    )
    if classes.numel() == 0:
        return disp_np.astype(np.float32), []

    boxes = _map_boxes_right_top_pad(boxes, H0, W0, th, tw)
    nms_keep = batched_nms(boxes, conf, classes, iou_threshold)
    if nms_keep.numel() > max_detections:
        nms_keep = nms_keep[:max_detections]
    classes = classes[nms_keep]
    conf = conf[nms_keep]
    boxes = boxes[nms_keep]
    mask_coeff = mask_coeff[nms_keep]

    masks = _process_mask_right_top_pad(
        proto, mask_coeff, boxes, (H0, W0), th, tw
    )

    instances: List[InstancePrediction] = []
    for i in range(len(classes)):
        cls_id = int(classes[i].item())
        name = (
            class_names[cls_id]
            if 0 <= cls_id < len(class_names)
            else f"class_{cls_id}"
        )
        # Title-case for open-world canonical display when lowercase.
        if name and name[0].islower():
            name = name[:1].upper() + name[1:]
        mask_i = masks[i].detach().cpu().numpy().astype(np.uint8)
        instances.append(
            InstancePrediction(
                instance_id=i + 1,
                class_name=name,
                score=float(conf[i].item()),
                bbox_xyxy=[float(x) for x in boxes[i].detach().cpu().tolist()],
                mask=mask_i,
                source="known",
            )
        )
    return disp_np.astype(np.float32), instances
