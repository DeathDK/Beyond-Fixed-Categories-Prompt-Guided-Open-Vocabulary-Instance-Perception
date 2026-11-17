import math
from typing import Any, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import dist2bbox, make_anchors

# Inference helpers that output bbox / seg
class tensor2SEG:
    def __init__(self,):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        #h = model.args  # hyperparameters
        #self.hyp = h
        self.stride = [8, 16, 32]  # model strides
        self.reg_max = 16
        reg_max = 16
        self.proj = torch.arange(reg_max, dtype=torch.float)

    def decode_scores(self, pred_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode class scores; return class indices and confidence."""
        pred_probs = torch.sigmoid(pred_scores)  
        max_scores, max_classes = torch.max(pred_probs, dim=-1)
        return max_classes, max_scores


    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""

        b, a, c = pred_dist.shape  # batch, anchors, channels
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.to(pred_dist))
        # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    
    def __call__(self, preds: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        #feats = preds[1] if isinstance(preds, tuple) else preds
        output_channels = feats[0].shape[1]
        class_count = output_channels - self.reg_max * 4
        if class_count <= 0:
            raise ValueError(f"Invalid segmentation output width: {output_channels}")
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], output_channels, -1) for xi in feats], 2
        ).split(
            (self.reg_max * 4, class_count), 1
        )


        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()
        #print("Raw mask values:", pred_masks)

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_bboxes =pred_bboxes*stride_tensor
        pred_classes, pred_confidences = self.decode_scores(pred_scores)
        proto = torch.squeeze(proto,0)

        return pred_classes,pred_confidences,pred_bboxes,pred_masks,proto
    








def process_mask_native(protos, masks_in, bboxes, shape):
    """
    Apply masks to bounding boxes using mask head output with native upsampling.

    Args:
        protos (torch.Tensor): Mask prototypes with shape (mask_dim, mask_h, mask_w).
        masks_in (torch.Tensor): Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS.
        bboxes (torch.Tensor): Bounding boxes with shape (N, 4) where N is number of masks after NMS.
        shape (tuple): Input image size as (height, width).

    Returns:
        (torch.Tensor): Binary mask tensor with shape (H, W, N).
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = scale_masks(masks[None], shape)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.0)


def remove_right_top_padding(masks, original_shape, padded_shape):
    """Crop RightTopPad output back to the original top/right-aligned image."""
    original_height, original_width = original_shape
    padded_height, padded_width = padded_shape
    if original_height > padded_height or original_width > padded_width:
        raise ValueError(
            f"original shape {original_shape} exceeds padded shape {padded_shape}"
        )
    pad_top = padded_height - original_height
    return masks[:, pad_top:pad_top + original_height, :original_width]


def scale_masks(masks, shape, padding: bool = True):
    """
    Rescale segment masks to target shape.

    Args:
        masks (torch.Tensor): Masks with shape (N, C, H, W).
        shape (tuple): Target height and width as (height, width).
        padding (bool): Whether masks are based on YOLO-style augmented images with padding.

    Returns:
        (torch.Tensor): Rescaled masks.
    """
    mh, mw = masks.shape[2:]
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # wh padding
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))) if padding else (0, 0)  # y, x
    bottom, right = (
        mh - int(round(pad[1] + 0.1)),
        mw - int(round(pad[0] + 0.1)),
    )
    masks = masks[..., top:bottom, left:right]

    masks = F.interpolate(masks, shape, mode="bilinear", align_corners=False)  # NCHW
    return masks


def crop_mask(masks, boxes):
    """
    Crop masks to bounding box regions.

    Args:
        masks (torch.Tensor): Masks with shape (N, H, W).
        boxes (torch.Tensor): Bounding box coordinates with shape (N, 4) in relative point form.

    Returns:
        (torch.Tensor): Cropped masks.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def clip_boxes(boxes, shape):
    """
    Clip bounding boxes to image boundaries.

    Args:
        boxes (torch.Tensor | numpy.ndarray): Bounding boxes to clip.
        shape (tuple): Image shape as (height, width).

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped bounding boxes.
    """
    if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding: bool = True, xywh: bool = False):
    """
    Rescale bounding boxes from one image shape to another.

    Rescales bounding boxes from img1_shape to img0_shape, accounting for padding and aspect ratio changes.
    Supports both xyxy and xywh box formats.

    Args:
        img1_shape (tuple): Shape of the source image (height, width).
        boxes (torch.Tensor): Bounding boxes to rescale in format (N, 4).
        img0_shape (tuple): Shape of the target image (height, width).
        ratio_pad (tuple, optional): Tuple of (ratio, pad) for scaling. If None, calculated from image shapes.
        padding (bool): Whether boxes are based on YOLO-style augmented images with padding.
        xywh (bool): Whether box format is xywh (True) or xyxy (False).

    Returns:
        (torch.Tensor): Rescaled bounding boxes in the same format as input.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)
