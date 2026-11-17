import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union

from stereo.modeling.common.edge_utils import (
    disparity_to_edge_map,
    proto_to_edge_map,
    quantile_normalize,
    gaussian_kernel1d,
    gaussian_blur_2d,
    SobelFilter2d,
)

__all__ = ["EdgeGuidedProtoRefiner", "EdgeAlignLossProto", "EdgeAlignLossInstance"]

class EdgeGuidedProtoRefiner(nn.Module):
    """
    Stronger, more controllable edge-guided refinement in prototype space p.
    Inputs:
      - p: (B, nm, Hp, Wp)
      - edge_map: (B,1,H*,W*) geometric edge map (any resolution; will be aligned)
      - (optional) downstream seg_logits for boundary_align_loss
    Main knobs:
      - guidance_gain: amplify gated guidance strength
      - delta_scale: amplify residual correction strength
      - edge_band_sigma: Gaussian-smooth edge_map into a soft band, tolerating small misalignment
      - edge_norm: minmax / quantile-normalize edge_map for numerical stability
      - gate_temperature: temperature-sharpen the gate; <1 is sharper
      - detach_edge: block gradients from the seg branch back into geometric/stereo edges (default True)
      - boundary_align_loss: optional boundary consistency loss (Dice/L1) for weighted training
    """
    def __init__(
        self,
        nm: int,
        mid: int = 64,
        dilations=(1, 2, 3),
        guidance_gain: float = 2,  # gate / edge-guidance strength
        delta_scale: float = 1.0,
        edge_band_sigma: float = 1.0,  # at 1.0 the edge forms a band region
        edge_norm: str = "minmax",      # "none" | "minmax" | "qnorm"
        gate_temperature: float = 0.9,  # <1 sharper, >1 softer
        detach_edge: bool = True,
    ):
        super().__init__()
        self.nm = nm
        self.guidance_gain = guidance_gain
        self.delta_scale = delta_scale
        self.edge_band_sigma = edge_band_sigma
        self.edge_norm = edge_norm
        self.gate_temperature = gate_temperature
        self.detach_edge = detach_edge
        # Encode edges into a gate
        self.edge_enc = nn.Sequential(
            nn.Conv2d(1, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, nm, 3, padding=1),
            nn.Sigmoid(),  # gate in [0,1]
        )
        # Fusion branch (multi-dilation)
        blocks = []
        in_ch = nm + nm  # concat(gated_p, p)
        for d in dilations:
            blocks += [
                nn.Conv2d(in_ch, mid, 3, padding=d, dilation=d),
                nn.ReLU(inplace=True),
            ]
            in_ch = mid
        blocks += [nn.Conv2d(mid, nm, 1)]
        self.fuse = nn.Sequential(*blocks)
        # Start as an exact identity mapping so the randomly initialized SSEG
        # branch cannot destroy pretrained segmentation prototypes.
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)
        # Utilities
        self._sobel = SobelFilter2d()
    def _preprocess_edge(self, edge_map: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        # Align resolution
        if edge_map.shape[-2:] != target_hw:
            edge_map = F.interpolate(edge_map, size=target_hw, mode="bilinear", align_corners=False)
        # Optionally detach gradients back into the stereo/geometry branch
        if self.detach_edge:
            edge_map = edge_map.detach()
        # Optional normalization
        if self.edge_norm == "minmax":
            B = edge_map.size(0)
            flat = edge_map.view(B, -1)
            mn = flat.min(dim=1, keepdim=True)[0].view(B,1,1,1)
            mx = flat.max(dim=1, keepdim=True)[0].view(B,1,1,1)
            edge_map = (edge_map - mn) / (mx - mn + 1e-6)
            edge_map = edge_map.clamp_(0, 1)
        elif self.edge_norm == "qnorm":
            edge_map = quantile_normalize(edge_map)
        # Optionally form a soft band-like edge (Gaussian blur)
        if self.edge_band_sigma and self.edge_band_sigma > 0:
            edge_map = gaussian_blur_2d(edge_map, self.edge_band_sigma)
            edge_map = edge_map.clamp(0, 1)
        return edge_map
    @staticmethod
    def _sharpen01(x: torch.Tensor, temperature: float) -> torch.Tensor:
        # Temperature-scale [0,1] probabilities to sharpen / soften
        if temperature == 1.0:
            return x
        x = x.clamp(1e-6, 1 - 1e-6)
        return torch.sigmoid(torch.logit(x) / temperature)
    def forward(
        self,
        p: torch.Tensor,
        edge_map: torch.Tensor,
        return_aux: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Returns:
          - default: p_refined
          - if return_aux=True: (p_refined, aux_dict) for visualization / debugging
        """
        B, C, Hp, Wp = p.shape
        edge = self._preprocess_edge(edge_map, (Hp, Wp))  # (B,1,Hp,Wp)
        # Encode into a gate and optionally sharpen
        gate = self.edge_enc(edge)                       # (B,nm,Hp,Wp), [0,1]
        gate = self._sharpen01(gate, self.gate_temperature)
        # Stronger gated guidance
        gated_p = p * (1.0 + self.guidance_gain * gate)  # guidance_gain>1 -> stronger boost
        # Fusion and residual
        x = torch.cat([gated_p, p], dim=1)               # (B,2*nm,Hp,Wp)
        delta = self.fuse(x)                              # (B,nm,Hp,Wp)
        p_refined = p + self.delta_scale * delta
        if not return_aux:
            return p_refined
        aux = {
            "edge_preproc": edge,     # soft-band, normalized edge
            "gate": gate,             # gate map
            "gated_p": gated_p,       # amplified p
            "delta": delta,           # residual
        }
        return p_refined, aux
    

class EdgeAlignLossProto(nn.Module):
    """
    Default: dice + L1, with higher weights on edge locations.
    """
    def __init__(self, mode: str = "dice+l1", alpha: float = 2.0, smooth: float = 1e-6):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, p: torch.Tensor, disp_edge: torch.Tensor) -> torch.Tensor:
        # seg edge (from prototype aggregation)
        p_edge = proto_to_edge_map(p, normalize=True)         # (B,1,Hp,Wp)
        if disp_edge.shape[-2:] != p_edge.shape[-2:]:
            disp_edge = F.interpolate(disp_edge, size=p_edge.shape[-2:], mode="bilinear", align_corners=False)
        e = quantile_normalize(disp_edge)

        w = 1.0 + self.alpha * e  # edge weight
        losses = []

        if "dice" in self.mode:
            p_ = p_edge
            t_ = e
            inter = (p_ * t_ * w).sum(dim=(1, 2, 3))
            denom = (p_ * p_ * w).sum(dim=(1, 2, 3)) + (t_ * t_ * w).sum(dim=(1, 2, 3)) + self.smooth
            dice = 1.0 - (2 * inter + self.smooth) / denom
            losses.append(dice.mean())

        if "l1" in self.mode:
            l1 = (w * (p_edge - e).abs()).mean()
            losses.append(l1)

        return sum(losses) if len(losses) > 1 else losses[0]


class EdgeAlignLossInstance(nn.Module):
    """Supervise instance boundaries and use disparity edges only as weights.

    A disparity map contains object-internal depth discontinuities and background
    edges, so it is not a valid segmentation target by itself.  Ground-truth mask
    boundaries define *where* an instance must end; aligned disparity edges only
    increase the penalty near geometrically informative locations.
    """

    def __init__(self, mode: str = "dice+l1", alpha: float = 2.0, smooth: float = 1e-6):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.smooth = smooth
        self.sobel = SobelFilter2d()

    @staticmethod
    def _crop_to_boxes(values: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        count, height, width = values.shape
        rows = torch.arange(height, device=values.device, dtype=boxes.dtype)[None, :, None]
        cols = torch.arange(width, device=values.device, dtype=boxes.dtype)[None, None, :]
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        valid = (
            (cols >= x1[:, None, None])
            & (cols < x2[:, None, None])
            & (rows >= y1[:, None, None])
            & (rows < y2[:, None, None])
        )
        return values * valid.to(values.dtype)

    def forward(
        self,
        mask_logits: torch.Tensor,
        gt_masks: torch.Tensor,
        disparity_edge: torch.Tensor,
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        if mask_logits.numel() == 0:
            return mask_logits.sum()
        count, height, width = mask_logits.shape
        if gt_masks.shape != mask_logits.shape:
            raise ValueError(
                f"gt_masks must match mask_logits, got {tuple(gt_masks.shape)} "
                f"and {tuple(mask_logits.shape)}"
            )
        probabilities = mask_logits.sigmoid().unsqueeze(1)
        predicted_edge = self.sobel(probabilities)[2].squeeze(1)
        predicted_edge = predicted_edge / predicted_edge.flatten(1).amax(dim=1).clamp_min(self.smooth)[:, None, None]

        target_edge = self.sobel(gt_masks.float().unsqueeze(1))[2].squeeze(1)
        target_edge = target_edge / target_edge.flatten(1).amax(dim=1).clamp_min(self.smooth)[:, None, None]

        geometry_edge = F.interpolate(
            disparity_edge,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).expand(count, -1, -1, -1).squeeze(1)
        predicted_edge = self._crop_to_boxes(predicted_edge, boxes)
        target_edge = self._crop_to_boxes(target_edge, boxes)
        geometry_edge = self._crop_to_boxes(geometry_edge, boxes)
        valid = self._crop_to_boxes(torch.ones_like(predicted_edge), boxes)
        weights = 1.0 + self.alpha * geometry_edge

        losses = torch.zeros(count, device=mask_logits.device, dtype=mask_logits.dtype)
        if "dice" in self.mode:
            intersection = (predicted_edge * target_edge * weights).sum(dim=(1, 2))
            denominator = (
                (predicted_edge.square() * weights).sum(dim=(1, 2))
                + (target_edge.square() * weights).sum(dim=(1, 2))
            )
            losses = losses + 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if "l1" in self.mode:
            weighted_valid = weights * valid
            l1 = (weighted_valid * (predicted_edge - target_edge).abs()).sum(dim=(1, 2))
            l1 = l1 / weighted_valid.sum(dim=(1, 2)).clamp_min(self.smooth)
            losses = losses + l1
        return losses.sum()
