import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from stereo.modeling.common.edge_utils import disparity_to_edge_map
from stereo.modeling.seg.edge_refine import EdgeGuidedProtoRefiner
import time
__all__ = ["EdgeAwareSegPost"]
def _parse_segment_output(seg_out: Union[Tuple, list]):
    """
    Compatible with two return formats:
    1) training: (x, mc, p)
    2) inference (export=False): (fused, (x_raw, mc, p))
    Returns a unified dict:
      {
        "mode": "train" | "infer",
        "p": p,
        "mc": mc,
        "x": x or x_raw,
        "fused": fused or None,
        "raw": x_raw or None
      }
    """
    # Training mode: length-3 tuple
    if isinstance(seg_out, (tuple, list)) and len(seg_out) == 3 and not isinstance(seg_out[1], (tuple, list)):
        x, mc, p = seg_out
        return {"mode": "train", "p": p, "mc": mc, "x": x, "fused": None, "raw": None}
    # Inference mode (export=False): (fused, (x_raw, mc, p))
    assert isinstance(seg_out, (tuple, list)) and len(seg_out) == 2, \
        "Unexpected Segment output format. Expected (x, mc, p) or (fused, (x_raw, mc, p))."
    fused, tail = seg_out
    assert isinstance(tail, (tuple, list)) and len(tail) == 3, \
        "In infer mode, expected second element to be (x_raw, mc, p)."
    x_raw, mc, p = tail
    return {"mode": "infer", "p": p, "mc": mc, "x": None, "fused": fused, "raw": x_raw}
class EdgeAwareSegPost(nn.Module):
    """
    Prototype-space edge refinement + optional edge-alignment loss (training).
    Usage:
      post = EdgeAwareSegPost(nm=32, lambda_edge=1.0, mid=64)
      outputs, loss_dict = post(seg_head_outputs, disparity_map, training=self.training)
    """
    def __init__(self, nm: int, lambda_edge: float = 1.0, mid: int = 64, dilations=(1, 2), loss_mode="dice+l1", alpha=2.0):
        super().__init__()
        self.nm = nm
        self.lambda_edge = lambda_edge
        self.refiner = EdgeGuidedProtoRefiner(nm=nm, mid=mid, dilations=dilations)
    @torch.no_grad()
    def _make_disp_edge(self, disparity_map: torch.Tensor) -> torch.Tensor:
        # disparity_map: (B,1,H,W)
        return disparity_to_edge_map(disparity_map.detach(), normalize=True)
    def forward(self, seg_out: Union[Tuple, list], disparity_map: torch.Tensor, training: bool = True):

        parsed = _parse_segment_output(seg_out)
        p = parsed["p"]                       # (B, nm, Hp, Wp)
        #time0=time.time()
        disp_edge = self._make_disp_edge(disparity_map)  # (B,1,H,W)
        #print(time.time()-time0)
        # Refine prototypes
        p_refined = self.refiner(p, disp_edge)
        # Target assignment happens in SegmentationLoss.  Pass the detached edge
        # map there so alignment is computed on composed positive instance masks.
        loss_dict = {"disp_edge": disp_edge} if training and self.lambda_edge > 0.0 else {}
        # Return in the original format (only replace p -> p_refined)
        if parsed["mode"] == "train":
            return (parsed["x"], parsed["mc"], p_refined), loss_dict
        else:
            fused, x_raw, mc = parsed["fused"], parsed["raw"], parsed["mc"]
            return (fused, (x_raw, mc, p_refined)), loss_dict
