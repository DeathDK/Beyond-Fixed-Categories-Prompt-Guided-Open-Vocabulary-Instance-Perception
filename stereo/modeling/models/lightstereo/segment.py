import torch
import torch.nn as nn
from typing import List, Tuple
from .head_module import *
import math

def make_divisible(x, divisor=8):
    return int(math.ceil(x / divisor) * divisor)
def depth_round(n, depth_mult: float):
    return max(int(round(n * depth_mult)), 1)
def width_round(c, width_mult: float, max_channels: int):
    c_ = int(c * width_mult)
    c_ = make_divisible(c_, 8)
    return min(c_, max_channels)
class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, xs):
        return torch.cat(xs, dim=self.dim)
class Head(nn.Module):
    """
      11: Upsample
      12: Concat with layer 6
      13: C3k2 [512, False]
      14: Upsample
      15: Concat with layer 4
      16: C3k2 [256, False] -> output P3
      17: Conv [256, 3, 2]
      18: Concat with layer 13
      19: C3k2 [512, False] -> output P4
      20: Conv [512, 3, 2]
      21: Concat with layer 10
      22: C3k2 [1024, True] -> output P5
      23: Segment [nc, 32, 256] on (P3, P4, P5)
    """
    def __init__(
        self,
        in_ch_l4: int,   # channels of YAML layer 4 output
        in_ch_l6: int,   # channels of YAML layer 6 output
        in_ch_l10: int,  # channels of YAML layer 10 output (after C2PSA)
        nc: int = 80,
        depth_mult: float = 0.5,
        width_mult: float = 0.25,
        max_channels: int = 1024,
    ):
        super().__init__()
        # Three output channel sizes after YAML width scaling
        self.c3 = width_round(256, width_mult, max_channels)   # P3
        self.c4 = width_round(512, width_mult, max_channels)   # P4
        self.c5 = width_round(1024, width_mult, max_channels)  # P5
        # YOLO scales the nominal 256-channel prototype hidden width with the
        # same width multiplier as the neck.  For the nano/S configuration this
        # is 64, not Segment's unscaled default of 256.
        self.npr = width_round(256, width_mult, max_channels)
        # YAML repeats=2, scaled by depth_mult
        n2 = depth_round(2, depth_mult)
        # 11: upsample
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        # 12: concat with layer 6
        self.concat1 = Concat(1)
        # 13: C3k2 [512, False]
        self.c3k2_13 = C3k2(in_ch_l10 + in_ch_l6, self.c4, n=n2, c3k=False)
        # 14: upsample
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        # 15: concat with layer 4
        self.concat2 = Concat(1)
        # 16: C3k2 [256, False] -> P3
        self.c3k2_16 = C3k2(self.c4 + in_ch_l4, self.c3, n=n2, c3k=False)
        # 17: Conv [256, 3, 2]
        self.down_p3 = Conv(self.c3, self.c3, k=3, s=2)
        # 18: concat with layer 13
        self.concat3 = Concat(1)
        # 19: C3k2 [512, False] -> P4
        self.c3k2_19 = C3k2(self.c3 + self.c4, self.c4, n=n2, c3k=False)
        # 20: Conv [512, 3, 2]
        self.down_p4 = Conv(self.c4, self.c4, k=3, s=2)
        # 21: concat with layer 10
        self.concat4 = Concat(1)
        # 22: C3k2 [1024, True] -> P5
        self.c3k2_22 = C3k2(self.c4 + in_ch_l10, self.c5, n=n2, c3k=True)
        # 23: Segment [nc, nm, npr] on the explicit P3/P4/P5 channels.
        self.seg = Segment(
            nc=nc,
            nm=32,
            npr=self.npr,
            ch=(self.c3, self.c4, self.c5),
        )
    def forward(self, x4, x6=None, x10=None):
        # Compatible with two calling styles:
        # 1) forward(p3, p4, c5)
        # 2) forward([p2, p3, p4, c5]) or forward((p2, p3, p4, c5))
        if x6 is None and x10 is None:
            assert isinstance(x4, (list, tuple)) and len(x4) >= 4, \
                "If passing a single argument, it must be [p2, p3, p4, c5]"
            _, x4, x6, x10 = x4  # map to p3, p4, c5
        # 11-13
        y = self.upsample1(x10)
        y = self.concat1([y, x6])
        y13 = self.c3k2_13(y)
        # 14-16 -> P3
        y = self.upsample2(y13)
        y = self.concat2([y, x4])
        p3 = self.c3k2_16(y)
        # 17-19 -> P4
        y = self.down_p3(p3)
        y = self.concat3([y, y13])
        p4 = self.c3k2_19(y)
        # 20-22 -> P5
        y = self.down_p4(p4)
        y = self.concat4([y, x10])
        p5 = self.c3k2_22(y)
        return self.seg([p3, p4, p5])
