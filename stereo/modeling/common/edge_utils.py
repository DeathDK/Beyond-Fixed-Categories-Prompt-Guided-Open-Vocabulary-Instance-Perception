import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, List, Tuple, Union
from PIL import Image


__all__ = [
    "SobelFilter2d",
    "quantile_normalize",
    "disparity_to_edge_map",
    "proto_to_edge_map",
    "compose_instance_masks",
]

class SobelFilter2d(nn.Module):
    def __init__(self, normalize_kernel: bool = True):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]])
        ky = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]])
        if normalize_kernel:
            kx = kx / 8.0
            ky = ky / 8.0
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B,1,H,W) or (B,C,H,W)
        B, C, H, W = x.shape
        kx = self.kx.to(x)
        ky = self.ky.to(x)
        gx_list, gy_list = [], []
        for c in range(C):
            xc = x[:, c:c+1, :, :]
            gx_c = F.conv2d(xc, kx, padding=1)
            gy_c = F.conv2d(xc, ky, padding=1)
            gx_list.append(gx_c)
            gy_list.append(gy_c)
        gx = torch.stack(gx_list, dim=1).mean(dim=1)
        gy = torch.stack(gy_list, dim=1).mean(dim=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        return gx, gy, mag

_sobel = SobelFilter2d()

def quantile_normalize(x: torch.Tensor,
                       q_low: float = 0.01,
                       q_high: float = 0.99,
                       eps: float = 1e-6) -> torch.Tensor:
    """
    Per-sample quantile normalization to [0, 1].
    x: (B,1,H,W)
    """
    if x.dim() != 4 or x.size(1) != 1:
        raise ValueError(f"quantile_normalize expects (B,1,H,W), got {x.shape}")
    B = x.size(0)
    flat = x.view(B, -1)
    lo = torch.quantile(flat, q_low, dim=1, keepdim=True)
    hi = torch.quantile(flat, q_high, dim=1, keepdim=True)
    # Reshape back to (B,1,H,W)
    lo = lo.view(B, 1, 1, 1)
    hi = hi.view(B, 1, 1, 1)
    y = (x - lo) / (hi - lo + eps)
    return y.clamp_(0.0, 1.0)

def gaussian_kernel1d(sigma: float, ksize: Optional[int] = None, device=None, dtype=None):
    if sigma <= 0:
        return None
    if ksize is None:
        ksize = int(2 * math.ceil(3 * sigma) + 1)  # ~6σ coverage
    coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    kernel = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1)
def gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    # x: (B, C, H, W)
    if sigma is None or sigma <= 0:
        return x
    B, C, H, W = x.shape

    # Build a 1D Gaussian kernel on the same device/dtype as x
    k1 = gaussian_kernel1d(sigma, device=x.device, dtype=x.dtype)  # (1, 1, K)
    if k1 is None:
        return x
    K = k1.size(-1)

    # Horizontal conv: kernel shape (1, K) -> weight (C,1,1,K)
    weight_h = k1.view(1, 1, 1, K).expand(C, 1, 1, K).contiguous()
    x = F.conv2d(x, weight_h, padding=(0, K // 2), groups=C)

    # Vertical conv: kernel shape (K, 1) -> weight (C,1,K,1)
    weight_v = k1.view(1, 1, K, 1).expand(C, 1, K, 1).contiguous()
    x = F.conv2d(x, weight_v, padding=(K // 2, 0), groups=C)

    return x


def _to_uint8_numpy(img01: torch.Tensor):
    """
    img01: (H,W) or (1,H,W) tensor with values in [0,1]
    Returns uint8 numpy array (H,W)
    """
    if img01.dim() == 3 and img01.size(0) == 1:
        img01 = img01[0]
    img01 = img01.clamp(0, 1).detach().cpu().numpy()
    img255 = (img01 * 255.0 + 0.5).astype("uint8")
    return img255

def _save_gray_image(img01: torch.Tensor, path: str):
    arr = _to_uint8_numpy(img01)
    Image.fromarray(arr, mode="L").save(path)

def disparity_to_edge_map(
    disp: torch.Tensor,
    normalize: bool = True,
    return_xy: bool = False,
    visualize: bool = False,
    save_dir: Optional[str] = "./",
    filenames: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    disp: (B,1,H,W) float tensor (may be on CUDA)
    normalize: True -> quantile-normalize gradient magnitude to [0,1]
    return_xy: True -> also return (gx, gy)
    visualize: True -> save edge maps to save_dir (requires pillow)
    save_dir: directory path; if visualize=True and set, save each sample
    filenames: per-batch filenames (no extension); if None, use indices
    threshold: if set (e.g. 0.2), also save a thresholded binary map for inspection
    q_low, q_high: quantile bounds for normalization, default 1%–99%
    Returns:
      - by default: edge (B,1,H,W)
      - if return_xy=True: (edge, gx, gy), each (B,1,H,W)
    """
    if not disp.is_floating_point():
        disp = disp.float()
    # Compute Sobel
    gx, gy, mag = _sobel(disp)  # each tensor is (B,1,H,W)
    # Normalize (for visualization or as a [0,1] feature downstream)
    edge = quantile_normalize(mag, q_low=q_low, q_high=q_high) if normalize else mag
    # Visualization / saving
    if visualize and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        B = edge.size(0)
        for i in range(B):
            name = filenames[i] if (filenames is not None and i < len(filenames)) else f"{i:04d}"
            # Save grayscale edge map
            _save_gray_image(edge[i], os.path.join(save_dir, f"{name}_sobel_edge.png"))
            # Optionally save a thresholded binary map to inspect edge response
            if threshold is not None:
                bin_map = (edge[i] >= threshold).float()
                _save_gray_image(bin_map, os.path.join(save_dir, f"{name}_sobel_edge_bin_t{threshold:.2f}.png"))
            # Optionally save gx/gy visualizations (symmetrically normalized to [0,1])
            gx_i = gx[i]
            gy_i = gy[i]
            gx01 = (gx_i - gx_i.min()) / (gx_i.max() - gx_i.min() + 1e-6)
            gy01 = (gy_i - gy_i.min()) / (gy_i.max() - gy_i.min() + 1e-6)
            _save_gray_image(gx01, os.path.join(save_dir, f"{name}_sobel_gx.png"))
            _save_gray_image(gy01, os.path.join(save_dir, f"{name}_sobel_gy.png"))
    if return_xy:
        return edge, gx, gy
    return edge


def proto_to_edge_map(p: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    p: (B, nm, Hp, Wp)
    Returns: (B,1,Hp,Wp) — max edge magnitude across channels
    Notes:
      - Uses existing per-channel _sobel and keeps only the magnitude (3rd return)
      - Optionally quantile-normalizes the (B,1,H,W) result if normalize=True
    """
    # If p is logits, sigmoid is more stable for edges; if already probabilities use prob = p.clamp(0,1)
    prob = torch.sigmoid(p)
    B, C, H, W = prob.shape

    # Per-channel Sobel magnitude
    mags = []
    for c in range(C):
        mag_c = _sobel(prob[:, c:c+1, :, :])[2]  # (B,1,H,W)
        mags.append(mag_c)

    # Aggregate to a single-channel edge map via max over channels
    edge = torch.max(torch.cat(mags, dim=1), dim=1, keepdim=True)[0]  # (B,1,H,W)

    # Optional normalization to [0,1]
    if normalize:
        edge = quantile_normalize(edge)  # existing signature (q_low, q_high, eps)
    return edge

def compose_instance_masks(p: torch.Tensor, mc: torch.Tensor) -> torch.Tensor:
    """
    p:  (B, nm, Hp, Wp)
    mc: (B, nm, N)
    returns: (B, N, Hp, Wp)
    """
    B, nm, Hp, Wp = p.shape
    _, nm2, N = mc.shape
    assert nm == nm2, f"nm mismatch: {nm} vs {nm2}"
    p_flat = p.view(B, nm, Hp * Wp)            # (B, nm, L)
    # mc: (B, nm, N) -> want (B, N, nm) for bmm
    mc_t = mc.transpose(1, 2)                  # (B, N, nm)
    # result (B, N, L) = (B, N, nm) x (B, nm, L)
    masks = torch.bmm(mc_t, p_flat)            # (B, N, L)
    masks = masks.view(B, N, Hp, Wp)           # (B, N, Hp, Wp)
    return masks