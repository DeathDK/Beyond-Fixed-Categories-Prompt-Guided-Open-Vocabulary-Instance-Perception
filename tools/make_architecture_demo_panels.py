#!/usr/bin/env python
"""Generate academic demo panels: RGB, disparity, known instances, instance point cloud.

Uses real KITTI2015 images + disp GT + IS seg_label (not fake illustrations).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, "./")

# Load leaf modules without importing openworld package __init__ (avoids torch).
import importlib.util


def _load_mod(name: str, rel: str):
    path = Path(rel).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_instances = _load_mod("ow_instances", "openworld/instances.py")
_label_mapping = _load_mod("ow_label_mapping", "openworld/label_mapping.py")
_pointcloud = _load_mod("ow_pointcloud", "openworld/pointcloud.py")
InstancePrediction = _instances.InstancePrediction
annotation_id_to_canonical = _label_mapping.annotation_id_to_canonical
reconstruct_instances = _pointcloud.reconstruct_instances
from stereo.utils.disp_color import disp_to_color

CLASS_COLORS_BGR = {
    "Car": (0, 0, 255),
    "Truck": (0, 255, 0),
    "Bus": (0, 220, 255),
    "Bicycle": (255, 80, 0),
    "Person": (0, 255, 255),
}

# Paper-style class colors matching common open-world legends (RGB 0-1).
CLASS_COLORS_RGB = {
    "Car": (0.20, 0.72, 0.70),
    "Truck": (0.92, 0.55, 0.18),
    "Bus": (0.55, 0.32, 0.78),
    "Bicycle": (0.88, 0.28, 0.25),
    "Person": (0.62, 0.62, 0.22),
    "Unknown": (0.38, 0.32, 0.68),
}

# Per-instance accent colors (reference-style multi-car scene).
INSTANCE_PALETTE = [
    (0.18, 0.72, 0.70),
    (0.35, 0.78, 0.28),
    (0.92, 0.78, 0.18),
    (0.92, 0.35, 0.72),
    (0.25, 0.35, 0.85),
    (0.92, 0.55, 0.18),
    (0.55, 0.32, 0.78),
    (0.88, 0.28, 0.25),
]


def load_instances(label_path: Path, hw):
    H, W = hw
    out = []
    if not label_path.exists():
        return out
    with label_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            name = annotation_id_to_canonical(int(float(parts[0])))
            if name == "none":
                continue
            pts = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= W
            pts[:, 1] *= H
            poly = np.round(pts).astype(np.int32)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 1)
            xs, ys = poly[:, 0], poly[:, 1]
            out.append(
                InstancePrediction(
                    instance_id=i + 1,
                    class_name=name,
                    score=1.0,
                    bbox_xyxy=[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                    mask=mask,
                    source="gt",
                )
            )
    return out


def draw_known_contours(image_bgr, instances, upsample=2):
    h, w = image_bgr.shape[:2]
    canvas = cv2.resize(image_bgr, (w * upsample, h * upsample), interpolation=cv2.INTER_LINEAR)
    for inst in instances:
        color = CLASS_COLORS_BGR.get(inst.class_name, (200, 200, 200))
        ys, xs = np.where(inst.mask > 0)
        if xs.size == 0:
            continue
        # contour from mask for clean outline
        cnts, _ = cv2.findContours(
            (inst.mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        for c in cnts:
            c2 = (c.astype(np.float32) * upsample).astype(np.int32)
            cv2.polylines(canvas, [c2], True, color, 1, cv2.LINE_AA)
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)


def densify_disparity_for_viz(disp: np.ndarray) -> np.ndarray:
    """KITTI GT disparity is sparse; inpaint for dense paper-style visualization only."""
    d = disp.astype(np.float32).copy()
    valid = np.isfinite(d) & (d > 0)
    mask = (~valid).astype(np.uint8) * 255
    # Normalize to 16-bit for inpaint stability, then scale back.
    scale = 256.0
    src = np.clip(d * scale, 0, 65535).astype(np.uint16)
    # OpenCV inpaint wants 8-bit; do two-pass on high/low bytes approx via float32 convert.
    src8 = np.clip(d / max(float(np.percentile(d[valid], 99)), 1.0) * 255.0, 0, 255).astype(np.uint8)
    filled8 = cv2.inpaint(src8, mask, 3, cv2.INPAINT_NS)
    # Map filled8 back using original value range.
    vmax = float(np.percentile(d[valid], 99))
    filled = filled8.astype(np.float32) / 255.0 * vmax
    filled[valid] = d[valid]
    return filled


def colorize_disparity(disp: np.ndarray, densify: bool = False) -> np.ndarray:
    """Official-style sparse disparity color map (BGR). Invalid -> black."""
    d = densify_disparity_for_viz(disp) if densify else disp.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return np.zeros((*disp.shape, 3), dtype=np.uint8)
    vmax = float(np.percentile(d[valid], 98))
    vmax = max(vmax, 1.0)
    # Prefer KITTI repo colormap on sparse GT.
    tmp = np.zeros_like(d)
    tmp[valid] = np.clip(d[valid], 0, vmax)
    colored = disp_to_color(tmp, max_disp=vmax)
    colored = np.clip(colored, 0, 255).astype(np.uint8)
    colored[~valid] = 0
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)


def densify_disp_inside_mask(disp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fully fill disparity inside mask for dense instance surfaces (viz only)."""
    d = disp.astype(np.float32).copy()
    m = mask.astype(bool)
    if m.sum() < 20:
        return d
    valid = m & np.isfinite(d) & (d > 0)
    if valid.sum() < 8:
        return d
    med = float(np.median(d[valid]))
    ys, xs = np.where(m)
    y0, y1 = max(0, ys.min() - 4), min(d.shape[0], ys.max() + 5)
    x0, x1 = max(0, xs.min() - 4), min(d.shape[1], xs.max() + 5)
    crop = d[y0:y1, x0:x1].copy()
    crop_m = m[y0:y1, x0:x1]
    crop_valid = crop_m & (crop > 0) & np.isfinite(crop)
    vmax = max(float(np.percentile(crop[crop_valid], 99)), med, 1.0)
    src8 = np.zeros(crop.shape, np.uint8)
    src8[crop_valid] = np.clip(crop[crop_valid] / vmax * 255.0, 0, 255).astype(np.uint8)
    hole = (crop_m & ~crop_valid).astype(np.uint8) * 255
    if hole.any():
        filled = cv2.inpaint(src8, hole, 12, cv2.INPAINT_TELEA)
        crop[crop_m & ~crop_valid] = filled[crop_m & ~crop_valid].astype(np.float32) / 255.0 * vmax
    still = crop_m & ~(crop > 0)
    crop[still] = med
    # Kill KITTI GT scanline banding: tall vertical blur, then isotropic polish.
    # (Sparse GT is dense along rows, sparse across rows → horizontal stripes in 3D.)
    work = crop.copy()
    work[~crop_m] = med
    work = cv2.GaussianBlur(work, (5, 31), 0)
    work = cv2.GaussianBlur(work, (11, 11), 0)
    crop[crop_m] = work[crop_m]
    d[y0:y1, x0:x1][crop_m] = crop[crop_m]
    return d


def build_pretty_instance_clouds(
    disp: np.ndarray,
    left_rgb: np.ndarray,
    instances: list,
    calib: dict,
    min_points: int = 200,
):
    clouds = {}
    for inst in instances:
        if inst.mask is None or (inst.mask > 0).sum() < 80:
            continue
        # Skip masks with almost no GT overlap (densify would invent depth).
        if ((inst.mask > 0) & (disp > 0) & np.isfinite(disp)).sum() < 30:
            continue
        d_inst = densify_disp_inside_mask(disp, inst.mask)
        d_masked = np.zeros_like(d_inst)
        m = inst.mask > 0
        d_masked[m] = d_inst[m]
        one = reconstruct_instances(d_masked, left_rgb, [inst], calib, min_disp=1.0)
        cloud = one.get(inst.instance_id)
        if cloud is None or cloud["num_points"] < min_points:
            continue
        clouds[inst.instance_id] = cloud
    return clouds


def _project_scene(xyz: np.ndarray, yaw_deg: float = -28.0, pitch_deg: float = 18.0, f: float = 820.0):
    """Orbit camera for a paper-style street scene point cloud."""
    X, Ydown, Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    Yup = -Ydown
    yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    x1 = c * X + s * Z
    z1 = -s * X + c * Z
    cp, sp = np.cos(pitch), np.sin(pitch)
    y2 = cp * Yup - sp * z1
    z2 = sp * Yup + cp * z1
    z2_safe = np.clip(z2, 0.5, None)
    return f * x1 / z2_safe, -f * y2 / z2_safe, z2_safe


def _draw_legend(canvas: np.ndarray, x0: int, y0: int):
    names = ["Car", "Truck", "Bus", "Bicycle", "Person", "Unknown"]
    for i, name in enumerate(names):
        y = y0 + i * 36
        rgb = CLASS_COLORS_RGB[name]
        bgr = tuple(int(255 * c) for c in rgb[::-1])
        cv2.circle(canvas, (x0 + 12, y), 8, bgr, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x0 + 12, y), 8, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(
            canvas, name, (x0 + 28, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (45, 45, 45), 1, cv2.LINE_AA,
        )


def render_pointcloud_figure(
    clouds: dict,
    out_path: Path,
    disp: np.ndarray | None = None,
    left_rgb: np.ndarray | None = None,
    instances: list | None = None,
    calib: dict | None = None,
    max_points: int = 18000,
):
    """Scene point cloud: gray background + solid instance colors + legend (reference style)."""
    del clouds, max_points
    if disp is None or left_rgb is None or instances is None or calib is None:
        canvas = np.full((480, 720, 3), 255, np.uint8)
        cv2.putText(canvas, "missing inputs", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        save_bgr(out_path, canvas)
        return

    Himg, Wimg = disp.shape[:2]
    # Densify sparse GT for solid paper-style surfaces (viz only).
    d_dense = densify_disparity_for_viz(disp)
    # Also densify inside each mask for cleaner cars.
    for inst in instances:
        if inst.mask is None:
            continue
        if ((inst.mask > 0) & (disp > 0)).sum() >= 30:
            d_dense = densify_disp_inside_mask(d_dense, inst.mask)

    fx, fy = float(calib["fx"]), float(calib["fy"])
    cx, cy = float(calib["cx"]), float(calib["cy"])
    baseline = float(calib["baseline"])

    # Instance id map + class / color.
    id_map = np.zeros((Himg, Wimg), np.int32)
    inst_meta = {}  # iid -> (class_name, rgb_tuple)
    # Prefer larger instances first so small ones can overwrite edges if needed.
    ordered = sorted(
        [x for x in instances if x.mask is not None and (x.mask > 0).sum() >= 80],
        key=lambda x: (x.mask > 0).sum(),
        reverse=True,
    )
    for rank, inst in enumerate(ordered[:12]):
        m = inst.mask > 0
        if (m & (disp > 0)).sum() < 20:
            continue
        id_map[m] = inst.instance_id
        # Multi-car scenes: distinct instance colors (reference style); non-Car use class color.
        if inst.class_name == "Car":
            col = INSTANCE_PALETTE[rank % len(INSTANCE_PALETTE)]
            if rank == 0:
                col = CLASS_COLORS_RGB["Car"]
        else:
            col = CLASS_COLORS_RGB.get(inst.class_name, INSTANCE_PALETTE[rank % len(INSTANCE_PALETTE)])
        inst_meta[inst.instance_id] = (inst.class_name, col)

    # Sample points: denser on instances, sparser on background.
    rgb = left_rgb
    if rgb.max() <= 1.5:
        rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    else:
        rgb_u8 = rgb.astype(np.uint8)
    gray = (0.299 * rgb_u8[..., 0] + 0.587 * rgb_u8[..., 1] + 0.114 * rgb_u8[..., 2]).astype(np.float32)

    ys_i, xs_i = np.where((id_map > 0) & (d_dense > 1.0) & np.isfinite(d_dense))
    bg_mask = (id_map == 0) & (d_dense > 1.0) & np.isfinite(d_dense)
    ys_b, xs_b = np.where(bg_mask)
    if ys_b.size > 0:
        stride_sel = (ys_b % 2 == 0) & (xs_b % 2 == 0)
        ys_b, xs_b = ys_b[stride_sel], xs_b[stride_sel]
        if ys_b.size > 80000:
            sel = np.random.RandomState(1).choice(ys_b.size, 80000, replace=False)
            ys_b, xs_b = ys_b[sel], xs_b[sel]

    # Instance: every pixel (already densified) — subsample if huge.
    if ys_i.size > 120000:
        sel = np.random.RandomState(2).choice(ys_i.size, 120000, replace=False)
        ys_i, xs_i = ys_i[sel], xs_i[sel]

    def backproj(ys, xs):
        d = d_dense[ys, xs]
        z = fx * baseline / d
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        return np.stack([x, y, z], 1).astype(np.float32)

    pts, cols, is_inst = [], [], []
    if ys_b.size:
        xyz_b = backproj(ys_b, xs_b)
        # Light gray from luminance (reference style).
        g = np.clip(gray[ys_b, xs_b] / 255.0 * 0.72 + 0.18, 0.15, 0.85)
        col_b = np.stack([g, g, g], 1)
        pts.append(xyz_b)
        cols.append(col_b)
        is_inst.append(np.zeros(xyz_b.shape[0], np.uint8))
    if ys_i.size:
        xyz_i = backproj(ys_i, xs_i)
        iids = id_map[ys_i, xs_i]
        col_i = np.zeros((xyz_i.shape[0], 3), np.float32)
        for iid, (_, rgb_t) in inst_meta.items():
            m = iids == iid
            col_i[m] = rgb_t
        pts.append(xyz_i)
        cols.append(col_i)
        is_inst.append(np.ones(xyz_i.shape[0], np.uint8))

    if not pts:
        canvas = np.full((480, 720, 3), 255, np.uint8)
        cv2.putText(canvas, "no points", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        save_bgr(out_path, canvas)
        return

    xyz = np.concatenate(pts, 0)
    colors = np.concatenate(cols, 0)
    flag = np.concatenate(is_inst, 0)

    # Crop to a sensible street volume.
    keep = (
        (xyz[:, 2] > 2.5) & (xyz[:, 2] < 45.0)
        & (np.abs(xyz[:, 0]) < 16.0)
        & (xyz[:, 1] > -3.0) & (xyz[:, 1] < 2.0)
    )
    xyz, colors, flag = xyz[keep], colors[keep], flag[keep]

    W_scene, W_leg, H = 980, 160, 560
    canvas = np.full((H, W_scene + W_leg, 3), 255, np.uint8)
    scene = canvas[:, :W_scene]

    u, v, zcam = _project_scene(xyz)
    umin, umax = np.percentile(u, 1.5), np.percentile(u, 98.5)
    vmin, vmax = np.percentile(v, 1.5), np.percentile(v, 98.5)
    mid_u, mid_v = 0.5 * (umin + umax), 0.5 * (vmin + vmax)
    pad = 40
    scale = min((W_scene - 2 * pad) / max(umax - umin, 1e-6), (H - 2 * pad) / max(vmax - vmin, 1e-6)) * 0.95
    px = ((u - mid_u) * scale + W_scene * 0.48).astype(np.int32)
    py = ((v - mid_v) * scale + H * 0.52).astype(np.int32)

    # Draw background first (far→near), then instances on top with slightly larger points.
    order = np.argsort(-zcam)
    for i in order:
        x, y = int(px[i]), int(py[i])
        if not (0 <= x < W_scene and 0 <= y < H):
            continue
        bgr = tuple(int(255 * c) for c in colors[i][::-1])
        r = 2 if flag[i] else 1
        cv2.circle(scene, (x, y), r, bgr, -1, cv2.LINE_AA)

    _draw_legend(canvas, W_scene + 18, 48)
    cv2.putText(
        canvas, "Instance Point Cloud", (22, 36),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (40, 40, 40), 2, cv2.LINE_AA,
    )
    save_bgr(out_path, canvas)


def save_bgr(path: Path, img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    path.write_bytes(buf.tobytes())


def _load_disp(path: Path) -> np.ndarray:
    d = np.array(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), dtype=np.float32)
    if d is None or d.size == 0:
        raise FileNotFoundError(path)
    if d.ndim == 3:
        d = d[:, :, 0]
    return d / 256.0


def warp_instances_to_right(instances: list, disp_left: np.ndarray) -> list:
    """Warp left IS masks to right view via left disparity (x_r = x_l - d)."""
    h, w = disp_left.shape
    out = []
    for inst in instances:
        if inst.mask is None:
            continue
        ys, xs = np.where(inst.mask > 0)
        if ys.size == 0:
            continue
        d = disp_left[ys, xs]
        valid = np.isfinite(d) & (d > 0)
        ys, xs, d = ys[valid], xs[valid], d[valid]
        if ys.size < 30:
            continue
        xr = np.clip(np.round(xs - d).astype(np.int32), 0, w - 1)
        mask_r = np.zeros((h, w), np.uint8)
        mask_r[ys, xr] = 1
        # densify thin warped mask a bit
        mask_r = cv2.dilate(mask_r, np.ones((3, 3), np.uint8), iterations=1)
        ys2, xs2 = np.where(mask_r > 0)
        out.append(
            InstancePrediction(
                instance_id=inst.instance_id,
                class_name=inst.class_name,
                score=inst.score,
                bbox_xyxy=[
                    float(xs2.min()),
                    float(ys2.min()),
                    float(xs2.max()),
                    float(ys2.max()),
                ],
                mask=mask_r,
                source="gt_warped",
            )
        )
    return out


def _save_lr_disp_known(
    out: Path,
    stem: str,
    side: str,
    rgb_bgr: np.ndarray,
    disp: np.ndarray,
    instances: list,
):
    """Save RGB / disparity / known-instance panels for one stereo side (no point cloud)."""
    known_vis = draw_known_contours(rgb_bgr, instances, upsample=2)
    disp_vis = colorize_disparity(disp)
    save_bgr(out / f"{stem}_{side}_01_rgb.jpg", rgb_bgr)
    save_bgr(out / f"{stem}_{side}_02_disparity.jpg", disp_vis)
    save_bgr(out / f"{stem}_{side}_03_known_instances.jpg", known_vis)

    def resize_h(img, th=280):
        scale = th / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), th))

    titles = [
        f"{side.upper()} RGB",
        f"{side.upper()} Disparity",
        f"{side.upper()} Known Instances",
    ]
    panels = [resize_h(rgb_bgr), resize_h(disp_vis), resize_h(known_vis)]
    labeled = []
    for img, title in zip(panels, titles):
        canvas = np.full((img.shape[0] + 28, img.shape[1], 3), 255, dtype=np.uint8)
        canvas[28:] = img
        cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        labeled.append(canvas)
    mh = max(x.shape[0] for x in labeled)
    for i, x in enumerate(labeled):
        if x.shape[0] < mh:
            pad = np.full((mh - x.shape[0], x.shape[1], 3), 255, dtype=np.uint8)
            labeled[i] = np.concatenate([x, pad], 0)
    gap = np.full((mh, 8, 3), 255, dtype=np.uint8)
    strip = labeled[0]
    for x in labeled[1:]:
        strip = np.concatenate([strip, gap, x], 1)
    save_bgr(out / f"{stem}_{side}_strip.jpg", strip)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data/KITTI15")
    p.add_argument("--stem", default="000089_10")
    p.add_argument("--output_dir", default="./architecture_figs/demo_panels")
    p.add_argument(
        "--stereo_panels",
        action="store_true",
        help="(deprecated) old L/R disp+known export; ignored — use default layout.",
    )
    p.add_argument("--skip_pointcloud", action="store_true")
    args = p.parse_args()

    root = Path(args.data_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = args.stem

    left = cv2.imread(str(root / "training" / "image_2" / f"{stem}.png"))
    right = cv2.imread(str(root / "training" / "image_3" / f"{stem}.png"))
    if left is None:
        raise FileNotFoundError(stem)
    if right is None:
        raise FileNotFoundError(f"right image missing for {stem}")
    h, w = left.shape[:2]
    disp_l = _load_disp(root / "training" / "disp_occ_0" / f"{stem}.png")
    instances_l = load_instances(root / "training" / "seg_label" / f"{stem}.txt", (h, w))

    # Layout: Left RGB + Right RGB (stereo originals); one Disp / Known / PointCloud (left view).
    known_vis = draw_known_contours(left, instances_l, upsample=2)
    disp_vis = colorize_disparity(disp_l)

    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    calib = {"fx": 721.5377, "fy": 721.5377, "cx": 609.5593, "cy": 172.854, "baseline": 0.54}
    clouds = build_pretty_instance_clouds(disp_l, left_rgb, instances_l, calib)

    save_bgr(out / f"{stem}_01_left_rgb.jpg", left)
    save_bgr(out / f"{stem}_01b_right_rgb.jpg", right)
    # keep legacy alias for left
    save_bgr(out / f"{stem}_01_rgb.jpg", left)
    save_bgr(out / f"{stem}_02_disparity.jpg", disp_vis)
    save_bgr(out / f"{stem}_03_known_instances.jpg", known_vis)
    if not args.skip_pointcloud:
        pc_path = out / f"{stem}_04_instance_pointcloud.jpg"
        render_pointcloud_figure(
            clouds, pc_path, disp=disp_l, left_rgb=left_rgb, instances=instances_l, calib=calib
        )
    else:
        pc_path = None

    def resize_h(img, th=280):
        scale = th / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), th))

    panels = [resize_h(left), resize_h(right), resize_h(disp_vis), resize_h(known_vis)]
    titles = ["Left RGB (IL)", "Right RGB (IR)", "Disparity Map", "Known Instances"]
    if pc_path is not None and pc_path.exists():
        panels.append(resize_h(cv2.imread(str(pc_path))))
        titles.append("Instance Point Cloud")
    labeled = []
    for img, title in zip(panels, titles):
        canvas = np.full((img.shape[0] + 28, img.shape[1], 3), 255, dtype=np.uint8)
        canvas[28:] = img
        cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        labeled.append(canvas)
    mh = max(x.shape[0] for x in labeled)
    for i, x in enumerate(labeled):
        if x.shape[0] < mh:
            pad = np.full((mh - x.shape[0], x.shape[1], 3), 255, dtype=np.uint8)
            labeled[i] = np.concatenate([x, pad], 0)
    gap = np.full((mh, 8, 3), 255, dtype=np.uint8)
    strip = labeled[0]
    for x in labeled[1:]:
        strip = np.concatenate([strip, gap, x], 1)
    save_bgr(out / f"{stem}_strip.jpg", strip)

    # Remove obsolete per-side disp/known panels if present.
    for side in ("left", "right"):
        for name in (
            f"{stem}_{side}_01_rgb.jpg",
            f"{stem}_{side}_02_disparity.jpg",
            f"{stem}_{side}_03_known_instances.jpg",
            f"{stem}_{side}_strip.jpg",
        ):
            old = out / name
            if old.exists():
                old.unlink()

    n_pts = sum(c["num_points"] for c in clouds.values())
    print(f"saved panels to {out.resolve()}")
    print(f"instances={len(instances_l)} total_points={n_pts}")
    print("layout: Left RGB | Right RGB | Disparity | Known | PointCloud(optional)")


if __name__ == "__main__":
    main()
