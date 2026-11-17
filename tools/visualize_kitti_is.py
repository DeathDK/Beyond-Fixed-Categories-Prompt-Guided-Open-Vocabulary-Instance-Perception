#!/usr/bin/env python
"""Visualize KITTI2015-IS instance polygons as thin colored contours (paper style)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "./")

from openworld.label_mapping import annotation_id_to_canonical

# Thin contour colors (BGR).
CLASS_COLORS_BGR = {
    "Car": (0, 0, 255),
    "Truck": (0, 255, 0),
    "Bus": (0, 220, 255),
    "Bicycle": (255, 80, 0),
    "Person": (0, 255, 255),
}


def load_from_yolo_seg(label_path: Path, hw):
    H, W = hw
    instances = []
    if not label_path.exists():
        return instances
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            ann_id = int(float(parts[0]))
            name = annotation_id_to_canonical(ann_id)
            if name == "none":
                continue
            pts = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= W
            pts[:, 1] *= H
            instances.append((name, np.round(pts).astype(np.int32)))
    return instances


def load_from_json(json_path: Path):
    instances = []
    if not json_path.exists():
        return instances
    data = json.loads(json_path.read_text(encoding="utf-8"))
    for shape in data.get("shapes", []):
        raw = shape.get("label", "0")
        try:
            name = annotation_id_to_canonical(int(float(raw)))
        except Exception:
            name = str(raw)
        pts = np.array(shape.get("points", []), dtype=np.float32)
        if pts.size < 6:
            continue
        instances.append((name, np.round(pts).astype(np.int32)))
    return instances


def draw_instances(image_bgr, instances, thickness=1, upsample=4):
    """Paper style: hairline colored outlines only (no fill / text).

    OpenCV minimum integer thickness is 1px; draw on an upsampled canvas then
    downscale so effective stroke is ~1/upsample pixels (very thin).
    """
    h, w = image_bgr.shape[:2]
    scale = max(1, int(upsample))
    if scale == 1:
        out = image_bgr.copy()
        for name, poly in instances:
            if poly.shape[0] < 3:
                continue
            color = CLASS_COLORS_BGR.get(name, (200, 200, 200))
            cv2.polylines(
                out, [poly], True, color, thickness=max(1, thickness), lineType=cv2.LINE_AA
            )
        return out

    canvas = cv2.resize(image_bgr, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    for name, poly in instances:
        if poly.shape[0] < 3:
            continue
        color = CLASS_COLORS_BGR.get(name, (200, 200, 200))
        poly_up = np.round(poly.astype(np.float32) * scale).astype(np.int32)
        cv2.polylines(
            canvas, [poly_up], True, color, thickness=1, lineType=cv2.LINE_AA
        )
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data/KITTI15")
    p.add_argument(
        "--ids",
        type=str,
        default="000000_10,000019_10,000045_10,000120_10",
    )
    p.add_argument("--source", choices=["seg_label", "json"], default="seg_label")
    p.add_argument(
        "--output_dir",
        type=str,
        default="./KITTI2015-IS_visualization",
    )
    p.add_argument("--thickness", type=int, default=1)
    p.add_argument(
        "--upsample",
        type=int,
        default=2,
        help="draw at Nx then downscale for thin contours (default 2; larger= thinner)",
    )
    p.add_argument("--grid", action="store_true")
    p.add_argument("--random", type=int, default=0, help="if >0, pick N random *_10 frames")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.random > 0:
        import random

        random.seed(args.seed)
        label_dir = root / "training" / "seg_label"
        cands = []
        for p in label_dir.glob("*_10.txt"):
            n = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
            if n >= 2:
                cands.append(p.stem)
        random.shuffle(cands)
        stems = cands[: args.random]
        print("random ids:", ",".join(stems))
    else:
        stems = [s.strip() for s in args.ids.split(",") if s.strip()]

    panels = []
    for stem in stems:
        img_path = root / "training" / "image_2" / f"{stem}.png"
        if not img_path.exists():
            print(f"[skip] missing {img_path}")
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[skip] unreadable {img_path}")
            continue
        h, w = image.shape[:2]
        if args.source == "json":
            instances = load_from_json(root / "training" / "json" / f"{stem}.json")
        else:
            instances = load_from_yolo_seg(
                root / "training" / "seg_label" / f"{stem}.txt", (h, w)
            )
        vis = draw_instances(
            image,
            instances,
            thickness=max(1, args.thickness),
            upsample=max(1, args.upsample),
        )
        out_path = out_dir / f"{stem}_is.jpg"
        # Use imencode to avoid Windows OpenCV path-encoding issues on non-ASCII folders.
        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"encode failed for {stem}")
        out_path.write_bytes(buf.tobytes())
        print(f"saved {out_path.resolve()} ({len(instances)} instances)")
        panels.append(vis)

    if args.grid and panels:
        cols = 3 if len(panels) >= 6 else 2
        target_h = min(p.shape[0] for p in panels)
        cells = []
        for p in panels:
            scale = target_h / p.shape[0]
            cells.append(cv2.resize(p, (int(p.shape[1] * scale), target_h)))
        cell_w = max(c.shape[1] for c in cells)

        def pad_w(img, mw=cell_w):
            if img.shape[1] >= mw:
                return img[:, :mw]
            return np.concatenate(
                [img, np.zeros((img.shape[0], mw - img.shape[1], 3), dtype=np.uint8)],
                1,
            )

        cells = [pad_w(c) for c in cells]
        while len(cells) % cols != 0:
            cells.append(np.zeros((target_h, cell_w, 3), dtype=np.uint8))
        rows = []
        for i in range(0, len(cells), cols):
            rows.append(np.concatenate(cells[i : i + cols], 1))
        grid = np.concatenate(rows, 0)
        grid_path = out_dir / "kitti2015_is_grid.jpg"
        ok, buf = cv2.imencode(".jpg", grid, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        grid_path.write_bytes(buf.tobytes())
        print(f"saved {grid_path.resolve()}")


if __name__ == "__main__":
    main()
