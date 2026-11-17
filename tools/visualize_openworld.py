#!/usr/bin/env python
"""Visualizations for open-world experiments (paper figure helpers).

Modes:
  1) Stack existing panel_*.jpg from a directory.
  2) Render a comparison row from JSON results produced by infer_openworld.
  3) Draw F5 / K-only / unknown / fused overlays from saved masks + result.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "./")

from openworld.instances import InstancePrediction


def draw_instances(image_rgb, instances, title=""):
    img = image_rgb.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    overlay = img.copy()
    colors = {
        "Car": (0, 0, 255),
        "Truck": (0, 165, 255),
        "Bus": (0, 255, 255),
        "Bicycle": (255, 0, 255),
        "Person": (0, 255, 0),
        "unknown1": (255, 128, 0),
        "unknown2": (128, 0, 255),
        "unknown3": (255, 255, 0),
        "unknown4": (0, 255, 128),
    }
    for inst in instances:
        color = colors.get(inst.class_name, (200, 200, 200))
        if inst.mask is not None and inst.mask.shape[:2] == overlay.shape[:2]:
            m = inst.mask.astype(bool)
            overlay[m] = (overlay[m].astype(np.float32) * 0.5 + np.array(color) * 0.5).astype(
                np.uint8
            )
        x1, y1, x2, y2 = [int(v) for v in inst.bbox_xyxy]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"{inst.class_name}:{inst.score:.2f}",
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    if title:
        cv2.putText(
            overlay, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
    return overlay


def stack_row(images):
    h = min(im.shape[0] for im in images)
    resized = []
    for im in images:
        scale = h / im.shape[0]
        w = int(im.shape[1] * scale)
        resized.append(cv2.resize(im, (w, h)))
    return np.concatenate(resized, axis=1)


def load_instances_from_result(result_json: Path, mask_dir: Path):
    data = json.loads(result_json.read_text(encoding="utf-8"))
    instances = []
    for item in data.get("instances", []):
        mask = None
        mf = item.get("mask_file")
        if mf:
            mp = mask_dir / mf
            if mp.exists():
                mask = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        instances.append(
            InstancePrediction(
                instance_id=int(item["instance_id"]),
                class_name=item["class_name"],
                score=float(item["score"]),
                bbox_xyxy=list(item["bbox_xyxy"]),
                mask=mask,
                prompt_slot=item.get("prompt_slot"),
                source=item.get("source", "fused"),
            )
        )
    return instances, data


def render_from_result(args):
    left = cv2.cvtColor(cv2.imread(args.left_img), cv2.COLOR_BGR2RGB)
    result_json = Path(args.result_json)
    mask_dir = Path(args.mask_dir) if args.mask_dir else result_json.parent / f"{result_json.stem.replace('_result','')}_masks"
    # Prefer sibling *_masks next to result
    stem = result_json.name.replace("_result.json", "")
    alt = result_json.parent / f"{stem}_masks"
    if alt.exists():
        mask_dir = alt
    instances, data = load_instances_from_result(result_json, mask_dir)
    known = [i for i in instances if not str(i.class_name).startswith("unknown")]
    unknown = [i for i in instances if str(i.class_name).startswith("unknown")]
    panels = [
        left,
        draw_instances(left, known, title="Known-only"),
        draw_instances(left, unknown, title="Unknown"),
        draw_instances(left, instances, title=f"Fused ({data.get('split', '')})"),
    ]
    # Convert RGB panels to BGR for writing
    panels_bgr = [cv2.cvtColor(p, cv2.COLOR_RGB2BGR) if p.ndim == 3 else p for p in panels]
    row = stack_row(panels_bgr)
    cv2.imwrite(args.output, row)
    print(f"saved {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_img", type=str, default="")
    parser.add_argument("--panel_dir", type=str, default="",
                        help="Directory with panel_*.jpg already rendered")
    parser.add_argument("--result_json", type=str, default="",
                        help="infer_openworld *_result.json for overlay panels")
    parser.add_argument("--mask_dir", type=str, default="")
    parser.add_argument("--output", type=str, default="openworld_compare.jpg")
    parser.add_argument("--titles", type=str, default="RGB,Known,Unknown,Fused")
    args = parser.parse_args()

    if args.result_json:
        if not args.left_img:
            raise SystemExit("--left_img is required with --result_json")
        render_from_result(args)
        return

    if not args.panel_dir:
        raise SystemExit("Provide --panel_dir or --result_json")

    panel_dir = Path(args.panel_dir)
    panels = sorted(panel_dir.glob("panel_*.jpg")) + sorted(panel_dir.glob("panel_*.png"))
    if not panels:
        if not args.left_img:
            raise SystemExit("No panels found and no --left_img")
        img = cv2.imread(args.left_img)
        cv2.putText(
            img,
            "Place panel_*.jpg from infer runs into --panel_dir",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.imwrite(args.output, img)
        print(f"wrote placeholder {args.output}")
        return

    images = [cv2.imread(str(p)) for p in panels]
    titles = [t.strip() for t in args.titles.split(",")]
    for i, im in enumerate(images):
        title = titles[i] if i < len(titles) else panels[i].stem
        cv2.putText(im, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    row = stack_row(images)
    cv2.imwrite(args.output, row)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
