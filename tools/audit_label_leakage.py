#!/usr/bin/env python
"""Audit open-world training label leakage and model class-head width."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from easydict import EasyDict

sys.path.insert(0, "./")

from openworld.label_mapping import AnnotationSchema, annotation_id_to_canonical
from openworld.splits import get_split
from stereo.utils import common_utils


def audit_label_dir(label_dir: Path, schema: AnnotationSchema) -> Counter:
    counts = Counter()
    for path in sorted(label_dir.glob("*.txt")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(float(parts[0]))
                # Filtered openworld labels store train IDs already.
                if 0 <= cid < schema.num_classes:
                    name = schema.train_id_to_canonical(cid)
                    counts[f"train_id:{name}"] += 1
                else:
                    # Raw annotation space
                    try:
                        name = annotation_id_to_canonical(cid)
                    except ValueError:
                        name = f"id{cid}"
                    counts[f"unexpected:{name}"] += 1
    return counts


def audit_checkpoint(ckpt_path: Path, expected_nc: int) -> None:
    if not ckpt_path.exists():
        print(f"[ckpt] missing {ckpt_path}")
        return
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    cls_keys = [k for k in state if "cv3" in k and "weight" in k and "seg." in k]
    # Fallback: any class-head like tensors with out channels == nc or 5
    found = False
    for key, val in state.items():
        if not hasattr(val, "shape"):
            continue
        name = key.lower()
        if "cls" in name or "cv3" in name:
            if len(val.shape) >= 1:
                out_ch = int(val.shape[0])
                if out_ch in (expected_nc, 5, expected_nc * 2):
                    print(f"[ckpt] {key}: shape={tuple(val.shape)}")
                    found = True
    if not found:
        print(f"[ckpt] no obvious cls tensors; total tensors={len(state)}")
    else:
        print(f"[ckpt] expected NUM_CLASSES={expected_nc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="K4")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/KITTI15",
    )
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="",
        help="Optional yaml to print NUM_CLASSES / KNOWN_CLASSES",
    )
    args = parser.parse_args()

    split = get_split(args.split)
    schema = AnnotationSchema(split.known_classes)
    print(f"split={split.name} known={list(split.known_classes)} nc={schema.num_classes}")
    print(f"unknown_prompts={list(split.unknown_prompts)}")

    if args.cfg_file:
        cfg = EasyDict(common_utils.config_loader(args.cfg_file))
        print(
            f"cfg NUM_CLASSES={cfg.MODEL.NUM_CLASSES} "
            f"KNOWN_CLASSES={cfg.MODEL.get('KNOWN_CLASSES', None)}"
        )
        if int(cfg.MODEL.NUM_CLASSES) != schema.num_classes:
            raise RuntimeError("NUM_CLASSES does not match split")

    label_dir = Path(args.data_root) / "training" / f"seg_label_openworld_{split.name.lower()}"
    if label_dir.exists():
        counts = audit_label_dir(label_dir, schema)
        for k in sorted(counts):
            print(f"  {k}: {counts[k]}")
        for canon in split.unknown_canonical_classes:
            if counts.get(f"train_id:{canon}", 0) > 0:
                raise RuntimeError(f"LEAK: unknown class {canon} present in filtered labels")
        print("label audit OK")
    else:
        print(f"filtered label dir missing: {label_dir} (run prepare_openworld.py)")

    if args.ckpt:
        audit_checkpoint(Path(args.ckpt), schema.num_classes)


if __name__ == "__main__":
    main()
