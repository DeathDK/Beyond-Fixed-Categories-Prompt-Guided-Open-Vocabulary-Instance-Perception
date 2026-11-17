#!/usr/bin/env python
"""Per-GT unknown recall diagnosis for one known ckpt + YOLOE settings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, "./")

from openworld.fusion import assign_public_unknown_labels
from openworld.instances import InstancePrediction
from openworld.label_mapping import normalize_canonical_name
from openworld.metrics import mask_iou
from openworld.prompt_expert import YOLOEPromptExpert
from openworld.splits import expand_unknown_prompts_for_yoloe, get_split
from stereo.datasets.dataset_template import build_transform_by_cfg
from stereo.modeling.models.lightstereo.trainer import Trainer
from stereo.utils import common_utils
from tools.eval_openworld import load_gt_from_yolo_label
from tools.infer_openworld import run_openworld


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="K4-2")
    p.add_argument("--data_root", default="./data/KITTI15")
    p.add_argument(
        "--list_file", default="./data/KITTI15/kitti15_val20_with_seglabel.txt"
    )
    p.add_argument("--yoloe_ckpt", default="yoloe-26x-seg.pt")
    p.add_argument("--known_conf", type=float, default=0.001)
    p.add_argument("--yoloe_conf", type=float, default=0.05)
    p.add_argument("--no_synonyms", action="store_true")
    p.add_argument("--multipass", action="store_true", help="One YOLOE call per prompt.")
    p.add_argument("--imgsz", type=int, default=0, help="0 = ultralytics default")
    p.add_argument("--mask_iou_conflict", type=float, default=0.75)
    p.add_argument("--unknown_score_boost", type=float, default=0.15)
    p.add_argument("--g0_edge_band", type=int, default=3)
    p.add_argument("--out_json", default="")
    return p.parse_args()


def main():
    args = parse_args()
    logger = common_utils.create_logger(None, rank=0)
    split = get_split(args.split)
    prompts, slots = expand_unknown_prompts_for_yoloe(
        split, use_synonyms=not args.no_synonyms
    )
    cfgs = EasyDict(common_utils.config_loader(args.cfg_file))
    cfgs.MODEL.PRETRAINED_MODEL = ""
    cfgs.MODEL.YOLO_PRETRAINED_MODEL = None
    if "F5_INIT" in cfgs.MODEL:
        cfgs.MODEL.F5_INIT.ENABLED = False
    from cfgs.data_basic import DATA_PATH_DICT

    for each in cfgs.DATA_CONFIG.DATA_INFOS:
        dataset_name = each.DATASET
        if dataset_name == "KittiDataset":
            dataset_name = "KittiDataset15"
        each.DATA_PATH = DATA_PATH_DICT.get(dataset_name, args.data_root)

    train_args = argparse.Namespace(
        dist_mode=False,
        run_mode="infer",
        workers=0,
        pin_memory=False,
        save_root_dir="./output",
        extra_tag="diag_unknown",
        fix_random_seed=False,
    )
    trainer = Trainer(train_args, cfgs, 0, 0, logger, None)
    model = trainer.model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    transform = build_transform_by_cfg(cfgs.DATA_CONFIG.DATA_TRANSFORM.EVALUATING)
    expert = YOLOEPromptExpert(
        checkpoint=args.yoloe_ckpt,
        conf_threshold=float(args.yoloe_conf),
        device=device if device == "cuda" else None,
    )
    if args.imgsz > 0:
        expert.imgsz = int(args.imgsz)
    expert.multipass = bool(args.multipass)

    data_root = Path(args.data_root)
    unknown_canon = {
        normalize_canonical_name(c) for c in split.unknown_canonical_classes
    }
    rows = []
    tp = 0
    n_gt = 0
    for line in Path(args.list_file).read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        left_rel, right_rel, _, label_rel = parts[:4]
        raw_name = Path(label_rel.replace("\\", "/")).name
        raw_label = data_root / "training" / "seg_label" / raw_name
        image_id = Path(left_rel).stem
        out = run_openworld(
            model,
            transform,
            str(data_root / left_rel),
            str(data_root / right_rel),
            split,
            expert,
            list(prompts),
            known_conf=float(args.known_conf),
            yoloe_conf=float(args.yoloe_conf),
            prompt_slots=list(slots),
            use_geometry=True,
            geometry_mode="g0",
            g0_edge_band_pixels=int(args.g0_edge_band),
            mask_iou_conflict=float(args.mask_iou_conflict),
            unknown_score_boost=float(args.unknown_score_boost),
            device=device,
        )
        h, w = out["left_rgb"].shape[:2]
        gts = load_gt_from_yolo_label(raw_label, (h, w))
        u_gts = []
        for g in gts:
            if normalize_canonical_name(g.class_name) not in unknown_canon:
                continue
            for slot, canon in enumerate(split.unknown_canonical_classes):
                if normalize_canonical_name(canon) == normalize_canonical_name(
                    g.class_name
                ):
                    u_gts.append(
                        InstancePrediction(
                            instance_id=g.instance_id,
                            class_name=split.unknown_public_labels[slot],
                            score=1.0,
                            bbox_xyxy=g.bbox_xyxy,
                            mask=g.mask,
                            prompt_slot=slot,
                            source="gt",
                        )
                    )
                    break
        u_preds = [
            p for p in out["fused"] if str(p.class_name).startswith("unknown")
        ]
        u_pre = assign_public_unknown_labels(list(out["unknown"]), split)
        used = set()
        for gi, g in enumerate(u_gts):
            n_gt += 1
            best_post = (-1.0, None, None)
            best_pre = (-1.0, None, None)
            for pi, p in enumerate(u_preds):
                if p.mask is None or g.mask is None:
                    continue
                iou = mask_iou(p.mask, g.mask)
                if iou > best_post[0]:
                    best_post = (iou, pi, float(p.score))
            for pi, p in enumerate(u_pre):
                if p.mask is None or g.mask is None:
                    continue
                iou = mask_iou(p.mask, g.mask)
                if iou > best_pre[0]:
                    best_pre = (iou, pi, float(p.score))
            hit = best_post[0] >= 0.5 and best_post[1] not in used
            if hit:
                used.add(best_post[1])
                tp += 1
            area = int(g.mask.sum()) if g.mask is not None else 0
            rows.append(
                {
                    "image_id": image_id,
                    "gt_id": g.instance_id,
                    "gt_area": area,
                    "best_iou_post": float(best_post[0]),
                    "best_score_post": best_post[2],
                    "best_iou_pre": float(best_pre[0]),
                    "best_score_pre": best_pre[2],
                    "hit_iou50": bool(hit),
                }
            )

    miss = [r for r in rows if not r["hit_iou50"]]
    report = {
        "settings": vars(args),
        "prompts": list(prompts),
        "tp": tp,
        "n_gt": n_gt,
        "fn": n_gt - tp,
        "u_recall_micro": float(tp / n_gt) if n_gt else None,
        "per_gt": rows,
        "misses": miss,
    }
    print(
        json.dumps(
            {
                "tp": tp,
                "n_gt": n_gt,
                "fn": n_gt - tp,
                "u_recall_micro": report["u_recall_micro"],
                "n_miss": len(miss),
                "misses": miss,
            },
            indent=2,
        )
    )
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
