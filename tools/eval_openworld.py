#!/usr/bin/env python
"""Offline open-world evaluation (reads full 5-class GT; inference itself must not)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, "./")

from openworld.instances import InstancePrediction
from openworld.label_mapping import annotation_id_to_canonical, normalize_canonical_name
from openworld.metrics import (
    classwise_mask_ap,
    map_unknown_preds_to_canonical,
    unknown_recall_and_miou,
)
from openworld.prompt_expert import YOLOEPromptExpert
from openworld.vlm_expert import GroundingSAM2Expert
from openworld.splits import get_split
from stereo.datasets.dataset_template import build_transform_by_cfg
from stereo.modeling.models.lightstereo.trainer import Trainer
from stereo.utils import common_utils
from tools.infer_openworld import run_openworld


def load_gt_from_yolo_label(label_path: Path, image_hw) -> list:
    H, W = image_hw
    gts = []
    if not label_path.exists():
        return gts
    with label_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            ann_id = int(float(parts[0]))
            name = annotation_id_to_canonical(ann_id)
            if name == "none":
                continue
            poly = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
            poly[:, 0] *= W
            poly[:, 1] *= H
            poly_i = np.round(poly).astype(np.int32)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [poly_i], 1)
            xs, ys = poly[:, 0], poly[:, 1]
            bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            gts.append(
                InstancePrediction(
                    instance_id=i + 1,
                    class_name=name,
                    score=1.0,
                    bbox_xyxy=bbox,
                    mask=mask,
                    source="gt",
                )
            )
    return gts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", type=str, default="cfgs/openworld/known4.yaml")
    p.add_argument("--pretrained_model", type=str, required=True)
    p.add_argument("--split", type=str, default="K4")
    p.add_argument("--data_root", type=str, default="./data/KITTI15")
    p.add_argument(
        "--list_file",
        type=str,
        default="./data/KITTI15/kitti15_val20_with_seglabel.txt",
    )
    p.add_argument("--yoloe_ckpt", type=str, default="yoloe-26x-seg.pt")
    p.add_argument(
        "--unknown_expert",
        type=str,
        default="gdino_sam2",
        choices=["gdino_sam2", "yoloe"],
    )
    p.add_argument("--box_threshold", type=float, default=0.25)
    p.add_argument("--text_threshold", type=float, default=0.20)
    p.add_argument("--gdino_model", type=str, default="IDEA-Research/grounding-dino-base")
    p.add_argument("--sam2_model", type=str, default="facebook/sam2.1-hiera-large")
    p.add_argument("--output_json", type=str, default="./openworld_eval.json")
    p.add_argument("--max_images", type=int, default=-1)
    p.add_argument(
        "--conf_threshold",
        type=float,
        default=0.001,
        help="Known-head decode conf (and YOLOE conf when --unknown_expert yoloe).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.dist_mode = False
    args.fix_random_seed = False
    args.workers = 0
    args.pin_memory = False
    args.save_root_dir = "./output"
    args.extra_tag = "eval_openworld"
    args.run_mode = "infer"

    split = get_split(args.split)
    cfgs = EasyDict(common_utils.config_loader(args.cfg_file))
    cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model
    from cfgs.data_basic import DATA_PATH_DICT

    for each in cfgs.DATA_CONFIG.DATA_INFOS:
        dataset_name = each.DATASET
        if dataset_name == "KittiDataset":
            dataset_name = "KittiDataset15"
        each.DATA_PATH = DATA_PATH_DICT.get(dataset_name, args.data_root)

    logger = common_utils.create_logger(None, rank=0)
    trainer = Trainer(args, cfgs, 0, 0, logger, None)
    model = trainer.model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    transform = build_transform_by_cfg(cfgs.DATA_CONFIG.DATA_TRANSFORM.EVALUATING)
    if args.unknown_expert == "yoloe":
        expert = YOLOEPromptExpert(
            checkpoint=args.yoloe_ckpt,
            conf_threshold=float(args.conf_threshold),
            device=device if device == "cuda" else None,
        )
        unk_thr = float(args.conf_threshold)
    else:
        expert = GroundingSAM2Expert(
            gdino_model=args.gdino_model,
            sam2_model=args.sam2_model,
            box_threshold=float(args.box_threshold),
            text_threshold=float(args.text_threshold),
            device=device,
        )
        unk_thr = float(args.box_threshold)

    data_root = Path(args.data_root)
    preds_by_image = {}
    gts_by_image = {}
    u_preds_by_image = {}
    u_gts_by_image = {}

    lines = Path(args.list_file).read_text(encoding="utf-8").strip().splitlines()
    if args.max_images > 0:
        lines = lines[: args.max_images]

    unknown_canon = {normalize_canonical_name(c) for c in split.unknown_canonical_classes}
    known_canon = {normalize_canonical_name(c) for c in split.known_classes}

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        left_rel, right_rel, _, label_rel = parts[:4]
        label_rel = label_rel.replace("\\", "/")
        raw_name = Path(label_rel).name
        raw_label = data_root / "training" / "seg_label" / raw_name
        left_path = str(data_root / left_rel)
        right_path = str(data_root / right_rel)
        image_id = Path(left_rel).stem

        out = run_openworld(
            model,
            transform,
            left_path,
            right_path,
            split,
            expert,
            list(split.unknown_prompts),
            known_conf=0.001,
            yoloe_conf=unk_thr,
            use_geometry=True,
            device=device,
        )
        left_rgb = out["left_rgb"]
        h, w = left_rgb.shape[:2]
        gts = load_gt_from_yolo_label(raw_label, (h, w))
        gts_by_image[image_id] = gts
        preds_by_image[image_id] = map_unknown_preds_to_canonical(out["fused"], split)

        u_gts = []
        for g in gts:
            if normalize_canonical_name(g.class_name) not in unknown_canon:
                continue
            for slot, canon in enumerate(split.unknown_canonical_classes):
                if normalize_canonical_name(canon) == normalize_canonical_name(g.class_name):
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
        u_gts_by_image[image_id] = u_gts
        u_preds_by_image[image_id] = [
            p for p in out["fused"] if str(p.class_name).startswith("unknown")
        ]

    known_preds = {
        k: [
            p
            for p in v
            if normalize_canonical_name(p.class_name) in known_canon
        ]
        for k, v in preds_by_image.items()
    }
    known_gts = {
        k: [
            g
            for g in v
            if normalize_canonical_name(g.class_name) in known_canon
        ]
        for k, v in gts_by_image.items()
    }
    known_ap = classwise_mask_ap(
        known_preds,
        known_gts,
        [normalize_canonical_name(c) for c in split.known_classes],
    )
    full_ap = classwise_mask_ap(
        preds_by_image,
        gts_by_image,
        ["Car", "Truck", "Bus", "Bicycle", "Person"],
    )

    u_recall_vals, u_miou_vals = [], []
    total_tp, total_gt = 0.0, 0.0
    for img_id in u_gts_by_image:
        stats = unknown_recall_and_miou(
            u_preds_by_image.get(img_id, []),
            u_gts_by_image[img_id],
            iou_thr=0.5,
            require_prompt_slot_match=True,
        )
        if not np.isnan(stats["u_recall"]):
            u_recall_vals.append(stats["u_recall"])
        if not np.isnan(stats["u_matched_miou"]):
            u_miou_vals.append(stats["u_matched_miou"])
        total_tp += stats["tp"]
        total_gt += stats["n_gt"]

    report = {
        "split": split.name,
        "num_images": len(gts_by_image),
        "known_ap": known_ap,
        "full5_ap": full_ap,
        "unknown": {
            "u_recall_macro_image": float(np.nanmean(u_recall_vals)) if u_recall_vals else None,
            "u_matched_miou_macro_image": float(np.nanmean(u_miou_vals)) if u_miou_vals else None,
            "u_recall_micro": float(total_tp / total_gt) if total_gt > 0 else None,
            "support_gt": int(total_gt),
            "tp": int(total_tp),
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
