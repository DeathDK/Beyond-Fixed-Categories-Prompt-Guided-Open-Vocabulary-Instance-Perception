#!/usr/bin/env python
"""Open-world eval on all multi-best known ckpts; report own-best metrics.

Same protocol as eval_best_suite.py, but with YOLOE+G0(+conflict) unknown branch:
  1) Run eval_openworld on each of checkpoint_best_{epe,d1_all,mAP,w_mAP}.pth
  2) Final report = best value of each open-world metric across those runs

Example:
  python tools/eval_openworld_best_suite.py \\
    --cfg_file cfgs/openworld/k4_2_leave_truck.yaml \\
    --run_dir output/KittiDataset/LightStereo/k4_2_leave_truck/k4_2_from_f5 \\
    --split K4-2 \\
    --yoloe_ckpt yoloe-26x-seg.pt \\
    --ckpt_tags best_epe \\
    --known_conf 0.001 --yoloe_conf 0.15 --use_synonyms
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, "./")

from openworld.fusion import assign_public_unknown_labels
from openworld.instances import InstancePrediction
from openworld.label_mapping import normalize_canonical_name
from openworld.metrics import (
    classwise_mask_ap,
    map_unknown_preds_to_canonical,
    unknown_recall_and_miou,
)
from openworld.prompt_expert import YOLOEPromptExpert
from openworld.vlm_expert import GroundingSAM2Expert
from openworld.splits import expand_unknown_prompts_for_yoloe, get_split
from stereo.datasets.dataset_template import build_transform_by_cfg
from stereo.modeling.models.lightstereo.trainer import Trainer
from stereo.utils import common_utils
from PIL import Image

from stereo.evaluation.metric_per_image import d1_metric, epe_metric
from tools.eval_openworld import load_gt_from_yolo_label
from tools.infer_openworld import run_openworld

BEST_CKPT_SPECS: List[Tuple[str, str]] = [
    ("best_epe", "checkpoint_best_epe.pth"),
    ("best_d1_all", "checkpoint_best_d1_all.pth"),
    ("best_mAP", "checkpoint_best_mAP.pth"),
    ("best_w_mAP", "checkpoint_best_w_mAP.pth"),
]

LOWER_BETTER = {"epe", "d1_all"}

# Final report metrics (stereo + open-world segmentation).
OW_REPORT_METRICS = (
    "epe",
    "d1_all",
    "known_mAP50_95",
    "known_w_mAP",
    "full5_mAP50_95",
    "full5_w_mAP",
    "u_recall_micro",
    "u_recall_macro_image",
    "u_matched_miou_macro_image",
    "u_recall_micro_pre_fusion",
    "u_recall_macro_image_pre_fusion",
)


def parse_args():
    p = argparse.ArgumentParser(description="Open-world multi-best suite eval")
    p.add_argument("--cfg_file", type=str, required=True)
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument(
        "--ckpt_tags",
        type=str,
        default="",
        help="Comma-separated tags to eval (e.g. best_epe). Empty = all four.",
    )
    p.add_argument("--split", type=str, default="K4-2")
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
    # Backward-compat: --conf_threshold maps to known if known_conf omitted.
    p.add_argument("--conf_threshold", type=float, default=None)
    p.add_argument(
        "--known_conf",
        type=float,
        default=0.001,
        help="Known-head decode conf (aligns with train SEG_AP_CONF).",
    )
    p.add_argument(
        "--yoloe_conf",
        type=float,
        default=0.25,
        help="YOLOE detection confidence (only when --unknown_expert yoloe).",
    )
    p.add_argument(
        "--box_threshold",
        type=float,
        default=0.10,
        help="Grounding DINO box score threshold (lower = higher recall).",
    )
    p.add_argument("--text_threshold", type=float, default=0.10)
    p.add_argument(
        "--gdino_model",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="Largest official HF Grounding DINO (base; no larger release).",
    )
    p.add_argument(
        "--sam2_model",
        type=str,
        default="facebook/sam2.1-hiera-large",
        help="Largest SAM2.1 (hiera-large).",
    )
    p.add_argument(
        "--use_synonyms",
        action="store_true",
        default=False,
        help="Expand leave-out prompts with synonym groups (default off).",
    )
    p.add_argument("--no_synonyms", action="store_true", help="Disable synonym expansion.")
    # GDINO boxes often overlap Car; prefer unknowns so fusion does not erase recall.
    p.add_argument("--mask_iou_conflict", type=float, default=0.90)
    p.add_argument("--unknown_score_boost", type=float, default=0.50)
    p.add_argument(
        "--no_fusion",
        action="store_true",
        help="Skip Known↔Unknown conflict fusion; keep G0 unknowns + known as-is.",
    )
    p.add_argument("--g0_edge_band", type=int, default=3)
    p.add_argument("--max_images", type=int, default=-1)
    p.add_argument("--max_disp", type=float, default=192.0)
    p.add_argument("--save_root_dir", type=str, default="./eval_output")
    p.add_argument("--extra_tag", type=str, default="", help="Suffix for output folder.")
    p.add_argument(
        "--checkpoint_state",
        choices=["model_state", "raw_model_state"],
        default="model_state",
    )
    args = p.parse_args()
    if args.conf_threshold is not None:
        args.known_conf = float(args.conf_threshold)
    if args.no_synonyms:
        args.use_synonyms = False
    return args


def _discover_ckpts(ckpt_dir: str, tags: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    wanted = set(tags) if tags else None
    found = []
    for tag, fname in BEST_CKPT_SPECS:
        if wanted is not None and tag not in wanted:
            continue
        path = os.path.join(ckpt_dir, fname)
        if os.path.isfile(path):
            found.append((tag, path))
    return found


def _load_ckpt(model, ckpt_path: str, state_key: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get(state_key, ckpt.get("model_state", ckpt))
    model.load_state_dict(state, strict=False)


def _aggregate_unknown_recall(
    u_preds_by_image: Dict[str, List[InstancePrediction]],
    u_gts_by_image: Dict[str, List[InstancePrediction]],
) -> Dict[str, Any]:
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
    return {
        "u_recall_macro_image": float(np.nanmean(u_recall_vals))
        if u_recall_vals
        else None,
        "u_matched_miou_macro_image": float(np.nanmean(u_miou_vals))
        if u_miou_vals
        else None,
        "u_recall_micro": float(total_tp / total_gt) if total_gt > 0 else None,
        "support_gt": int(total_gt),
        "tp": int(total_tp),
    }


def _flatten_openworld_report(report: Dict[str, Any]) -> Dict[str, float]:
    flat = {}
    stereo = report.get("stereo") or {}
    for key in ("epe", "d1_all"):
        if stereo.get(key) is not None and math.isfinite(float(stereo[key])):
            flat[key] = float(stereo[key])
    known = report.get("known_ap") or {}
    full5 = report.get("full5_ap") or {}
    unknown = report.get("unknown") or {}
    pre = report.get("unknown_pre_fusion") or {}
    if "mAP50_95" in known and known["mAP50_95"] is not None:
        flat["known_mAP50_95"] = float(known["mAP50_95"])
    if known.get("w_mAP") is not None and math.isfinite(float(known["w_mAP"])):
        flat["known_w_mAP"] = float(known["w_mAP"])
    if "mAP50_95" in full5 and full5["mAP50_95"] is not None:
        flat["full5_mAP50_95"] = float(full5["mAP50_95"])
    if full5.get("w_mAP") is not None and math.isfinite(float(full5["w_mAP"])):
        flat["full5_w_mAP"] = float(full5["w_mAP"])
    for key in (
        "u_recall_micro",
        "u_recall_macro_image",
        "u_matched_miou_macro_image",
    ):
        if unknown.get(key) is not None and math.isfinite(float(unknown[key])):
            flat[key] = float(unknown[key])
    if pre.get("u_recall_micro") is not None and math.isfinite(float(pre["u_recall_micro"])):
        flat["u_recall_micro_pre_fusion"] = float(pre["u_recall_micro"])
    if pre.get("u_recall_macro_image") is not None and math.isfinite(
        float(pre["u_recall_macro_image"])
    ):
        flat["u_recall_macro_image_pre_fusion"] = float(pre["u_recall_macro_image"])
    fusion = report.get("fusion_totals") or {}
    for key in (
        "unknown_suppressed_by_known",
        "known_to_unknown",
        "unknown_kept",
    ):
        if key in fusion:
            flat[f"fusion_{key}"] = float(fusion[key])
    return flat


def _build_own_best(per_ckpt: Dict[str, Dict[str, float]]) -> Dict[str, dict]:
    keys = set()
    for m in per_ckpt.values():
        keys.update(m.keys())
    own_best = {}
    for key in sorted(keys):
        best_val, best_tag = None, None
        lower_better = key in LOWER_BETTER
        for tag, metrics in per_ckpt.items():
            if key not in metrics:
                continue
            value = float(metrics[key])
            if not math.isfinite(value):
                continue
            if best_val is None:
                best_val, best_tag = value, tag
                continue
            better = value < best_val if lower_better else value > best_val
            if better:
                best_val, best_tag = value, tag
        if best_val is not None:
            own_best[key] = {
                "value": best_val,
                "source_ckpt": best_tag,
                "higher_is_better": not lower_better,
            }
    return own_best


def _load_disp_gt(disp_path: Path) -> Optional[np.ndarray]:
    if not disp_path.is_file():
        return None
    disp = np.array(Image.open(disp_path), dtype=np.float32) / 256.0
    return disp


def _stereo_metrics_one(disp_pred: np.ndarray, disp_gt: np.ndarray, max_disp: float = 192.0):
    """Return epe / d1_all for one image (KITTI-style valid mask)."""
    if disp_pred.shape != disp_gt.shape:
        h = min(disp_pred.shape[0], disp_gt.shape[0])
        w = min(disp_pred.shape[1], disp_gt.shape[1])
        disp_pred = disp_pred[:h, :w]
        disp_gt = disp_gt[:h, :w]
    pred_t = torch.from_numpy(disp_pred.astype(np.float32))[None]
    gt_t = torch.from_numpy(disp_gt.astype(np.float32))[None]
    mask = (gt_t > 0) & (gt_t < max_disp)
    if int(mask.sum().item()) == 0:
        return None, None
    epe = float(epe_metric(pred_t, gt_t, mask)[0].item())
    d1 = float(d1_metric(pred_t, gt_t, mask)[0].item())
    return epe, d1


def evaluate_one_ckpt(
    model,
    transform,
    expert,
    split,
    args,
    device: str,
    prompts: Tuple[str, ...],
    prompt_slots: Tuple[int, ...],
) -> Dict[str, Any]:
    data_root = Path(args.data_root)
    preds_by_image = {}
    gts_by_image = {}
    u_preds_by_image = {}
    u_preds_pre_by_image = {}
    u_gts_by_image = {}
    epe_vals: List[float] = []
    d1_vals: List[float] = []
    fusion_totals = {
        "known_to_unknown": 0,
        "unknown_suppressed_by_known": 0,
        "unknown_kept": 0,
        "known_kept": 0,
    }

    lines = Path(args.list_file).read_text(encoding="utf-8").strip().splitlines()
    if args.max_images > 0:
        lines = lines[: args.max_images]

    unknown_canon = {
        normalize_canonical_name(c) for c in split.unknown_canonical_classes
    }
    known_canon = {normalize_canonical_name(c) for c in split.known_classes}
    max_disp = float(getattr(args, "max_disp", 192.0))

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        left_rel, right_rel, disp_rel, label_rel = parts[:4]
        label_rel = label_rel.replace("\\", "/")
        disp_rel = disp_rel.replace("\\", "/")
        raw_name = Path(label_rel).name
        raw_label = data_root / "training" / "seg_label" / raw_name
        left_path = str(data_root / left_rel)
        right_path = str(data_root / right_rel)
        image_id = Path(left_rel).stem

        unk_thr = (
            float(args.yoloe_conf)
            if getattr(args, "unknown_expert", "gdino_sam2") == "yoloe"
            else float(args.box_threshold)
        )
        out = run_openworld(
            model,
            transform,
            left_path,
            right_path,
            split,
            expert,
            list(prompts),
            known_conf=float(args.known_conf),
            yoloe_conf=unk_thr,
            prompt_slots=list(prompt_slots),
            use_geometry=True,
            geometry_mode="g0",
            g0_edge_band_pixels=int(args.g0_edge_band),
            mask_iou_conflict=float(args.mask_iou_conflict),
            unknown_score_boost=float(args.unknown_score_boost),
            skip_fusion=bool(getattr(args, "no_fusion", False)),
            device=device,
        )
        left_rgb = out["left_rgb"]
        h, w = left_rgb.shape[:2]
        gts = load_gt_from_yolo_label(raw_label, (h, w))
        gts_by_image[image_id] = gts
        preds_by_image[image_id] = map_unknown_preds_to_canonical(out["fused"], split)

        for k in fusion_totals:
            fusion_totals[k] += int(out["stats"].get(k, 0))

        disp_gt = _load_disp_gt(data_root / disp_rel)
        if disp_gt is not None and out.get("disparity") is not None:
            epe_i, d1_i = _stereo_metrics_one(
                np.asarray(out["disparity"], dtype=np.float32),
                disp_gt,
                max_disp=max_disp,
            )
            if epe_i is not None:
                epe_vals.append(epe_i)
            if d1_i is not None:
                d1_vals.append(d1_i)

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
        u_gts_by_image[image_id] = u_gts
        u_preds_by_image[image_id] = [
            p for p in out["fused"] if str(p.class_name).startswith("unknown")
        ]
        u_preds_pre_by_image[image_id] = assign_public_unknown_labels(
            list(out["unknown"]), split
        )

    known_preds = {
        k: [p for p in v if normalize_canonical_name(p.class_name) in known_canon]
        for k, v in preds_by_image.items()
    }
    known_gts = {
        k: [g for g in v if normalize_canonical_name(g.class_name) in known_canon]
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

    unknown_post = _aggregate_unknown_recall(u_preds_by_image, u_gts_by_image)
    unknown_pre = _aggregate_unknown_recall(u_preds_pre_by_image, u_gts_by_image)

    return {
        "split": split.name,
        "num_images": len(gts_by_image),
        "settings": {
            "known_conf": float(args.known_conf),
            "yoloe_conf": float(args.yoloe_conf),
            "use_synonyms": bool(args.use_synonyms),
            "prompts": list(prompts),
            "prompt_slots": list(prompt_slots),
            "mask_iou_conflict": float(args.mask_iou_conflict),
            "unknown_score_boost": float(args.unknown_score_boost),
            "g0_edge_band": int(args.g0_edge_band),
            "no_fusion": bool(getattr(args, "no_fusion", False)),
        },
        "stereo": {
            "epe": float(np.mean(epe_vals)) if epe_vals else None,
            "d1_all": float(np.mean(d1_vals)) if d1_vals else None,
            "num_images_with_disp": len(epe_vals),
        },
        "known_ap": known_ap,
        "full5_ap": full_ap,
        "unknown": unknown_post,
        "unknown_pre_fusion": unknown_pre,
        "fusion_totals": fusion_totals,
    }


def main():
    args = parse_args()
    args.dist_mode = False
    args.fix_random_seed = False
    args.workers = 0
    args.pin_memory = False
    args.save_root_dir_train = "./output"
    args.extra_tag_train = "eval_openworld_best_suite"
    args.run_mode = "infer"

    ckpt_dir = args.ckpt_dir or os.path.join(args.run_dir, "ckpt")
    ckpt_dir = os.path.abspath(ckpt_dir)
    tag_filter = (
        [t.strip() for t in args.ckpt_tags.split(",") if t.strip()]
        if args.ckpt_tags
        else None
    )
    found = _discover_ckpts(ckpt_dir, tag_filter)
    if not found:
        raise FileNotFoundError(f"No best ckpts under {ckpt_dir} (tags={tag_filter})")

    run_name = os.path.basename(os.path.abspath(args.run_dir.rstrip("\\/")))
    if args.extra_tag:
        run_name = f"{run_name}_{args.extra_tag}"
    out_dir = Path(args.save_root_dir) / "openworld_best_suite" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / (
        "eval_ow_best_suite_%s.log"
        % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = common_utils.create_logger(str(log_file), rank=0)

    split = get_split(args.split)
    prompts, prompt_slots = expand_unknown_prompts_for_yoloe(
        split, use_synonyms=bool(args.use_synonyms)
    )
    logger.info(
        "Settings: expert=%s known_conf=%.4f box/yoloe_thr=%.4f text_thr=%.4f "
        "synonyms=%s prompts=%s conflict=%.2f boost=%.2f g0_band=%d no_fusion=%s",
        args.unknown_expert,
        args.known_conf,
        args.box_threshold
        if args.unknown_expert == "gdino_sam2"
        else args.yoloe_conf,
        args.text_threshold,
        args.use_synonyms,
        list(prompts),
        args.mask_iou_conflict,
        args.unknown_score_boost,
        args.g0_edge_band,
        bool(args.no_fusion),
    )

    cfgs = EasyDict(common_utils.config_loader(args.cfg_file))
    cfgs.MODEL.PRETRAINED_MODEL = ""
    cfgs.MODEL.YOLO_PRETRAINED_MODEL = None
    if "F5_INIT" in cfgs.MODEL:
        cfgs.MODEL.F5_INIT.ENABLED = False
    cfgs.MODEL.CHECKPOINT_STATE = args.checkpoint_state

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
        extra_tag="eval_openworld_best_suite",
        fix_random_seed=False,
    )
    trainer = Trainer(train_args, cfgs, 0, 0, logger, None)
    model = trainer.model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    transform = build_transform_by_cfg(cfgs.DATA_CONFIG.DATA_TRANSFORM.EVALUATING)
    if args.unknown_expert == "yoloe":
        expert = YOLOEPromptExpert(
            checkpoint=args.yoloe_ckpt,
            conf_threshold=float(args.yoloe_conf),
            device=device if device == "cuda" else None,
        )
    else:
        expert = GroundingSAM2Expert(
            gdino_model=args.gdino_model,
            sam2_model=args.sam2_model,
            box_threshold=float(args.box_threshold),
            text_threshold=float(args.text_threshold),
            device=device,
        )

    per_ckpt_raw = {}
    per_ckpt_flat = {}
    for tag, path in found:
        logger.info("=" * 60)
        logger.info("Open-world eval on %s (%s)", tag, path)
        logger.info("=" * 60)
        _load_ckpt(model, path, args.checkpoint_state)
        raw = evaluate_one_ckpt(
            model, transform, expert, split, args, device, prompts, prompt_slots
        )
        flat = _flatten_openworld_report(raw)
        per_ckpt_raw[tag] = raw
        per_ckpt_flat[tag] = flat
        logger.info("Flat metrics [%s]: %s", tag, flat)
        logger.info(
            "  pre_fusion recall_micro=%.4f  post=%.4f  suppressed=%s",
            flat.get("u_recall_micro_pre_fusion", float("nan")),
            flat.get("u_recall_micro", float("nan")),
            flat.get("fusion_unknown_suppressed_by_known", 0),
        )
        with open(out_dir / f"{tag}_openworld.json", "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, default=str)

    own_best = _build_own_best(per_ckpt_flat)
    final_report = {
        metric: own_best[metric]
        for metric in OW_REPORT_METRICS
        if metric in own_best
    }

    report = {
        "protocol": (
            "Run open-world eval (default GDINO+SAM2+G0) on each multi-best "
            "known ckpt (best_epe / best_d1_all / best_mAP / best_w_mAP); "
            "final report takes the best value of each metric across ckpts "
            "(epe/d1_all lower-better; others higher-better). "
            "Not a single model — each line cites its source_ckpt."
        ),
        "cfg_file": args.cfg_file,
        "run_dir": os.path.abspath(args.run_dir),
        "ckpt_dir": ckpt_dir,
        "split": args.split,
        "yoloe_ckpt": args.yoloe_ckpt,
        "settings": {
            "known_conf": float(args.known_conf),
            "yoloe_conf": float(args.yoloe_conf),
            "use_synonyms": bool(args.use_synonyms),
            "prompts": list(prompts),
            "mask_iou_conflict": float(args.mask_iou_conflict),
            "unknown_score_boost": float(args.unknown_score_boost),
            "g0_edge_band": int(args.g0_edge_band),
            "no_fusion": bool(args.no_fusion),
            "ckpt_tags": tag_filter,
        },
        "per_checkpoint_flat": per_ckpt_flat,
        "per_checkpoint_raw": per_ckpt_raw,
        "final_report": final_report,
        "own_best_all_metrics": own_best,
    }
    out_json = out_dir / "openworld_best_suite_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL OPEN-WORLD REPORT (own-best across selected ckpts)")
    logger.info("=" * 60)
    for metric, item in final_report.items():
        logger.info("  %-36s = %.6f  <- %s", metric, item["value"], item["source_ckpt"])
    logger.info("Wrote %s", out_json)


if __name__ == "__main__":
    main()
