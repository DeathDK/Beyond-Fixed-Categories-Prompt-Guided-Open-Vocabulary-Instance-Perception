#!/usr/bin/env python
"""Evaluate all independent best checkpoints and report per-metric own-best.

Protocol (matches multi-best saving):
  1) Run val on every available best ckpt (epe / d1_all / mAP / w_mAP).
  2) Final report = for each metric, take the best value among those ckpts
     (lower-better for epe/d1/thres; higher-better for mAP/w_mAP).
  3) Also keep the full ckpt x metric matrix for transparency.

Example:
  python tools/eval_best_suite.py \\
    --cfg_file cfgs/openworld/k4_2_leave_truck.yaml \\
    --run_dir output/KittiDataset/LightStereo/k4_2_leave_truck/k4_2_from_f5
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import torch
from easydict import EasyDict

sys.path.insert(0, "./")

from stereo.modeling.models.lightstereo.trainer import Trainer
from stereo.utils import common_utils

# Core four metrics the user tracks + extras computed in val.
REPORT_METRICS = ("epe", "d1_all", "mAP", "w_mAP")
LOWER_BETTER = {"epe", "d1_all", "thres_1", "thres_2", "thres_3"}

BEST_CKPT_SPECS: List[Tuple[str, str]] = [
    ("best_epe", "checkpoint_best_epe.pth"),
    ("best_d1_all", "checkpoint_best_d1_all.pth"),
    ("best_mAP", "checkpoint_best_mAP.pth"),
    ("best_w_mAP", "checkpoint_best_w_mAP.pth"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multi-best checkpoint suite")
    parser.add_argument("--cfg_file", type=str, required=True)
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Training run directory that contains ckpt/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Optional explicit ckpt dir (default: <run_dir>/ckpt)",
    )
    parser.add_argument(
        "--checkpoint_state",
        choices=["model_state", "raw_model_state"],
        default="model_state",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--save_root_dir", type=str, default="./eval_output")
    args = parser.parse_args()
    args.dist_mode = False
    args.run_mode = "eval"
    return args


def _to_float(v) -> float:
    if torch.is_tensor(v):
        return float(v.item())
    return float(v)


def _is_lower_better(metric: str) -> bool:
    return metric in LOWER_BETTER or metric.startswith("thres")


def _load_ckpt_into_model(model, ckpt_path: str, state_key: str, logger) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if state_key in ckpt:
        state = ckpt[state_key]
    elif "model_state" in ckpt:
        state = ckpt["model_state"]
        logger.warning("%s missing %s; falling back to model_state", ckpt_path, state_key)
    else:
        raise KeyError(f"No usable state in {ckpt_path}; keys={list(ckpt.keys())}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(
        "Loaded %s (%s); missing=%d unexpected=%d epoch=%s",
        ckpt_path,
        state_key,
        len(missing),
        len(unexpected),
        ckpt.get("epoch", "?"),
    )


def _discover_ckpts(ckpt_dir: str) -> List[Tuple[str, str]]:
    found = []
    for tag, fname in BEST_CKPT_SPECS:
        path = os.path.join(ckpt_dir, fname)
        if os.path.isfile(path):
            found.append((tag, path))
    return found


def build_own_best_report(per_ckpt: Dict[str, Dict[str, float]]) -> Dict[str, dict]:
    """For each metric, pick the best value across all evaluated ckpts."""
    keys = set()
    for metrics in per_ckpt.values():
        keys.update(metrics.keys())

    own_best = {}
    for key in sorted(keys):
        best_val = None
        best_tag = None
        for tag, metrics in per_ckpt.items():
            if key not in metrics:
                continue
            value = metrics[key]
            if value is None or not math.isfinite(float(value)):
                continue
            value = float(value)
            if best_val is None:
                best_val, best_tag = value, tag
                continue
            better = value < best_val if _is_lower_better(key) else value > best_val
            if better:
                best_val, best_tag = value, tag
        if best_val is not None:
            own_best[key] = {
                "value": best_val,
                "source_ckpt": best_tag,
                "higher_is_better": not _is_lower_better(key),
            }
    return own_best


def main():
    args = parse_args()
    cfgs = EasyDict(common_utils.config_loader(args.cfg_file))
    cfgs.MODEL.CHECKPOINT_STATE = args.checkpoint_state
    cfgs.MODEL.PRETRAINED_MODEL = ""
    cfgs.MODEL.YOLO_PRETRAINED_MODEL = None
    if "F5_INIT" in cfgs.MODEL:
        cfgs.MODEL.F5_INIT.ENABLED = False

    ckpt_dir = args.ckpt_dir or os.path.join(args.run_dir, "ckpt")
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"ckpt dir not found: {ckpt_dir}")

    found = _discover_ckpts(ckpt_dir)
    if not found:
        raise FileNotFoundError(
            f"No best checkpoints under {ckpt_dir}. "
            "Expected checkpoint_best_epe/d1_all/mAP/w_mAP.pth"
        )

    run_name = os.path.basename(os.path.abspath(args.run_dir.rstrip("\\/")))
    args.output_dir = os.path.abspath(
        os.path.join(args.save_root_dir, "best_suite", run_name)
    )
    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(
        args.output_dir,
        "eval_best_suite_%s.log" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    logger = common_utils.create_logger(log_file, rank=0)
    for key, val in vars(args).items():
        logger.info("%-20s %s", key, val)
    logger.info("Discovered checkpoints:")
    for tag, path in found:
        logger.info("  %-14s %s", tag, path)

    torch.cuda.set_device(0)
    common_utils.set_random_seed(0)

    trainer = Trainer(args, cfgs, 0, 0, logger, None)
    model = trainer.model
    model.eval()

    per_ckpt: Dict[str, Dict[str, float]] = {}
    for tag, path in found:
        logger.info("=" * 60)
        logger.info("Evaluating %s", tag)
        logger.info("=" * 60)
        _load_ckpt_into_model(model, path, args.checkpoint_state, logger)
        metrics = trainer.evaluate(current_epoch=0)
        per_ckpt[tag] = {k: _to_float(v) for k, v in metrics.items()}
        logger.info("Results [%s]: %s", tag, per_ckpt[tag])

    own_best = build_own_best_report(per_ckpt)
    final_report = {
        metric: own_best[metric]
        for metric in REPORT_METRICS
        if metric in own_best
    }

    report = {
        "protocol": (
            "Evaluate all multi-best ckpts; final numbers take the best value "
            "of each metric across ckpts (own-best suite report)."
        ),
        "cfg_file": args.cfg_file,
        "run_dir": os.path.abspath(args.run_dir),
        "ckpt_dir": ckpt_dir,
        "checkpoint_state": args.checkpoint_state,
        "per_checkpoint": per_ckpt,
        "final_report": final_report,
        "own_best_all_metrics": own_best,
    }

    out_json = os.path.join(args.output_dir, "best_suite_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", out_json)

    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL REPORT (own-best across 4 ckpts)")
    logger.info("=" * 60)
    for metric in REPORT_METRICS:
        if metric not in final_report:
            continue
        item = final_report[metric]
        logger.info(
            "  %-8s = %.6f  <- %s",
            metric,
            item["value"],
            item["source_ckpt"],
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
