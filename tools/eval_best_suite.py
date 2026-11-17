#!/usr/bin/env python
"""Evaluate selected best checkpoint(s) under a training run.

Paper tables: pass --ckpt_tags best_mAP so every metric comes from the same
weights (checkpoint_best_mAP.pth). Multiple tags are only for optional
side-by-side comparison; do not mix them into one paper table row.

Example (paper):
  python tools/eval_best_suite.py \\
    --cfg_file cfgs/openworld/k4_2_leave_truck.yaml \\
    --run_dir output/KittiDataset/LightStereo/k4_2_leave_truck/k4_2_from_f5 \\
    --ckpt_tags best_mAP
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

REPORT_METRICS = ("epe", "d1_all", "mAP", "w_mAP")
LOWER_BETTER = {"epe", "d1_all", "thres_1", "thres_2", "thres_3"}

BEST_CKPT_SPECS: List[Tuple[str, str]] = [
    ("best_epe", "checkpoint_best_epe.pth"),
    ("best_d1_all", "checkpoint_best_d1_all.pth"),
    ("best_mAP", "checkpoint_best_mAP.pth"),
    ("best_w_mAP", "checkpoint_best_w_mAP.pth"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate selected checkpoint(s)")
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
        "--ckpt_tags",
        type=str,
        default="best_mAP",
        help="Comma-separated tags to eval (default: best_mAP). "
        "Empty string = all four files if present.",
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


def _parse_tags(raw: str) -> List[str]:
    text = str(raw).strip()
    if not text:
        return [tag for tag, _ in BEST_CKPT_SPECS]
    return [t.strip() for t in text.split(",") if t.strip()]


def _discover_ckpts(ckpt_dir: str, tags: List[str]) -> List[Tuple[str, str]]:
    wanted = set(tags)
    found = []
    for tag, fname in BEST_CKPT_SPECS:
        if tag not in wanted:
            continue
        path = os.path.join(ckpt_dir, fname)
        if os.path.isfile(path):
            found.append((tag, path))
    return found


def build_report_from_single_or_compare(per_ckpt: Dict[str, Dict[str, float]]) -> Dict[str, dict]:
    """If one ckpt: use its metrics. If several: keep a compare matrix only."""
    if len(per_ckpt) == 1:
        tag, metrics = next(iter(per_ckpt.items()))
        out = {}
        for key, value in metrics.items():
            if value is None or not math.isfinite(float(value)):
                continue
            out[key] = {
                "value": float(value),
                "source_ckpt": tag,
                "higher_is_better": not _is_lower_better(key),
            }
        return out

    # Multi-ckpt compare: still pick a readable summary, but source_ckpt is labeled.
    keys = set()
    for metrics in per_ckpt.values():
        keys.update(metrics.keys())
    summary = {}
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
            summary[key] = {
                "value": best_val,
                "source_ckpt": best_tag,
                "higher_is_better": not _is_lower_better(key),
                "note": "compare-only; not used for paper tables",
            }
    return summary


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

    tags = _parse_tags(args.ckpt_tags)
    found = _discover_ckpts(ckpt_dir, tags)
    if not found:
        raise FileNotFoundError(
            f"No matching checkpoints under {ckpt_dir} for tags={tags}. "
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
    logger.info("Selected checkpoints:")
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

    summary = build_report_from_single_or_compare(per_ckpt)
    final_report = {
        metric: summary[metric]
        for metric in REPORT_METRICS
        if metric in summary
    }

    single = len(per_ckpt) == 1
    report = {
        "protocol": (
            "Single-checkpoint report (paper): all metrics from one selected ckpt."
            if single
            else "Multi-checkpoint compare only; paper tables should use one --ckpt_tags."
        ),
        "cfg_file": args.cfg_file,
        "run_dir": os.path.abspath(args.run_dir),
        "ckpt_dir": ckpt_dir,
        "ckpt_tags": tags,
        "checkpoint_state": args.checkpoint_state,
        "per_checkpoint": per_ckpt,
        "final_report": final_report,
        "summary_all_metrics": summary,
    }

    out_json = os.path.join(args.output_dir, "best_suite_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", out_json)

    logger.info("")
    logger.info("=" * 60)
    if single:
        logger.info("FINAL REPORT (single checkpoint: %s)", next(iter(per_ckpt)))
    else:
        logger.info("COMPARE SUMMARY (not for paper tables)")
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
