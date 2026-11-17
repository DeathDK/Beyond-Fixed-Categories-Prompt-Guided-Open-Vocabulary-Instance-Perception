#!/usr/bin/env python
"""Fixed unknown-module migration eval across K4¡úK1 (same YOLOE/PG-GUIC).

Does not retrain the unknown expert. Only swaps known-model checkpoints and
unknown_prompts from each split. Writes a table skeleton JSON for the paper.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# Example subset; full combinatorial matrix is in cfgs/openworld/TRAINING_MATRIX.md
SPLIT_CFG = {
    "K4-2": "cfgs/openworld/k4_2_leave_truck.yaml",
    "K3-5": "cfgs/openworld/k3_5_leave_truck_bus.yaml",
    "K2-7": "cfgs/openworld/k2_7_leave_truck_bus_bicycle.yaml",
    "K1-5": "cfgs/openworld/k1_5_leave_truck_bus_bicycle_person.yaml",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpts",
        type=str,
        required=True,
        help="JSON map split->ckpt, e.g. '{\"K4\":\"a.pth\",\"K3\":\"b.pth\"}'",
    )
    p.add_argument("--yoloe_ckpt", type=str, default="yoloe-26x-seg.pt")
    p.add_argument("--output_json", type=str, default="./openworld_migrate_table.json")
    p.add_argument("--max_images", type=int, default=20)
    p.add_argument("--list_file", type=str, default="./data/KITTI15/kitti15_val20_with_seglabel.txt")
    return p.parse_args()


def main():
    args = parse_args()
    ckpts = json.loads(args.ckpts)
    rows = {}
    root = Path(__file__).resolve().parents[1]
    for split, cfg in SPLIT_CFG.items():
        ckpt = ckpts.get(split)
        if not ckpt:
            rows[split] = {"status": "missing_ckpt"}
            continue
        out_json = Path(args.output_json).with_name(f"migrate_{split.lower()}.json")
        cmd = [
            sys.executable,
            str(root / "tools" / "eval_openworld.py"),
            "--cfg_file",
            cfg,
            "--pretrained_model",
            ckpt,
            "--split",
            split,
            "--yoloe_ckpt",
            args.yoloe_ckpt,
            "--list_file",
            args.list_file,
            "--max_images",
            str(args.max_images),
            "--output_json",
            str(out_json),
        ]
        print(" ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(root))
        if proc.returncode != 0:
            rows[split] = {"status": "failed", "returncode": proc.returncode}
            continue
        if out_json.exists():
            rows[split] = json.loads(out_json.read_text(encoding="utf-8"))
            rows[split]["status"] = "ok"
        else:
            rows[split] = {"status": "no_output"}

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
