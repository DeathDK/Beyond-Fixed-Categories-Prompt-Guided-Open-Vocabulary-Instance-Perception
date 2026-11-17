#!/usr/bin/env python
"""Contrastive entry: naive YOLOE / +G0 / +G1 / +V1 on the same stereo pair.

Records t_main / t_open_vocab / t_geometry / t_fusion / t_3D for each mode.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", type=str, default="cfgs/openworld/known4.yaml")
    p.add_argument("--pretrained_model", type=str, required=True)
    p.add_argument("--left_img_path", type=str, required=True)
    p.add_argument("--right_img_path", type=str, required=True)
    p.add_argument("--yoloe_ckpt", type=str, default="yoloe-26x-seg.pt")
    p.add_argument("--g1_ckpt", type=str, default="")
    p.add_argument("--v1_ckpt", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./openworld_ablation")
    p.add_argument("--modes", type=str, default="naive,g0,g1,v1")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    summary = {}
    for mode in modes:
        out_dir = Path(args.output_dir) / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(root / "tools" / "infer_openworld.py"),
            "--cfg_file",
            args.cfg_file,
            "--pretrained_model",
            args.pretrained_model,
            "--left_img_path",
            args.left_img_path,
            "--right_img_path",
            args.right_img_path,
            "--yoloe_ckpt",
            args.yoloe_ckpt,
            "--output_dir",
            str(out_dir),
            "--ablation",
            mode if mode != "naive" else "naive",
        ]
        if mode == "g1" and args.g1_ckpt:
            cmd += ["--g1_ckpt", args.g1_ckpt]
        if mode == "v1" and args.v1_ckpt:
            cmd += ["--v1_ckpt", args.v1_ckpt]
        if mode == "v1" and not args.v1_ckpt:
            print(f"[skip] v1 requires --v1_ckpt")
            summary[mode] = {"status": "skipped_no_v1_ckpt"}
            continue
        if mode == "g1" and not args.g1_ckpt:
            print("[warn] g1 without --g1_ckpt falls back to G0 inside refiner")
        print(" ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(root))
        # Collect timing from result json if present
        results = list(out_dir.glob("*_result.json"))
        if proc.returncode != 0 or not results:
            summary[mode] = {"status": "failed", "returncode": proc.returncode}
            continue
        payload = json.loads(results[0].read_text(encoding="utf-8"))
        timing = payload.get("timing_ms", {})
        summary[mode] = {
            "status": "ok",
            "t_main": timing.get("main"),
            "t_open_vocab": timing.get("open_vocab"),
            "t_geometry": timing.get("geometry"),
            "t_fusion": timing.get("fusion"),
            "t_3D": timing.get("pointcloud"),
            "num_instances": len(payload.get("instances", [])),
            "fusion_stats": payload.get("fusion_stats"),
        }
    out = Path(args.output_dir) / "ablation_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
