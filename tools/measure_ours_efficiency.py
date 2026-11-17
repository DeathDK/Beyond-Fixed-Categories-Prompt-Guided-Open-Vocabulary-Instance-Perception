"""Measure FLOPs / Params / runtime for the known LightStereo backbone (Ours).

Usage (Death env):
  python tools/measure_ours_efficiency.py --cfg_file cfgs/openworld/f5.yaml
  python tools/measure_ours_efficiency.py --cfg_file cfgs/openworld/k4_1_leave_car.yaml

Notes:
  - Counts the trainable multi-task network only (same convention as LSI tables).
  - Frozen GDINO/SAM2 and rule-based G0/Fusion are reported separately (params/time if requested).
  - Runtime uses CUDA events + synchronize on the current GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stereo.modeling.models.lightstereo.trainer import Trainer  # noqa: E402
from stereo.utils import common_utils  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", type=str, default="cfgs/openworld/f5.yaml")
    p.add_argument("--h", type=int, default=384, help="eval pad H (KITTI RightTopPad)")
    p.add_argument("--w", type=int, default=1248, help="eval pad W")
    p.add_argument("--legacy_shape", action="store_true", help="also measure 544x960")
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--out", type=str, default="experiment/exp1_efficiency_accuracy/data/ours_efficiency.json")
    return p.parse_args()


def count_params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_m": total / 1e6,
        "trainable_params_m": trainable / 1e6,
        "params": int(total),
        "trainable_params": int(trainable),
    }


@torch.no_grad()
def measure_flops(model, shape_hw, device) -> float | None:
    try:
        import thop
    except ImportError:
        print("[warn] thop not installed")
        return None
    h, w = shape_hw
    inputs = {
        "left": torch.randn(1, 3, h, w, device=device),
        "right": torch.randn(1, 3, h, w, device=device),
    }
    model.eval()
    try:
        flops, _ = thop.profile(model, inputs=(inputs,), verbose=False)
        return float(flops / 1e9)
    except Exception as e:
        print(f"[warn] thop.profile failed: {e}")
        return None


@torch.no_grad()
def measure_runtime_ms(model, shape_hw, device, reps=100, warmup=20) -> dict:
    h, w = shape_hw
    inputs = {
        "left": torch.randn(1, 3, h, w, device=device),
        "right": torch.randn(1, 3, h, w, device=device),
    }
    model.eval()
    for _ in range(warmup):
        _ = model(inputs)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        starter.record()
        _ = model(inputs)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms
    arr = torch.tensor(times)
    return {
        "time_ms_mean": float(arr.mean()),
        "time_ms_std": float(arr.std()),
        "time_ms_median": float(arr.median()),
        "reps": reps,
        "warmup": warmup,
    }


def build_model(cfg_file: str, device: torch.device):
    yaml_config = common_utils.config_loader(cfg_file)
    cfgs = EasyDict(yaml_config)
    args = EasyDict(
        {
            "cfg_file": cfg_file,
            "dist_mode": False,
            "run_mode": "measure",
            "extra_tag": "measure",
            "pretrained_model": None,
            "ckpt": None,
        }
    )
    # Avoid loading missing YOLO/F5 weights for structure-only measure if possible.
    if "MODEL" in cfgs:
        # Keep structure; empty pretrained so random init is fine for FLOPs/time.
        cfgs.MODEL.PRETRAINED_MODEL = ""
        cfgs.MODEL.YOLO_PRETRAINED_MODEL = ""
        if "F5_INIT" in cfgs.MODEL:
            cfgs.MODEL.F5_INIT.ENABLED = False
    trainer = Trainer(args, cfgs, local_rank=0, global_rank=0, logger=None, tb_writer=None)
    model = trainer.model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model = model.to(device).eval()
    return model, cfgs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for runtime measurement")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"cfg: {args.cfg_file}")

    model, cfgs = build_model(args.cfg_file, device)
    nc = int(cfgs.MODEL.get("NUM_CLASSES", -1))
    param_info = count_params(model)
    print(
        f"Params: {param_info['params_m']:.4f} M "
        f"(trainable {param_info['trainable_params_m']:.4f} M), NUM_CLASSES={nc}"
    )

    shapes = [(args.h, args.w)]
    if args.legacy_shape:
        shapes.append((544, 960))

    results = {
        "gpu": gpu_name,
        "cfg_file": args.cfg_file,
        "num_classes": nc,
        "params": param_info,
        "shapes": {},
        "notes": [
            "FLOPs/Params/runtime are for the trainable LightStereo multi-task network only.",
            "Frozen Grounding DINO + SAM2 and rule-based G0/Fusion are not included in FLOPs.",
            "G0/Fusion have 0 learnable parameters.",
        ],
    }

    for hw in shapes:
        print(f"\n=== Input 1x3x{hw[0]}x{hw[1]} ===")
        flops_g = measure_flops(model, hw, device)
        print(f"FLOPs: {flops_g:.4f} G" if flops_g is not None else "FLOPs: N/A")
        rt = measure_runtime_ms(model, hw, device, reps=args.reps, warmup=args.warmup)
        print(
            f"Runtime: {rt['time_ms_mean']:.3f} ± {rt['time_ms_std']:.3f} ms "
            f"(median {rt['time_ms_median']:.3f})"
        )
        results["shapes"][f"{hw[0]}x{hw[1]}"] = {
            "flops_g": flops_g,
            "runtime": rt,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
