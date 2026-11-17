"""Measure full open-world system Params / runtime (+ FLOPs when possible).

Includes: LightStereo known backbone + frozen Grounding DINO-base + SAM2.1-large
+ G0 + Fusion.

  python tools/measure_full_system_efficiency.py \\
      --cfg_file cfgs/openworld/k4_1_leave_car.yaml \\
      --ckpt pretrained/f5_checkpoint_best.pth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from easydict import EasyDict
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openworld.fusion import fuse_known_unknown  # noqa: E402
from openworld.geometry_refine import refine_unknown_instances_g0  # noqa: E402
from openworld.known_decode import decode_known_instances  # noqa: E402
from openworld.label_mapping import AnnotationSchema  # noqa: E402
from openworld.splits import get_split  # noqa: E402
from openworld.vlm_expert import GroundingSAM2Expert  # noqa: E402
from stereo.datasets.dataset_template import build_transform_by_cfg  # noqa: E402
from stereo.modeling.models.lightstereo.trainer import Trainer  # noqa: E402
from stereo.utils import common_utils  # noqa: E402
from tools.infer_openworld import preprocess_pair  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", default="cfgs/openworld/k4_1_leave_car.yaml")
    p.add_argument("--ckpt", default="pretrained/f5_checkpoint_best.pth")
    p.add_argument("--list_file", default="data/KITTI15/kitti15_val20_with_seglabel.txt")
    p.add_argument("--data_root", default="data/KITTI15")
    p.add_argument("--split", default="K4-1")
    p.add_argument("--num_images", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps_per_image", type=int, default=1)
    p.add_argument("--known_conf", type=float, default=0.25)
    p.add_argument("--box_threshold", type=float, default=0.25)
    p.add_argument("--text_threshold", type=float, default=0.20)
    p.add_argument(
        "--out",
        default="experiment/exp1_efficiency_accuracy/data/ours_full_system_efficiency.json",
    )
    return p.parse_args()


def count_params(module: torch.nn.Module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


class _NullLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def exception(self, *a, **k):
        pass


def build_known_model(cfg_file: str, ckpt: str, device: str):
    yaml_config = common_utils.config_loader(cfg_file)
    cfgs = EasyDict(yaml_config)
    args = EasyDict(
        {
            "cfg_file": cfg_file,
            "dist_mode": False,
            "run_mode": "measure",
            "extra_tag": "measure_full",
            "pretrained_model": None,
            "ckpt": None,
        }
    )
    # Structure-only for speed; skip weight load to avoid logger/ckpt coupling.
    cfgs.MODEL.PRETRAINED_MODEL = ""
    cfgs.MODEL.YOLO_PRETRAINED_MODEL = ""
    if "F5_INIT" in cfgs.MODEL:
        cfgs.MODEL.F5_INIT.ENABLED = False
    trainer = Trainer(
        args, cfgs, local_rank=0, global_rank=0, logger=_NullLogger(), tb_writer=None
    )
    model = trainer.model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    # Optional weight load (skip shape-mismatched tensors; timing does not need them).
    if ckpt and Path(ckpt).is_file():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        model_sd = model.state_dict()
        filtered = {
            k: v
            for k, v in state.items()
            if k in model_sd and getattr(v, "shape", None) == model_sd[k].shape
        }
        model.load_state_dict(filtered, strict=False)
    return model.to(device).eval(), cfgs


def load_image_pairs(list_file: str, data_root: str, n: int):
    pairs = []
    root = Path(data_root)
    for line in Path(list_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        left = root / parts[0]
        right = root / parts[1]
        if left.is_file() and right.is_file():
            pairs.append((str(left), str(right)))
        if len(pairs) >= n:
            break
    return pairs


@torch.no_grad()
def time_one(
    model,
    transform,
    expert: GroundingSAM2Expert,
    left_path,
    right_path,
    split,
    prompts,
    device,
    known_conf: float = 0.25,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
):
    left_t, right_t, orig_hw, left_rgb = preprocess_pair(left_path, right_path, transform)
    left_t = left_t.to(device)
    right_t = right_t.to(device)
    known_names = list(AnnotationSchema(split.known_classes).known_classes)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pred = model({"left": left_t, "right": right_t})
    torch.cuda.synchronize()
    t_main = (time.perf_counter() - t0) * 1000.0

    disp, known_inst = decode_known_instances(
        pred, class_names=known_names, orig_hw=orig_hw, conf_threshold=known_conf
    )

    left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    unknown_raw = expert.predict_instances(
        left_bgr,
        prompts=prompts,
        image_size_hw=orig_hw,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    torch.cuda.synchronize()
    t_ov = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    unknown_ref = refine_unknown_instances_g0(unknown_raw, disp, edge_band_pixels=3)
    t_g0 = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    fused, _ = fuse_known_unknown(
        known_inst,
        unknown_ref,
        split,
        mask_iou_conflict=0.90,
        unknown_score_boost=0.50,
    )
    t_fuse = (time.perf_counter() - t0) * 1000.0

    return {
        "main_ms": t_main,
        "open_vocab_ms": t_ov,
        "geometry_ms": t_g0,
        "fusion_ms": t_fuse,
        "online_ms": t_main + t_ov + t_g0 + t_fuse,
        "n_known": len(known_inst),
        "n_unknown": len(unknown_ref),
        "n_fused": len(fused),
    }


def try_expert_flops(expert: GroundingSAM2Expert, sample_rgb_path: str, prompts) -> dict:
    """Best-effort FLOPs for one GDINO forward + one SAM2 forward (1 box)."""
    out = {"gdino_g": None, "sam2_g_per_box": None, "method": None}
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore
    except Exception:
        try:
            import subprocess

            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "fvcore", "-q"]
            )
            from fvcore.nn import FlopCountAnalysis  # type: ignore
        except Exception as e:
            out["error"] = f"fvcore unavailable: {e}"
            return out

    expert._ensure_models()
    image = Image.open(sample_rgb_path).convert("RGB")
    text = ". ".join(prompts) + "."
    inputs = expert._gdino_processor(images=image, text=text, return_tensors="pt").to(
        expert.device
    )

    class _G(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values, input_ids, attention_mask):
            return self.m(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

    try:
        gwrap = _G(expert._gdino).eval()
        flops = FlopCountAnalysis(
            gwrap,
            (
                inputs["pixel_values"],
                inputs["input_ids"],
                inputs["attention_mask"],
            ),
        )
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        out["gdino_g"] = float(flops.total() / 1e9)
        out["method"] = "fvcore"
    except Exception as e:
        out["gdino_error"] = str(e)

    # SAM2: one dummy box covering center.
    w, h = image.size
    box = [[w * 0.25, h * 0.25, w * 0.75, h * 0.75]]
    sam_inputs = expert._sam2_processor(
        images=image, input_boxes=[box], return_tensors="pt"
    ).to(expert.device)

    class _S(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, **kwargs):
            return self.m(**kwargs, multimask_output=True)

    try:
        # FlopCountAnalysis prefers tensor args; call through a thin lambda-like module.
        class _SFixed(torch.nn.Module):
            def __init__(self, m, packed):
                super().__init__()
                self.m = m
                self.packed = packed

            def forward(self, dummy):
                return self.m(**self.packed, multimask_output=True)

        # Keep all SAM2 inputs on the expert device for fvcore.
        packed = {
            k: (v.to(expert.device) if hasattr(v, "to") else v)
            for k, v in sam_inputs.items()
        }
        swrap = _SFixed(expert._sam2, packed).eval().to(expert.device)
        flops = FlopCountAnalysis(swrap, (torch.zeros(1, device=expert.device),))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        out["sam2_g_per_box"] = float(flops.total() / 1e9)
    except Exception as e:
        out["sam2_error"] = str(e)

    return out


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "CUDA required"
    print("GPU:", torch.cuda.get_device_name(0))

    model, cfgs = build_known_model(args.cfg_file, args.ckpt, device)
    transform = build_transform_by_cfg(cfgs.DATA_CONFIG.DATA_TRANSFORM.EVALUATING)
    split = get_split(args.split)
    prompts = list(split.unknown_prompts)

    expert = GroundingSAM2Expert(
        gdino_model="IDEA-Research/grounding-dino-base",
        sam2_model="facebook/sam2.1-hiera-large",
        box_threshold=0.25,
        text_threshold=0.20,
        device=device,
    )
    expert._ensure_models()

    known_m = count_params(model)
    gdino_m = count_params(expert._gdino)
    sam2_m = count_params(expert._sam2)
    total_m = known_m + gdino_m + sam2_m
    print(
        f"Params M | known={known_m:.3f}  GDINO={gdino_m:.3f}  "
        f"SAM2={sam2_m:.3f}  total={total_m:.3f}"
    )

    pairs = load_image_pairs(args.list_file, args.data_root, args.num_images)
    if not pairs:
        raise FileNotFoundError("No image pairs found for timing")
    print(f"Timing on {len(pairs)} images, prompts={prompts}")

    time_kw = dict(
        known_conf=float(args.known_conf),
        box_threshold=float(args.box_threshold),
        text_threshold=float(args.text_threshold),
    )
    # Warmup
    for i in range(min(args.warmup, len(pairs))):
        _ = time_one(
            model,
            transform,
            expert,
            pairs[i][0],
            pairs[i][1],
            split,
            prompts,
            device,
            **time_kw,
        )

    rows = []
    for left, right in pairs:
        for _ in range(args.reps_per_image):
            rows.append(
                time_one(
                    model,
                    transform,
                    expert,
                    left,
                    right,
                    split,
                    prompts,
                    device,
                    **time_kw,
                )
            )

    def mean_key(k):
        return float(np.mean([r[k] for r in rows]))

    timing = {
        "main_ms": mean_key("main_ms"),
        "open_vocab_ms": mean_key("open_vocab_ms"),
        "geometry_ms": mean_key("geometry_ms"),
        "fusion_ms": mean_key("fusion_ms"),
        "online_ms": mean_key("online_ms"),
        "n_images": len(rows),
        "avg_n_unknown": mean_key("n_unknown"),
    }
    print(
        "Runtime ms | main={main_ms:.1f}  OV(DINO+SAM2)={open_vocab_ms:.1f}  "
        "G0={geometry_ms:.1f}  fuse={fusion_ms:.1f}  TOTAL={online_ms:.1f}".format(
            **timing
        )
    )

    # Known FLOPs from prior thop; expert FLOPs best-effort.
    known_flops = 28.2208  # @384x1248 F5
    expert_flops = try_expert_flops(expert, pairs[0][0], prompts)
    print("Expert FLOPs attempt:", expert_flops)

    avg_boxes = max(1.0, timing["avg_n_unknown"])
    gdino_g = expert_flops.get("gdino_g")
    sam2_g = expert_flops.get("sam2_g_per_box")
    if gdino_g is not None and sam2_g is not None:
        # SAM2 cost scales roughly with number of boxes in one batched call;
        # measured per 1-box call — scale by average detections.
        full_flops = known_flops + gdino_g + sam2_g * avg_boxes
    elif gdino_g is not None:
        full_flops = known_flops + gdino_g
    else:
        full_flops = None

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "cfg_file": args.cfg_file,
        "split": args.split,
        "prompts": prompts,
        "params_m": {
            "known": round(known_m, 4),
            "grounding_dino_base": round(gdino_m, 4),
            "sam2_1_hiera_large": round(sam2_m, 4),
            "g0_fusion": 0.0,
            "total": round(total_m, 4),
        },
        "flops_g": {
            "known_384x1248": known_flops,
            "gdino": gdino_g,
            "sam2_per_box": sam2_g,
            "avg_unknown_boxes": avg_boxes,
            "total_estimated": None if full_flops is None else round(full_flops, 2),
            "expert_detail": expert_flops,
        },
        "runtime_ms": {k: round(v, 2) if isinstance(v, float) else v for k, v in timing.items()},
        "averages_for_table": {
            "FLOPs_G": None if full_flops is None else round(full_flops, 2),
            "Params_M": round(total_m, 2),
            "Time_ms": round(timing["online_ms"], 2),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== TABLE AVERAGES (full system) ===")
    print(json.dumps(report["averages_for_table"], indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
