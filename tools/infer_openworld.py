#!/usr/bin/env python
"""Open-world inference: known LightStereo-Instance(K) + YOLOE prompts + PG-GUIC.

Does NOT read evaluation GT. Prompts decide unknown semantics; public labels are
unknown1/unknown2/...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from easydict import EasyDict
from PIL import Image

sys.path.insert(0, "./")

from openworld.fusion import fuse_known_unknown
from openworld.geometry_refine import (
    G1BoundaryRefiner,
    refine_unknown_instances_g0,
    refine_unknown_instances_g1,
)
from openworld.instances import instances_to_jsonable
from openworld.known_decode import decode_known_instances
from openworld.label_mapping import AnnotationSchema
from openworld.pointcloud import reconstruct_instances, union_instance_point_count
from openworld.prompt_expert import YOLOEPromptExpert
from openworld.vlm_expert import GroundingSAM2Expert
from openworld.splits import expand_unknown_prompts_for_yoloe, get_split
from openworld.student_v1 import PromptStudentV1A
from stereo.datasets.dataset_template import build_transform_by_cfg
from stereo.modeling.models.lightstereo.trainer import Trainer
from stereo.utils import common_utils


def parse_args():
    p = argparse.ArgumentParser(description="Open-world infer")
    p.add_argument("--cfg_file", type=str, default="cfgs/openworld/known4.yaml")
    p.add_argument("--pretrained_model", type=str, required=True)
    p.add_argument("--left_img_path", type=str, required=True)
    p.add_argument("--right_img_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./openworld_output")
    p.add_argument("--split", type=str, default="")
    p.add_argument("--prompts", type=str, default="", help="comma-separated prompts")
    p.add_argument("--yoloe_ckpt", type=str, default="yoloe-26x-seg.pt")
    p.add_argument(
        "--conf_threshold",
        type=float,
        default=0.001,
        help="Known-head decode confidence (SEG_AP-aligned).",
    )
    p.add_argument("--known_conf", type=float, default=None, help="Alias for known decode conf.")
    p.add_argument(
        "--unknown_expert",
        type=str,
        default="gdino_sam2",
        choices=["gdino_sam2", "yoloe"],
        help="Frozen unknown expert: Grounding DINO+SAM2 (default) or YOLOE.",
    )
    p.add_argument("--unknown_conf", type=float, default=0.25, help="YOLOE conf (yoloe expert only).")
    p.add_argument("--yoloe_conf", type=float, default=None, help="Alias for YOLOE conf.")
    p.add_argument("--box_threshold", type=float, default=0.10, help="Grounding DINO box threshold.")
    p.add_argument("--text_threshold", type=float, default=0.10, help="Grounding DINO text threshold.")
    p.add_argument(
        "--gdino_model",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="HF id or local path for Grounding DINO (base = largest official).",
    )
    p.add_argument(
        "--sam2_model",
        type=str,
        default="facebook/sam2.1-hiera-large",
        help="HF id or local path for SAM2 (hiera-large = largest).",
    )
    p.add_argument("--use_synonyms", action="store_true", default=False)
    p.add_argument("--no_synonyms", action="store_true")
    p.add_argument("--mask_iou_conflict", type=float, default=0.90)
    p.add_argument("--unknown_score_boost", type=float, default=0.50)
    p.add_argument(
        "--no_fusion",
        action="store_true",
        help="Skip Known↔Unknown conflict fusion (concat known + G0 unknowns).",
    )
    p.add_argument("--g0_edge_band", type=int, default=3)
    p.add_argument(
        "--geometry_mode",
        type=str,
        default="g0",
        choices=["none", "g0", "g1"],
        help="PG-GUIC geometry refine mode",
    )
    p.add_argument("--g1_ckpt", type=str, default="")
    p.add_argument("--v1_ckpt", type=str, default="", help="Optional V1-A student ckpt")
    p.add_argument(
        "--ablation",
        type=str,
        default="",
        choices=["", "naive", "g0", "g1", "v1"],
        help="Shortcut: naive=YOLOE only; g0/g1/v1 enable corresponding modules",
    )
    p.add_argument("--no_geometry", action="store_true")
    p.add_argument("--no_pointcloud", action="store_true")
    p.add_argument("--fx", type=float, default=721.5377)
    p.add_argument("--fy", type=float, default=721.5377)
    p.add_argument("--cx", type=float, default=609.5593)
    p.add_argument("--cy", type=float, default=172.854)
    p.add_argument("--baseline", type=float, default=0.54)
    return p.parse_args()


def load_known_model(args, cfgs):
    cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model
    local_rank = 0
    logger = common_utils.create_logger(None, rank=local_rank)
    trainer = Trainer(args, cfgs, local_rank, 0, logger, None)
    model = trainer.model
    model.eval()
    return model, trainer


def preprocess_pair(left_path, right_path, transform):
    left = np.array(Image.open(left_path).convert("RGB"), dtype=np.float32)
    right = np.array(Image.open(right_path).convert("RGB"), dtype=np.float32)
    h0, w0 = left.shape[:2]
    sample = transform({"left": left, "right": right})
    left_t = sample["left"].unsqueeze(0)
    right_t = sample["right"].unsqueeze(0)
    return left_t, right_t, (h0, w0), left.astype(np.uint8)


def _load_g1_refiner(ckpt_path: str, device: str):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model = G1BoundaryRefiner()
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def _student_unknown_instances(student, text_emb, left_rgb, prompts, orig_hw, device):
    """Optional V1-A student proposals (mask from objectness x prompt_sim)."""
    if student is None or text_emb is None or not prompts:
        return []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = left_rgb.astype(np.float32) / 255.0
    img = (img - mean) / std
    t = torch.from_numpy(img.transpose(2, 0, 1)[None]).float().to(device)
    with torch.no_grad():
        out = student(t, text_emb.to(device))
        obj = out["objectness"][0, 0].cpu().numpy()
        sim = out["prompt_sim"][0].cpu().numpy()  # M,H,W
        mask_prob = torch.sigmoid(out["mask_logits"][0, 0]).cpu().numpy()
    H0, W0 = orig_hw
    instances = []
    for slot, prompt in enumerate(prompts):
        heat = obj * (1.0 / (1.0 + np.exp(-sim[slot]))) * mask_prob
        if heat.shape != (H0, W0):
            heat = cv2.resize(heat, (W0, H0), interpolation=cv2.INTER_LINEAR)
        binary = (heat > 0.35).astype(np.uint8)
        if binary.sum() < 32:
            continue
        ys, xs = np.where(binary > 0)
        bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        from openworld.instances import InstancePrediction

        instances.append(
            InstancePrediction(
                instance_id=slot + 1,
                class_name=prompt,
                score=float(heat.max()),
                bbox_xyxy=bbox,
                mask=binary,
                prompt_slot=slot,
                prompt_text=prompt,
                source="unknown",
            )
        )
    return instances


def run_openworld(
    model,
    transform,
    left_path,
    right_path,
    split,
    expert: YOLOEPromptExpert,
    prompts,
    conf_threshold=0.45,
    known_conf=None,
    yoloe_conf=None,
    prompt_slots=None,
    use_geometry=True,
    geometry_mode="g0",
    g0_edge_band_pixels=3,
    g1_refiner=None,
    student=None,
    student_text_emb=None,
    prefer_student=False,
    mask_iou_conflict=0.6,
    unknown_score_boost=0.0,
    skip_fusion=False,
    device="cuda",
):
    timing = {}
    left_t, right_t, orig_hw, left_rgb = preprocess_pair(left_path, right_path, transform)
    left_t = left_t.to(device)
    right_t = right_t.to(device)

    # Decode expects names aligned with train IDs; use schema order.
    known_names = list(AnnotationSchema(split.known_classes).known_classes)
    known_thr = float(conf_threshold if known_conf is None else known_conf)
    yoloe_thr = float(expert.conf_threshold if yoloe_conf is None else yoloe_conf)

    t0 = time.perf_counter()
    with torch.no_grad():
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        pred = model({"left": left_t, "right": right_t})
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    timing["main"] = (time.perf_counter() - t0) * 1000.0

    disp, known_inst = decode_known_instances(
        pred,
        class_names=known_names,
        orig_hw=orig_hw,
        conf_threshold=known_thr,
    )

    t0 = time.perf_counter()
    if prefer_student and student is not None:
        unknown_raw = _student_unknown_instances(
            student, student_text_emb, left_rgb, prompts, orig_hw, device
        )
    else:
        left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
        unknown_raw = expert.predict_instances(
            left_bgr,
            prompts=prompts,
            image_size_hw=orig_hw,
            prompt_slots=prompt_slots,
            conf_threshold=yoloe_thr,
        )
    timing["open_vocab"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    mode = (geometry_mode or "g0").lower()
    edge_band = int(g0_edge_band_pixels)
    if (not use_geometry) or mode in ("none", "off"):
        unknown_ref = unknown_raw
    elif mode == "g1":
        unknown_ref = refine_unknown_instances_g1(
            unknown_raw,
            disp,
            rgb=left_rgb,
            refiner=g1_refiner,
            device=device,
            edge_band_pixels=edge_band,
        )
    else:
        unknown_ref = refine_unknown_instances_g0(
            unknown_raw, disp, edge_band_pixels=edge_band
        )
    timing["geometry"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    fused, stats = fuse_known_unknown(
        known_inst,
        unknown_ref,
        split,
        mask_iou_conflict=float(mask_iou_conflict),
        unknown_score_boost=float(unknown_score_boost),
        skip_conflict=bool(skip_fusion),
    )
    timing["fusion"] = (time.perf_counter() - t0) * 1000.0
    timing["online"] = sum(timing[k] for k in ("main", "open_vocab", "geometry", "fusion"))

    return {
        "disparity": disp,
        "known": known_inst,
        "unknown": unknown_ref,
        "fused": fused,
        "stats": stats,
        "timing_ms": timing,
        "left_rgb": left_rgb,
        "orig_hw": orig_hw,
        "geometry_mode": mode if use_geometry else "none",
        "known_conf": known_thr,
        "yoloe_conf": yoloe_thr,
        "g0_edge_band_pixels": edge_band,
    }


def visualize(left_rgb, instances, out_path):
    img = left_rgb.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    overlay = img.copy()
    palette = [
        (0, 0, 255),
        (0, 165, 255),
        (0, 255, 255),
        (255, 0, 255),
        (0, 255, 0),
        (255, 128, 0),
        (128, 0, 255),
    ]
    for inst in instances:
        color = palette[inst.instance_id % len(palette)]
        if inst.mask is not None:
            m = inst.mask.astype(bool)
            if m.shape[:2] == overlay.shape[:2]:
                overlay[m] = (
                    overlay[m].astype(np.float32) * 0.5 + np.array(color) * 0.5
                ).astype(np.uint8)
        x1, y1, x2, y2 = [int(v) for v in inst.bbox_xyxy]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{inst.class_name}:{inst.score:.2f}"
        cv2.putText(
            overlay, label, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    # Minimal args object expected by Trainer
    args.dist_mode = False
    args.fix_random_seed = False
    args.workers = 0
    args.pin_memory = False
    args.save_root_dir = "./output"
    args.extra_tag = "infer_openworld"
    args.run_mode = "infer"

    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)
    # Resolve data path like train.py
    from cfgs.data_basic import DATA_PATH_DICT

    for each in cfgs.DATA_CONFIG.DATA_INFOS:
        dataset_name = each.DATASET
        if dataset_name == "KittiDataset":
            dataset_name = (
                "KittiDataset15"
                if "kitti15" in each.DATA_SPLIT.EVALUATING
                else "KittiDataset12"
            )
        each.DATA_PATH = DATA_PATH_DICT.get(dataset_name, each.get("DATA_PATH", "./data/KITTI15"))

    split_name = args.split or cfgs.MODEL.get("OPENWORLD_SPLIT", "K4")
    split = get_split(split_name)
    known_conf = float(args.known_conf if args.known_conf is not None else args.conf_threshold)
    yoloe_conf = float(args.yoloe_conf if args.yoloe_conf is not None else args.unknown_conf)
    use_synonyms = bool(getattr(args, "use_synonyms", False)) and not bool(
        getattr(args, "no_synonyms", False)
    )
    if args.prompts:
        prompts = [x.strip() for x in args.prompts.split(",") if x.strip()]
        prompt_slots = list(range(len(prompts)))
    else:
        prompts_t, slots_t = expand_unknown_prompts_for_yoloe(
            split, use_synonyms=use_synonyms
        )
        prompts = list(prompts_t)
        prompt_slots = list(slots_t)

    # Ablation shortcuts
    geometry_mode = args.geometry_mode
    prefer_student = False
    if args.ablation == "naive" or args.no_geometry:
        geometry_mode = "none"
    elif args.ablation == "g0":
        geometry_mode = "g0"
    elif args.ablation == "g1":
        geometry_mode = "g1"
    elif args.ablation == "v1":
        prefer_student = True
        geometry_mode = "g0"

    os.makedirs(args.output_dir, exist_ok=True)
    model, trainer = load_known_model(args, cfgs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    transform = build_transform_by_cfg(cfgs.DATA_CONFIG.DATA_TRANSFORM.EVALUATING)

    if args.unknown_expert == "yoloe":
        expert = YOLOEPromptExpert(
            checkpoint=args.yoloe_ckpt,
            conf_threshold=yoloe_conf,
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
    g1_refiner = _load_g1_refiner(args.g1_ckpt, device) if geometry_mode == "g1" else None
    student, student_text_emb = None, None
    if prefer_student or args.v1_ckpt:
        if args.v1_ckpt and os.path.isfile(args.v1_ckpt):
            ckpt = torch.load(args.v1_ckpt, map_location=device, weights_only=False)
            student = PromptStudentV1A(freeze_encoder=True).to(device).eval()
            student.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
            student_text_emb = ckpt.get("text_embedding")
            if student_text_emb is not None:
                student_text_emb = student_text_emb.to(device)
            prefer_student = True

    result = run_openworld(
        model,
        transform,
        args.left_img_path,
        args.right_img_path,
        split,
        expert,
        prompts,
        known_conf=known_conf,
        yoloe_conf=yoloe_conf if args.unknown_expert == "yoloe" else float(args.box_threshold),
        prompt_slots=prompt_slots,
        use_geometry=geometry_mode != "none",
        geometry_mode=geometry_mode,
        g0_edge_band_pixels=int(args.g0_edge_band),
        g1_refiner=g1_refiner,
        student=student,
        student_text_emb=student_text_emb,
        prefer_student=prefer_student,
        mask_iou_conflict=float(args.mask_iou_conflict),
        unknown_score_boost=float(args.unknown_score_boost),
        skip_fusion=bool(args.no_fusion),
        device=device,
    )

    stem = Path(args.left_img_path).stem
    vis_path = os.path.join(args.output_dir, f"{stem}_openworld.jpg")
    visualize(result["left_rgb"], result["fused"], vis_path)
    np.save(os.path.join(args.output_dir, f"{stem}_disp.npy"), result["disparity"])

    mask_dir = Path(args.output_dir) / f"{stem}_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_files = {}
    for inst in result["fused"]:
        if inst.mask is None:
            continue
        mp = mask_dir / f"inst_{inst.instance_id}.png"
        cv2.imwrite(str(mp), (inst.mask.astype(np.uint8) * 255))
        mask_files[inst.instance_id] = str(mp.name)

    payload = {
        "image_id": stem,
        "split": split.name,
        "prompts": prompts,
        "instances": instances_to_jsonable(result["fused"], mask_files),
        "fusion_stats": result["stats"],
        "timing_ms": result["timing_ms"],
    }

    if not args.no_pointcloud:
        t0 = time.perf_counter()
        calib = {
            "fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy,
            "baseline": args.baseline,
        }
        clouds = reconstruct_instances(
            result["disparity"], result["left_rgb"], result["fused"], calib
        )
        payload["timing_ms"]["pointcloud"] = (time.perf_counter() - t0) * 1000.0
        payload["point_counts"] = {
            str(k): v["num_points"] for k, v in clouds.items()
        }
        payload["union_instance_points"] = union_instance_point_count(
            result["disparity"], result["fused"]
        )
        # Optional Open3D dump of union
        try:
            import open3d as o3d

            all_xyz, all_rgb = [], []
            for v in clouds.values():
                if v["num_points"] == 0:
                    continue
                all_xyz.append(v["xyz"])
                all_rgb.append(v["rgb"])
            if all_xyz:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_xyz, 0))
                pcd.colors = o3d.utility.Vector3dVector(np.concatenate(all_rgb, 0))
                o3d.io.write_point_cloud(
                    os.path.join(args.output_dir, f"{stem}_instances.ply"), pcd
                )
        except Exception as exc:
            payload["pointcloud_warning"] = str(exc)

    with open(os.path.join(args.output_dir, f"{stem}_result.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload["timing_ms"], indent=2))
    print(f"saved visualization to {vis_path}")


if __name__ == "__main__":
    main()
