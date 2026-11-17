#!/usr/bin/env python
"""Train shared PG-GUIC G1 refiner and/or V1-A unknown student once.

Supervision policy (P-A):
  - Never use left-out unknown KITTI instance GT (Truck/Bus/...).
  - G1 / student may use Car masks only (common known across K4..K1) or
    teacher YOLOE pseudo labels with confidence filtering.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, "./")

from openworld.geometry_refine import G1BoundaryRefiner, disparity_edge_numpy
from openworld.label_mapping import annotation_id_to_canonical
from openworld.prompt_expert import YOLOEPromptExpert
from openworld.student_v1 import PromptStudentV1A, student_distill_loss


class CarOnlyMaskDataset(Dataset):
    """KITTI left images with Car polygon masks only (+ optional disparity)."""

    def __init__(self, data_root: str, list_file: str, image_size=(384, 1248)):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.items = []
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                self.items.append(parts)

    def __len__(self):
        return len(self.items)

    def _car_mask(self, label_path: Path, hw) -> np.ndarray:
        H, W = hw
        mask = np.zeros((H, W), dtype=np.float32)
        if not label_path.exists():
            return mask
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                ann_id = int(float(parts[0]))
                if annotation_id_to_canonical(ann_id) != "Car":
                    continue
                poly = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
                poly[:, 0] *= W
                poly[:, 1] *= H
                cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1.0)
        return mask

    def __getitem__(self, idx):
        left_rel, right_rel, disp_rel, label_rel = self.items[idx][:4]
        left = np.array(Image.open(self.data_root / left_rel).convert("RGB"), dtype=np.uint8)
        H0, W0 = left.shape[:2]
        label_path = self.data_root / "training" / "seg_label" / Path(label_rel).name
        car_mask = self._car_mask(label_path, (H0, W0))
        disp = None
        disp_path = self.data_root / disp_rel
        if disp_path.exists():
            disp = np.array(Image.open(disp_path), dtype=np.float32) / 256.0
        # Resize to fixed size for batching.
        th, tw = self.image_size
        left_r = cv2.resize(left, (tw, th))
        mask_r = cv2.resize(car_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        if disp is not None:
            disp_r = cv2.resize(disp, (tw, th), interpolation=cv2.INTER_LINEAR)
        else:
            disp_r = np.zeros((th, tw), dtype=np.float32)
        # ImageNet-ish normalize
        img = left_r.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        return {
            "image": img,
            "car_mask": torch.from_numpy(mask_r[None]).float(),
            "disp": torch.from_numpy(disp_r[None]).float(),
            "left_u8": torch.from_numpy(left_r),  # RGB uint8 for teacher
        }


def rgb_edge(image_chw: torch.Tensor) -> torch.Tensor:
    # image_chw normalized; approximate edge from first channel after denorm skip
    x = image_chw[:, :1]
    kx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=x.dtype, device=x.device).unsqueeze(0)
    ky = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=x.dtype, device=x.device).unsqueeze(0)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    mag = mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return mag


def train_g1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CarOnlyMaskDataset(args.data_root, args.list_file)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    model = G1BoundaryRefiner().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        losses = []
        for batch in tqdm(loader, desc=f"G1 epoch {epoch}"):
            img = batch["image"].to(device)
            gt = batch["car_mask"].to(device)
            disp = batch["disp"].to(device)
            # Pseudo expert mask = GT car (allowed common-known supervision)
            expert = gt
            edge = []
            for i in range(disp.shape[0]):
                e = disparity_edge_numpy(disp[i, 0].detach().cpu().numpy())
                edge.append(torch.from_numpy(e)[None])
            edge = torch.stack(edge, 0).to(device)
            dmax = disp.amax(dim=(2, 3), keepdim=True).clamp_min(1.0)
            disp_norm = (disp / dmax).clamp(0, 1)
            rgb_e = rgb_edge(img)
            pred = model(expert, edge, disp_norm, rgb_e)
            loss = F.binary_cross_entropy(pred, gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"G1 epoch {epoch}: loss={np.mean(losses):.4f}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "kind": "g1"}, out)
    print(f"saved {out}")


def _yoloe_teacher_mask(expert: YOLOEPromptExpert, left_u8: np.ndarray, hw) -> np.ndarray:
    """Build a soft teacher mask from frozen YOLOE car detections (no unknown GT)."""
    H, W = hw
    mask = np.zeros((H, W), dtype=np.float32)
    bgr = cv2.cvtColor(left_u8, cv2.COLOR_RGB2BGR)
    preds = expert.predict_instances(bgr, prompts=["car"], image_size_hw=(H, W))
    for inst in preds:
        if inst.mask is None:
            continue
        m = inst.mask.astype(np.float32)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = np.maximum(mask, m * float(inst.score))
    return mask


def train_v1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CarOnlyMaskDataset(args.data_root, args.list_file)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    student = PromptStudentV1A(freeze_encoder=True).to(device)
    # Train heads only
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    expert = None
    # Cached random unit text embedding; student learns to match via distillation.
    text_emb = F.normalize(torch.randn(1, 128, device=device), dim=-1)
    if args.use_yoloe_teacher:
        expert = YOLOEPromptExpert(checkpoint=args.yoloe_ckpt, conf_threshold=0.25)
        expert.set_prompts(["car"])

    for epoch in range(args.epochs):
        losses = []
        for batch in tqdm(loader, desc=f"V1 epoch {epoch}"):
            img = batch["image"].to(device)
            gt = batch["car_mask"].to(device)
            teacher_mask = gt
            if expert is not None:
                # Distill from frozen YOLOE teacher on Car only (shared known).
                teacher_list = []
                left_batch = batch["left_u8"].numpy()
                for i in range(left_batch.shape[0]):
                    tm = _yoloe_teacher_mask(
                        expert, left_batch[i], (left_batch.shape[1], left_batch.shape[2])
                    )
                    # Fall back to Car GT if teacher empty (still no unknown GT).
                    if tm.sum() < 1:
                        tm = batch["car_mask"][i, 0].numpy()
                    teacher_list.append(torch.from_numpy(tm)[None])
                teacher_mask = torch.stack(teacher_list, 0).to(device)
            teacher_obj = (teacher_mask > 0.5).float()
            out = student(img, text_emb)
            loss = student_distill_loss(out, teacher_mask, teacher_obj, prompt_index=0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"V1 epoch {epoch}: loss={np.mean(losses):.4f}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": student.state_dict(),
            "kind": "v1a",
            "text_embedding": text_emb.detach().cpu(),
        },
        out,
    )
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["g1", "v1"], default="g1")
    parser.add_argument("--data_root", type=str, default="./data/KITTI15")
    parser.add_argument(
        "--list_file",
        type=str,
        default="./data/KITTI15/kitti15_train180_with_seglabel.txt",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="./pretrained/pg_guic_g1.pt")
    parser.add_argument("--use_yoloe_teacher", action="store_true")
    parser.add_argument("--yoloe_ckpt", type=str, default="yoloe-26x-seg.pt")
    args = parser.parse_args()
    if args.mode == "g1":
        train_g1(args)
    else:
        if args.output == "./pretrained/pg_guic_g1.pt":
            args.output = "./pretrained/unknown_student_v1a.pt"
        train_v1(args)


if __name__ == "__main__":
    import numpy as np

    main()
