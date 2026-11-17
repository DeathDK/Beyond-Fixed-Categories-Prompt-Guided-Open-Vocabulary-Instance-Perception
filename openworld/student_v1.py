"""V1-A shared unknown student: independent frozen lightweight encoder + prompt match."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightEncoder(nn.Module):
    """Tiny Conv encoder (independent of LightStereo backbone)."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class PromptStudentV1A(nn.Module):
    """Class-agnostic objectness + visual embedding matched to cached text vectors."""

    def __init__(self, feat_dim: int = 128, embed_dim: int = 128, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = LightweightEncoder(out_dim=feat_dim)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.objectness = nn.Conv2d(feat_dim, 1, 1)
        self.embed_head = nn.Conv2d(feat_dim, embed_dim, 1)
        self.mask_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, 1, 1),
        )
        self.temperature = nn.Parameter(torch.tensor(10.0))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> dict:
        """
        Args:
            images: BCHW normalized
            text_embeddings: [M, D] L2-normalized prompt embeddings
        """
        feat = self.encode_image(images)
        obj = torch.sigmoid(self.objectness(feat))
        emb = F.normalize(self.embed_head(feat), dim=1)
        mask_logits = self.mask_head(feat)
        # Dense prompt matching: [B,M,H,W]
        # emb: B,D,H,W ; text: M,D
        b, d, h, w = emb.shape
        emb_flat = emb.view(b, d, -1)  # B,D,HW
        txt = F.normalize(text_embeddings, dim=-1)  # M,D
        sim = torch.einsum("bdn,md->bmn", emb_flat, txt) * self.temperature
        sim = sim.view(b, -1, h, w)
        return {
            "objectness": obj,
            "embeddings": emb,
            "mask_logits": mask_logits,
            "prompt_sim": sim,
        }


def student_distill_loss(
    student_out: dict,
    teacher_mask: torch.Tensor,
    teacher_obj: torch.Tensor,
    prompt_index: int,
) -> torch.Tensor:
    """Simple mask + objectness distillation onto one prompt channel."""
    obj = student_out["objectness"]
    mask_logits = student_out["mask_logits"]
    sim = student_out["prompt_sim"][:, prompt_index : prompt_index + 1]
    if teacher_mask.shape[-2:] != mask_logits.shape[-2:]:
        teacher_mask = F.interpolate(teacher_mask, size=mask_logits.shape[-2:], mode="nearest")
        teacher_obj = F.interpolate(teacher_obj, size=obj.shape[-2:], mode="nearest")
    loss_mask = F.binary_cross_entropy_with_logits(mask_logits, teacher_mask)
    loss_obj = F.binary_cross_entropy(obj, teacher_obj.clamp(0, 1))
    loss_sim = F.binary_cross_entropy_with_logits(sim, teacher_mask)
    return loss_mask + loss_obj + 0.5 * loss_sim
