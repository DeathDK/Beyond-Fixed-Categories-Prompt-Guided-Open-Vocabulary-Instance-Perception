"""Unified instance container for known / unknown open-world outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class InstancePrediction:
    instance_id: int
    class_name: str
    score: float
    bbox_xyxy: List[float]
    mask: Optional[np.ndarray] = None
    prompt_slot: Optional[int] = None
    prompt_text: Optional[str] = None
    source: str = "known"  # known | unknown | fused
    geometry_score: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self, include_mask_path: Optional[str] = None) -> Dict[str, Any]:
        payload = {
            "instance_id": int(self.instance_id),
            "class_name": self.class_name,
            "score": float(self.score),
            "bbox_xyxy": [float(x) for x in self.bbox_xyxy],
            "source": self.source,
        }
        if self.prompt_slot is not None:
            payload["prompt_slot"] = int(self.prompt_slot)
        if self.prompt_text is not None:
            payload["prompt_text"] = self.prompt_text
        if self.geometry_score is not None:
            payload["geometry_score"] = float(self.geometry_score)
        if include_mask_path is not None:
            payload["mask_file"] = include_mask_path
        return payload


def instances_to_jsonable(
    instances: List[InstancePrediction],
    mask_files: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    mask_files = mask_files or {}
    return [
        inst.to_public_dict(include_mask_path=mask_files.get(inst.instance_id))
        for inst in instances
    ]


def clone_instance(inst: InstancePrediction, **overrides) -> InstancePrediction:
    data = asdict(inst)
    data.update(overrides)
    # asdict converts ndarray oddly; keep mask reference unless overridden.
    if "mask" not in overrides:
        data["mask"] = inst.mask
    return InstancePrediction(**data)
