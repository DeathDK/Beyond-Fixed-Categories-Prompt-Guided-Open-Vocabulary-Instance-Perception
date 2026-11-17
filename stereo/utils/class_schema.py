"""Shared class metadata and ID conversion for KITTI2015-IS.

Annotation files keep their historical IDs 1--5 and reserve 0 as ``none``.
YOLO heads use contiguous foreground IDs with no explicit background output.

Historical annotation name for ID 2 is ``trunk``; canonical public name is
``truck`` / ``Truck``. Open-world experiments should use
``openworld.label_mapping`` for K-class remapping.
"""

from __future__ import annotations

from typing import Optional, Sequence

KITTI_INSTANCE_CLASS_NAMES = (
    "none",
    "car",
    "truck",  # was historically stored as "trunk" in some docs/code
    "bus",
    "bicycle",
    "person",
)

# Keep legacy alias for callers that still look up "trunk".
KITTI_INSTANCE_CLASS_ALIASES = {
    "trunk": "truck",
    "lorry": "truck",
    "pedestrian": "person",
}

KITTI_INSTANCE_MODEL_CLASS_NAMES = KITTI_INSTANCE_CLASS_NAMES[1:]
KITTI_INSTANCE_NUM_OUTPUTS = len(KITTI_INSTANCE_MODEL_CLASS_NAMES)


def normalize_class_name(name: str) -> str:
    key = str(name).strip().lower()
    return KITTI_INSTANCE_CLASS_ALIASES.get(key, key)


def annotation_id_to_model_id(class_id: int) -> int:
    """Map an annotation ID in [1, 5] to a YOLO model ID in [0, 4]."""
    class_id = int(class_id)
    if not 1 <= class_id < len(KITTI_INSTANCE_CLASS_NAMES):
        raise ValueError(f"Invalid KITTI2015-IS annotation class ID: {class_id}")
    return class_id - 1


def model_id_to_annotation_id(class_id: int) -> int:
    """Map a YOLO model ID in [0, 4] back to the annotation schema."""
    class_id = int(class_id)
    if not 0 <= class_id < KITTI_INSTANCE_NUM_OUTPUTS:
        raise ValueError(f"Invalid KITTI2015-IS model class ID: {class_id}")
    return class_id + 1


def model_ids_to_annotation_ids(class_ids):
    """Vectorized ID conversion for tensors or NumPy-like arrays."""
    return class_ids + 1


def annotation_id_to_name(class_id: int) -> str:
    class_id = int(class_id)
    if not 0 <= class_id < len(KITTI_INSTANCE_CLASS_NAMES):
        raise ValueError(f"Invalid KITTI2015-IS annotation class ID: {class_id}")
    return KITTI_INSTANCE_CLASS_NAMES[class_id]


def build_known_name_list(known_classes: Optional[Sequence[str]] = None):
    """Return model-facing lowercase names for a known-class subset.

    If ``known_classes`` is None, return the full five-class list.
    """
    if known_classes is None:
        return list(KITTI_INSTANCE_MODEL_CLASS_NAMES)
    names = []
    for name in known_classes:
        n = normalize_class_name(name)
        if n not in KITTI_INSTANCE_MODEL_CLASS_NAMES:
            raise ValueError(f"Unsupported known class: {name!r}")
        names.append(n)
    return names
