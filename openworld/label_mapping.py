"""Canonical class mapping for KITTI2015-IS open-world experiments.

Annotation files historically use IDs 1--5 with name ``trunk`` for trucks.
Public experiment naming uses ``Truck``. Training for a K-class known model
remaps only known classes to contiguous IDs in ``[0, K-1]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Annotation ID order used by seg_label / JSON (1-indexed; 0 reserved as none).
ANNOTATION_ID_TO_LEGACY_NAME = {
    0: "none",
    1: "car",
    2: "trunk",  # historical misspelling of truck
    3: "bus",
    4: "bicycle",
    5: "person",
}

# Canonical names used by open-world configs and papers.
CANONICAL_CLASSES: Tuple[str, ...] = (
    "Car",
    "Truck",
    "Bus",
    "Bicycle",
    "Person",
)

_ALIAS_TO_CANONICAL = {
    "none": "none",
    "car": "Car",
    "truck": "Truck",
    "trunk": "Truck",
    "bus": "Bus",
    "bicycle": "Bicycle",
    "bike": "Bicycle",
    "person": "Person",
    "pedestrian": "Person",
    "lorry": "Truck",
    "pickup": "Truck",
    "pickup truck": "Truck",
    "semi truck": "Truck",
    "semi-truck": "Truck",
    "box truck": "Truck",
    "freight truck": "Truck",
    "delivery truck": "Truck",
    "tractor trailer": "Truck",
    "city bus": "Bus",
    "automobile": "Car",
    "sedan": "Car",
    "vehicle": "Car",
}


def normalize_canonical_name(name: str) -> str:
    """Map arbitrary string / alias to a canonical class name."""
    key = str(name).strip()
    if key in CANONICAL_CLASSES or key == "none":
        return key
    lowered = key.lower()
    if lowered in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[lowered]
    # Title-case fallback for already-near-canonical strings.
    titled = key[:1].upper() + key[1:].lower() if key else key
    if titled in CANONICAL_CLASSES:
        return titled
    raise ValueError(f"Unknown class name or alias: {name!r}")


def annotation_id_to_canonical(class_id: int) -> str:
    class_id = int(class_id)
    if class_id not in ANNOTATION_ID_TO_LEGACY_NAME:
        raise ValueError(f"Invalid annotation class ID: {class_id}")
    legacy = ANNOTATION_ID_TO_LEGACY_NAME[class_id]
    if legacy == "none":
        return "none"
    return normalize_canonical_name(legacy)


def canonical_to_annotation_id(name: str) -> int:
    canonical = normalize_canonical_name(name)
    if canonical == "none":
        return 0
    # Prefer the historical ID order, not CANONICAL_CLASSES order.
    for ann_id, legacy in ANNOTATION_ID_TO_LEGACY_NAME.items():
        if legacy == "none":
            continue
        if normalize_canonical_name(legacy) == canonical:
            return int(ann_id)
    raise ValueError(f"No annotation ID for canonical class {canonical!r}")


@dataclass(frozen=True)
class ExperimentSplit:
    """One nested known/unknown experiment configuration."""

    name: str
    known_classes: Tuple[str, ...]
    unknown_prompts: Tuple[str, ...]
    unknown_public_labels: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        known = tuple(normalize_canonical_name(c) for c in self.known_classes)
        object.__setattr__(self, "known_classes", known)
        prompts = tuple(str(p).strip().lower() for p in self.unknown_prompts)
        object.__setattr__(self, "unknown_prompts", prompts)
        if self.unknown_public_labels:
            publics = tuple(self.unknown_public_labels)
        else:
            publics = tuple(f"unknown{i + 1}" for i in range(len(prompts)))
        if len(publics) != len(prompts):
            raise ValueError(
                f"{self.name}: unknown_public_labels length "
                f"{len(publics)} != prompts {len(prompts)}"
            )
        object.__setattr__(self, "unknown_public_labels", publics)

    @property
    def num_classes(self) -> int:
        return len(self.known_classes)

    @property
    def unknown_canonical_classes(self) -> Tuple[str, ...]:
        """Map prompts to canonical GT classes for offline evaluation only."""
        return tuple(normalize_canonical_name(p) for p in self.unknown_prompts)

    def prompt_to_public(self, prompt: str) -> str:
        key = str(prompt).strip().lower()
        for p, pub in zip(self.unknown_prompts, self.unknown_public_labels):
            if p == key or normalize_canonical_name(p) == normalize_canonical_name(key):
                return pub
        raise KeyError(f"Prompt {prompt!r} not in split {self.name}")

    def public_to_canonical(self, public_label: str) -> str:
        """Evaluator-only reverse map unknownj -> Truck/..."""
        for pub, canon in zip(self.unknown_public_labels, self.unknown_canonical_classes):
            if pub == public_label:
                return canon
        raise KeyError(f"Public label {public_label!r} not in split {self.name}")


@dataclass
class AnnotationSchema:
    """Dynamic remapper for a specific K-class known model."""

    known_classes: Tuple[str, ...]

    def __post_init__(self):
        known = tuple(normalize_canonical_name(c) for c in self.known_classes)
        if len(set(known)) != len(known):
            raise ValueError(f"Duplicate known classes: {known}")
        self.known_classes = known
        self.name_to_train_id: Dict[str, int] = {
            name: i for i, name in enumerate(known)
        }
        self.train_id_to_name: Dict[int, str] = {
            i: name for name, i in self.name_to_train_id.items()
        }

    @property
    def num_classes(self) -> int:
        return len(self.known_classes)

    def is_known(self, name_or_id) -> bool:
        if isinstance(name_or_id, (int, float)):
            name = annotation_id_to_canonical(int(name_or_id))
        else:
            name = normalize_canonical_name(str(name_or_id))
        return name in self.name_to_train_id

    def annotation_id_to_train_id(self, class_id: int) -> Optional[int]:
        """Return train ID in [0, K-1], or None if the class is left-out."""
        name = annotation_id_to_canonical(int(class_id))
        if name == "none":
            return None
        return self.name_to_train_id.get(name)

    def train_id_to_canonical(self, train_id: int) -> str:
        return self.train_id_to_name[int(train_id)]

    def filter_training_labels(self, labels: Iterable[dict]) -> List[dict]:
        """Drop unknown-class instances and remap class_id to train IDs.

        Accepts either annotation IDs in ``[1, 5]`` or already-remapped train
        IDs in ``[0, K-1]`` (from ``prepare_openworld`` filtered label dirs).
        """
        filtered = []
        for label in labels:
            raw_id = label.get("class_id", label.get("category_id"))
            if raw_id is None:
                continue
            if hasattr(raw_id, "item"):
                raw_id = int(raw_id.item())
            else:
                raw_id = int(raw_id)

            train_id = None
            # Prefer annotation-space IDs 1..5 when present.
            if 1 <= raw_id <= 5:
                train_id = self.annotation_id_to_train_id(raw_id)
            elif label.get("id_space") == "train" or label.get("remapped_to_train"):
                if 0 <= raw_id < self.num_classes:
                    train_id = raw_id
            elif 0 <= raw_id < self.num_classes:
                # Prepared openworld label files already store train IDs.
                train_id = raw_id
            else:
                train_id = None

            if train_id is None:
                continue
            new_label = dict(label)
            new_label["class_id"] = train_id
            new_label["canonical_name"] = self.train_id_to_canonical(train_id)
            new_label["id_space"] = "train"
            new_label["remapped_to_train"] = True
            filtered.append(new_label)
        return filtered


def build_experiment_split(
    name: str,
    known_classes: Sequence[str],
    unknown_prompts: Sequence[str],
    unknown_public_labels: Optional[Sequence[str]] = None,
) -> ExperimentSplit:
    return ExperimentSplit(
        name=name,
        known_classes=tuple(known_classes),
        unknown_prompts=tuple(unknown_prompts),
        unknown_public_labels=tuple(unknown_public_labels or ()),
    )
