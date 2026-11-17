"""Combinatorial leave-out splits for open-world experiments.

Protocol (user redesign):
  - Full set C = {Car, Truck, Bus, Bicycle, Person}
  - For known-budget K in {4,3,2,1}: every combination of (5-K) left-out
    classes becomes one experiment; each needs its own trained known model.
  - F5: all five known (closed-set reference).
  - Unknown expert (YOLOE) is frozen and shared across all experiments.

Example K4 (5 models):
  K4-1 leave Car, K4-2 leave Truck, ..., K4-5 leave Person.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

from .label_mapping import ExperimentSplit, build_experiment_split

# Annotation ID order: 1 car, 2 truck, 3 bus, 4 bicycle, 5 person
ALL_CLASSES: Tuple[str, ...] = ("Car", "Truck", "Bus", "Bicycle", "Person")

_PROMPT_TEXT = {
    "Car": "car",
    "Truck": "truck",
    "Bus": "bus",
    "Bicycle": "bicycle",
    "Person": "person",
}

# Synonym groups for YOLOE text prompts. All synonyms of one leave-out class
# share the same unknownj slot / public label.
PROMPT_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "Car": ("car", "automobile", "sedan"),
    # Keep short; long multi-word lists can collapse YOLOE text-prompt quality.
    "Truck": ("truck", "lorry", "pickup", "semi truck"),
    "Bus": ("bus", "city bus"),
    "Bicycle": ("bicycle", "bike"),
    "Person": ("person", "pedestrian"),
}


def expand_unknown_prompts_for_yoloe(
    split: ExperimentSplit,
    use_synonyms: bool = True,
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    """Flat YOLOE prompt list + per-prompt slot into split.unknown_public_labels."""
    texts: List[str] = []
    slots: List[int] = []
    for slot, (prompt, canon) in enumerate(
        zip(split.unknown_prompts, split.unknown_canonical_classes)
    ):
        if use_synonyms:
            syns = PROMPT_SYNONYMS.get(canon, (prompt,))
        else:
            syns = (prompt,)
        for s in syns:
            texts.append(str(s).strip().lower())
            slots.append(slot)
    return tuple(texts), tuple(slots)


def _known_in_annotation_order(known: Sequence[str]) -> Tuple[str, ...]:
    known_set = {str(x) for x in known}
    return tuple(c for c in ALL_CLASSES if c in known_set)


def _unknown_in_annotation_order(unknown: Sequence[str]) -> Tuple[str, ...]:
    unknown_set = {str(x) for x in unknown}
    return tuple(c for c in ALL_CLASSES if c in unknown_set)


def make_leave_out_split(
    known_budget: int,
    unknown_classes: Sequence[str],
    index: int,
) -> ExperimentSplit:
    """Build one leave-out split. ``index`` is 1-based within that K."""
    unknown = _unknown_in_annotation_order(unknown_classes)
    if len(unknown) != 5 - known_budget:
        raise ValueError(
            f"K={known_budget} expects {5 - known_budget} unknowns, got {unknown}"
        )
    known = _known_in_annotation_order([c for c in ALL_CLASSES if c not in set(unknown)])
    if len(known) != known_budget:
        raise ValueError(f"known size mismatch: {known}")
    prompts = tuple(_PROMPT_TEXT[c] for c in unknown)
    publics = tuple(f"unknown{i + 1}" for i in range(len(prompts)))
    name = f"K{known_budget}-{index}"
    return build_experiment_split(
        name=name,
        known_classes=known,
        unknown_prompts=prompts,
        unknown_public_labels=publics,
    )


def build_all_combinatorial_splits() -> Dict[str, ExperimentSplit]:
    splits: Dict[str, ExperimentSplit] = {
        "F5": build_experiment_split(
            name="F5",
            known_classes=list(ALL_CLASSES),
            unknown_prompts=[],
            unknown_public_labels=[],
        )
    }
    for k in (4, 3, 2, 1):
        n_unknown = 5 - k
        combos = list(combinations(ALL_CLASSES, n_unknown))
        for i, unk in enumerate(combos, start=1):
            split = make_leave_out_split(k, unk, i)
            splits[split.name] = split
            # Alias: K4_leave_Truck, K3_leave_Car_Bus, ...
            alias = f"K{k}_leave_" + "_".join(unk)
            splits[alias] = split
    return splits


EXPERIMENT_SPLITS: Dict[str, ExperimentSplit] = build_all_combinatorial_splits()


def get_split(name: str) -> ExperimentSplit:
    key = str(name).strip()
    # Accept K4-2, k4-2, K4_leave_Truck, etc.
    if key.upper().startswith("K") and "-" in key:
        # normalize K4-2
        parts = key.replace("_", "-").split("-")
        if len(parts) >= 2 and parts[0][1:].isdigit():
            key = f"{parts[0].upper()}-{parts[1]}"
    if key not in EXPERIMENT_SPLITS:
        # try alias case variants
        for cand in (key, key.upper(), key.replace("-", "_")):
            if cand in EXPERIMENT_SPLITS:
                return EXPERIMENT_SPLITS[cand]
        raise KeyError(
            f"Unknown split {name!r}; examples: F5, K4-1..K4-5, K3-1..K3-10, "
            f"K4_leave_Truck. Total registered={len(set(EXPERIMENT_SPLITS.values()))}"
        )
    return EXPERIMENT_SPLITS[key]


def list_splits_for_k(known_budget: int) -> List[ExperimentSplit]:
    """Unique splits for a given K, ordered by index."""
    seen = set()
    out: List[ExperimentSplit] = []
    for name, split in EXPERIMENT_SPLITS.items():
        if not name.startswith(f"K{known_budget}-"):
            continue
        if split.name in seen:
            continue
        seen.add(split.name)
        out.append(split)
    out.sort(key=lambda s: int(s.name.split("-")[1]))
    return out


def experiment_count_summary() -> Dict[str, int]:
    return {
        "F5": 1,
        "K4": len(list_splits_for_k(4)),
        "K3": len(list_splits_for_k(3)),
        "K2": len(list_splits_for_k(2)),
        "K1": len(list_splits_for_k(1)),
        "total_known_models": 1
        + len(list_splits_for_k(4))
        + len(list_splits_for_k(3))
        + len(list_splits_for_k(2))
        + len(list_splits_for_k(1)),
    }
