"""Initialize open-world K-class models from a frozen F5 checkpoint.

The F5 file is opened read-only for ``torch.load`` only. Training always writes
to ``output/...``; never pass the F5 path as ``--resume_dir`` or ckpt save dir.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .splits import ALL_CLASSES

# Keys that carry a leading class dimension equal to ``nc`` (Ultralytics Detect/Seg).
_CLASS_HEAD_MARKERS = (".cv3.", ".one2one_cv3.")


def _norm_name(name: str) -> str:
    return str(name).strip()


def _is_class_logit_key(key: str) -> bool:
    lowered = key.lower()
    return any(m in lowered for m in _CLASS_HEAD_MARKERS)


def _class_axis_size(tensor: torch.Tensor) -> Optional[int]:
    if tensor.ndim == 0:
        return None
    # Conv weight [nc, C, k, k], bias [nc], or linear [nc, ...]
    return int(tensor.shape[0])


def build_f5_to_k_index(
    source_classes: Sequence[str],
    known_classes: Sequence[str],
) -> List[Tuple[int, int]]:
    """Return list of (src_idx, dst_idx) for classes present in both lists."""
    src = {_norm_name(c): i for i, c in enumerate(source_classes)}
    pairs: List[Tuple[int, int]] = []
    for dst_i, name in enumerate(known_classes):
        key = _norm_name(name)
        if key not in src:
            raise KeyError(
                f"Known class {name!r} not found in F5 source classes {list(source_classes)}"
            )
        pairs.append((src[key], dst_i))
    return pairs


def remap_class_tensor(
    src: torch.Tensor,
    dst_like: torch.Tensor,
    index_pairs: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    """Copy selected class channels from ``src`` into a new tensor shaped like ``dst_like``."""
    if _class_axis_size(src) is None or _class_axis_size(dst_like) is None:
        raise ValueError(f"Cannot remap class tensor shapes {tuple(src.shape)} -> {tuple(dst_like.shape)}")
    out = dst_like.detach().clone()
    # Keep destination init for any missing slot; overwrite mapped classes.
    for src_i, dst_i in index_pairs:
        out[dst_i] = src[src_i].to(dtype=out.dtype, device=out.device)
    return out


def apply_f5_class_head_init(
    model: nn.Module,
    f5_state: Dict[str, torch.Tensor],
    known_classes: Sequence[str],
    source_classes: Optional[Sequence[str]] = None,
    mode: str = "remap",
    logger=None,
) -> Dict[str, int]:
    """Initialize K-class Detect class towers from an F5 (5-class) state dict.

    Parameters
    ----------
    mode:
      - ``remap``: copy F5 channels for each remaining known class into the K head.
      - ``reset``: leave class-logit layers at their current (random) init.
    """
    source_classes = list(source_classes) if source_classes else list(ALL_CLASSES)
    mode = str(mode).strip().lower()
    if mode not in {"remap", "reset"}:
        raise ValueError(f"F5 class head mode must be 'remap' or 'reset', got {mode!r}")

    stats = {"remapped": 0, "reset_left": 0, "skipped": 0}
    if mode == "reset":
        for key, _ in model.named_parameters():
            if _is_class_logit_key(key) and key.endswith(("weight", "bias")):
                stats["reset_left"] += 1
        if logger:
            logger.info(
                "F5_INIT CLASS_HEAD_MODE=reset: leaving %d class-logit tensors at random init",
                stats["reset_left"],
            )
        return stats

    index_pairs = build_f5_to_k_index(source_classes, known_classes)
    if logger:
        mapping_str = ", ".join(
            f"{source_classes[s]}[{s}]->known[{d}]({known_classes[d]})" for s, d in index_pairs
        )
        logger.info("F5_INIT class channel map: %s", mapping_str)

    name_to_param = dict(model.named_parameters())
    with torch.no_grad():
        for key, param in name_to_param.items():
            if not (_is_class_logit_key(key) and key.endswith(("weight", "bias"))):
                continue
            if key not in f5_state:
                stats["skipped"] += 1
                continue
            src = f5_state[key]
            if not torch.is_tensor(src):
                stats["skipped"] += 1
                continue
            src_nc = _class_axis_size(src)
            dst_nc = _class_axis_size(param.data)
            if src_nc != len(source_classes) or dst_nc != len(known_classes):
                # Intermediate cv3 layers (not final nc conv) — already loaded by shape match.
                stats["skipped"] += 1
                continue
            if src.shape[1:] != param.data.shape[1:]:
                stats["skipped"] += 1
                if logger:
                    logger.warning(
                        "F5_INIT skip %s: trailing shape %s vs %s",
                        key,
                        tuple(src.shape),
                        tuple(param.data.shape),
                    )
                continue
            param.data.copy_(remap_class_tensor(src, param.data, index_pairs))
            stats["remapped"] += 1

    if logger:
        logger.info(
            "F5_INIT class head done: remapped=%d reset_left=%d skipped=%d",
            stats["remapped"],
            stats["reset_left"],
            stats["skipped"],
        )
    return stats


def assert_not_overwriting_pretrained(
    save_path: str,
    pretrained_paths: Iterable[Optional[str]],
) -> None:
    """Refuse to write checkpoints onto any pretrained/F5 file path."""
    save_abs = os.path.abspath(save_path)
    for raw in pretrained_paths:
        if not raw:
            continue
        pre_abs = os.path.abspath(str(raw))
        if save_abs == pre_abs:
            raise RuntimeError(
                f"Refusing to overwrite pretrained weights: save_path={save_abs} "
                f"equals pretrained={pre_abs}. Use a separate output/ directory."
            )


def resolve_f5_state_dict(checkpoint: dict, state_key: str = "model_state") -> Dict[str, torch.Tensor]:
    if state_key in checkpoint:
        state = checkpoint[state_key]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state = checkpoint["model"]
    else:
        # Raw state_dict
        state = checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"F5 checkpoint state is not a dict (key={state_key!r})")
    return state
