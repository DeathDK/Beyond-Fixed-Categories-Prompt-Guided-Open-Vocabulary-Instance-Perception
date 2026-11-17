#!/usr/bin/env python
"""Filter YOLO seg_label files for open-world K-class experiments.

Images and disparity files are never deleted. Only instance label lines whose
annotation class is outside the known set are removed when writing filtered
copies under data/KITTI15/training/seg_label_openworld/<split>/.

Example:
  python tools/prepare_openworld.py --split K4
  python tools/prepare_openworld.py --split all --audit
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "./")

from openworld.label_mapping import AnnotationSchema, annotation_id_to_canonical
from openworld.splits import (
    experiment_count_summary,
    get_split,
    list_splits_for_k,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def filter_label_file(
    src: Path,
    dst: Path,
    schema: AnnotationSchema,
) -> Counter:
    counts = Counter()
    lines_out = []
    if not src.exists():
        return counts
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            ann_id = int(float(parts[0]))
            canonical = annotation_id_to_canonical(ann_id)
            counts[f"raw:{canonical}"] += 1
            train_id = schema.annotation_id_to_train_id(ann_id)
            if train_id is None:
                counts[f"dropped:{canonical}"] += 1
                continue
            counts[f"kept:{canonical}"] += 1
            parts[0] = str(train_id)
            lines_out.append(" ".join(parts))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
        if lines_out:
            f.write("\n")
    return counts


def rewrite_split_list(
    src_list: Path,
    dst_list: Path,
    label_subdir: str,
) -> int:
    """Rewrite the 4th path column to point at filtered labels."""
    n = 0
    lines = []
    with src_list.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                lines.append(line.rstrip("\n"))
                continue
            old_label = parts[3].replace("\\", "/")
            filename = Path(old_label).name
            parts[3] = f"training/{label_subdir}/{filename}"
            lines.append(" ".join(parts))
            n += 1
    dst_list.parent.mkdir(parents=True, exist_ok=True)
    with dst_list.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return n


def prepare_split(split_name: str, data_root: Path, audit: bool) -> None:
    split = get_split(split_name)
    schema = AnnotationSchema(split.known_classes)
    src_label_dir = data_root / "training" / "seg_label"
    dst_label_dir = data_root / "training" / f"seg_label_openworld_{split.name.lower()}"
    if dst_label_dir.exists():
        shutil.rmtree(dst_label_dir)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    total = Counter()
    label_files = sorted(src_label_dir.glob("*.txt"))
    for src in label_files:
        total.update(filter_label_file(src, dst_label_dir / src.name, schema))

    for list_name in (
        "kitti15_train180_with_seglabel.txt",
        "kitti15_val20_with_seglabel.txt",
    ):
        src_list = data_root / list_name
        if not src_list.exists():
            continue
        dst_list = data_root / list_name.replace(
            "_with_seglabel.txt",
            f"_with_seglabel_{split.name.lower()}.txt",
        )
        n = rewrite_split_list(
            src_list,
            dst_list,
            label_subdir=dst_label_dir.name,
        )
        print(f"[{split.name}] wrote {dst_list.name} ({n} rows)")

    print(f"[{split.name}] known={list(split.known_classes)} nc={schema.num_classes}")
    print(f"[{split.name}] label dir -> {dst_label_dir}")
    for key in sorted(total):
        print(f"  {key}: {total[key]}")

    if audit:
        dropped_unknown = sum(
            v for k, v in total.items() if k.startswith("dropped:")
        )
        kept = sum(v for k, v in total.items() if k.startswith("kept:"))
        # Ensure no unknown canonical remains in kept counts.
        for canon in split.unknown_canonical_classes:
            kept_key = f"kept:{canon}"
            if total.get(kept_key, 0) != 0:
                raise RuntimeError(
                    f"AUDIT FAIL {split.name}: kept unknown class {canon}"
                )
        print(
            f"[{split.name}] AUDIT OK kept={kept} dropped_unknown={dropped_unknown}"
        )


def main():
    parser = argparse.ArgumentParser(description="Prepare open-world seg labels")
    parser.add_argument(
        "--split",
        type=str,
        default="K4",
        help="F5 / K4-2 / K4_leave_Truck / ... or 'all'",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(_project_root() / "data" / "KITTI15"),
    )
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not (data_root / "training" / "seg_label").exists():
        raise FileNotFoundError(f"Missing seg_label under {data_root}")

    if args.split.lower() == "all":
        names = ["F5"]
        for k in (4, 3, 2, 1):
            names.extend(s.name for s in list_splits_for_k(k))
        print("preparing", experiment_count_summary())
    else:
        names = [args.split]
    for name in names:
        prepare_split(name, data_root, audit=args.audit or True)


if __name__ == "__main__":
    main()
