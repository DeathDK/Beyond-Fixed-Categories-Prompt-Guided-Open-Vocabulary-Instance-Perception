import math
import os
from collections import Counter

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, Sampler


def collect_image_classes(dataset):
    """Return the set of class IDs present in every dataset image.

    When ``dataset.annotation_schema`` is set (open-world K-class leave-out),
    only known classes are counted and IDs are remapped to train space
    ``[0, K-1]`` so rare unknown classes do not inflate repeat factors.
    """
    if isinstance(dataset, ConcatDataset):
        image_classes = []
        for child in dataset.datasets:
            image_classes.extend(collect_image_classes(child))
        return image_classes

    if not hasattr(dataset, "data_list") or not hasattr(dataset, "root"):
        raise TypeError(
            "Class-aware sampling requires datasets with 'data_list' and 'root' attributes"
        )

    schema = getattr(dataset, "annotation_schema", None)
    image_classes = []
    for item in dataset.data_list:
        classes = set()
        if len(item) > 3:
            label_path = os.path.join(dataset.root, item[3])
            if os.path.isfile(label_path):
                with open(label_path, "r", encoding="utf-8") as label_file:
                    for line in label_file:
                        fields = line.strip().split()
                        if not fields:
                            continue
                        try:
                            raw_id = int(float(fields[0]))
                        except ValueError as error:
                            raise ValueError(
                                f"Invalid class ID in segmentation label: {label_path}"
                            ) from error
                        if schema is not None:
                            # Annotation IDs 1..5 → train ID; already-train IDs kept.
                            if 1 <= raw_id <= 5:
                                train_id = schema.annotation_id_to_train_id(raw_id)
                            elif 0 <= raw_id < schema.num_classes:
                                train_id = raw_id
                            else:
                                train_id = None
                            if train_id is None:
                                continue
                            classes.add(int(train_id))
                        else:
                            classes.add(raw_id)
        image_classes.append(classes)
    return image_classes


def capped_repeat_factors(image_classes, repeat_threshold=0.1, max_repeat_factor=3.0):
    """Compute capped LVIS-style repeat factors from per-image class presence."""
    if repeat_threshold <= 0:
        raise ValueError("repeat_threshold must be positive")
    if max_repeat_factor < 1:
        raise ValueError("max_repeat_factor must be at least 1")

    image_count = len(image_classes)
    if image_count == 0:
        raise ValueError("Cannot sample an empty dataset")

    class_image_counts = Counter(
        class_id for classes in image_classes for class_id in set(classes)
    )
    class_frequencies = {
        class_id: count / image_count
        for class_id, count in sorted(class_image_counts.items())
    }
    class_repeat_factors = {
        class_id: min(
            float(max_repeat_factor),
            max(1.0, math.sqrt(float(repeat_threshold) / frequency)),
        )
        for class_id, frequency in class_frequencies.items()
    }
    image_repeat_factors = [
        max((class_repeat_factors[class_id] for class_id in classes), default=1.0)
        for classes in image_classes
    ]
    return image_repeat_factors, class_frequencies, class_repeat_factors


class CappedClassAwareSampler(Sampler):
    """Preserve all images and append a capped number of rare-class repeats."""

    def __init__(
        self,
        dataset,
        repeat_threshold=0.1,
        max_repeat_factor=3.0,
        max_extra_fraction=0.1,
        seed=0,
        num_replicas=None,
        rank=None,
        image_classes=None,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank {rank} is invalid for {num_replicas} replicas")

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.image_classes = (
            collect_image_classes(dataset) if image_classes is None else image_classes
        )
        if len(self.image_classes) != len(dataset):
            raise ValueError("image_classes must contain one entry per dataset image")

        repeat_factors, class_frequencies, class_repeat_factors = capped_repeat_factors(
            self.image_classes,
            repeat_threshold=repeat_threshold,
            max_repeat_factor=max_repeat_factor,
        )
        self.weights = torch.as_tensor(repeat_factors, dtype=torch.double)
        if max_extra_fraction < 0:
            raise ValueError("max_extra_fraction must be non-negative")
        self.max_extra_fraction = float(max_extra_fraction)
        repeat_capacities = torch.ceil(self.weights).to(torch.long) - 1
        requested_extra_samples = int(round(len(dataset) * self.max_extra_fraction))
        self.extra_samples = min(
            requested_extra_samples,
            int(repeat_capacities.sum().item()),
        )
        self.epoch_size = len(dataset) + self.extra_samples
        self.num_samples = math.ceil(self.epoch_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.class_frequencies = class_frequencies
        self.class_repeat_factors = class_repeat_factors
        probabilities = self.weights / self.weights.sum()
        self.expected_class_frequencies = {
            class_id: sum(
                probabilities[index].item()
                for index, classes in enumerate(self.image_classes)
                if class_id in classes
            )
            for class_id in class_frequencies
        }

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        # Every original sample appears once. Extra slots are reserved only for
        # rare-class images and remain bounded by each image's repeat factor.
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        capacities = torch.ceil(self.weights).to(torch.long) - 1
        extra_weights = (self.weights - 1.0).clamp_min(0.0)
        for _ in range(self.extra_samples):
            probabilities = extra_weights * (capacities > 0).to(extra_weights.dtype)
            if probabilities.sum().item() <= 0:
                break
            index = int(torch.multinomial(probabilities, 1, generator=generator).item())
            indices.append(index)
            capacities[index] -= 1

        order = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[index] for index in order]
        if len(indices) < self.total_size:
            indices.extend(indices[:self.total_size - len(indices)])
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def summary(self):
        return {
            class_id: {
                "original_frequency": self.class_frequencies[class_id],
                "expected_frequency": self.expected_class_frequencies[class_id],
                "repeat_factor": self.class_repeat_factors[class_id],
            }
            for class_id in self.class_frequencies
        }
