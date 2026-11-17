# @Time    : 2025/6/24 
# @Author  : glyy
import os
import random
import torch.utils.data as torch_data
import torch
from .dataset_utils import stereo_trans
import cv2
import numpy as np
import time
import multiprocessing as mp

class DataVisualizer:
    """Data augmentation visualizer (supports instance segmentation)."""

    def __init__(self, save_dir, class_names, max_samples=100, save_prob=0.1,
                 always_save_first_n=5, verbose=True):
        """
        Args:
            save_dir (str): Directory to save images.
            class_names (list): List of class names.
            max_samples (int): Maximum number of samples to save.
            save_prob (float): Save probability (0-1).
            always_save_first_n (int): Always save the first N samples for debugging.
            verbose (bool): Whether to print logs.
        """
        self.save_dir = save_dir
        self.class_names = class_names
        self.max_samples = int(max_samples)
        self.save_prob = float(save_prob)
        self.always_save_first_n = int(always_save_first_n)
        self.verbose = bool(verbose)
        self.counter = 0

        # Create directory in a process-safe way
        try:
            is_main = (mp.current_process().name == 'MainProcess')
        except Exception:
            is_main = True
        if is_main and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        # Extra safety
        os.makedirs(self.save_dir, exist_ok=True)

        if self.verbose:
            print(f"Visualization output will be saved to: {self.save_dir} | "
                  f"max_samples={self.max_samples}, save_prob={self.save_prob}, "
                  f"always_save_first_n={self.always_save_first_n}")

    def visualize_sample(self, sample, index=None):
        """
        Visualize and save a sample (supports instance segmentation).
        Args:
            sample (dict): Data sample expected to contain 'left' and optional 'seg_labels'.
            index (int): Sample index (optional).
        """
        try:
            # Stop condition
            if self.counter >= self.max_samples:
                if self.verbose and self.counter == self.max_samples:
                    print("Reached max save count; stopping saves.")
                return

            # Required field checks
            if not isinstance(sample, dict):
                if self.verbose:
                    print("Visualization skipped: sample is not a dict.")
                return
            if 'left' not in sample:
                if self.verbose:
                    print("Visualization skipped: sample missing key 'left'.")
                return

            # Save gate: always save first N, then by probability
            must_save = (self.counter < self.always_save_first_n)
            if (not must_save) and (random.random() >= self.save_prob):
                return

            # Process image
            left_img = self._process_image(sample['left'])
            if left_img is None:
                if self.verbose:
                    print("Visualization skipped: _process_image returned None.")
                return

            # Draw instance segmentation if present
            if 'seg_labels' in sample and sample['seg_labels'] is not None:
                try:
                    self._draw_segmentation(left_img, sample['seg_labels'])
                except Exception as e:
                    print(f"Failed to draw instances (overall call): {str(e)}")

            # Generate unique filename (avoid collisions)
            ts_ns = time.time_ns()
            pid = os.getpid()
            cnt = self.counter
            idx = (index if index is not None else cnt)
            fname = f"left_{ts_ns}_{pid}_{idx}.png"
            save_path = os.path.join(self.save_dir, fname)

            # Save and validate
            ok = cv2.imwrite(save_path, left_img)
            if not ok:
                print(f"cv2.imwrite failed, path: {save_path}")
            else:
                if self.verbose:
                    print(f"[Save OK] {save_path}")

            # Increment counter only after a save attempt (counts save attempts)
            self.counter += 1

        except Exception as e:
            # Top-level fallback to avoid silent failures
            print(f"Visualization save failed: {str(e)}")

    def _process_image(self, img):
        """Process image data into a savable BGR uint8 format."""
        out = None

        # torch.Tensor
        if isinstance(img, torch.Tensor):
            t = img.detach().cpu()
            # Denormalize (simple heuristic: if range exceeds [0,1], apply ImageNet denorm)
            if t.min().item() < 0.0 or t.max().item() > 1.0:
                t = self._denormalize_tensor(t)
            # CHW -> HWC
            if t.ndim == 3 and t.shape[0] in (1, 3):
                t = t if t.shape[0] == 3 else t.expand(3, -1, -1)
                arr = t.numpy().transpose(1, 2, 0)  # HWC, RGB in [0,1]
            elif t.ndim == 2:
                arr = t.numpy()[..., None]  # H, W, 1
            else:
                arr = t.numpy()
            # Convert to uint8
            if arr.dtype != np.uint8:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            # Convert to BGR
            if arr.ndim == 3 and arr.shape[2] == 3:
                out = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                out = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.ndim == 2:
                out = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                out = arr

        # numpy.ndarray
        elif isinstance(img, np.ndarray):
            arr = img
            # If CHW, convert to HWC
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
                arr = arr if arr.shape[0] != 1 else np.repeat(arr, 3, axis=0)
                arr = arr.transpose(1, 2, 0)  # HWC

            # Normalize to uint8
            if arr.dtype != np.uint8:
                # Assume 0..1 float or other range
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

            # If single-channel, convert to BGR
            if arr.ndim == 2:
                out = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                out = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.ndim == 3 and arr.shape[2] == 3:
                # Default treat as RGB -> BGR for OpenCV
                out = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                out = arr
        else:
            # Unsupported type
            return None

        return out

    def _denormalize_tensor(self, tensor):
        """Denormalize image tensor (ImageNet mean/std)."""
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            return tensor * std + mean
        elif tensor.ndim == 2:
            return tensor  # leave grayscale unchanged
        else:
            return tensor

    def _draw_segmentation(self, image, seg_labels):
        """Draw instance segmentation on image (bbox from normalized xywh bbox_norm)."""
        h, w = image.shape[:2]
        drawn_image = image.copy()  # copy to avoid overwriting in-place mid-draw

        # Normalize seg_labels to an iterable list of dicts
        if isinstance(seg_labels, dict):
            seg_iter = [seg_labels]
        else:
            seg_iter = list(seg_labels)

        for label in seg_iter:
            try:
                # Process polygon points
                polygon = label.get('polygon', [])
                if isinstance(polygon, torch.Tensor):
                    polygon = polygon.detach().cpu().numpy()
                polygon = np.array(polygon, dtype=np.float32)
                if polygon.size == 0:
                    continue
                polygon = polygon.reshape(-1, 2).astype(np.float32)
                if len(polygon) < 3:
                    continue  # need at least 3 points

                # Class ID
                class_id = label.get('class_id', 0)
                if isinstance(class_id, torch.Tensor):
                    class_id = int(class_id.item())
                else:
                    class_id = int(class_id)

                # Use bbox_norm: [xc, yc, w, h], normalized to [0,1]
                bn = label.get('bbox_norm', None)
                if isinstance(bn, torch.Tensor):
                    bn = bn.detach().cpu().numpy().tolist()

                if bn is None:
                    # Fallback: use polygon bounding box
                    x1 = int(np.clip(np.min(polygon[:, 0]), 0, w - 1))
                    y1 = int(np.clip(np.min(polygon[:, 1]), 0, h - 1))
                    x2 = int(np.clip(np.max(polygon[:, 0]), 0, w - 1))
                    y2 = int(np.clip(np.max(polygon[:, 1]), 0, h - 1))
                else:
                    xc_n, yc_n, bw_n, bh_n = [float(v) for v in bn]
                    xc_n = float(np.clip(xc_n, 0.0, 1.0))
                    yc_n = float(np.clip(yc_n, 0.0, 1.0))
                    bw_n = max(0.0, float(bw_n))
                    bh_n = max(0.0, float(bh_n))

                    x1f = (xc_n - bw_n / 2.0) * w
                    y1f = (yc_n - bh_n / 2.0) * h
                    x2f = (xc_n + bw_n / 2.0) * w
                    y2f = (yc_n + bh_n / 2.0) * h

                    x1 = int(np.clip(round(x1f), 0, w - 1))
                    y1 = int(np.clip(round(y1f), 0, h - 1))
                    x2 = int(np.clip(round(x2f), 0, w - 1))
                    y2 = int(np.clip(round(y2f), 0, h - 1))

                    if x2 <= x1 or y2 <= y1:
                        x1 = int(np.clip(np.min(polygon[:, 0]), 0, w - 1))
                        y1 = int(np.clip(np.min(polygon[:, 1]), 0, h - 1))
                        x2 = int(np.clip(np.max(polygon[:, 0]), 0, w - 1))
                        y2 = int(np.clip(np.max(polygon[:, 1]), 0, h - 1))

                # Color
                color = self._get_color(class_id)

                # Draw polygon outline and semi-transparent fill
                poly_i32 = polygon.astype(np.int32)
                cv2.polylines(drawn_image, [poly_i32], isClosed=True, color=color, thickness=2)
                overlay = drawn_image.copy()
                cv2.fillPoly(overlay, [poly_i32], color)
                drawn_image = cv2.addWeighted(overlay, 0.3, drawn_image, 0.7, 0)

                # Draw bounding box
                cv2.rectangle(drawn_image, (x1, y1), (x2, y2), color, 2)

                # Add label text
                class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else str(class_id)
                area = float(cv2.contourArea(poly_i32))
                label_text = f"{class_name} {area:.1f}px"
                text_org = (x1, max(0, y1 - 10))
                cv2.putText(drawn_image, label_text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness=2)

            except Exception as e:
                print(f"Failed to draw instance: {str(e)}")
                continue

        # Copy drawn result back onto the original image
        image[:] = drawn_image[:]

    def _get_color(self, class_id):
        """Generate a fixed color for each class."""
        colors = [
            (255, 0, 0),    # red
            (0, 255, 0),    # green
            (0, 0, 255),    # blue
            (255, 255, 0),  # yellow
            (255, 0, 255),  # magenta
            (0, 255, 255),  # cyan
            (255, 128, 0),  # orange
            (128, 0, 255)   # purple-red
        ]
        return colors[int(class_id) % len(colors)]





def build_transform_by_cfg(transform_config):
    transform_compose = []
    for cur_cfg in transform_config:
        cur_augmentor = getattr(stereo_trans, cur_cfg.NAME)(config=cur_cfg)
        transform_compose.append(cur_augmentor)
    return stereo_trans.Compose(transform_compose)


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__()
        self.data_info = data_info
        self.data_cfg = data_cfg
        self.mode = mode
        self.root = self.data_info.DATA_PATH

        self.split_file = self.data_info.DATA_SPLIT[self.mode.upper()]
        self.data_list = []
        if os.path.exists(self.split_file):
            with open(self.split_file, 'r') as fp:
                self.data_list.extend([x.strip().split(' ') for x in fp.readlines()])

        transform_config = self.data_cfg.DATA_TRANSFORM[self.mode.upper()]
        self.transform = build_transform_by_cfg(transform_config)

        
    # visualizer = DataVisualizer(
    #     save_dir="visualization",
    #     class_names=["bus", "car", "trunk"], 
    #     max_samples=500,
    #     save_prob=0.8
    # )

    def __len__(self):
        return len(self.data_list)


def get_size(base_size, w_range, h_range, random_type):
    if random_type == 'range':
        w = random.randint(int(w_range[0] * base_size[1]), int(w_range[1] * base_size[1]))
        h = random.randint(int(h_range[0] * base_size[0]), int(h_range[1] * base_size[0]))
    elif random_type == 'choice':
        w = random.choice(w_range) if isinstance(w_range, list) else w_range
        h = random.choice(h_range) if isinstance(h_range, list) else h_range
    else:
        raise NotImplementedError
    return int(h), int(w)



# Custom collate because each image may have a different number of labels
def custom_collate(data_list, concat_dataset, batch_uniform=False,
                   random_type=None, h_range=None, w_range=None):
    # Handle batch_uniform logic (keep existing behavior)
    if batch_uniform:
        for each_dataset in concat_dataset.datasets:
            for cur_t in each_dataset.transform.transforms:
                if type(cur_t).__name__ == 'RandomCrop':
                    base_size = cur_t.base_size
                    cur_t.crop_size = get_size(base_size, w_range, h_range, random_type)
                    break

    # Separate stackable and non-stackable fields
    collated_batch = {
        # Stackable fields (images / disparity maps, etc.)
        'left': torch.stack([x['left'] for x in data_list]),
        'right': torch.stack([x['right'] for x in data_list]),
        'disp': torch.stack([x['disp'] for x in data_list]),
        
        # Keep as list (instance segmentation labels)
        'seg_labels': [x.get('seg_labels', []) for x in data_list],
        
        # Other metadata
        'index': torch.tensor([x['index'] for x in data_list]),
        'name': [x['name'] for x in data_list],
        'orig_size': [x.get('orig_size') for x in data_list],
    }

    # Special handling for pos field if present
    if 'pos' in data_list[0]:
        collated_batch['pos'] = torch.stack([x['pos'] for x in data_list])

    return collated_batch
