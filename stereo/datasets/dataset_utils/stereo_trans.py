# @Time    : 2025/8/9 
# @Author  : glyy
import random
import torch
import numpy as np
import cv2
from torchvision.transforms.functional import normalize
from PIL import Image
from torchvision.transforms import ColorJitter


def clip_polygon_to_rect(polygon, xmin, ymin, xmax, ymax):
    """Clip an Nx2 polygon to an axis-aligned rectangle."""
    points = [np.asarray(point, dtype=np.float32) for point in polygon]

    def clip_edge(vertices, inside, intersect):
        if not vertices:
            return []
        output = []
        previous = vertices[-1]
        previous_inside = inside(previous)
        for current in vertices:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersect(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersect(previous, current))
            previous = current
            previous_inside = current_inside
        return output

    def vertical_intersection(first, second, x_value):
        delta = second - first
        denominator = delta[0] if abs(delta[0]) > 1e-12 else 1.0
        ratio = (x_value - first[0]) / denominator
        return np.array([x_value, first[1] + ratio * delta[1]], dtype=np.float32)

    def horizontal_intersection(first, second, y_value):
        delta = second - first
        denominator = delta[1] if abs(delta[1]) > 1e-12 else 1.0
        ratio = (y_value - first[1]) / denominator
        return np.array([first[0] + ratio * delta[0], y_value], dtype=np.float32)

    points = clip_edge(points, lambda p: p[0] >= xmin, lambda a, b: vertical_intersection(a, b, xmin))
    points = clip_edge(points, lambda p: p[0] <= xmax, lambda a, b: vertical_intersection(a, b, xmax))
    points = clip_edge(points, lambda p: p[1] >= ymin, lambda a, b: horizontal_intersection(a, b, ymin))
    points = clip_edge(points, lambda p: p[1] <= ymax, lambda a, b: horizontal_intersection(a, b, ymax))
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class TransposeImage(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        sample['left'] = sample['left'].transpose((2, 0, 1))
        sample['right'] = sample['right'].transpose((2, 0, 1))
        return sample


class ToTensor(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        for k in list(sample.keys()):
            if isinstance(sample[k], np.ndarray):
                if k == 'super_pixel_label':
                    sample[k] = torch.from_numpy(sample[k].copy()).to(torch.int32)
                elif k in ['occ_mask', 'occ_mask_2']:
                    sample[k] = torch.from_numpy(sample[k].copy()).to(torch.bool)
                else:
                    sample[k] = torch.from_numpy(sample[k].copy()).to(torch.float32)
            elif isinstance(sample[k], (list, tuple)) and k == 'seg_labels':
                processed_labels = []
                for obj in sample[k]:
                    if not all(key in obj for key in ['class_id', 'bbox', 'bbox_norm', 'polygon']):
                        continue
                    # Flatten polygon to 1D list then to tensor
                    poly = np.array(obj['polygon'], dtype=np.float32).reshape(-1)
                    bbox = np.array(obj['bbox'], dtype=np.float32).reshape(4)
                    bbox_norm = np.array(obj['bbox_norm'], dtype=np.float32).reshape(4)

                    processed_obj = {
                        'class_id': torch.tensor(int(obj['class_id']), dtype=torch.int64),
                        'polygon': torch.from_numpy(poly).to(torch.float32),   # 1D tensor: [x0,y0,x1,y1,...]
                        'bbox': torch.from_numpy(bbox).to(torch.float32),     # [x_min, y_min, x_max, y_max]
                        'bbox_norm': torch.from_numpy(bbox_norm).to(torch.float32)
                    }
                    # Keep open-world remapping flags so loss uses train IDs.
                    if obj.get('id_space') is not None:
                        processed_obj['id_space'] = obj['id_space']
                    if obj.get('remapped_to_train'):
                        processed_obj['remapped_to_train'] = True
                    if obj.get('canonical_name') is not None:
                        processed_obj['canonical_name'] = obj['canonical_name']
                    processed_labels.append(processed_obj)
                sample[k] = processed_labels
            elif isinstance(sample[k], (int, float)):
                sample[k] = torch.tensor(sample[k], dtype=torch.float32)
        return sample


class NormalizeImage(object):
    def __init__(self, config):
        self.mean = config.MEAN
        self.std = config.STD

    def __call__(self, sample):
        sample['left'] = normalize(sample['left'] / 255.0, self.mean, self.std)
        sample['right'] = normalize(sample['right'] / 255.0, self.mean, self.std)
        return sample


class RandomCrop(object):
    def __init__(self, config):
        self.crop_size = config.SIZE
        self.base_size = config.SIZE
        self.y_jitter = config.get('Y_JITTER', False)

    def __call__(self, sample):
        crop_h, crop_w = self.crop_size
        height, width = sample['left'].shape[:2]
        if crop_w > width or crop_h > height:
            return sample

        n_pixels = 2 if (self.y_jitter and np.random.rand() < 0.5) else 0
        y1 = random.randint(n_pixels, height - crop_h - n_pixels)
        x1 = random.randint(0, width - crop_w)

        # Unified crop region: y1:y1+crop_h, x1:x1+crop_w
        y0 = y1
        x0 = x1
        y_end = y0 + crop_h
        x_end = x0 + crop_w

        for k in list(sample.keys()):
            if k == 'seg_labels':
                continue
            if k in ['right']:
                sample[k] = sample[k][y0:y_end, x0:x_end]
            elif k in ['pos']:
                sample[k] = sample[k][:, y0:y_end, x0:x_end]
            elif isinstance(sample[k], np.ndarray):
                sample[k] = sample[k][y0:y_end, x0:x_end]

        # Process instance segmentation labels
        if 'seg_labels' in sample:
            new_labels = []
            for label in sample['seg_labels']:
                poly = np.array(label['polygon']).reshape(-1, 2)
                orig_area = max(1e-6, abs(cv2.contourArea(poly.astype(np.float32))))

                # Shift to crop coordinates and compute the actual polygon-box
                # intersection instead of clamping vertices independently.
                poly[:, 0] -= x0
                poly[:, 1] -= y0
                poly = clip_polygon_to_rect(
                    poly, 0.0, 0.0, crop_w - 1.0, crop_h - 1.0
                )
                if len(poly) < 3:
                    continue
                clipped_area = abs(cv2.contourArea(poly.astype(np.float32)))

                if clipped_area / orig_area > 0.3:
                    x_min_new, y_min_new = poly[:, 0].min(), poly[:, 1].min()
                    x_max_new, y_max_new = poly[:, 0].max(), poly[:, 1].max()
                    xc = (x_min_new + x_max_new) / 2.0 / crop_w
                    yc = (y_min_new + y_max_new) / 2.0 / crop_h
                    bw = (x_max_new - x_min_new) / crop_w
                    bh = (y_max_new - y_min_new) / crop_h
                    new_label = {
                        'class_id': label['class_id'],
                        'polygon': poly.flatten().tolist(),
                        'bbox': [float(x_min_new), float(y_min_new), float(x_max_new), float(y_max_new)],
                        'bbox_norm': [float(xc), float(yc), float(bw), float(bh)]
                    }
                    if label.get('id_space') is not None:
                        new_label['id_space'] = label['id_space']
                    if label.get('remapped_to_train'):
                        new_label['remapped_to_train'] = True
                    if label.get('canonical_name') is not None:
                        new_label['canonical_name'] = label['canonical_name']
                    new_labels.append(new_label)
            sample['seg_labels'] = new_labels
        return sample


class RandomSparseScale(object):
    def __init__(self, config):
        self.crop_size = config.SIZE
        self.min_scale = config.MIN_SCALE
        self.max_scale = config.MAX_SCALE
        self.scale_prob = config.SCALE_PROB

    def __call__(self, sample):
        ht, wd = sample['left'].shape[:2]
        min_scale = np.maximum((self.crop_size[0] + 1) / float(ht), (self.crop_size[1] + 1) / float(wd))
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)
        if np.random.rand() < self.scale_prob:
            # Process images
            for k in sample.keys():
                if k in ['left', 'right']:
                    sample[k] = cv2.resize(sample[k], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                elif k in ['disp', 'disp_right']:
                    sample[k] = self.sparse_disp_map_reisze(sample[k], fx=scale_x, fy=scale_y)

            # Update instance segmentation annotations
            if 'seg_labels' in sample:
                for label in sample['seg_labels']:
                    polygon = np.array(label['polygon']).reshape(-1, 2)
                    polygon[:, 0] *= scale_x
                    polygon[:, 1] *= scale_y
                    label['polygon'] = polygon.flatten().tolist()

                    x_min, y_min, x_max, y_max = label['bbox']
                    x_min *= scale_x; x_max *= scale_x
                    y_min *= scale_y; y_max *= scale_y
                    label['bbox'] = [x_min, y_min, x_max, y_max]

                    # Recompute bbox_norm using scaled image size
                    new_w = wd * scale_x
                    new_h = ht * scale_y
                    # Avoid division by zero
                    if new_w > 0 and new_h > 0:
                        xc = (x_min + x_max) / 2.0
                        yc = (y_min + y_max) / 2.0
                        bw = max(0.0, x_max - x_min)
                        bh = max(0.0, y_max - y_min)
                        label['bbox_norm'] = [xc / new_w, yc / new_h, bw / new_w, bh / new_h]
        return sample

    @staticmethod
    def sparse_disp_map_reisze(disp, fx=1.0, fy=1.0):
        ht, wd = disp.shape[:2]
        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))
        valid = disp > 0.0
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1).reshape(-1, 2).astype(np.float32)
        disp = disp.reshape(-1).astype(np.float32)
        valid = valid.reshape(-1)
        coords = coords[valid]
        disp = disp[valid]
        coords = coords * [fx, fy]
        disp = disp * fx
        xx = np.round(coords[:, 0]).astype(np.int32)
        yy = np.round(coords[:, 1]).astype(np.int32)
        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]; yy = yy[v]; disp = disp[v]
        resized_disp = np.zeros([ht1, wd1], dtype=np.float32)
        resized_disp[yy, xx] = disp
        return resized_disp


# Keep only boxes that are not inside erased regions
class RandomErase(object):
    def __init__(self, config):
        self.eraser_aug_prob = config.PROB
        self.max_erase_time = config.MAX_TIME
        self.bounds = config.BOUNDS

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        ht, wd = img1.shape[:2]

        # Initialize erase region params (default: no erase)
        erase_regions = []  # store all erase regions (x0, y0, dx, dy)

        # Handle super_pixel_label (keep original logic)
        if 'super_pixel_label' in sample:
            occ_mask_2 = np.zeros((ht, wd), dtype=bool)

        # Apply erase augmentation
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            erase_times = np.random.randint(1, self.max_erase_time + 1)
            
            for _ in range(erase_times):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(self.bounds[0], self.bounds[1])
                dy = np.random.randint(self.bounds[0], self.bounds[1])
                
                # Perform erase
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
                erase_regions.append((x0, y0, dx, dy))  # record erase region
                
                if 'super_pixel_label' in sample:
                    occ_mask_2 = np.zeros((ht, wd), dtype=bool)

        # Update sample data
        sample['left'] = img1
        sample['right'] = img2
        if 'super_pixel_label' in sample:
            sample['occ_mask_2'] = occ_mask_2

        # RandomErase simulates occlusion in the right stereo view only.  The
        # unchanged left view supplies segmentation supervision, so its instance
        # labels must remain untouched.

        return sample


# Color-only augmentation
class StereoColorJitter(object):
    def __init__(self, config):
        self.brightness = list(config.BRIGHTNESS)
        self.contrast = list(config.CONTRAST)
        self.saturation = list(config.SATURATION)
        if isinstance(config.HUE, float):
            config.HUE = [-config.HUE, config.HUE]
        self.hue = list(config.HUE)
        self.asymmetric_color_aug_prob = config.ASYMMETRIC_PROB
        self.color_jitter = ColorJitter(brightness=self.brightness, contrast=self.contrast,
                                        saturation=self.saturation, hue=[x / 3.14 for x in self.hue])

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.color_jitter(Image.fromarray(img1.astype(np.uint8))), dtype=np.uint8)
            img2 = np.array(self.color_jitter(Image.fromarray(img2.astype(np.uint8))), dtype=np.uint8)
        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0).astype(np.uint8)
            image_stack = np.array(self.color_jitter(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        sample['left'] = img1
        sample['right'] = img2

        return sample


class RightTopPad(object):
    def __init__(self, config):
        # Expected as (th, tw)
        self.size = config.SIZE

    def __call__(self, sample):
        # Original image size
        H0, W0 = sample['left'].shape[:2]
        th, tw = self.size
        # Keep original size for eval metrics that unpad predictions / GT masks.
        sample['orig_size'] = (int(H0), int(W0))

        # Pad only right/top (no cropping)
        pad_left   = 0
        pad_right  = max(0, tw - W0)
        pad_top    = max(0, th - H0)
        pad_bottom = 0

        # Pad images / regular arrays
        for k in list(sample.keys()):
            if k in ['left', 'right']:
                # HWC
                sample[k] = np.pad(sample[k],
                                   ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                   mode='edge')
            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                # HW
                sample[k] = np.pad(sample[k],
                                   ((pad_top, pad_bottom), (pad_left, pad_right)),
                                   mode='constant', constant_values=0)

        # New image size after padding
        H1 = H0 + pad_top + pad_bottom
        W1 = W0 + pad_left + pad_right

        # Fix instance segmentation annotations
        if 'seg_labels' in sample:
            for label in sample['seg_labels']:
                # 1) polygon: add pad_top to y (x pad is on the right, so x unchanged)
                if 'polygon' in label and len(label['polygon']) > 0:
                    poly = label['polygon']
                    if isinstance(poly, torch.Tensor):
                        poly = poly.detach().cpu().numpy()
                    else:
                        poly = np.array(poly, dtype=np.float32)
                    poly = poly.reshape(-1, 2)
                    poly[:, 0] += pad_left
                    poly[:, 1] += pad_top
                    label['polygon'] = poly.reshape(-1).tolist()

                # 2) bbox (xyxy): add pad_top to y1/y2; x unchanged
                if 'bbox' in label and label['bbox'] is not None and len(label['bbox']) == 4:
                    bbox = label['bbox']
                    if isinstance(bbox, torch.Tensor):
                        bbox = bbox.detach().cpu().numpy()
                    bbox = np.array(bbox, dtype=np.float32)
                    # x1, y1, x2, y2
                    bbox[0] += pad_left
                    bbox[1] += pad_top
                    bbox[2] += pad_left
                    bbox[3] += pad_top
                    # Optional: clip to new image bounds
                    bbox[0] = float(np.clip(bbox[0], 0, W1 - 1))
                    bbox[2] = float(np.clip(bbox[2], 0, W1 - 1))
                    bbox[1] = float(np.clip(bbox[1], 0, H1 - 1))
                    bbox[3] = float(np.clip(bbox[3], 0, H1 - 1))
                    label['bbox'] = bbox.tolist()

                # 3) bbox_norm (xc, yc, w, h in [0,1])
                # Recompute with new size
                if 'bbox_norm' in label and label['bbox_norm'] is not None and len(label['bbox_norm']) == 4:
                    bn = label['bbox_norm']
                    if isinstance(bn, torch.Tensor):
                        bn = bn.detach().cpu().numpy().tolist()
                    xc_n, yc_n, bw_n, bh_n = [float(v) for v in bn]

                    # Old normalized -> pixels (based on old size W0, H0)
                    xc_pix = xc_n * W0
                    yc_pix = yc_n * H0
                    bw_pix = bw_n * W0
                    bh_pix = bh_n * H0

                    # Add pad offset (objects are translated, not scaled)
                    xc_pix += pad_left
                    yc_pix += pad_top

                    # Re-normalize by new size W1, H1
                    xc_new = xc_pix / W1
                    yc_new = yc_pix / H1
                    bw_new = bw_pix / W1
                    bh_new = bh_pix / H1

                    # Clip to [0,1]
                    xc_new = float(np.clip(xc_new, 0.0, 1.0))
                    yc_new = float(np.clip(yc_new, 0.0, 1.0))
                    bw_new = float(np.clip(bw_new, 0.0, 1.0))
                    bh_new = float(np.clip(bh_new, 0.0, 1.0))

                    label['bbox_norm'] = [xc_new, yc_new, bw_new, bh_new]

                # If bbox_norm is missing, recover from bbox or polygon (for consistency; optional)
                elif 'bbox_norm' not in label or label['bbox_norm'] is None:
                    if 'bbox' in label and label['bbox'] is not None and len(label['bbox']) == 4:
                        x1, y1, x2, y2 = [float(v) for v in label['bbox']]
                        xc = (x1 + x2) / 2.0
                        yc = (y1 + y2) / 2.0
                        bw = max(0.0, x2 - x1)
                        bh = max(0.0, y2 - y1)
                        label['bbox_norm'] = [
                            float(np.clip(xc / W1, 0.0, 1.0)),
                            float(np.clip(yc / H1, 0.0, 1.0)),
                            float(np.clip(bw / W1, 0.0, 1.0)),
                            float(np.clip(bh / H1, 0.0, 1.0)),
                        ]
                    elif 'polygon' in label and len(label['polygon']) > 0:
                        poly = np.array(label['polygon'], dtype=np.float32).reshape(-1, 2)
                        x1, y1 = float(np.min(poly[:, 0])), float(np.min(poly[:, 1]))
                        x2, y2 = float(np.max(poly[:, 0])), float(np.max(poly[:, 1]))
                        xc = (x1 + x2) / 2.0
                        yc = (y1 + y2) / 2.0
                        bw = max(0.0, x2 - x1)
                        bh = max(0.0, y2 - y1)
                        label['bbox_norm'] = [
                            float(np.clip(xc / W1, 0.0, 1.0)),
                            float(np.clip(yc / H1, 0.0, 1.0)),
                            float(np.clip(bw / W1, 0.0, 1.0)),
                            float(np.clip(bh / H1, 0.0, 1.0)),
                        ]

        return sample
