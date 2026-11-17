import os
import torch.utils.data as torch_data
import numpy as np
from PIL import Image
from .dataset_template import DatasetTemplate
from stereo.utils.common_utils import get_pos_fullres

try:
    from openworld.label_mapping import AnnotationSchema
except ImportError:  # pragma: no cover
    AnnotationSchema = None


class KittiDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.return_right_disp = self.data_info.RETURN_RIGHT_DISP
        self.use_noc = self.data_info.get('USE_NOC', False)
        if hasattr(self.data_info, 'RETURN_POS'):
            self.retrun_pos = self.data_info.RETURN_POS
        else:
            self.retrun_pos = False
            
        # Instance segmentation related params
        self.max_points = 100  # max polygon points
        self.min_points = 4   # min polygon points

        # Open-world K-class filtering (strict leave-out of unknown labels).
        known_classes = None
        if hasattr(self.data_info, 'KNOWN_CLASSES') and self.data_info.KNOWN_CLASSES:
            known_classes = list(self.data_info.KNOWN_CLASSES)
        elif hasattr(data_cfg, 'MODEL') and data_cfg.MODEL.get('KNOWN_CLASSES', None):
            known_classes = list(data_cfg.MODEL.KNOWN_CLASSES)
        self.annotation_schema = None
        if known_classes is not None:
            if AnnotationSchema is None:
                raise ImportError("openworld.label_mapping.AnnotationSchema is required")
            self.annotation_schema = AnnotationSchema(known_classes)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths[:3]

        if self.use_noc:
            disp_img_path = disp_img_path.replace('disp_occ', 'disp_noc')
        # image
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        # disp
        disp_img = np.array(Image.open(disp_img_path), dtype=np.float32) / 256.0

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
        }

        # Modified: load instance segmentation labels
        label_path = os.path.join(self.root, item[3]).replace('\\', '/') if len(item) > 3 else None
        if label_path:
            sample['seg_labels'] = self._load_yolo_seg_labels(label_path, left_img.shape[:2])
            if self.annotation_schema is not None:
                sample['seg_labels'] = self.annotation_schema.filter_training_labels(
                    sample['seg_labels']
                )
                for lab in sample['seg_labels']:
                    lab['id_space'] = 'train'
        else:
            sample['seg_labels'] = []
        #print(sample['seg_labels'])

        if self.return_right_disp:
            disp_img_right_path = disp_img_path.replace('c_0', 'c_1')
            disp_img_right = np.array(Image.open(disp_img_right_path), dtype=np.float32) / 256.0
            sample['disp_right'] = disp_img_right

        if self.retrun_pos and self.mode == 'training':
            sample['pos'] = get_pos_fullres(800, sample['left'].shape[1], sample['left'].shape[0])

        sample = self.transform(sample)  # apply data augmentation
        # DatasetTemplate.visualizer.visualize_sample(sample)
        sample['index'] = idx
        sample['name'] = left_img_path
        #print(sample)

        return sample
    
    def _load_yolo_seg_labels(self, label_path, img_shape):
        """Load YOLO-format instance segmentation label file."""
        labels = []
        img_height, img_width = img_shape
        
        # If label file does not exist, return empty list
        if not os.path.exists(label_path):
            return labels
            
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if not parts:
                    continue
                    
                try:
                    # Parse class ID and polygon points
                    class_id = int(parts[0])
                    polygon_norm = list(map(float, parts[1:]))
                    
                    # Convert to pixel coordinates
                    polygon_pixel = []
                    for i in range(0, len(polygon_norm), 2):
                        if i + 1 >= len(polygon_norm):
                            break
                        x = polygon_norm[i] * img_width
                        y = polygon_norm[i+1] * img_height
                        polygon_pixel.extend([x, y])
                    
                    # Ensure polygon point count is within a reasonable range
                    if len(polygon_pixel) < self.min_points * 2:
                        continue
                    if len(polygon_pixel) > self.max_points * 2:
                        polygon_pixel = polygon_pixel[:self.max_points*2]
                    
                    # Compute bounding box from polygon
                    xs = polygon_pixel[0::2]  # all x coords
                    ys = polygon_pixel[1::2]  # all y coords
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    # Store label info
                    labels.append({
                        'class_id': class_id,
                        'polygon': polygon_pixel,  # polygon points in pixel coords
                        'bbox': [x_min, y_min, x_max, y_max],  # bounding box
                        'bbox_norm': [  # normalized bounding box
                            (x_min + x_max) / 2 / img_width,  # x_center
                            (y_min + y_max) / 2 / img_height,  # y_center
                            (x_max - x_min) / img_width,  # width
                            (y_max - y_min) / img_height   # height
                        ]
                    })
                except Exception as e:
                    print(f"Error parsing line in {label_path}: {line.strip()} - {str(e)}")
                    continue
                    
        return labels