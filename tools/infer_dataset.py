import sys
import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
from easydict import EasyDict
from PIL import Image
import cv2
from torchvision.ops import batched_nms
import matplotlib.pyplot as plt
import glob
import torch.nn.functional as F

sys.path.insert(0, './')

from stereo.utils import common_utils
from stereo.modeling.models.lightstereo.trainer import Trainer
from stereo.utils.disp_color import disp_to_color
from stereo.datasets.dataset_template import build_transform_by_cfg
from stereo.modeling.models.lightstereo.infer_module import crop_mask, tensor2SEG, scale_boxes, process_mask_native, remove_right_top_padding
from stereo.utils.class_schema import KITTI_INSTANCE_CLASS_NAMES, model_ids_to_annotation_ids
import time

total_time = 0

# Class names
class_names = list(KITTI_INSTANCE_CLASS_NAMES)

# Fixed BGR color per class
# Unmapped / extended classes fall back to a reproducible hashed color
CLASS_COLORS = {
    0: (128, 128, 128),  # none -> gray
    1: (0, 0, 255),      # car -> red
    2: (0, 165, 255),    # trunk(=truck) -> orange
    3: (0, 255, 255),    # bus -> yellow
    4: (255, 0, 255),    # bicycle -> magenta
    5: (0, 255, 0),      # person -> green
}

def _hashed_color(idx: int):
    # Generate a reproducible hashed BGR color (avoid overly dark colors)
    rng = np.random.default_rng(seed=idx + 12345)
    base = rng.integers(low=64, high=256, size=3, dtype=np.int32)
    return int(base[0]), int(base[1]), int(base[2])

def get_class_color(cls_idx: int):
    # Return BGR color for the class
    if cls_idx in CLASS_COLORS:
        return CLASS_COLORS[cls_idx]
    return _hashed_color(int(cls_idx))

def map_boxes_back_right_top_pad(boxes_xyxy, in_h, in_w, H0, W0, th, tw):
    """
    Map boxes from RightTopPad space back to the original image.
    boxes_xyxy: model output boxes (in padded image coordinates)
    in_h, in_w: model input height/width (padded size)
    H0, W0: original image height/width
    th, tw: target height/width (384, 1248)
    """
    if not torch.is_tensor(boxes_xyxy):
        boxes = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
    else:
        boxes = boxes_xyxy.clone().float()
    
    # Compute padding amounts (consistent with RightTopPad)
    pad_top = max(0, th - H0)  # top padding
    pad_left = 0               # left padding (always 0)
    
    # Compute scale factor if any
    # Note: RightTopPad only pads, never rescales, so r=1
    r = 1.0
    
    # Map coordinates
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes[:, :4] /= r
    
    # Clamp to the original image bounds
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp_(0, W0 - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp_(0, H0 - 1)
    
    return boxes

def process_mask_native_right_top_pad(protos, masks_in, bboxes, shape, th, tw):
    """
    Mask post-processing tailored for RightTopPad.
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    
    # Upsample to target size (th, tw)
    masks = F.interpolate(masks.unsqueeze(1), (th, tw), mode="bilinear", align_corners=False).squeeze(1)
    
    # Compute padding amounts
    H0, W0 = shape
    pad_top = max(0, th - H0)
    pad_right = max(0, tw - W0)
    
    # Crop away padding and keep only the original image region
    pad_right = max(0, tw - W0)
    masks = remove_right_top_padding(masks, (H0, W0), (th, tw))
    
    # Further crop the mask using the bounding box
    masks = crop_mask(masks, bboxes)
    
    return masks.gt_(0.0)


def draw_boxes_and_masks_on_image_bgr(
    image_bgr,
    classes,
    confidence,
    boxes,
    masks=None,
    class_names=None,
    confidence_threshold=0.5,
    mask_alpha=0.5,
    out_path="result.jpg"
):
    img = image_bgr.copy()
    H, W = img.shape[:2]

    if torch.is_tensor(classes):
        classes = classes.detach().cpu().numpy().astype(int)
    else:
        classes = np.asarray(classes).astype(int)
    if torch.is_tensor(confidence):
        confidence = confidence.detach().cpu().numpy().astype(float)
    else:
        confidence = np.asarray(confidence).astype(float)
    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy().astype(float)
    else:
        boxes = np.asarray(boxes).astype(float)

    overlay = img.copy()

    if masks is not None:
        # Sort by confidence (high to low) so high-confidence instances render first
        valid_indices = [i for i in range(len(classes)) if confidence[i] >= confidence_threshold]
        if len(valid_indices) == 0:
            valid_indices = range(len(classes))
        order = np.argsort(-confidence[valid_indices])  # high-confidence first
        valid_indices = [valid_indices[i] for i in order]

        painted_mask = np.zeros((H, W), dtype=bool)  # mark already painted mask regions

        for i in valid_indices:
            # Extract mask
            if torch.is_tensor(masks):
                mask_i = masks[i].detach().cpu().numpy()
            else:
                mask_i = masks[i]
            # Ensure correct spatial size
            if mask_i.ndim != 2:
                mask_i = cv2.resize(
                    mask_i.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST
                )
            m_bool = (mask_i > 0.5)
            # Paint only regions not yet occupied by another instance
            m_unpainted = np.logical_and(m_bool, np.logical_not(painted_mask))
            if not m_unpainted.any():
                continue
            painted_mask[m_unpainted] = True

            cls_idx = int(classes[i])
            color = get_class_color(cls_idx)
            a = float(mask_alpha)
            overlay_region = overlay[m_unpainted]
            if overlay_region.size > 0:
                color_arr = np.array(color, dtype=np.float32)
                overlay_region = overlay_region.astype(np.float32) * (1.0 - a) + color_arr * a
                overlay[m_unpainted] = overlay_region.astype(np.uint8)

    img = overlay

    # Draw boxes and labels
    for i in range(len(classes)):
        score = float(confidence[i]) if i < len(confidence) else 1.0
        if score < confidence_threshold:
            continue
        box = boxes[i]
        x1, y1, x2, y2 = box.astype(int)
        x1 = np.clip(x1, 0, W - 1); x2 = np.clip(x2, 0, W - 1)
        y1 = np.clip(y1, 0, H - 1); y2 = np.clip(y2, 0, H - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        cls_idx = int(classes[i])
        color = get_class_color(cls_idx)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name = class_names[cls_idx] if (class_names and 0 <= cls_idx < len(class_names)) else str(cls_idx)
        label = f"{name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - th - 4)
        cv2.rectangle(img, (x1, y_text), (x1 + tw, y_text + th + 4), color, -1)
        cv2.putText(img, label, (x1, y_text + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if out_path:
        cv2.imwrite(out_path, img)

    return img


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default="cfgs/lightstereo/lightstereo_s_kitti.yaml", help='specify the config for eval')
    parser.add_argument('--pretrained_model', type=str, default="s-with-yolopt.pth", help='pretrained_model')
    parser.add_argument('--output_dir', type=str, default="./visual", help='output directory for results')
    parser.add_argument('--disable_amp', action='store_true', help='disable automatic mixed precision')
    args = parser.parse_args()

    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)
    if args.pretrained_model is not None:
        cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model

    args.run_mode = 'infer'
    return args, cfgs

tensor2SEG = tensor2SEG()

def process_image_pair(model, transform, left_path, right_path, output_dir, use_amp=True):
    global total_time
    # Create output subdirectories
    os.makedirs(os.path.join(output_dir, 'disparity'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'segmentation'), exist_ok=True)

    # Load images
    left_img = np.array(Image.open(left_path).convert('RGB'), dtype=np.float32)
    height, width, channels = left_img.shape
    right_img = np.array(Image.open(right_path).convert('RGB'), dtype=np.float32)

    # Prepare sample
    sample = {
        'left': left_img,
        'right': right_img,
    }
    sample = transform(sample)
    sample['left'] = sample['left'].unsqueeze(0)
    sample['right'] = sample['right'].unsqueeze(0)

    # Move to GPU if available and ensure float32
    for k, v in sample.items():
        if torch.is_tensor(v):
            sample[k] = v.cuda().float()  # ensure float32

    # Inference
    with torch.cuda.amp.autocast(enabled=False):
        time0 = time.time()
        model_pred = model(sample)
        infer_time = time.time()-time0
        print("Inference time:", infer_time, "ms")
        total_time += infer_time

    # Process disparity
    disp_pred = model_pred['disp_pred'].squeeze().cpu().numpy()
    img_color = disp_to_color(disp_pred, max_disp=192)
    img_color = img_color.astype('uint8')

    # Save disparity
    base_name = os.path.basename(left_path)
    disp_path = os.path.join(output_dir, 'disparity', base_name)
    cv2.imwrite(disp_path, cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR))

    # Process segmentation
    seg_pred = model_pred['seg_pred']
    classes, confidence, boxes, masks, proto = tensor2SEG(seg_pred)

    H0, W0 = height, width
    in_h, in_w = sample['left'].shape[-2:]
    th, tw = 384, 1248

    if len(boxes) > 0:
        boxes_net = boxes.squeeze(0)
        conf_1d = confidence.squeeze(0)
        classes_1d = model_ids_to_annotation_ids(classes.squeeze(0))
        masks_3d = masks.squeeze(0)

        # Use the RightTopPad-aware box mapping
        boxes_img = map_boxes_back_right_top_pad(boxes_net, in_h, in_w, H0, W0, th, tw)
        keep = batched_nms(boxes_img, conf_1d, classes_1d, iou_threshold=0.4)

        boxes_img = boxes_img[keep]
        conf_1d = conf_1d[keep]
        classes_1d = classes_1d[keep]
        masks_3d = masks_3d[keep]

        # Use the RightTopPad-aware mask processing
        mask_tensor = process_mask_native_right_top_pad(proto, masks_3d, boxes_img, (H0, W0), th, tw)

        image_bgr = cv2.imread(left_path)
        seg_path = os.path.join(output_dir, 'segmentation', base_name)

        draw_boxes_and_masks_on_image_bgr(
            image_bgr=image_bgr,
            classes=classes_1d,
            confidence=conf_1d,
            boxes=boxes_img,
            masks=mask_tensor,
            class_names=class_names,
            confidence_threshold=0.45,
            mask_alpha=0.8,
            out_path=seg_path
        )

@torch.no_grad()
def main():

    args, cfgs = parse_config()

    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
    else:
        local_rank = 0
        global_rank = 0

    torch.cuda.set_device(local_rank)
    seed = 0 if not args.dist_mode else dist.get_rank()
    common_utils.set_random_seed(seed=seed)

    logger = common_utils.create_logger(log_file=None, rank=local_rank)
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)

    trainer = Trainer(args, cfgs, local_rank, global_rank, logger, None)
    model = trainer.model
    model.eval()

    # Ensure model parameters are float32
    model = model.float()

    transform_config = cfgs.DATA_CONFIG.DATA_TRANSFORM.EVALUATING
    transform = build_transform_by_cfg(transform_config)

    # Prepare image pairs
    left_dir = "./data/KITTI15/training/image_2"
    right_dir = "./data/KITTI15/training/image_3"
    left_images = sorted(glob.glob(os.path.join(left_dir, "*.png")))
    right_images = sorted(glob.glob(os.path.join(right_dir, "*.png")))

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each image pair
    for left_path, right_path in zip(left_images, right_images):
        if os.path.basename(left_path) != os.path.basename(right_path):
            logger.warning(f"Filename mismatch: {left_path} and {right_path}")
            continue

        logger.info(f"Processing image pair: {left_path} and {right_path}")
        #try:
        process_image_pair(
            model,
            transform,
            left_path,
            right_path,
            args.output_dir,
            use_amp=not args.disable_amp  # whether to use mixed precision (currently unused)
        )
        # except Exception as e:
        #     logger.error(f"Error processing {left_path}: {str(e)}")
        #     continue
    print("avg infer time", total_time/400, "ms")

if __name__ == '__main__':
    main()
