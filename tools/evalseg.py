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
import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict

sys.path.insert(0, './')

from stereo.utils import common_utils
from stereo.modeling.models.lightstereo.trainer import Trainer
from stereo.utils.disp_color import disp_to_color
from stereo.datasets.dataset_template import build_transform_by_cfg
from stereo.modeling.models.lightstereo.infer_module import crop_mask, tensor2SEG, scale_boxes, process_mask_native
from stereo.evaluation.mask_ap import MaskAPCalculator as ImageLocalMaskAPCalculator
from stereo.utils.class_schema import KITTI_INSTANCE_CLASS_NAMES, model_ids_to_annotation_ids

total_time = 0

# Class names
class_names = list(KITTI_INSTANCE_CLASS_NAMES)

def map_boxes_back_right_top_pad(boxes_xyxy, in_h, in_w, H0, W0, th, tw):
    """
    Map boxes from RightTopPad space back to the original image.
    """
    if not torch.is_tensor(boxes_xyxy):
        boxes = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
    else:
        boxes = boxes_xyxy.clone().float()
    
    pad_top = max(0, th - H0)
    pad_left = 0
    r = 1.0
    
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes[:, :4] /= r
    
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp_(0, W0 - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp_(0, H0 - 1)
    return boxes

def process_mask_native_right_top_pad(protos, masks_in, bboxes, shape, th, tw):
    """
    Mask post-processing tailored for RightTopPad.
    """
    c, mh, mw = protos.shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = F.interpolate(masks.unsqueeze(1), (th, tw), mode="bilinear", align_corners=False).squeeze(1)
    
    H0, W0 = shape
    pad_top = max(0, th - H0)
    pad_right = max(0, tw - W0) # Corrected logic for right padding
    
    # Crop the padded area
    masks = masks[:, pad_top:pad_top+H0, 0:W0] # RightTopPad means padding is on top and right
                                              # We should remove padding from top and right, so we select from left (0) to W0
                                              
    masks = crop_mask(masks, bboxes)
    return masks.gt_(0.0)

def generate_predictions(
    model,
    image_path,
    transform,
    confidence_threshold=0.45,
    max_detections=300,
):
    """
    Generates structured instance segmentation predictions for a given stereo image pair.

    This function takes the path to a left image, infers the right image path,
    loads and preprocesses the stereo pair, and runs the model to get predictions.
    The raw output is then post-processed, including coordinate mapping,
    Non-Maximum Suppression (NMS), confidence thresholding, and mask generation.

    Args:
        model: The trained instance segmentation model to use for inference.
        image_path (str): The file path to the left image. The right image is
                          expected to be in a corresponding 'image_3' directory.
        transform: The transformation pipeline to apply to the images before
                   feeding them to the model.
        confidence_threshold (float): The minimum confidence score required for
                                      a detection to be included in the final output.

    Returns:
        Dict: A dictionary containing the processed predictions with the following keys:
            'boxes' (np.ndarray): An array of bounding boxes in [x1, y1, x2, y2] format.
            'scores' (np.ndarray): An array of confidence scores for each detection.
            'classes' (np.ndarray): An array of class IDs for each detection.
            'masks' (np.ndarray): A boolean array of shape (N, H, W) with instance masks.
            'image_id' (str): The base name of the processed image file.
            'image_size' (tuple): The original (height, width) of the image.

    Raises:
        FileNotFoundError: If the corresponding right image cannot be found.
    """
    left_dir = os.path.dirname(image_path)
    right_dir = left_dir.replace("image_2", "image_3")
    right_path = os.path.join(right_dir, os.path.basename(image_path))
    if not os.path.exists(right_path):
        raise FileNotFoundError(f"Right image not found: {right_path}")

    left_img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
    height, width, channels = left_img.shape
    right_img = np.array(Image.open(right_path).convert('RGB'), dtype=np.float32)
    
    sample = {'left': left_img, 'right': right_img}
    sample = transform(sample)
    sample['left'] = sample['left'].unsqueeze(0)
    sample['right'] = sample['right'].unsqueeze(0)

    for k, v in sample.items():
        if torch.is_tensor(v):
            sample[k] = v.cuda().float()

    with torch.cuda.amp.autocast(enabled=False):
        model_pred = model(sample)

    seg_pred = model_pred['seg_pred']
    classes, confidence, boxes, masks, proto = tensor2SEG(seg_pred)
    H0, W0 = height, width
    in_h, in_w = sample['left'].shape[-2:]
    th, tw = 384, 1248

    boxes_img = torch.empty((0, 4), dtype=torch.float32, device='cpu')
    conf_1d = torch.empty((0,), dtype=torch.float32, device='cpu')
    classes_1d = torch.empty((0,), dtype=torch.int64, device='cpu')
    mask_tensor = torch.empty((0, H0, W0), dtype=torch.bool, device='cpu')

    if len(boxes) > 0:
        boxes_net = boxes.squeeze(0)
        conf_1d = confidence.squeeze(0)
        classes_1d = model_ids_to_annotation_ids(classes.squeeze(0))
        masks_3d = masks.squeeze(0)

        boxes_img = map_boxes_back_right_top_pad(boxes_net, in_h, in_w, H0, W0, th, tw)
        
        keep = batched_nms(boxes_img, conf_1d, classes_1d, iou_threshold=0.4)
        if max_detections > 0:
            keep = keep[:max_detections]
        boxes_img = boxes_img[keep]
        conf_1d = conf_1d[keep]
        classes_1d = classes_1d[keep]
        masks_3d = masks_3d[keep]

        if confidence_threshold > 0:
            conf_mask = conf_1d >= confidence_threshold
            boxes_img = boxes_img[conf_mask]
            conf_1d = conf_1d[conf_mask]
            classes_1d = classes_1d[conf_mask]
            masks_3d = masks_3d[conf_mask]

        if len(boxes_img) > 0:
            mask_tensor = process_mask_native_right_top_pad(proto, masks_3d, boxes_img, (H0, W0), th, tw)

    predictions = {
        'boxes': boxes_img.cpu().numpy() if torch.is_tensor(boxes_img) else boxes_img,
        'scores': conf_1d.cpu().numpy() if torch.is_tensor(conf_1d) else conf_1d,
        'classes': classes_1d.cpu().numpy() if torch.is_tensor(classes_1d) else classes_1d,
        'masks': mask_tensor.cpu().numpy() if torch.is_tensor(mask_tensor) else mask_tensor,
        'image_id': os.path.basename(image_path),
        'image_size': (H0, W0)
    }
    return predictions

def save_predictions_to_json(predictions_list: List[Dict], output_path: str):
    json_ready = []
    for pred in predictions_list:
        json_pred = {
            'image_id': pred['image_id'],
            'image_size': pred['image_size'],
            'boxes': pred['boxes'].tolist() if hasattr(pred['boxes'], 'tolist') else pred['boxes'],
            'scores': pred['scores'].tolist() if hasattr(pred['scores'], 'tolist') else pred['scores'],
            'classes': pred['classes'].tolist() if hasattr(pred['classes'], 'tolist') else pred['classes'],
            'masks': pred['masks'].tolist() if hasattr(pred['masks'], 'tolist') else pred['masks']
        }
        json_ready.append(json_pred)

    with open(output_path, 'w') as f:
        json.dump(json_ready, f, indent=2)

def load_ground_truths(gt_file_path: str, base_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all ground-truth annotations.
    Args:
        gt_file_path: path to the validation txt file list.
        base_dir: root directory of the KITTI dataset.
    Returns:
        A dict mapping image_id to its ground-truth data.
    """
    gts = {}
    with open(gt_file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            image_path = parts[0]
            label_path = parts[3]
            
            image_id = os.path.basename(image_path)
            full_label_path = os.path.join(base_dir, label_path)
            
            if not os.path.exists(full_label_path):
                print(f"Warning: Label file not found: {full_label_path}")
                continue

            # Get image size
            full_image_path = os.path.join(base_dir, image_path)
            with Image.open(full_image_path) as img:
                W, H = img.size

            gt_masks = []
            gt_classes = []
            
            with open(full_label_path, 'r') as lf:
                for obj_line in lf.readlines():
                    obj_parts = obj_line.strip().split()
                    class_id = int(obj_parts[0])
                    # YOLO polygon format starts with class_id, followed by normalized coords
                    poly_norm = np.array(obj_parts[1:], dtype=np.float32).reshape(-1, 2)
                    
                    # Denormalize polygon
                    poly = np.copy(poly_norm)
                    poly[:, 0] *= W
                    poly[:, 1] *= H
                    poly = poly.astype(np.int32)

                    # Create binary mask from polygon
                    mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(mask, [poly], 1)
                    
                    gt_masks.append(mask.astype(bool))
                    gt_classes.append(class_id)
            
            gts[image_id] = {
                'masks': np.array(gt_masks) if gt_masks else np.empty((0, H, W), dtype=bool),
                'classes': np.array(gt_classes) if gt_classes else np.empty((0,), dtype=int)
            }
    return gts

def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

class _LegacyMaskAPCalculator:
    """
    Helper class for computing instance-segmentation mAP.
    """
    def __init__(self, iou_thresholds: List[float] = None):
        self.iou_thresholds = iou_thresholds or np.linspace(0.5, 0.95, 10).tolist()
    
    def calculate_ap_per_class(self, preds: List[Dict], gts: List[Dict], iou_thresh: float, class_id: int):
        # Filter preds and gts for the current class
        class_preds = [p for p in preds if p['class_id'] == class_id]
        class_gts = [g for g in gts if g['class_id'] == class_id]

        if not class_preds:
            # No predictions for this class but GTs exist -> AP is 0
            return 0.0, 0, len(class_gts)
            
        num_gt = len(class_gts)
        # No GTs for this class -> skip / treat AP as 0 per evaluation convention
        if num_gt == 0:
            return 0.0, 0, 0  # alternatively return None and handle upstream

        # Sort predictions by score
        class_preds.sort(key=lambda x: x['score'], reverse=True)
        
        tps = np.zeros(len(class_preds))
        fps = np.zeros(len(class_preds))
        gt_matched = np.zeros(num_gt)

        for i, pred in enumerate(class_preds):
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, gt in enumerate(class_gts):
                if gt_matched[j]:
                    continue
                iou = calculate_mask_iou(pred['mask'], gt['mask'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_thresh:
                if not gt_matched[best_gt_idx]:
                    tps[i] = 1
                    gt_matched[best_gt_idx] = 1
                else:
                    fps[i] = 1 # Matched a GT that was already taken by a higher-score pred
            else:
                fps[i] = 1

        # Calculate precision and recall
        tp_cumsum = np.cumsum(tps)
        fp_cumsum = np.cumsum(fps)
        
        if num_gt > 0:
            recalls = tp_cumsum / num_gt
        else:
            recalls = np.zeros_like(tp_cumsum)
            
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6) # Avoid division by zero
        
        # Standard COCO-style AP calculation (area under PR curve)
        precisions = np.concatenate(([1.], precisions))
        recalls = np.concatenate(([0.], recalls))
        
        # Replace each precision value with the maximum precision value to its right
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i+1])

        # Find indices where recall changes
        i_recall_change = np.where(recalls[1:] != recalls[:-1])[0]

        # Calculate AP
        ap = np.sum((recalls[i_recall_change + 1] - recalls[i_recall_change]) * precisions[i_recall_change + 1])

        return ap, tp_cumsum[-1] if len(tp_cumsum) > 0 else 0, num_gt

    def evaluate(self, all_predictions: List[Dict], all_gts: Dict[str, Dict]):
        # (Upper half of this method is unchanged; computes per-class AP)
        # ...
        
        # <<<<<<<<<<<<<<<<<<<< START: NO CHANGE (EXISTING CODE) <<<<<<<<<<<<<<<<<<<<
        all_aps_per_iou = {iou_thresh: {} for iou_thresh in self.iou_thresholds}
        unique_classes = [i for i, name in enumerate(class_names) if name != "none"]

        preds_by_image = defaultdict(list)
        for pred_group in all_predictions:
            image_id = pred_group['image_id']
            for i in range(len(pred_group['scores'])):
                preds_by_image[image_id].append({
                    'score': pred_group['scores'][i],
                    'class_id': pred_group['classes'][i],
                    'mask': pred_group['masks'][i]
                })

        gts_by_image = defaultdict(list)
        for image_id, gt_group in all_gts.items():
            for i in range(len(gt_group['classes'])):
                gts_by_image[image_id].append({
                    'class_id': gt_group['classes'][i],
                    'mask': gt_group['masks'][i]
                })

        for iou_thresh in self.iou_thresholds:
            print(f"\\n--- Calculating AP for IoU threshold: {iou_thresh:.2f} ---")
            for class_id in unique_classes:
                class_preds_all_images = []
                class_gts_all_images = []
                
                relevant_image_ids = {img_id for img_id, gt_list in gts_by_image.items() if any(g['class_id'] == class_id for g in gt_list)}
                relevant_image_ids.update({img_id for img_id, pred_list in preds_by_image.items() if any(p['class_id'] == class_id for p in pred_list)})

                for image_id in relevant_image_ids:
                    class_preds_all_images.extend([p for p in preds_by_image.get(image_id, []) if p['class_id'] == class_id])
                    class_gts_all_images.extend([g for g in gts_by_image.get(image_id, []) if g['class_id'] == class_id])

                ap, total_tp, total_gt = self.calculate_ap_per_class(class_preds_all_images, class_gts_all_images, iou_thresh, class_id)
                all_aps_per_iou[iou_thresh][class_id] = ap
                
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                print(f"AP for class '{class_name}': {ap:.4f} (TPs: {int(total_tp)}, GTs: {total_gt})")
        
        mean_aps = {}
        for iou_thresh, class_aps in all_aps_per_iou.items():
            valid_aps = [ap for ap in class_aps.values() if ap is not None]
            if valid_aps:
                mean_aps[iou_thresh] = np.mean(valid_aps)
            else:
                mean_aps[iou_thresh] = 0.0

        map50 = mean_aps.get(0.50, 0.0)
        map75 = mean_aps.get(0.75, 0.0)
        map50_95 = np.mean(list(mean_aps.values())) if mean_aps else 0.0
        # >>>>>>>>>>>>>>>>>>>> END: NO CHANGE (EXISTING CODE) >>>>>>>>>>>>>>>>>>>>


        # =================================================================
        #  Added: compute an "all-class" mAP treating every class as one
        # =================================================================
        print("\\n" + "--- Calculating AP for 'all' classes combined ---")
        
        all_class_aps_per_iou = {}
        for iou_thresh in self.iou_thresholds:
            # 1. Prepare data by forcing every class_id to 0
            all_preds_single_class = []
            for pred_group in all_predictions:
                for i in range(len(pred_group['scores'])):
                    all_preds_single_class.append({
                        'score': pred_group['scores'][i],
                        'class_id': 0,  # force a single shared class
                        'mask': pred_group['masks'][i]
                    })

            all_gts_single_class = []
            for img_id, gt_group in all_gts.items():
                 for i in range(len(gt_group['classes'])):
                    all_gts_single_class.append({
                        'class_id': 0,  # force a single shared class
                        'mask': gt_group['masks'][i]
                    })
            
            # 2. Call the AP computation routine
            ap, total_tp, total_gt = self.calculate_ap_per_class(
                all_preds_single_class, 
                all_gts_single_class, 
                iou_thresh, 
                class_id=0  # evaluate this virtual class
            )
            all_class_aps_per_iou[iou_thresh] = ap
            
            # Print details only at iou=0.50 for reference
            if abs(iou_thresh - 0.50) < 1e-4:
                 print(f"AP for 'all' @0.50: {ap:.4f} (TPs: {int(total_tp)}, GTs: {total_gt})")

        # 3. Compute the final "all-class" mAP scores
        all_map50 = all_class_aps_per_iou.get(0.50, 0.0)
        all_map75 = all_class_aps_per_iou.get(0.75, 0.0)
        all_map50_95 = np.mean(list(all_class_aps_per_iou.values())) if all_class_aps_per_iou else 0.0

        # 4. Pack all results into a dict and return
        final_results = {
            "mAP@.50": map50, 
            "mAP@.75": map75, 
            "mAP@.50-.95": map50_95,
            "all_mAP@.50": all_map50,
            "all_mAP@.75": all_map75,
            "all_mAP@.50-.95": all_map50_95
        }
        
        return final_results
    
    
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default="cfgs/lightstereo/lightstereo_s_kitti.yaml", help='specify the config for eval')
    parser.add_argument('--pretrained_model', type=str, default="checkpoint_epoch_250.pth", help='pretrained_model')
    parser.add_argument(
        '--checkpoint_state',
        choices=['model_state', 'raw_model_state'],
        default='model_state',
        help='checkpoint state to evaluate (EMA model_state or raw training state)',
    )
    parser.add_argument('--output_dir', type=str, default="./predictions", help='output directory for results')
    parser.add_argument('--kitti_root', type=str, default="./data/KITTI15", help='Path to the root of the KITTI dataset')
    parser.add_argument('--gt_file', type=str, default="data/KITTI15/kitti15_val20_with_seglabel.txt", help='Path to the ground truth file list')
    parser.add_argument('--disable_amp', action='store_true', help='disable automatic mixed precision')
    parser.add_argument('--confidence_threshold', type=float, default=0.001, help='low pre-filter threshold used before AP ranking')
    parser.add_argument('--max_detections', type=int, default=300, help='maximum ranked detections retained per image')
    parser.add_argument(
        '--save_format',
        type=str,
        choices=['json', 'pkl', 'npy', 'none'],
        default='json',
        help='prediction serialization format; use none for metric-only evaluation',
    )
    
    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    if args.pretrained_model is not None:
        cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model
    cfgs.MODEL.CHECKPOINT_STATE = args.checkpoint_state
    args.run_mode = 'infer'
    return args, cfgs

tensor2SEG = tensor2SEG()

@torch.no_grad()
def main():
    args, cfgs = parse_config()
    
    # Setup distributed training if enabled
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
    
    # Load Ground Truths first
    logger.info("Loading ground truth data...")
    ground_truths = load_ground_truths(args.gt_file, args.kitti_root)
    logger.info(f"Loaded {len(ground_truths)} ground truth annotations.")

    # Build model
    trainer = Trainer(args, cfgs, local_rank, global_rank, logger, None)
    model = trainer.model
    model.eval()
    model = model.float()

    # Build transform
    transform_config = cfgs.DATA_CONFIG.DATA_TRANSFORM.EVALUATING
    transform = build_transform_by_cfg(transform_config)

    # Prepare image list from GT keys to ensure we only process images with labels
    image_dir = os.path.join(args.kitti_root, "training/image_2")
    left_images = sorted([os.path.join(image_dir, img_id) for img_id in ground_truths.keys() if os.path.exists(os.path.join(image_dir, img_id))])

    os.makedirs(args.output_dir, exist_ok=True)
    all_predictions = []

    for left_path in left_images:
        logger.info(f"Processing image: {left_path}")
        try:
            predictions = generate_predictions(
                model=model,
                image_path=left_path,
                transform=transform,
                confidence_threshold=args.confidence_threshold,
                max_detections=args.max_detections,
            )
            all_predictions.append(predictions)
            logger.info(f"Detected {len(predictions['boxes'])} instances.")
        except Exception as e:
            logger.error(f"Error processing {left_path}: {str(e)}")
            continue
            
    # Save predictions if needed
    if args.save_format == 'json':
        output_path = os.path.join(args.output_dir, 'predictions.json')
        save_predictions_to_json(all_predictions, output_path)
        logger.info(f"Predictions saved to: {output_path}")

    logger.info(f"Finished processing {len(all_predictions)} images.")
    
    # Calculate mAP
    if all_predictions and ground_truths:
        logger.info("\n" + "="*40)
        logger.info("Calculating Mask mAP...")
        logger.info("="*40)
        ap_calculator = ImageLocalMaskAPCalculator(class_names)
        map_scores = ap_calculator.evaluate(all_predictions, ground_truths)
        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as metrics_file:
            json.dump(map_scores, metrics_file, indent=2)
        logger.info(f"Metrics saved to: {metrics_path}")
        
        logger.info("\\n" + "="*50)
        logger.info("      Mask mAP Evaluation Results (Per-Class Average)")
        logger.info("="*50)
        logger.info(f"  mAP@.50         : {map_scores['mAP@.50']:.4f}")
        logger.info(f"  mAP@.75         : {map_scores['mAP@.75']:.4f}")
        logger.info(f"  mAP@.50:.95     : {map_scores['mAP@.50-.95']:.4f}")
        logger.info("="*50)
        
        logger.info("\\n" + "="*50)
        logger.info("      Mask mAP Evaluation Results (All Classes Combined)")
        logger.info("="*50)
        logger.info(f"  mAP@.50 (all)   : {map_scores['all_mAP@.50']:.4f}")
        logger.info(f"  mAP@.75 (all)   : {map_scores['all_mAP@.75']:.4f}")
        logger.info(f"  mAP@.50:.95 (all): {map_scores['all_mAP@.50-.95']:.4f}")
        logger.info("="*50)
    else:
        logger.warning("No predictions or ground truths available for mAP calculation.")

if __name__ == '__main__':
    main()
