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
import open3d as o3d
from scipy import ndimage

sys.path.insert(0, './')

from stereo.utils import common_utils
from stereo.modeling.models.lightstereo.trainer import Trainer
from stereo.utils.disp_color import disp_to_color
from stereo.datasets.dataset_template import build_transform_by_cfg
from stereo.modeling.models.lightstereo.infer_module import crop_mask, tensor2SEG, scale_boxes, process_mask_native, remove_right_top_padding
from stereo.utils.class_schema import KITTI_INSTANCE_CLASS_NAMES, model_ids_to_annotation_ids
import time

total_time = 0
pointcloud_time = 0  # track point-cloud reconstruction time


def crop_from_right_top_pad(arr_2d: np.ndarray, H0: int, W0: int, th: int = 384, tw: int = 1248) -> np.ndarray:
    """Crop RightTopPad output (top+right pad to th*tw) back to the original image (H0*W0)."""
    if arr_2d.ndim != 2:
        return arr_2d
    if arr_2d.shape == (H0, W0):
        return arr_2d
    pad_top = max(0, th - H0)
    # Original region is at: y=[pad_top, pad_top+H0), x=[0, W0)
    if arr_2d.shape[0] >= pad_top + H0 and arr_2d.shape[1] >= W0:
        return arr_2d[pad_top:pad_top + H0, 0:W0]
    return arr_2d

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

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default="cfgs/lightstereo/lightstereo_s_kitti.yaml", help='specify the config for eval')
    parser.add_argument('--pretrained_model', type=str, default="s-with-yolopt.pth", help='pretrained_model')
    parser.add_argument('--output_dir', type=str, default="./singleoutput", help='output directory for results')
    parser.add_argument('--disable_amp', action='store_true', help='disable automatic mixed precision')
    # Single-image path arguments
    parser.add_argument('--left_img_path', type=str, default="./testdata/17l.png", help='path to left image')
    parser.add_argument('--right_img_path', type=str, default="./testdata/17r.png", help='path to right image')
    # Optional camera / disparity parameters
    parser.add_argument('--min_disp', type=float, default=3.0, help='minimum disparity threshold')
    parser.add_argument('--no_pointcloud', action='store_true', help='disable pointcloud generation')
    
    # Point-cloud post-processing parameters
    parser.add_argument('--display_pointcloud', default=True, action='store_true', help='display pointcloud after generation')
    parser.add_argument('--display_full_pointcloud', default=True, action='store_true', help='display full-scene pointcloud for comparison')
    parser.add_argument('--remove_occluded', default=True, action='store_true', help='remove occluded points from disparity map')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='confidence threshold for instance detection')
    
    args = parser.parse_args()

    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)
    if args.pretrained_model is not None:
        cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model

    args.run_mode = 'infer'
    return args, cfgs

tensor2SEG = tensor2SEG()


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

def create_camera_matrix():
    """
    Create the camera re-projection matrix Q.
    Q = [ 1,  0,  0,    -609.5593 ]
        [ 0,  1,  0,    -172.8540 ]
        [ 0,  0,  0,     721.5377 ]
        [ 0,  0,  1/384.381472,  0 ]
    """
    Q = np.array([
        [1.0, 0.0, 0.0, -609.5593],
        [0.0, 1.0, 0.0, -172.8540],
        [0.0, 0.0, 0.0, 721.5377],
        [0.0, 0.0, 1.0/384.381472, 0.0]
    ])
    return Q

def remove_occluded_points(disparity):
    """
    Remove occluded / invisible points.
    In stereo matching, some points visible in the right view may be invisible in the left view.
    """
    if torch.is_tensor(disparity):
        disparity = disparity.detach().cpu().numpy()
    
    H, W = disparity.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    us_right = xx - disparity
    invalid = us_right < 0
    disparity[invalid] = np.inf
    
    return disparity


def display_pointcloud(pcd, window_name="Point Cloud"):
    """
    Display a point cloud with Open3D.

    Args:
        pcd: point cloud object
        window_name: window title
    """
    if pcd is None or len(pcd.points) == 0:
        print("Cannot display an empty point cloud")
        return
        
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=800, height=600)
    
    # Add point cloud
    vis.add_geometry(pcd)
    
    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([255, 255, 255])  
    opt.point_size = 2.7
    
    # Set camera viewpoint
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_lookat([0, 0, 0])
    ctr.set_zoom(0.8)
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def disparity_to_pointcloud_with_color(disparity, mask, left_image, min_disp=3.0, remove_occluded=False):
    """
    Convert a disparity map to a colored point cloud with camera matrix Q,
    processing only the masked region. No extra filtering; keep the raw points.

    Args:
        disparity: disparity map (H, W)
        mask: binary mask; True marks pixels that should become points
        left_image: original left image (H, W, 3) in BGR
        min_disp: minimum disparity threshold to avoid infinite-depth points
        remove_occluded: whether to remove occluded points

    Returns:
        open3d.geometry.PointCloud: colored point cloud
    """
    global pointcloud_time
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    if torch.is_tensor(disparity):
        disparity = disparity.detach().cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    if torch.is_tensor(left_image):
        left_image = left_image.detach().cpu().numpy()
    
    H, W = disparity.shape
    
    # Optionally remove occluded points from the disparity map first
    if remove_occluded:
        disparity = remove_occluded_points(disparity)
    
    # Ensure mask is boolean and matches disparity shape
    mask = mask.astype(bool)
    
    # Resize mask if its shape does not match disparity
    if mask.shape != disparity.shape:
        mask_resized = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        mask = mask_resized > 0.5
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Apply mask and filter invalid disparities
    valid_disp_mask = (disparity > min_disp).astype(bool)
    valid_mask = np.logical_and(mask, valid_disp_mask)
    
    if not np.any(valid_mask):
        print("Warning: no valid point-cloud data")
        pointcloud_time += time.time() - start_time
        return None
    
    # Extract coordinates and disparities of valid points
    valid_y = y_coords[valid_mask]
    valid_x = x_coords[valid_mask]
    valid_disp = disparity[valid_mask]
    
    # Create camera matrix Q
    Q = create_camera_matrix()
    
    # Reproject with matrix Q
    # Build homogeneous coordinates [x, y, disp, 1]
    points_homo = np.column_stack([valid_x, valid_y, valid_disp, np.ones_like(valid_x)])
    
    # Apply re-projection: points_3d_homo = Q * points_homo.T
    points_3d_homo = Q @ points_homo.T
    
    # Convert to Cartesian coordinates (X/W, Y/W, Z/W)
    points_3d = points_3d_homo[:3] / points_3d_homo[3]
    points_3d = points_3d.T  # transpose to (N, 3)
    
    # Extract corresponding colors from the left image
    colors = left_image[valid_y, valid_x]  # shape (N, 3) BGR
    
    # Convert BGR to RGB
    colors_rgb = colors[:, [2, 1, 0]] / 255.0
    
    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
    
    pointcloud_time += time.time() - start_time
    return pcd

def save_combined_pointcloud_for_objects(disparity, masks, classes, confidence, left_image_path, output_dir, base_name, args):
    """
    Generate point clouds for all detected objects and merge them into one file.

    Args:
        disparity: disparity map
        masks: list of object masks
        classes: class id list
        confidence: confidence list
        left_image_path: left image path (for color extraction)
        output_dir: output directory
        base_name: base filename
        args: command-line arguments

    Returns:
        combined_pcd: merged point cloud object
    """
    global pointcloud_time
    start_time = time.time()
    
    # Create point-cloud output directory
    pointcloud_dir = os.path.join(output_dir, 'pointclouds')
    os.makedirs(pointcloud_dir, exist_ok=True)
    
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Load left image for color extraction
    left_image = cv2.imread(left_image_path)
    if left_image is None:
        print(f"Error: failed to load left image {left_image_path}")
        return None
    
    all_pointclouds = []
    
    for i, (cls_idx, conf, mask) in enumerate(zip(classes, confidence, masks)):
        if conf < args.confidence_threshold:
            continue
            
        # Get class name
        cls_name = class_names[int(cls_idx)] if int(cls_idx) < len(class_names) else f"class_{int(cls_idx)}"
        
        print(f"Generating point cloud for {cls_name} (confidence: {conf:.2f})...")
        
        # Ensure mask / class tensors are numpy arrays
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        if torch.is_tensor(cls_idx):
            cls_idx = cls_idx.detach().cpu().numpy()
        
        # Generate a colored point cloud
        pcd = disparity_to_pointcloud_with_color(
            disparity=disparity,
            mask=mask,
            left_image=left_image,
            min_disp=args.min_disp,
            remove_occluded=args.remove_occluded
        )
        
        if pcd is not None and len(pcd.points) > 0:        
            all_pointclouds.append(pcd)
            print(f"Point cloud for {cls_name} generated successfully ({len(pcd.points)} points)")
        else:
            print(f"Warning: failed to generate point cloud for {cls_name}")
    
    # Merge all point clouds
    if all_pointclouds:
        combined_pcd = all_pointclouds[0]
        for pcd in all_pointclouds[1:]:
            combined_pcd += pcd
        
        # Save the merged point cloud
        combined_filename = f"{name_without_ext}_combined.ply"
        combined_path = os.path.join(pointcloud_dir, combined_filename)
        o3d.io.write_point_cloud(combined_path, combined_pcd)
        
        
        total_pc_time = time.time() - start_time
        print(f"Merged point cloud saved: {combined_path} (total points: {len(combined_pcd.points)})")
        print(f"Total point-cloud generation time: {total_pc_time:.4f} sec")
        
        # Optionally display the point cloud
        if args.display_pointcloud:
            print("Displaying point cloud...")
            display_pointcloud(combined_pcd, f"Point Cloud - {base_name}")
        
        return combined_pcd
    else:
        print("No point cloud was generated")
        pointcloud_time += time.time() - start_time
        return None


def process_image_pair(model, transform, left_path, right_path, output_dir, args, use_amp=True):
    global total_time, pointcloud_time
    # Create output subdirectories
    os.makedirs(os.path.join(output_dir, 'disparity'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'segmentation'), exist_ok=True)
    if not args.no_pointcloud:
        os.makedirs(os.path.join(output_dir, 'pointclouds'), exist_ok=True)

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
        print("Inference time:", infer_time, "sec")
        total_time += infer_time

    # Process disparity
    disp_pred = model_pred['disp_pred'].squeeze().cpu().numpy()
    # Model output is usually in RightTopPad space (384,1248); crop back to original size to match mask/color
    disp_pred = crop_from_right_top_pad(disp_pred, height, width, th=384, tw=1248)
    img_color = disp_to_color(disp_pred, max_disp=192)
    img_color = img_color.astype('uint8')

    # Save disparity
    base_name = os.path.basename(left_path)
    name_without_ext = os.path.splitext(base_name)[0]
    disp_path = os.path.join(output_dir, 'disparity', f"{name_without_ext}_disparity.png")
    cv2.imwrite(disp_path, cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR))
    print(f"Disparity map saved to: {disp_path}")

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
        seg_path = os.path.join(output_dir, 'segmentation', f"{name_without_ext}_segmentation.png")

        draw_boxes_and_masks_on_image_bgr(
            image_bgr=image_bgr,
            classes=classes_1d,
            confidence=conf_1d,
            boxes=boxes_img,
            masks=mask_tensor,
            class_names=class_names,
            confidence_threshold=args.confidence_threshold,
            mask_alpha=0.8,
            out_path=seg_path
        )
        print(f"Segmentation result saved to: {seg_path}")
        
        # Generate instance-level object point clouds
        if not args.no_pointcloud:
            print("Generating object point clouds...")
            
            # Ensure correct data types
            if torch.is_tensor(classes_1d):
                classes_1d = classes_1d.detach().cpu().numpy()
            if torch.is_tensor(conf_1d):
                conf_1d = conf_1d.detach().cpu().numpy()
            
            # Reset point-cloud timer
            pointcloud_time = 0
            
            combined_pcd = save_combined_pointcloud_for_objects(
                disparity=disp_pred,
                masks=mask_tensor,
                classes=classes_1d,
                confidence=conf_1d,
                left_image_path=left_path,  # left image path for color extraction
                output_dir=output_dir,
                base_name=base_name,
                args=args
            )
            
            if combined_pcd is not None:
                print(f"Total point-cloud reconstruction time: {pointcloud_time:.4f} sec")
                print(f"Generated a merged point cloud with {len(combined_pcd.points)} points")

            # Generate and show a full-scene point cloud (for comparison with instance-level)
            if args.display_full_pointcloud:
                print("Generating full-scene point cloud...")
                left_image_bgr = cv2.imread(left_path)
                if left_image_bgr is None:
                    print(f"Error: failed to load left image {left_path}")
                else:
                    full_mask = np.ones((height, width), dtype=bool)
                    full_pcd = disparity_to_pointcloud_with_color(
                        disparity=disp_pred,
                        mask=full_mask,
                        left_image=left_image_bgr,
                        min_disp=args.min_disp,
                        remove_occluded=args.remove_occluded
                    )
                    if full_pcd is not None and len(full_pcd.points) > 0:
                        pointcloud_dir = os.path.join(output_dir, 'pointclouds')
                        os.makedirs(pointcloud_dir, exist_ok=True)
                        full_filename = f"{name_without_ext}_full.ply"
                        full_path = os.path.join(pointcloud_dir, full_filename)
                        o3d.io.write_point_cloud(full_path, full_pcd)
                        print(f"Full-scene point cloud saved: {full_path} (points: {len(full_pcd.points)})")
                        if args.display_pointcloud:
                            print("Displaying full-scene point cloud (close the window to continue)...")
                            display_pointcloud(full_pcd, f"Full-scene Point Cloud - {base_name}")
                    else:
                        print("Warning: failed to generate full-scene point cloud")
    else:
        print("No objects detected")

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

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Check that input images exist
    if not os.path.exists(args.left_img_path):
        logger.error(f"Left image not found: {args.left_img_path}")
        return
    if not os.path.exists(args.right_img_path):
        logger.error(f"Right image not found: {args.right_img_path}")
        return

    # Process a single stereo image pair
    logger.info(f"Processing image pair: {args.left_img_path} and {args.right_img_path}")
    try:
        process_image_pair(
            model,
            transform,
            args.left_img_path,
            args.right_img_path,
            args.output_dir,
            args,
            use_amp=not args.disable_amp
        )
        print(f"Inference finished! Results saved to: {args.output_dir}")
        print(f"Total inference time: {total_time:.4f} sec")
        print(f"Total point-cloud reconstruction time: {pointcloud_time:.4f} sec")
    except Exception as e:
        logger.error(f"Error while processing images: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
