import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import defaultdict

class ModuleTimer:
    def __init__(self):
        self.module_times = defaultdict(list)
        self.enabled = False
        
    def record_time(self, module_name, elapsed_time):
        if self.enabled:
            self.module_times[module_name].append(elapsed_time)
        
    def get_avg_times(self):
        return {name: sum(times)/len(times) for name, times in self.module_times.items()}
        
    def reset(self):
        self.module_times.clear()
        
    def enable(self):
        self.enabled = True
        
    def disable(self):
        self.enabled = False

from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.cost_volume.cost_volume import correlation_volume
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.modeling.disp_refinement.disp_refinement import context_upsample
from stereo.modeling.seg.edge_aware_pipeline import EdgeAwareSegPost

from .backbone import Backbone, FPNLayer
from .aggregation import Aggregation
from .segment import Head
from .loss import SegmentationLoss,prepare_seg_batch




class LightStereo(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT
        self.loss_balancing = str(cfgs.get('LOSS_BALANCING', 'uncertainty')).lower()
        if self.loss_balancing not in {'uncertainty', 'fixed'}:
            raise ValueError(
                f"Unsupported LOSS_BALANCING={self.loss_balancing!r}; "
                "expected 'uncertainty' or 'fixed'"
            )
        fixed_weights = cfgs.get('LOSS_WEIGHTS', [1.0, 1.0])
        if len(fixed_weights) != 2:
            raise ValueError('LOSS_WEIGHTS must contain stereo and segmentation weights')
        self.register_buffer(
            'fixed_task_weights',
            torch.tensor(fixed_weights, dtype=torch.float32),
            persistent=False,
        )

        # backbobe
        self.backbone = Backbone(cfgs.get('BACKCONE', 'MobileNetv2'))
        # Projection to adapt neck -> head
        self.proj_p3 = nn.Conv2d(32, 128, kernel_size=1)  # Map p3 from 32 to 128
        self.proj_p4 = nn.Conv2d(96, 128, kernel_size=1)  # Map p4 from 96 to 128
        self.proj_p5 = nn.Conv2d(160, 256, kernel_size=1) # Map c5 from 160 to 256
        
        # detection head
        self.head = Head(
            in_ch_l4=128,
            in_ch_l6=128,
            in_ch_l10=256,
            nc=int(cfgs.NUM_CLASSES),
        )

        # state_dict = self.head.state_dict()
        # for key in state_dict.keys():
        #     print(key)

        # aggregation
        self.cost_agg = Aggregation(in_channels=48,
                                    left_att=self.left_att,
                                    blocks=cfgs.AGGREGATION_BLOCKS,
                                    expanse_ratio=cfgs.EXPANSE_RATIO,
                                    backbone_channels=self.backbone.output_channels)

        # disp refine
        self.refine_1 = nn.Sequential(
            BasicConv2d(self.backbone.output_channels[0], 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU))

        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.refine_2 = FPNLayer(24, 16)

        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)
        self.seg_post = EdgeAwareSegPost(nm=32, lambda_edge=1.0, mid=64, dilations=(1,2), loss_mode="dice+l1", alpha=2.0)
        self.edge_loss_weight = float(cfgs.get('EDGE_LOSS_WEIGHT', 0.05))
        self.edge_loss_start_epoch = int(cfgs.get('EDGE_LOSS_START_EPOCH', 10))
        self.edge_loss_warmup_epochs = int(cfgs.get('EDGE_LOSS_WARMUP_EPOCHS', 10))
        self.train_epoch = 0
        #loss
        self.seg_loss_function = SegmentationLoss(nc=int(cfgs.NUM_CLASSES))

        self.uncertainty_loss = MultiTaskUncertaintyLoss(num_tasks=2)

    def set_train_epoch(self, epoch):
        """Set the epoch used by the delayed edge-regularization schedule."""
        self.train_epoch = int(epoch)

    def current_edge_loss_weight(self):
        if self.edge_loss_weight <= 0.0 or self.train_epoch < self.edge_loss_start_epoch:
            return 0.0
        if self.edge_loss_warmup_epochs <= 0:
            return self.edge_loss_weight
        elapsed = self.train_epoch - self.edge_loss_start_epoch + 1
        progress = min(1.0, elapsed / self.edge_loss_warmup_epochs)
        return self.edge_loss_weight * progress


    def forward(self, data):
        if not hasattr(self, 'timer'):
            self.timer = ModuleTimer()
            
        image1 = data['left']
        image2 = data['right']

        # Record backbone time
        if self.timer.enabled:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        features_left = self.backbone(image1)
        features_right = self.backbone(image2)
        if self.timer.enabled:
            torch.cuda.synchronize()
            self.timer.record_time('backbone', time.perf_counter() - t_start)

        p3 = features_left[1]  # channel 32
        p4 = features_left[2]  # channel 96
        c5 = features_left[3]  # channel 160

        # Record projection time
        if self.timer.enabled:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        p3_proj = self.proj_p3(features_left[1])  # 32 -> 128
        p4_proj = self.proj_p4(features_left[2])  # 96 -> 128
        p5_proj = self.proj_p5(features_left[3])  # 160 -> 256
        if self.timer.enabled:
            torch.cuda.synchronize()
            self.timer.record_time('projection', time.perf_counter() - t_start)

        # Record head time
        if self.timer.enabled:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        seg_pred = self.head(p3_proj, p4_proj, p5_proj)
        if self.timer.enabled:
            torch.cuda.synchronize()
            self.timer.record_time('head', time.perf_counter() - t_start)
        #print("seg time",(time.time()-time0)*1000,"ms")
        #seg_pred = self.head(features_left[1], features_left[2], features_left[3])  # p3, p4, c5


        # Record cost-volume generation time
        if self.timer.enabled:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        gwc_volume = correlation_volume(features_left[0], features_right[0], self.max_disp // 4)
        if self.timer.enabled:
            torch.cuda.synchronize()
            self.timer.record_time('cost_volume', time.perf_counter() - t_start)

        # Record cost aggregation time
        if self.timer.enabled:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        encoding_volume = self.cost_agg(gwc_volume, features_left)  # [bz, 1, max_disp/4, H/4, W/4]
        squeezed_encoding = encoding_volume[0].reshape(encoding_volume[0].size(0), -1, encoding_volume[0].size(2), encoding_volume[0].size(3))
        if self.timer.enabled:
            torch.cuda.synchronize()
            self.timer.record_time('cost_aggregation', time.perf_counter() - t_start)

        # Record disparity estimation time
        if self.timer.enabled:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        prob = F.softmax(squeezed_encoding, dim=1)
        init_disp = disparity_regression(prob, self.max_disp // 4)  # [bz, 1, H/4, W/4]
        if self.timer.enabled:
            torch.cuda.synchronize()
            self.timer.record_time('disparity_estimation', time.perf_counter() - t_start)

        # Record refinement time
        if self.timer.enabled:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(image1))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)  # [bz, 9, H, W]
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # [bz, 1, H, W]
        if self.timer.enabled:
            torch.cuda.synchronize()
            self.timer.record_time('refinement', time.perf_counter() - t_start)
        seg_pred, loss_edge_dict = self.seg_post(seg_pred, disp_pred, training=self.training)
        result = {'disp_pred': disp_pred, 'seg_pred': seg_pred}

        if self.training:
            disp_4 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            disp_4 *= 4
            result['disp_4'] = disp_4
            if loss_edge_dict is not None and 'disp_edge' in loss_edge_dict:
                result['disp_edge'] = loss_edge_dict['disp_edge']
        #print("infer time:",(time.time()-time0)*1000,"ms")

        return result


    def get_loss(self, model_pred, input_data, collect_metrics=True):
        # Stereo matching loss
        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]

        disp_pred = model_pred['disp_pred']
        stereo_loss = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')

        if 'disp_4' in model_pred:
            disp_4 = model_pred['disp_4']
            stereo_loss += 0.3 * F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')

        # Initialize detection loss
        seg_loss = 0

        #total_loss = stereo_loss  # debug
        
        # Object detection loss
        seg_pred = model_pred['seg_pred']
        targets = input_data["seg_labels"]  # detection label dict
        #print("input format", targets)
        if not isinstance(seg_pred, (tuple, list)) or len(seg_pred) != 3:
            raise RuntimeError(
                "Training expects segmentation output (features, coefficients, prototypes)"
            )
        proto = seg_pred[2]
        image_height, image_width = input_data["left"].shape[-2:]
        seg_batch_input = prepare_seg_batch(
            targets,
            img_size=(image_width, image_height),
            mask_size=proto.shape[-2:],
            device=proto.device,
        )
        #print("batch format", seg_batch_input)

        seg_loss, seg_loss_components, edge_loss = self.seg_loss_function(
            seg_pred,
            seg_batch_input,
            edge_map=model_pred.get('disp_edge'),
        )
        seg_loss_info = {}
        if collect_metrics:
            component_names = ('box', 'mask', 'cls', 'dfl')
            seg_loss_info = {
                f'scalar/train/loss_seg_{name}': component.item()
                for name, component in zip(component_names, seg_loss_components)
            }
        # print(f"SEGMENT Loss: {seg_loss}")
        # print(f"Box/Seg/Cls/DFL Loss: {seg_loss_components}")
        # print(f"stereo loss:{stereo_loss}")


        edge_weight = self.current_edge_loss_weight()
        seg_objective = seg_loss + edge_weight * edge_loss
        task_losses = [stereo_loss, seg_objective]
        if self.loss_balancing == 'uncertainty':
            total_loss = self.uncertainty_loss(task_losses)
            task_weights = torch.exp(-self.uncertainty_loss.log_vars.detach())
            if collect_metrics:
                for name, weight, log_var in zip(
                    ('disp', 'seg'),
                    task_weights,
                    self.uncertainty_loss.log_vars.detach(),
                ):
                    seg_loss_info[f'scalar/train/task_weight_{name}'] = weight.item()
                    seg_loss_info[f'scalar/train/task_log_var_{name}'] = log_var.item()
        else:
            total_loss = sum(
                weight.to(loss) * loss
                for weight, loss in zip(self.fixed_task_weights, task_losses)
            )
            if collect_metrics:
                for name, weight in zip(('disp', 'seg'), self.fixed_task_weights):
                    seg_loss_info[f'scalar/train/task_weight_{name}'] = weight.item()

        if collect_metrics:
            seg_loss_info['scalar/train/edge_loss_weight'] = edge_weight
            seg_loss_info['scalar/train/loss_seg_objective'] = seg_objective.item()

        #print(f"total loss:{total_loss}")

        
        loss_info = {}
        if collect_metrics:
            loss_info = {
                'scalar/train/loss_disp': stereo_loss.item(),
                'scalar/train/loss_det': seg_loss.item(),
                'scalar/train/loss_edge': edge_loss.item(),
                'scalar/train/total_loss': total_loss.item()
            }
            loss_info.update(seg_loss_info)
        
        return total_loss, loss_info
    
class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))  # updated by gradients
        
    def forward(self, losses):
        """
        losses: list of per-task loss tensors [loss1, loss2, ...]
        Returns: weighted total loss
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss
