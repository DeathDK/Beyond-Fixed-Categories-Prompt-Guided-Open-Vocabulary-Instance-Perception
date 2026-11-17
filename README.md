# Beyond Fixed Categories: Prompt-Guided Open-Vocabulary Instance Perception for Stereo Point Cloud Reconstruction

Official PyTorch implementation for prompt-guided open-vocabulary instance perception on stereo imagery, jointly producing:

1. dense disparity,
2. known-class instance masks from a LightStereo-Instance detector,
3. text-prompted unknown instances (default: Grounding DINO + SAM2),
4. geometry-aware refinement (G0) and calibrated known/unknown fusion,
5. instance-level stereo point clouds.

This README is written so that the main paper experiments can be re-run from the repository root with the commands below.

**Dataset note.** The `data/` folder is **not** shipped with this repository (size). Please obtain the KITTI2015-IS data package from [LightStereo-Instance](https://github.com/laserquit/LightStereo-Instance) and place it as described in [Section 5](#5-dataset-preparation-kitti2015-is).

If you meet any issue while reproducing the results, please contact: **zhouyiqiao0913@163.com**

---

## Table of contents

1. [Method overview](#1-method-overview)
2. [Repository layout](#2-repository-layout)
3. [Hardware & software](#3-hardware--software)
4. [Installation](#4-installation)
5. [Dataset preparation (KITTI2015-IS)](#5-dataset-preparation-kitti2015-is)
6. [Weights and checkpoints](#6-weights-and-checkpoints)
7. [Experiment protocol](#7-experiment-protocol)
8. [Training](#8-training)
9. [Evaluation](#9-evaluation)
10. [Inference & visualization](#10-inference--visualization)
11. [Efficiency measurement](#11-efficiency-measurement)
12. [Ablations](#12-ablations)
13. [Important hyperparameters](#13-important-hyperparameters)
14. [Troubleshooting](#14-troubleshooting)
15. [Contact](#15-contact)
16. [References](#16-references)

---

## 1. Method overview

**Known path.** A LightStereo backbone with a YOLO-style instance head is trained on KITTI2015-IS under combinatorial leave-out budgets:

| Budget | #models | Meaning |
|--------|--------:|---------|
| F5 | 1 | all 5 classes known (closed reference) |
| K4 | 5 | leave exactly 1 class out |
| K3 | 10 | leave exactly 2 classes out |
| K2 | 10 | leave exactly 3 classes out |
| K1 | 5 | leave exactly 4 classes out |

**Total: 31 known models.** Canonical classes: `Car, Truck, Bus, Bicycle, Person`.

**Unknown path (frozen).** Left-out classes are recovered at test time by text prompts:

- default expert: **Grounding DINO** (boxes) + **SAM2** (masks),
- optional ablation expert: frozen **YOLOE-seg** (`yoloe-26x-seg.pt`).

**Post-process.** G0 geometry refine (disparity edge band) + calibrated mask fusion (`--mask_iou_conflict 0.90` in paper experiments), then triangulation to instance point clouds.

Full leave-out matrix: [`cfgs/openworld/TRAINING_MATRIX.md`](cfgs/openworld/TRAINING_MATRIX.md).  
Open-world config notes: [`cfgs/openworld/README.md`](cfgs/openworld/README.md).

---

## 2. Repository layout

```text
.
├── README.md
├── requirements.txt
├── stereo/                      # LightStereo model, losses, trainer, KITTI dataset
├── openworld/                   # splits, experts, G0, fusion, metrics, point cloud
├── cfgs/
│   ├── lightstereo/             # closed-set LightStereo YAMLs
│   └── openworld/               # F5 + 30 leave-out YAMLs + unknown_module.yaml
├── tools/                       # train / eval / infer / prepare / measure
├── data/                        # NOT in this repo — obtain separately (see Section 5)
│   └── KITTI15/                 # expected local dataset root after setup
├── pretrained/                  # read-only F5 init slot: f5_checkpoint_best.pth
├── testdata/                    # stereo init ckpt + sample stereo pair
├── myyolo.pt                    # YOLO head init for F5 cold start
└── yoloe-26x-seg.pt             # optional YOLOE unknown expert
```

Key scripts:

| Script | Purpose |
|--------|---------|
| `tools/prepare_openworld.py` | filter YOLO labels for leave-out splits |
| `tools/generate_openworld_cfgs.py` | regenerate leave-out YAMLs from `openworld/splits.py` |
| `tools/train.py` | train F5 / leave-out known models |
| `tools/eval_best_suite.py` | known-only own-best evaluation |
| `tools/eval_openworld_best_suite.py` | open-world own-best evaluation |
| `tools/eval_openworld.py` | single-checkpoint open-world eval |
| `tools/infer_openworld.py` | single stereo pair open-world infer + point cloud |
| `tools/measure_ours_efficiency.py` | known-path FLOPs / params / latency |
| `tools/measure_full_system_efficiency.py` | full system (known + GDINO + SAM2) |
| `tools/ablation_openworld.py` | G0 / fusion mode ablation entry |

---

## 3. Hardware & software

Recommended for full reproduction:

- **GPU**: NVIDIA GPU with ≥ 24 GB VRAM for training (`BATCH_SIZE_PER_GPU: 8` at crop `320×736`). Smaller GPUs: lower batch size in the YAML.
- **Open-world eval**: Grounding DINO + SAM2 benefit from ≥ 16–24 GB VRAM; first run also needs disk space for HF weights (~several GB).
- **OS**: Linux or Windows.
- **Python**: 3.9 recommended (3.8–3.11 should work).
- **PyTorch**: 2.x with CUDA matching your driver.

Approximate training cost per leave-out run: 200 epochs, eval every 5 epochs, AdamW `lr=2e-4`, EMA enabled (see each YAML under `cfgs/openworld/`).

---

## 4. Installation

All commands below assume the **repository root** as the working directory.

### 4.1 Create environment

```bash
conda create -n openworld-stereo python=3.9 -y
conda activate openworld-stereo
```

### 4.2 Install PyTorch (choose one)

CUDA 12.8 example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

CUDA 11.8 example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 4.3 Install project dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `timm==0.4.12` (required for the stereo backbone) and installs `ultralytics`, `transformers`, `accelerate`, Open3D, etc. Comments inside that file explain each group.

### 4.4 Hugging Face models (open-world path)

On the first call to `tools/eval_openworld*.py` / `tools/infer_openworld.py` with `--unknown_expert gdino_sam2`, the following are downloaded automatically:

- `IDEA-Research/grounding-dino-base`
- `facebook/sam2.1-hiera-large`

Optional cache override:

```bash
# Linux / macOS
export HF_HOME=/path/to/hf_cache

# Windows PowerShell
$env:HF_HOME="D:\hf_cache"
```

Offline machines: pre-download those repos with `huggingface-cli download ...` into the cache, then run eval/infer without network.

---

## 5. Dataset preparation (KITTI2015-IS)

> **This repository does not include `data/`.**  
> Please get the KITTI2015-IS dataset (images, disparity, instance labels, and list files) from:  
> **https://github.com/laserquit/LightStereo-Instance**  
> Then put it under this project as `data/KITTI15/` (same layout as that repo).

### 5.1 Obtain the dataset

1. Clone or download [LightStereo-Instance](https://github.com/laserquit/LightStereo-Instance).
2. Follow its README to prepare KITTI 2015 stereo + KITTI2015-IS annotations (or copy its ready `data/KITTI15` package).
3. Official KITTI stereo images (if still needed separately): [KITTI Scene Flow / stereo](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).
4. Place / symlink the dataset so this project sees:

```text
<data of this repo>/data/KITTI15/
```

Example (Linux / macOS):

```bash
# after preparing data inside LightStereo-Instance
ln -s /path/to/LightStereo-Instance/data ./data
```

Example (Windows PowerShell, junction):

```powershell
cmd /c mklink /J data "D:\path\to\LightStereo-Instance\data"
```

Or simply copy the `data` folder into this repository root.

### 5.2 Instance labels and splits

This project uses YOLO-polygon instance labels for five classes (`Car, Truck, Bus, Bicycle, Person`) under:

```text
data/KITTI15/training/seg_label/
```

Train / val index files (provided with the dataset package):

- `data/KITTI15/kitti15_train180_with_seglabel.txt` (180 training frames)
- `data/KITTI15/kitti15_val20_with_seglabel.txt` (20 validation frames)

Each line lists left image, right image, disparity, and label path.

### 5.3 Expected directory tree

```text
data/KITTI15/
  training/
    image_2/                 # left RGB
    image_3/                 # right RGB
    disp_occ_0/  disp_noc_0/ ...
    seg_label/               # full 5-class YOLO polygons
    seg_label_openworld_f5/  # created by prepare_openworld.py
    seg_label_openworld_k4/
    seg_label_openworld_k3/
    seg_label_openworld_k2/
    seg_label_openworld_k1/
    json/                    # optional JSON annotations
  testing/
    image_2/
    image_3/
  kitti15_train180_with_seglabel.txt
  kitti15_val20_with_seglabel.txt
  kitti15_train180_with_seglabel_{f5,k4,k3,k2,k1}.txt   # generated here
  kitti15_val20_with_seglabel_{f5,k4,k3,k2,k1}.txt      # generated here
```

### 5.4 Build leave-out label copies (run in this repo)

After `data/KITTI15` is in place, generate open-world filtered labels. Images and disparity files are never modified; only filtered label folders and rewritten list files are written:

```bash
python tools/prepare_openworld.py --split all --audit
```

Single budget example:

```bash
python tools/prepare_openworld.py --split K4 --audit
```

After this step, leave-out YAMLs can train without seeing left-out classes as known supervision. Open-world evaluation still reads the **full** 5-class GT under `seg_label/` for metrics only (`USE_FULL_GT_FOR_METRICS_ONLY`).

Optional integrity check on image paths:

```bash
python tools/check_data.py \
  --input_path_file data/KITTI15/kitti15_train180_with_seglabel.txt \
  --error_path_file data/KITTI15/image_error.txt
```

---

## 6. Weights and checkpoints

| Path | Required for | Description |
|------|--------------|-------------|
| `testdata/s.ckpt` | F5 cold start | LightStereo stereo pretrained init |
| `myyolo.pt` (repo root) or `data/yolo.pt` | F5 cold start | YOLO segmentation head init |
| `pretrained/f5_checkpoint_best.pth` | leave-out fine-tune | trained F5 checkpoint (**copy** here; do not move your only copy) |
| `yoloe-26x-seg.pt` | YOLOE ablation only | frozen open-vocab seg expert |
| HF cache (auto) | default open-world eval | Grounding DINO + SAM2 |

Leave-out configs set:

```yaml
PRETRAINED_MODEL: "pretrained/f5_checkpoint_best.pth"
F5_INIT:
  ENABLED: true
  CLASS_HEAD_MODE: remap
```

Training always writes under `output/...` and never overwrites `pretrained/`. See [`pretrained/README.md`](pretrained/README.md).

After F5 training finishes, copy one best checkpoint, for example:

```bash
# Linux / macOS
cp output/KittiDataset/LightStereo/f5/<extra_tag>/ckpt/checkpoint_best_mAP.pth \
   pretrained/f5_checkpoint_best.pth

# Windows PowerShell
Copy-Item output\KittiDataset\LightStereo\f5\<extra_tag>\ckpt\checkpoint_best_mAP.pth `
          pretrained\f5_checkpoint_best.pth
```

Each training run saves four independent best files:

- `checkpoint_best_epe.pth`
- `checkpoint_best_d1_all.pth`
- `checkpoint_best_mAP.pth`
- `checkpoint_best_w_mAP.pth`

---

## 7. Experiment protocol

1. Prepare dataset + leave-out labels (`prepare_openworld.py`).
2. Train F5 → copy to `pretrained/f5_checkpoint_best.pth`.
3. Train each leave-out config (`k4_*` … `k1_*`), 30 models.
4. Known metrics: `eval_best_suite.py` (own-best across the four ckpts).
5. Open-world metrics: `eval_openworld_best_suite.py` with `--unknown_expert gdino_sam2` (default paper setting).
6. Optional: YOLOE expert, G0/fusion ablations, efficiency scripts.

**Own-best suite.** For each metric name, the reported number is the best value among the four saved best checkpoints (lower-is-better for `epe` / `d1_all`, higher-is-better for mAP / recall / mIoU). This avoids picking a single checkpoint for every metric.

Split name in CLI (e.g. `--split K4-2`) must match `OPENWORLD_SPLIT` in the YAML and the matrix in `TRAINING_MATRIX.md`.

---

## 8. Training

### 8.1 Closed F5 reference

```bash
python tools/train.py --cfg_file cfgs/openworld/f5.yaml --extra_tag f5_ref
```

Default optimization (from YAML):

- epochs: `200`
- batch size / GPU: `8`
- optimizer: AdamW, `lr=0.0002`, weight decay `5e-5`
- warmup: 5 epochs
- crop: `320×736` (train), pad `384×1248` (eval)
- max disparity: `192`
- loss: stereo + detection/seg with uncertainty balancing; edge loss weight `0.05` (starts at epoch 10)

Multi-GPU (optional):

```bash
torchrun --nproc_per_node=2 tools/train.py \
  --cfg_file cfgs/openworld/f5.yaml \
  --extra_tag f5_ref \
  --dist_mode
```

### 8.2 Leave-out known models

Example — K4 leave Truck (`KNOWN = Car, Bus, Bicycle, Person`):

```bash
python tools/train.py \
  --cfg_file cfgs/openworld/k4_2_leave_truck.yaml \
  --extra_tag k4_2_from_f5
```

More examples:

```bash
python tools/train.py --cfg_file cfgs/openworld/k4_1_leave_car.yaml      --extra_tag k4_1_from_f5
python tools/train.py --cfg_file cfgs/openworld/k3_1_leave_car_truck.yaml --extra_tag k3_1_from_f5
python tools/train.py --cfg_file cfgs/openworld/k1_5_leave_truck_bus_bicycle_person.yaml --extra_tag k1_5_from_f5
```

Use a **new** `--extra_tag` for each run so logs/ckpts do not overwrite each other.

Outputs:

```text
output/KittiDataset/LightStereo/<exp_name>/<extra_tag>/
  ckpt/
  events/          # TensorBoard
  ...
```

Resume an interrupted run:

```bash
python tools/train.py \
  --cfg_file cfgs/openworld/k4_2_leave_truck.yaml \
  --resume_dir output/KittiDataset/LightStereo/k4_2_leave_truck/k4_2_from_f5 \
  --ckpt best
```

Regenerate YAMLs after editing `openworld/splits.py`:

```bash
python tools/generate_openworld_cfgs.py
```

---

## 9. Evaluation

### 9.1 Known-only own-best suite

```bash
python tools/eval_best_suite.py \
  --cfg_file cfgs/openworld/k4_2_leave_truck.yaml \
  --run_dir output/KittiDataset/LightStereo/k4_2_leave_truck/k4_2_from_f5
```

Report JSON:

```text
eval_output/best_suite/<run>/best_suite_report.json
```

Inspect `final_report` for `epe`, `d1_all`, `mAP`, `w_mAP`.

### 9.2 Open-world own-best suite (paper setting)

Use the thresholds reported in the paper. These match the CLI defaults of
`tools/eval_openworld_best_suite.py` and `tools/infer_openworld.py`:

| Parameter | Paper / reproduce value | Notes |
|-----------|-------------------------|--------|
| Grounding DINO `--box_threshold` | `0.10` | not `0.25` |
| Grounding DINO `--text_threshold` | `0.10` | not `0.20` |
| Fusion `--mask_iou_conflict` | `0.90` | not the `0.6` placeholder in `unknown_module.yaml` |
| G0 `--g0_edge_band` | `3` | pixels |

```bash
python tools/eval_openworld_best_suite.py \
  --cfg_file cfgs/openworld/k4_2_leave_truck.yaml \
  --run_dir output/KittiDataset/LightStereo/k4_2_leave_truck/k4_2_from_f5 \
  --split K4-2 \
  --unknown_expert gdino_sam2 \
  --known_conf 0.001 \
  --box_threshold 0.10 \
  --text_threshold 0.10 \
  --mask_iou_conflict 0.90 \
  --g0_edge_band 3
```

`cfgs/openworld/unknown_module.yaml` keeps `MASK_IOU_CONFLICT: 0.6` only as a
module-template default. **Paper tables / open-world eval are reproduced with
`--mask_iou_conflict 0.90` on the CLI** (which overrides that YAML field).
Do not rely on `unknown_module.yaml` alone when reproducing reported numbers.

Main unknown metrics:

- `u_recall_micro`
- `u_recall_macro_image`
- `u_matched_miou_macro_image`
- pre-fusion counterparts: `u_recall_*_pre_fusion`

Also reports known / full-5 mask AP and stereo errors under the same own-best protocol.

Report JSON:

```text
eval_output/openworld_best_suite/<run>/openworld_best_suite_report.json
```

Fast debug (one checkpoint tag only):

```bash
python tools/eval_openworld_best_suite.py \
  --cfg_file cfgs/openworld/k4_2_leave_truck.yaml \
  --run_dir output/KittiDataset/LightStereo/k4_2_leave_truck/k4_2_from_f5 \
  --split K4-2 \
  --unknown_expert gdino_sam2 \
  --ckpt_tags best_epe
```

### 9.3 Optional YOLOE unknown expert

```bash
python tools/eval_openworld_best_suite.py \
  --cfg_file cfgs/openworld/k4_2_leave_truck.yaml \
  --run_dir output/KittiDataset/LightStereo/k4_2_leave_truck/k4_2_from_f5 \
  --split K4-2 \
  --unknown_expert yoloe \
  --yoloe_ckpt yoloe-26x-seg.pt \
  --known_conf 0.001
```

### 9.4 Single-checkpoint open-world eval

`tools/eval_openworld.py` may use different argparse defaults than the paper;
always pass the paper thresholds explicitly:

```bash
python tools/eval_openworld.py \
  --cfg_file cfgs/openworld/k4_2_leave_truck.yaml \
  --pretrained_model output/.../ckpt/checkpoint_best_mAP.pth \
  --split K4-2 \
  --unknown_expert gdino_sam2 \
  --box_threshold 0.10 \
  --text_threshold 0.10 \
  --list_file ./data/KITTI15/kitti15_val20_with_seglabel.txt
```

### 9.5 Closed-set stereo / segmentation (legacy LightStereo configs)

```bash
python tools/evalstereo.py \
  --cfg_file cfgs/lightstereo/lightstereo_s_kitti.yaml \
  --pretrained_model yourweights.pth

python tools/evalseg.py \
  --cfg_file cfgs/lightstereo/lightstereo_s_kitti.yaml \
  --pretrained_model yourweights.pth \
  --save_format none
```

---

## 10. Inference & visualization

### 10.1 Open-world single stereo pair

```bash
python tools/infer_openworld.py \
  --cfg_file cfgs/openworld/k4_2_leave_truck.yaml \
  --pretrained_model path/to/checkpoint_best_mAP.pth \
  --left_img_path data/KITTI15/training/image_2/000000_10.png \
  --right_img_path data/KITTI15/training/image_3/000000_10.png \
  --split K4-2 \
  --unknown_expert gdino_sam2 \
  --known_conf 0.001 \
  --box_threshold 0.10 \
  --text_threshold 0.10 \
  --mask_iou_conflict 0.90 \
  --output_dir ./openworld_output
```

Outputs under `--output_dir` include overlay images, JSON instance lists, timing, and instance point clouds when calibration is available.

Custom prompts (override split defaults):

```bash
... --prompts "truck,lorry"
```

### 10.2 Closed-set infer / point cloud

```bash
python tools/infer.py \
  --cfg_file cfgs/lightstereo/lightstereo_s_kitti.yaml \
  --left_img_path L.png \
  --right_img_path R.png \
  --pretrained_model yourweights.pth

python tools/inferpointcloud.py \
  --cfg_file cfgs/lightstereo/lightstereo_s_kitti.yaml \
  --left_img_path L.png \
  --right_img_path R.png \
  --pretrained_model yourweights.pth
```

### 10.3 Dataset visualization helpers

```bash
python tools/visualize_kitti_is.py
python tools/visualize_openworld.py
```

---

## 11. Efficiency measurement

Known path only:

```bash
python tools/measure_ours_efficiency.py
```

Full system (known + Grounding DINO + SAM2):

```bash
python tools/measure_full_system_efficiency.py
```

These scripts report parameters, FLOPs (where measurable), and latency. SAM2 FLOPs may be estimated when `thop` cannot wrap the HF module directly; see script logs/JSON for the method used.

---

## 12. Ablations

Unknown-module template: [`cfgs/openworld/unknown_module.yaml`](cfgs/openworld/unknown_module.yaml)
(`GEOMETRY_MODE: g0`, `FUSION_MODE: calibrated_mask_nms`, `EDGE_BAND_PIXELS: 3`).

For **paper reproduction**, always pass CLI thresholds explicitly (see §9.2):
`--box_threshold 0.10 --text_threshold 0.10 --mask_iou_conflict 0.90`.
The YAML field `MASK_IOU_CONFLICT: 0.6` is not the paper setting.

Contrastive modes on one stereo pair:

```bash
python tools/ablation_openworld.py \
  --cfg_file cfgs/openworld/k4_2_leave_truck.yaml \
  --pretrained_model path/to/ckpt.pth \
  --left_img_path L.png \
  --right_img_path R.png \
  --modes naive,g0
```

Fusion IoU / G0 edge-band sweeps can be done by changing CLI flags on `eval_openworld_best_suite.py` / `infer_openworld.py` (e.g. different `--g0_edge_band` or fusion IoU arguments exposed by those scripts) without retraining the known model.

---

## 13. Important hyperparameters

| Item | Paper setting | Where to set for reproduction |
|------|---------------|-------------------------------|
| Train crop | `320 × 736` | YAML `DATA_TRANSFORM` |
| Eval pad | `384 × 1248` | YAML `DATA_TRANSFORM` / `SEG_AP_PAD_HW` |
| Max disparity | `192` | YAML `MODEL.MAX_DISP` |
| Known decode conf (eval) | `0.001` | `--known_conf` / `SEG_AP_CONF` |
| GDINO box / text thr | `0.10` / `0.10` | `--box_threshold` / `--text_threshold` (CLI; suite defaults match paper) |
| G0 edge band | `3` px | `--g0_edge_band` |
| Fusion mask IoU conflict | `0.90` | `--mask_iou_conflict` (CLI; **not** the `0.6` in `unknown_module.yaml`) |
| Epochs / batch | `200` / `8` | YAML `OPTIMIZATION` |
| Val list | 20 frames | `kitti15_val20_with_seglabel.txt` |

---

## 14. Troubleshooting

| Symptom | What to check |
|---------|----------------|
| `FileNotFoundError` on `data/` or images | Dataset is not in this repo — get it from [LightStereo-Instance](https://github.com/laserquit/LightStereo-Instance) and place/symlink as `data/KITTI15/` |
| Wrong image paths after setup | Check `data/KITTI15/training/image_{2,3}` and columns in the list files |
| Empty known mAP after leave-out | ran `prepare_openworld.py`? YAML `KNOWN_CLASSES` matches split? |
| `pretrained/f5_checkpoint_best.pth` missing | train F5 first, then **copy** a best ckpt into `pretrained/` |
| CUDA OOM while training | lower `BATCH_SIZE_PER_GPU` in YAML; disable unused viz flags |
| CUDA OOM on open-world eval | eval one `--ckpt_tags`; reduce image size only for debugging |
| `transformers` / `Sam2Model` import error | `pip install -r requirements.txt` (`transformers>=4.45`) |
| HF download fails | set `HF_HOME`, use mirror, or pre-download models offline |
| YOLOE path fails | ensure `yoloe-26x-seg.pt` exists; use text-prompt `*-seg.pt`, not `*-seg-pf.pt` |
| Windows long-path errors | enable long paths, or place checkpoints under a short path |
| Metrics disagree with paper | confirm own-best suite (not a single ckpt), `--split` name, and GDINO thresholds |

Label-leakage audit (optional):

```bash
python tools/audit_label_leakage.py
```

---

## 15. Contact

For any questions or problems while reproducing the experiments in this repository, please email:

**zhouyiqiao0913@163.com**

---

## 16. References

Dataset / closed-set baseline package used by this project:

- [LightStereo-Instance](https://github.com/laserquit/LightStereo-Instance)

```bibtex
@inproceedings{guo2025lightstereo,
  title={LightStereo: Channel Boost Is All You Need for Efficient 2D Cost Aggregation},
  author={Guo, Xianda and Zhang, Chenming and Zhang, Youmin and Zheng, Wenzhao and Nie, Dujun and Chen, Long},
  booktitle={ICRA},
  year={2025}
}

@inproceedings{kitti2015,
  title={Object Scene Flow for Autonomous Vehicles},
  author={Menze, Moritz and Geiger, Andreas},
  booktitle={CVPR},
  year={2015}
}
```
