# Open-world combinatorial leave-out

## Design

Not nested "only Truck left out". For each known budget **K**, train **one model per left-out combination**:

| K | #models | Meaning |
|---|---:|---|
| F5 | 1 | all 5 known (closed reference) |
| K4 | 5 | leave exactly 1 class out |
| K3 | 10 | leave exactly 2 classes out |
| K2 | 10 | leave exactly 3 classes out |
| K1 | 5 | leave exactly 4 classes out |

**Total: 31 known models.** Default unknown expert is Grounding DINO + SAM2 (HF); optional frozen `yoloe-26x-seg.pt` for ablations.

Full table: [`TRAINING_MATRIX.md`](TRAINING_MATRIX.md)

Regenerate configs after editing `openworld/splits.py`:

```bash
python tools/generate_openworld_cfgs.py
```

## Train (one experiment at a time)

### Recommended: fine-tune from trained F5 (read-only init)

1. **Copy** (do not move) your trained F5 checkpoint to:
   `pretrained/f5_checkpoint_best.pth`
2. Leave-out configs load that file with `F5_INIT.ENABLED: true`:
   - stereo / backbone / matched head tensors from F5
   - class logits remapped F5 channels for remaining known classes (`CLASS_HEAD_MODE: remap`)
   - training writes only under `output/...` (never overwrites the F5 file)

Example — K4-1 leave Car (use a new `extra_tag` so it does not clash with cold-start runs):

```bash
python tools/train.py --cfg_file cfgs/openworld/k4_1_leave_car.yaml --extra_tag k4_1_from_f5
```

K4-2 leave Truck:

```bash
python tools/train.py --cfg_file cfgs/openworld/k4_2_leave_truck.yaml --extra_tag k4_2_from_f5
```

F5 reference (trains F5 itself; does **not** use `pretrained/f5_checkpoint_best.pth` as save path):

```bash
python tools/train.py --cfg_file cfgs/openworld/f5.yaml --extra_tag f5_ref
```

Cold start (legacy): set `PRETRAINED_MODEL: testdata/s.ckpt`, `YOLO_PRETRAINED_MODEL: myyolo.pt`, and `F5_INIT.ENABLED: false`.

## Checkpoint selection (paper tables)

Training may save several `checkpoint_best_*.pth` files. For paper tables,
**all metrics of a model come from the same checkpoint** (recommended:
`checkpoint_best_mAP.pth`). Pass `--ckpt_tags best_mAP` so suite scripts do not
mix results across different best-* files.

## Test known metrics (single checkpoint)

```bash
python tools/eval_best_suite.py \
  --cfg_file cfgs/openworld/k4_2_leave_truck.yaml \
  --run_dir output/KittiDataset/LightStereo/k4_2_leave_truck/k4_2_from_f5 \
  --ckpt_tags best_mAP
```

Report: `eval_output/best_suite/<run>/best_suite_report.json`.

## Test open-world (GDINO+SAM2+G0) on the same checkpoint

```bash
python tools/eval_openworld_best_suite.py \
  --cfg_file cfgs/openworld/k4_2_leave_truck.yaml \
  --run_dir output/KittiDataset/LightStereo/k4_2_leave_truck/k4_2_from_f5 \
  --split K4-2 \
  --ckpt_tags best_mAP \
  --unknown_expert gdino_sam2 \
  --known_conf 0.001 \
  --box_threshold 0.10 --text_threshold 0.10 \
  --mask_iou_conflict 0.90 \
  --g0_edge_band 3
```

Paper setting: GDINO box/text `0.10`/`0.10`, fusion conflict IoU `0.90`
(CLI overrides `unknown_module.yaml`, which still lists a `0.6` template default).

Default unknown expert is **Grounding DINO + SAM2** (HF), then G0 + fusion.
Optional legacy path: `--unknown_expert yoloe --yoloe_ckpt yoloe-26x-seg.pt`.
Report also logs pre-fusion vs post-fusion `u_recall_*`.

Deps: `pip install 'transformers>=4.45' accelerate` (SAM2 via transformers; no separate `sam2` package needed on Python 3.9).

Report: `eval_output/openworld_best_suite/<run>/openworld_best_suite_report.json`

Single-image debug still uses `tools/infer_openworld.py`.
