# Four-task CA-SAM continual-learning experiment

This is a compact reproduction of the paper's key idea: frozen SAM ViT-B, one 3-block task-specific Alignment Layer per task, and a VAE router that selects an adapter or rejects an unknown task. It is designed to measure continual-learning capacity on a free Colab T4, not to claim an exact reproduction of the paper's nine-task score.

## Tasks

The default order is `EBHI-SEG -> Kvasir-SEG -> DDTI -> STS-2D Tooth`. These are native 2D datasets and collectively small enough to work task-by-task in Colab. Do **not** pre-cache every image at 1024x1024 in Drive.

## Data contract

Preprocess each source into binary 2D image/mask pairs. The project deliberately separates source-specific conversion from training because the four sources use incompatible file formats (PNG/JPG, NIfTI, and MHD). Keep the official held-out test data separate, and create a deterministic validation partition only from the official training data:

```text
data/
  ebhi_seg/{images,masks,train.jsonl,val.jsonl,test.jsonl}
  kvasir_seg/{images,masks,train.jsonl,val.jsonl,test.jsonl}
  ddti/{images,masks,train.jsonl,val.jsonl,test.jsonl}
  sts_2d/{images,masks,train.jsonl,val.jsonl,test.jsonl}
```

Each JSONL line uses paths relative to its manifest. Start by making `train_all.jsonl` from the official training split, then run this for every task:

```bash
python split_manifest.py data/ebhi_seg/train_all.jsonl --val-fraction 0.15 --seed 2026
```

This writes `train.jsonl` (85%) and `val.jsonl` (15%) without accessing or changing `test.jsonl`. The same seed makes the split reproducible.

```json
{"image":"images/0001.png","mask":"masks/0001.png"}
```

For multi-class masks, create a foreground union first. This keeps the first experiment a binary continual-segmentation test. Preserve a held-out `test.jsonl` using the same format for evaluation.

## Usage

### 1. Clone and install

```bash
!git clone <REPOSITORY_URL>
%cd <REPOSITORY_DIRECTORY>
!pip install -r requirements.txt
```

### 2. Download the pretrained SAM ViT-B checkpoint

This is Meta's original pretrained SAM weight file. It remains frozen throughout CA-SAM training.

```bash
!mkdir -p /content/checkpoints
!wget -q --show-progress \
  -O /content/checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 3. Place the prepared dataset

The training data path must directly contain the four task folders. For the compact 200-sample experiment:

```text
/content/ca_sam_fixed_200/
  ebhi_seg/
  kvasir_seg/
  ddti/
  sts_2d/
```

### 4. Start training

This configuration uses 200 train / 40 validation samples per task (DDTI has 72 available test samples), trains each adapter for 20 epochs, and uses an effective batch size of 6.

```bash
!python train.py \
  --data /content/ca_sam_fixed_200 \
  --checkpoint /content/checkpoints/sam_vit_b_01ec64.pth \
  --output /content/ca_sam_runs/four_task_fixed_200 \
  --epochs 20 \
  --vae-epochs 5 \
  --batch-size 2 \
  --accumulation 3 \
  --workers 2
```

Use `--resume` after a disconnect, pointing to the same `--output` directory. Batch size 2 with accumulation 3 is selected for a free Colab T4 at SAM's 1024x1024 input resolution.

After every epoch, the console reports `epoch [current/total] : train acc/val acc | train loss/val loss`, plus IoU and Dice. Training does not read any `test.jsonl` file. The run saves a standalone `adapter_<task>.pt` and `vae_<task>.pt` after each task, and `ca_sam_four_task.pt` containing the complete model, VAE thresholds, task order, and training history. These weights are reserved for the separate test/inference step.

## What is implemented

- Frozen original SAM ViT-B image/prompt/mask components.
- Three CAResBlocks per Alignment Layer (about 3.54M parameters), with two 3x3 convolutions, efficient channel attention, residual connection, and LayerNorm2d.
- Independent 64-dimensional two-layer VAE routers with parameter-free L2 attention pooling.
- p97 task acceptance thresholds and an identity-adapter OOD fallback.
- Hybrid precision for the frozen encoder, gradient accumulation, and a checkpoint after every completed task.

The original paper does not publish the exact preprocessing or loss configuration. Here the adapter loss is BCE-with-logits plus soft Dice, and p97 is estimated after VAE training rather than the paper's 5-fold calibration. Those two differences should be kept in mind when interpreting results.
