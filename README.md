# Four-task CA-SAM continual-learning experiment

This is a compact reproduction of the paper's key idea: frozen SAM ViT-B, one 3-block task-specific Alignment Layer per task, and a VAE router that selects an adapter or rejects an unknown task. It is designed to measure continual-learning capacity on a free Colab T4, not to claim an exact reproduction of the paper's nine-task score.

## Tasks

The default order is `ACDC -> EBHI-SEG -> Kvasir-SEG -> MSD Prostate`. All are public and collectively small enough to work task-by-task in Colab. Do **not** pre-cache every image at 1024x1024 in Drive.

## Data contract

Preprocess each source into binary 2D image/mask pairs. The project deliberately separates source-specific conversion from training because the four sources use incompatible file formats (PNG/JPG, NIfTI, and MHD). Keep the official held-out test data separate, and create a deterministic validation partition only from the official training data:

```text
data/
  acdc/train.jsonl
  acdc/val.jsonl
  acdc/test.jsonl
  ebhi_seg/train.jsonl
  kvasir_seg/train.jsonl
  msd_prostate/train.jsonl
```

Each JSONL line uses paths relative to its manifest. Start by making `train_all.jsonl` from the official training split, then run this for every task:

```bash
python split_manifest.py data/acdc/train_all.jsonl --val-fraction 0.15 --seed 2026
```

This writes `train.jsonl` (85%) and `val.jsonl` (15%) without accessing or changing `test.jsonl`. The same seed makes the split reproducible.

```json
{"image":"images/0001.png","mask":"masks/0001.png"}
```

For multi-class masks, create a foreground union first. This keeps the first experiment a binary continual-segmentation test. Preserve a held-out `test.jsonl` using the same format for evaluation.

## Colab setup

```bash
!git clone <YOUR_REPOSITORY_URL>
%cd <YOUR_REPOSITORY_DIRECTORY>
!pip install -r requirements.txt
!wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
!python train.py --data /content/data --checkpoint checkpoints/sam_vit_b_01ec64.pth --batch-size 1 --accumulation 6 --output /content/drive/MyDrive/ca_sam_run
```

Use `--resume` after a disconnect. Batch size 1 plus accumulation 6 is intentional for 16 GB T4 memory. Start with `--epochs 2 --vae-epochs 1` to validate data and VRAM, then use the paper settings (24 and 10).

After every epoch, the console reports `epoch [current/total] : train acc/val acc | train loss/val loss`, plus IoU and Dice. Training does not read any `test.jsonl` file. The run saves a standalone `adapter_<task>.pt` and `vae_<task>.pt` after each task, and `ca_sam_four_task.pt` containing the complete model, VAE thresholds, task order, and training history. These weights are reserved for the separate test/inference step.

## What is implemented

- Frozen original SAM ViT-B image/prompt/mask components.
- Three CAResBlocks per Alignment Layer (about 3.54M parameters), with two 3x3 convolutions, efficient channel attention, residual connection, and LayerNorm2d.
- Independent 64-dimensional two-layer VAE routers with parameter-free L2 attention pooling.
- p97 task acceptance thresholds and an identity-adapter OOD fallback.
- AMP, gradient accumulation, and a checkpoint after every completed task.

The original paper does not publish the exact preprocessing or loss configuration. Here the adapter loss is BCE-with-logits plus soft Dice, and p97 is estimated after VAE training rather than the paper's 5-fold calibration. Those two differences should be kept in mind when interpreting results.
