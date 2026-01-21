# Temporal Re-Identification

Metric learning pipeline for sea star re-identification across time, using a temporal train/test split that evaluates whether models can match individuals across different outings (days/months/years apart).

## Quick Start

```bash
# Train a single model
python -m temporal_reid.train --dataset-root ./star_dataset --epochs 25 --batch-size 48

# Multi-GPU training (DataParallel)
python -m temporal_reid.train --dataset-root ./star_dataset --epochs 25 --batch-size 64 --gpus 0,1

# Grid search over loss configurations
python -m temporal_reid.grid_search --dataset-root ./star_dataset --epochs 25 --batch-size 64 --gpus 0,1
```

## Key Features

- **Temporal splitting**: Train on earlier outings, test on later outings (avg 354-day gap)
- **Negative-only identities**: Single-outing individuals used only as hard negatives
- **Loss functions**: Circle loss, triplet loss (hard mining), or combinations
- **Model**: SwinV2-small backbone with GeM pooling and learned embeddings
- **Evaluation**: Rank-1/5/10 accuracy and mAP on held-out temporal queries

## Folder Structure

```
temporal_reid/
├── train.py          # Main training entry point
├── grid_search.py    # Hyperparameter search over loss configs
├── config.py         # Configuration dataclasses
├── data/
│   ├── prepare.py    # Temporal split generation from metadata
│   ├── dataset.py    # Dataset and PK sampler with negative-only support
│   ├── transforms.py # Albumentations augmentation pipeline
│   └── cached_dataset.py  # RAM-cached evaluation for speed
├── models/
│   └── swin_reid.py  # SwinV2 + GeM + embedding head
├── training/
│   ├── trainer.py    # Training loop with AMP and DataParallel
│   └── losses.py     # Circle, triplet, and ArcFace losses
└── utils/
    ├── metrics.py    # CMC and mAP computation
    └── helpers.py    # Seed, device, parameter counting
```

## Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 32 | Batch size (P identities × K instances) |
| `--lr` | 1e-4 | Learning rate |
| `--gpus` | None | GPU IDs for DataParallel (e.g., "0,1") |
| `--cache-eval` | off | Cache eval images in RAM for faster validation |
| `--checkpoint-dir` | ./checkpoints/temporal | Where to save models |

## Grid Search

The grid search explores loss configurations:

```bash
# Preview available configs
python -m temporal_reid.grid_search --dry-run

# Run specific configs
python -m temporal_reid.grid_search --configs triplet_only circle_only balanced
```

Results saved to `./grid_search_results/` with per-experiment checkpoints and logs.

## Evaluation Protocol

- **Query set**: Test images from held-out outings
- **Gallery set**: Train images from earlier outings  
- **Task**: Given a query image, retrieve the same individual from the gallery
- **Metrics**: Rank-k accuracy (is correct match in top k?) and mAP


