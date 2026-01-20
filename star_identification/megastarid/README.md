# MegaStarID

Unified training module for sea star re-identification, integrating:
- **Wildlife10k** (140k images, 37 species) for broad visual feature learning
- **star_dataset** (8k images) for domain-specific training

## Augmentation Strategy

| Dataset | Library | Augmentations |
|---------|---------|---------------|
| Wildlife10k | torchvision | Standard: crop, flip, rotate, color jitter, blur, erasing |
| star_dataset | Albumentations | Underwater-optimized: perspective, affine, blue-shift color, CLAHE, motion blur, ISO noise, spatter, dropouts |

## Training Strategies

| Strategy | Description |
|----------|-------------|
| **Pre-train → Fine-tune** | Train on Wildlife10k first, then fine-tune on star_dataset |
| **Co-training** | Train on both datasets simultaneously with mixed batches |
| **Star-only** | Baseline training on star_dataset only |

## Quick Start

```bash
# Activate environment
conda activate wildlife

# Pre-train on Wildlife10k, then fine-tune on stars
python -m megastarid.pretrain --epochs 50
python -m megastarid.finetune --checkpoint checkpoints/megastarid/pretrain/best.pth --epochs 100

# Or co-train on both datasets together
python -m megastarid.cotrain --epochs 100 --star-batch-ratio 0.3

# Or baseline (star_dataset only)
python -m megastarid.finetune --epochs 100
```

## Grid Search Experiments

Compare training strategies and loss functions:

```bash
# Run full grid search (6 experiments: 2 losses × 3 strategies)
python -m megastarid.experiments.grid_search --epochs 25 --gpus 0,1

# Dry run to see experiment plan
python -m megastarid.experiments.grid_search --dry-run
```

## Module Structure

```
megastarid/
├── config.py          # PretrainConfig, FinetuneConfig, CotrainConfig
├── models.py          # Model creation and checkpoint loading
├── trainer.py         # Unified training loop
├── datasets/
│   └── combined.py    # StarDataset, CombinedDataset, dataloaders
├── pretrain.py        # Pre-training script (Wildlife10k)
├── finetune.py        # Fine-tuning script (star_dataset)
├── cotrain.py         # Co-training script (both)
└── experiments/
    └── grid_search.py # Grid search over strategies and losses
```

## Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 50/100 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-4 / 5e-5 | Learning rate (lower for fine-tuning) |
| `--circle-weight` | 0.7 | Circle loss weight (0 = triplet only) |
| `--triplet-weight` | 0.3 | Triplet loss weight (0 = circle only) |
| `--star-batch-ratio` | 0.3 | Fraction of batch from stars (co-training) |
| `--gpus` | 0,1 | GPU IDs for DataParallel |

## Evaluation Output

After training on star_dataset (finetune or cotrain), detailed per-identity and per-folder metrics are saved:

```
checkpoints/megastarid/finetune/
├── best.pth
├── star_identity_metrics.csv   # Per-identity CMC@K performance
├── star_folder_metrics.csv     # Per-folder aggregated CMC@K
└── star_evaluation_summary.json
```

**star_identity_metrics.csv** columns:
- `identity`, `folder`, `mean_rank`, `best_rank`, `worst_rank`, `num_queries`, `mAP`, `CMC@1`, `CMC@5`, `CMC@10`, `CMC@20`

**star_folder_metrics.csv** columns:
- `folder`, `num_identities`, `mean_rank`, `std_rank`, `num_queries`, `mAP`, `CMC@1`, `CMC@5`, `CMC@10`, `CMC@20`

