# Wildlife ReID-10k Module

Dataset handling and training infrastructure for the [WildlifeReID-10k](https://www.kaggle.com/datasets/wildlifedatasets/wildlifereid-10k) benchmark dataset.

## Overview

This module provides:
- **Dataset loading** with filtering by species/dataset
- **Per-dataset split strategies** (time-aware, cluster-aware, original)
- **Training infrastructure** matching `temporal_reid` (SwinV2 + Circle/Triplet loss)

## Dataset

Wildlife10k contains **140,488 images** across **37 sub-datasets** and **22 species** including:
- Sea turtles, cows, cats, dogs, tigers, whales, zebras, pandas, sea stars, etc.

Notably includes **SeaStarReID2023** (2,187 images) - directly relevant to our project.

## Quick Start

```python
from wildlife_reid import Wildlife10kLoader, Wildlife10kConfig, FilterConfig

# Load full dataset
loader = Wildlife10kLoader("./wildlifeReID")
train_ds, test_ds = loader.load()

# Or filter to specific datasets/species
config = Wildlife10kConfig(
    data_root="./wildlifeReID",
    filter=FilterConfig(include_datasets=["SeaStarReID2023", "BelugaID"]),
)
loader = Wildlife10kLoader("./wildlifeReID", config)
train_ds, test_ds = loader.load()
```

## Training

```bash
# Default training (full Wildlife10k)
python -m wildlife_reid.train

# Train on specific datasets
python -m wildlife_reid.train --include-datasets SeaStarReID2023 BelugaID

# Use per-dataset optimal splits
python -m wildlife_reid.train --split-strategy recommended --epochs 50

# See all options
python -m wildlife_reid.train --help
```

### Default Configuration
| Parameter | Value |
|-----------|-------|
| Model | `microsoft/swinv2-small-patch4-window16-256` |
| Image size | 384px |
| Embedding dim | 512 |
| Losses | Circle (0.7) + Triplet (0.3) |
| Batch size | 32 (8 identities × 4 instances) |

## Module Structure

```
wildlife_reid/
├── config.py          # Configuration dataclasses
├── registry.py        # Dataset registry (37 sub-datasets)
├── loader.py          # High-level data loader
├── train.py           # Training script
├── datasets/
│   ├── base.py        # BaseWildlifeDataset (PyTorch Dataset)
│   ├── wildlife10k.py # Main Wildlife10kDataset
│   └── subdataset.py  # Per-dataset split logic
├── models/
│   └── swin_reid.py   # SwinV2 + GeM + embedding head
├── training/
│   ├── losses.py      # Circle + Triplet loss
│   └── trainer.py     # Training loop with AMP
└── utils/
    ├── samplers.py    # P-K sampler for metric learning
    ├── transforms.py  # Train/test augmentations
    └── metrics.py     # mAP, CMC evaluation
```

## Split Strategies

| Strategy | Description |
|----------|-------------|
| `original` | Use provided train/test splits |
| `recommended` | Per-dataset optimal (time/cluster/original) |
| `time_aware` | Split by date (earlier→train, later→test) |
| `cluster_aware` | Keep similar images together |
| `random` | Random per-identity split |

## Dataset Registry

View all registered datasets:
```python
from wildlife_reid import DATASET_REGISTRY
print(DATASET_REGISTRY.summary())
```

Datasets with temporal info: `AmvrakikosTurtles`, `BelugaID`, `BirdIndividualID`  
Datasets with clusters: `AAUZebraFish`, `ATRW`, `CTai`, `CZoo`, etc.

## Testing

```bash
python -m wildlife_reid.test_module
```


