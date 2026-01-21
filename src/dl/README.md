# Deep Learning Integration for starBoard

This module integrates the `star_identification` (MegaStarID) deep learning project into the starBoard desktop application, providing visual re-identification capabilities for sunflower sea stars.

## Overview

The integration follows a **precomputation-first architecture** designed for small datasets (~200 stars). All embeddings, similarity matrices, and re-ranked scores are computed offline and stored, making runtime queries pure index lookups.

### Key Design Principles

1. **Minimal changes to starBoard** - The DL integration adds a new tab and enhances the First-order tab without modifying core starBoard logic
2. **Graceful degradation** - If PyTorch is not installed, DL features are simply disabled
3. **Hardware-adaptive** - Automatically optimizes for GPU vs CPU
4. **Precomputation-first** - No on-the-fly inference; all computations cached

## Architecture

```
starBoard/
├── src/
│   └── dl/                          # This module
│       ├── __init__.py              # DL availability check, device detection
│       ├── registry.py              # Model registry and precomputation tracking
│       ├── reid_adapter.py          # Interface to MegaStarID models
│       ├── image_cache.py           # YOLO-preprocessed image caching
│       ├── precompute.py            # Background precomputation worker
│       ├── similarity_lookup.py     # Fast similarity matrix lookups
│       └── outlier_detection.py     # Embedding-based outlier rejection
│
├── star_identification/             # MegaStarID project (mostly unchanged)
│   ├── megastarid/                  # Core model definitions
│   ├── wildlife_reid_inference/     # YOLO preprocessor, inference utils
│   ├── precompute_cache/            # YOLO-processed image cache (generated)
│   └── checkpoints/                 # Model weights
│
└── archive/
    └── dl_data/                     # Precomputed embeddings and similarity (generated)
        └── <model_key>/
            ├── embeddings/
            │   ├── gallery_embeddings.npz
            │   └── query_embeddings.npz
            └── similarity/
                ├── query_gallery_scores.npz
                ├── id_mapping.json
                └── metadata.json
```

## Components

### 1. `__init__.py` - Package Initialization

Checks for PyTorch availability and sets up device detection:

```python
try:
    import torch
    DL_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DL_AVAILABLE = False
    DEVICE = None
```

All other modules check `DL_AVAILABLE` before attempting DL operations.

### 2. `registry.py` - Model Registry

Manages `_dl_registry.json` which tracks:
- Registered models and their checkpoint paths
- Precomputation status for each model
- Currently active model for visual ranking
- Pending IDs that need precomputation after new data is added

```python
@dataclass
class ModelEntry:
    checkpoint_path: str
    checkpoint_hash: str
    display_name: str
    precomputed: bool = False
    last_computed: Optional[str] = None
    gallery_count: int = 0
    query_count: int = 0
```

### 3. `reid_adapter.py` - Model Interface

Provides a clean interface to MegaStarID models:

- **Automatic backbone detection** - Infers model architecture (DenseNet, SwinV2, ResNet) from checkpoint state dict keys
- **YOLO integration** - Loads and runs the YOLO instance segmentation model for star cropping
- **TTA support** - Test-time augmentation with horizontal/vertical flip
- **Batch processing** - Efficient batch embedding extraction

Key method:
```python
def extract_batch(
    image_paths: List[str],
    use_tta: bool = True,
    batch_size: int = 8,
    use_horizontal_flip: bool = True,
    use_vertical_flip: bool = True,
    use_yolo_preprocessing: bool = True
) -> Optional[np.ndarray]
```

### 4. `image_cache.py` - YOLO Preprocessing Cache

Creates a mirror dataset of YOLO-preprocessed images to speed up embedding extraction:

**Location:** `star_identification/precompute_cache/`

**Structure mirrors archive:**
```
precompute_cache/
├── gallery/
│   ├── anchovy/
│   │   ├── image1.png
│   │   └── image2.png
│   └── ...
└── queries/
    └── ...
```

**Benefits:**
- YOLO runs once per image, results cached
- Cached images are resized to 640px (smaller, faster to load)
- Subsequent precomputations skip Phase 1 if cache exists

### 5. `precompute.py` - Background Worker

Runs precomputation in a background `QThread` with progress signals:

**Two-Phase Pipeline:**

1. **Phase 1: Build Image Cache**
   - Load YOLO segmentation model
   - Process all images through YOLO (crop/segment stars)
   - Resize to 640px and save as PNG
   - Skip already-cached images

2. **Phase 2: Extract Embeddings**
   - Load re-ID model (auto-detect architecture)
   - Extract embeddings from cached images (skip YOLO)
   - Apply TTA if enabled
   - Aggregate with outlier rejection
   - Compute similarity matrix with optional re-ranking
   - Save results

**Hardware-Adaptive Settings:**

```python
@dataclass
class HardwareProfile:
    device: str           # "cuda" or "cpu"
    batch_size: int       # 16 for GPU, 4 for CPU
    use_tta: bool
    use_horizontal_flip: bool
    use_vertical_flip: bool   # Disabled on CPU for 2x speedup
    use_mixed_precision: bool
    estimated_img_per_sec: float
```

### 6. `similarity_lookup.py` - Fast Lookups

Provides cached access to precomputed similarity matrices:

```python
def get_visual_scores(query_id: str, model_key: str) -> Dict[str, float]:
    """Get visual similarity scores for a query against all gallery IDs."""
```

Used by First-order tab to blend visual and metadata scores.

### 7. `outlier_detection.py` - Embedding Outlier Rejection

Detects and excludes bad images before aggregating identity embeddings:

**Algorithm (Nearest-Neighbor + MAD):**
1. Extract embeddings for all images of an identity
2. Compute pairwise similarities between all embeddings
3. For each image, find its nearest-neighbor similarity (best match)
4. Compute median and MAD (Median Absolute Deviation) of NN similarities
5. Flag images whose NN similarity is >3 MAD below median (isolated points)
6. Aggregate inlier embeddings into final centroid

**Why this approach?**
- **Multi-modal friendly**: Front/back views of same animal stay connected to their clusters
- **Robust to contamination**: Median/MAD aren't affected by outliers (unlike mean/std)
- **Model-agnostic**: Works with triplet loss, circle loss, or any metric learning model
- **No hardcoded thresholds**: Uses relative statistics, not absolute distance values

**Safety constraints:**
- Minimum 1 inlier always kept
- Maximum 25% of images can be flagged as outliers
- Special handling for small sets (n ≤ 3)

**What gets flagged:**
- Wrong YOLO detections (different animal entirely)
- Severely corrupted images producing meaningless embeddings
- Mislabeled images

**What does NOT get flagged:**
- Different legitimate views (front vs back vs side)
- Different lighting conditions
- Different poses of the same animal

## UI Integration

### Deep Learning Tab (`src/ui/tab_dl.py`)

New tab added to the main window with:

1. **Status Section**
   - Device info (CPU/GPU, PyTorch version)
   - Active model
   - Precomputation status

2. **Model Management**
   - List of registered models
   - Import/Remove models
   - Set active model

3. **Precomputation Controls**
   - Scope selection (Gallery/Queries)
   - Speed mode (Auto/Fast/Quality)
   - TTA and re-ranking toggles
   - Progress bar with ETA

4. **Fine-Tuning Section** (placeholder for future)

### First-Order Tab Enhancement

Added visual ranking controls:

- **Visual checkbox** - Enable/disable DL visual similarity
- **Model dropdown** - Select which precomputed model to use
- **Fusion slider** - Blend between metadata (0%) and visual (100%) ranking

When enabled, the ranking becomes:
```python
fused_score = (alpha * visual_score) + ((1 - alpha) * metadata_score)
```

## Precomputation Pipeline

### Complete Flow

```
User clicks "Run Full Precomputation"
    │
    ▼
Phase 1: YOLO Preprocessing
    ├── Load YOLO model (starseg_best.pt)
    ├── For each identity:
    │   ├── Load original images
    │   ├── Run YOLO instance segmentation
    │   ├── Crop and mask star from background
    │   ├── Resize to 640px
    │   └── Save to precompute_cache/
    └── Skip if already cached
    │
    ▼
Phase 2: Embedding Extraction
    ├── Load re-ID model (auto-detect architecture)
    ├── For each identity:
    │   ├── Load cached images (small, preprocessed)
    │   ├── Apply test transforms (resize to 384, normalize)
    │   ├── Run through model with TTA
    │   ├── Get per-image embeddings
    │   ├── Detect and exclude outliers (distance > 0.6)
    │   └── Aggregate inliers: mean + L2-normalize
    │
    ▼
Phase 3: Similarity Computation
    ├── Stack all embeddings into matrices
    ├── Compute cosine similarity (query × gallery)
    ├── Optional: Apply k-reciprocal re-ranking
    └── Save matrices and ID mappings
    │
    ▼
Update registry, clear caches, emit completion signal
```

### Output Files

```
archive/dl_data/<model_key>/
├── embeddings/
│   ├── gallery_embeddings.npz   # {id: embedding} for gallery
│   └── query_embeddings.npz     # {id: embedding} for queries
└── similarity/
    ├── query_gallery_scores.npz  # (n_queries, n_gallery) matrix
    ├── id_mapping.json           # {"query_ids": [...], "gallery_ids": [...]}
    └── metadata.json             # {use_tta, use_reranking, embedding_dim, ...}
```

## Model Support

### Automatic Architecture Detection

The `ReIDAdapter` automatically detects the model architecture from checkpoint keys:

| Key Pattern | Detected Architecture |
|-------------|----------------------|
| `denselayer` | DenseNet121 |
| `layers.0.blocks` or `swinv2` | SwinV2-tiny |
| `layer1.0.conv1` + `layer4` | ResNet50 |
| `_blocks` or `features.0` | EfficientNet |

### Inferred Configuration

From checkpoint state dict:
- **Embedding dimension** - From `embedding_head.layer3.0.weight.shape[0]`
- **Use multiscale** - Presence of `fusion` keys
- **Use BN-neck** - Presence of `bnneck` keys

## Dependencies

Core starBoard dependencies plus optional DL packages:

**requirements-dl.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
ultralytics>=8.0.0      # For YOLO segmentation
opencv-python>=4.8.0
albumentations>=1.3.0
pandas>=1.5.0
scikit-learn>=1.2.0
```

## Usage

### First-Time Setup

1. Install DL dependencies: `pip install -r requirements-dl.txt`
2. Ensure model checkpoint exists (default path in registry)
3. Launch starBoard
4. Go to "Deep Learning" tab
5. Click "Run Full Precomputation"
6. Wait for completion (shows ETA)

### Using Visual Ranking

1. Ensure precomputation is complete (status shows ✓)
2. Go to "First-order" tab
3. Check "Visual" checkbox
4. Adjust "Fusion" slider (50% recommended)
5. Rankings now blend metadata and visual similarity

### Adding New Data

When new images are added via ingest:
1. Registry tracks pending IDs
2. User prompted to update precomputation
3. Can update for current model only or all models

## Performance

### Typical Timings (CPU)

| Phase | Speed | Notes |
|-------|-------|-------|
| YOLO preprocessing | ~5 img/s | First run only, cached thereafter |
| Embedding extraction (no TTA) | ~8 img/s | Fast mode |
| Embedding extraction (hflip only) | ~4 img/s | Auto mode on CPU |
| Embedding extraction (full TTA) | ~2 img/s | Quality mode |

### Typical Timings (GPU)

| Phase | Speed | Notes |
|-------|-------|-------|
| YOLO preprocessing | ~30 img/s | |
| Embedding extraction | ~50 img/s | With full TTA |

### Memory Usage

- YOLO model: ~50MB
- Re-ID model: ~100-200MB depending on backbone
- Image cache: ~500MB for typical dataset
- Similarity matrix: Negligible

## Troubleshooting

### "Failed to load model"
- Check checkpoint path exists
- Ensure architecture matches (auto-detection logs inferred config)
- Verify PyTorch version compatibility

### "YOLO preprocessor not available"
- Install ultralytics: `pip install ultralytics`
- Check `starseg_best.pt` exists in `wildlife_reid_inference/`

### Slow precomputation
- Use "Fast" speed mode on CPU
- Reduce batch size if memory-limited
- First run is slowest (building image cache)

### Outlier detection removing too many images
- Check YOLO detections (wrong crops will be outliers)
- Review logged outlier paths for patterns
- Threshold is 2× margin (0.6) - fairly generous

## Future Enhancements

1. **Fine-tuning UI** - Train on user's data from the app
2. **Per-image outlier review** - UI to manually include/exclude images
3. **Incremental precomputation** - Only process new images
4. **Multiple model comparison** - Side-by-side ranking from different models


