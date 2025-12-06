# starBoard

A Qt-based desktop application for **wildlife photo-ID matching**, designed to help researchers match unknown individuals (Queries) against a known population (Gallery) using a combination of visual comparison and metadata-based ranking.

Originally developed for sunflower sea star (*Pycnopodia helianthoides*) population monitoring, starBoard is adaptable to any photo-ID workflow where individuals can be described with structured metadata.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

---

## Features

- **Multi-modal ranking engine** — Combines numeric measurements, ordinal categories, color similarity (LAB color space), set-based codes, and text embeddings to rank potential matches
- **Sentence-transformer embeddings** — Uses `BAAI/bge-small-en-v1.5` (configurable) for semantic similarity of text descriptions
- **Interactive image comparison** — Side-by-side viewers with pan, zoom, and annotation tools
- **Decision workflow** — Label pairs as Yes/Maybe/No with notes; merge confirmed matches into the gallery
- **Merge & revert system** — Reversible merge operations with full history tracking
- **Reports & exports** — Generate master CSVs, summaries, timeline analytics, and decision matrices

---

## Core Concepts

| Term | Description |
|------|-------------|
| **Gallery** | Known individuals in your population database |
| **Query** | Unknown individuals to be matched against the gallery |
| **Encounter** | A dated observation folder (`MM_DD_YY[_suffix]`) containing images |
| **First-order** | Initial ranking of gallery candidates for a query |
| **Second-order** | Detailed side-by-side comparison for decision-making |

---

## Installation

### Requirements

- **Python 3.10+** (uses modern type syntax like `float | None`)
- **OS:** Windows, macOS, or Linux

### Quick Start with Conda

```bash
# Create environment
conda create -n starboard python=3.11 -y
conda activate starboard

# Install dependencies (all-pip approach)
pip install -U PySide6 sentence-transformers numpy pillow

# Optional but recommended for faster embedding model loading
pip install -U safetensors

# Optional for in-app visualizations
pip install -U matplotlib
```

### Alternative: Conda-first + pip

```bash
conda install -y -c conda-forge pyside6 numpy pillow matplotlib safetensors
pip install -U sentence-transformers
```

### GPU Acceleration (Optional)

For faster text embeddings with CUDA:

```bash
conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## Running the App

```bash
python main.py
```

The app will:
1. Create an `archive/` folder (or use `STARBOARD_ARCHIVE_DIR`)
2. Initialize logging to `archive/starboard.log`
3. Build embedding caches on first ranking operation

---

## Archive Structure

```
archive/
├── gallery/
│   ├── gallery_metadata.csv          # Metadata for known individuals
│   ├── _embeddings/                   # Cached text embeddings
│   │   └── text_embeddings_bge.json
│   ├── <gallery_id>/                  # Individual folders
│   │   ├── _merge_history.csv         # Merge operation log
│   │   └── 01_15_25/                  # Encounter folder (MM_DD_YY)
│   │       └── *.jpg
│   └── ...
├── queries/
│   ├── queries_metadata.csv           # Metadata for unknown individuals
│   ├── _embeddings/
│   │   └── text_embeddings_bge.json
│   ├── <query_id>/
│   │   ├── _second_order_labels.csv   # Decision records
│   │   ├── _pins_first_order.json     # Pinned candidates
│   │   ├── _SILENT.flag               # Marks merged/hidden queries
│   │   └── 12_05_24/
│   │       └── *.jpg
│   └── ...
├── reports/
│   ├── past_matches_master.csv
│   ├── past_matches_decisions.csv
│   └── summary_*.csv
└── starboard.log
```

---

## Metadata Schema (V2)

The app uses a structured annotation schema with five field types:

### Numeric Fields
| Field | Description |
|-------|-------------|
| `num_apparent_arms` | Visible arm count |
| `num_total_arms` | Total arms including small/hidden |
| `tip_to_tip_size_cm` | Diameter measurement |

### Ordinal Categorical Fields
| Field | Options |
|-------|---------|
| `stripe_order` | None → Mixed → Irregular → Regular |
| `stripe_prominence` | None → Weak → Medium → Strong → Strongest |
| `stripe_extent` | None → Quarter → Halfway → Three quarters → Full |
| `arm_thickness` | Thin → Medium → Thick |
| `rosette_prominence` | Weak → Medium → Strong |
| `reticulation_order` | None → Mixed → Meandering → Train tracks |

### Color Fields
- `stripe_color`, `arm_color`, `central_disc_color`, `papillae_central_disc_color`
- `rosette_color`, `papillae_stripe_color`, `madreporite_color`, `overall_color`

Uses perceptual similarity in LAB color space with fallback to exact matching.

### Set/Code Fields
| Field | Description |
|-------|-------------|
| `short_arm_code` | Position-aware arm coding, e.g., `tiny(2), small(10), short(3)` |

### Text Fields (Embedded)
| Field | Description |
|-------|-------------|
| `location` | Observation location |
| `unusual_observation` | Notable features |
| `health_observation` | Health-related notes |

---

## Application Tabs

### Setup Tab
- **Single Upload** — Ingest images for a new or existing ID with metadata
- **Batch Upload** — Discover and import multiple IDs from a folder structure
- **Metadata Editing Mode** — Browse and edit metadata with image preview and carry-over support

### First-order Tab
- Select a Query and configure which fields to use for ranking
- Adjust numeric offsets to account for growth/change over time
- View ranked Gallery candidates with per-field similarity breakdowns
- Pin promising candidates for Second-order review
- Filter by observation date range

### Second-order Tab
- Side-by-side comparison of Query and Gallery images
- Pan, zoom, and annotation tools for detailed inspection
- Record decisions (Yes/Maybe/No) with notes
- Jump directly from First-order selections

### Past Matches Tab
- Export master CSVs and summaries
- Visualizations: Totals, Timeline, By Query, By Gallery, Matrix view
- **Merge YES's** — Move confirmed query encounters into gallery folders
- **Revert merges** — Undo merge operations by batch

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STARBOARD_ARCHIVE_DIR` | Data directory path | `./archive` |
| `STARBOARD_EMBED_MODEL` | Hugging Face model ID | `BAAI/bge-small-en-v1.5` |
| `STARBOARD_LOG_LEVEL` | Logging verbosity | `INFO` |
| `STARBOARD_SESSION_ID` | Session tag for logs | Auto-generated |
| `STARBOARD_DUMP_RANK_CSV` | Export ranking CSVs | Unset (off) |

### Example (PowerShell)

```powershell
$env:STARBOARD_ARCHIVE_DIR = "D:\starboard\archive"
$env:STARBOARD_LOG_LEVEL = "DEBUG"
$env:STARBOARD_DUMP_RANK_CSV = "1"
python main.py
```

### Example (Bash)

```bash
export STARBOARD_ARCHIVE_DIR="$HOME/data/starboard"
export STARBOARD_LOG_LEVEL="DEBUG"
python main.py
```

---

## Scoring System

The ranking engine computes per-field similarity scores in `[0, 1]`:

| Field Type | Scoring Method |
|------------|----------------|
| Numeric | Gaussian decay based on median absolute deviation (MAD) |
| Ordinal | Gaussian decay (treats as ordered numeric) |
| Color | Perceptual distance in LAB color space |
| Set/Code | Position-aware fuzzy Jaccard for short arm codes |
| Text | Cosine similarity of sentence-transformer embeddings |

Final scores are computed as weighted averages of active fields. By default, all enabled fields contribute equally.

---

## Project Structure

```
starBoard/
├── main.py                 # Application entry point
├── src/
│   ├── data/               # I/O, CSV handling, validators, ingest
│   │   ├── archive_paths.py
│   │   ├── annotation_schema.py
│   │   ├── csv_io.py
│   │   ├── merge_yes.py
│   │   └── ...
│   ├── search/             # Ranking engine and field scorers
│   │   ├── engine.py
│   │   ├── embed_store.py
│   │   ├── fields_*.py
│   │   └── interfaces.py
│   └── ui/                 # Qt widgets and tabs
│       ├── main_window.py
│       ├── tab_setup.py
│       ├── tab_first_order.py
│       ├── tab_second_order.py
│       ├── tab_past_matches.py
│       └── ...
├── analysis/               # Evaluation scripts and outputs
│   ├── evaluate_metadata_*.py
│   └── plot_*.py
└── archive/                # Default data directory
```

---

## Troubleshooting

**"Embedding backend unavailable…"**  
Install `safetensors` or upgrade PyTorch to >= 2.6:
```bash
pip install safetensors
```

**Blank visualizations**  
Install matplotlib:
```bash
pip install matplotlib
```

**No IDs showing up**  
- Verify `STARBOARD_ARCHIVE_DIR` points to correct location
- Check that encounter folders follow `MM_DD_YY[_suffix]` naming
- Ensure images are in `gallery/<id>/<encounter>/` structure

**Ranking results seem wrong**  
- Click "Rebuild index" to refresh metadata and embeddings
- Verify CSV header names match exactly (case-sensitive)
- Check that relevant fields have values populated

---

## Development

The app runs directly from source. `main.py` adds the project root to `sys.path` for imports.

### Key Modules

- `src/search/engine.py` — Core ranking logic and scorer orchestration
- `src/search/embed_store.py` — Text embedding cache with hash-based invalidation
- `src/search/fields_*.py` — Individual field scorer implementations
- `src/data/merge_yes.py` — Merge/revert operations with history
- `src/ui/tab_first_order.py` — Main ranking interface

### Logging

Logs rotate automatically (1MB × 3 backups) at `archive/starboard.log`.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Developed for sunflower sea star population monitoring research. The annotation schema and scoring methods are designed around the morphological characteristics of *Pycnopodia helianthoides*.
