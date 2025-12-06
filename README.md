# starBoard

A desktop application for **wildlife photo-ID matching**, designed to help researchers match unknown individuals (Queries) against a known population (Gallery). 

While built specifically for the sunflower sea star (*Pycnopodia helianthoides*), the architecture supports other species with similar morphological traits. It combines visual comparison with a multi-modal ranking engine (morphometrics, color, and text descriptions).

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

---

## Table of Contents
- [Workflow Overview](#workflow-overview)
- [Installation](#installation)
- [Data Organization](#data-organization)
- [User Guide](#user-guide)
- [Metadata Reference](#metadata-reference)
- [Technical Details](#technical-details)
- [Development](#development)

---

## Workflow Overview

The application follows a four-step workflow that mirrors the user interface tabs:

1.  **Setup (Ingest):** Import new encounter folders and annotate them with metadata.
2.  **First-order (Ranking):** Select a Query individual. The system ranks the entire Gallery based on similarity scores (color, size, description). You "pin" the most likely candidates.
3.  **Second-order (Verification):** Perform detailed side-by-side visual comparisons between the Query and your pinned candidates. Mark decisions as **Match (Yes)**, **Potential (Maybe)**, or **Non-match (No)**.
4.  **Past Matches (Merge):** Review confirmed matches and merge the Query data into the existing Gallery ID, preserving the history.

---

## Installation

### Prerequisites
You need a Python environment manager. We recommend **Miniconda** or **Anaconda**.

### Quick Start
1.  Open your terminal (Anaconda Prompt on Windows, Terminal on macOS/Linux).
2.  Copy and paste the following block to create the environment and install dependencies:

```bash
# Create a clean environment named 'starboard'
conda create -n starboard python=3.11 -y

# Activate the environment
conda activate starboard

# Install core libraries
pip install -U PySide6 sentence-transformers numpy pillow

# Install helper for faster model loading
pip install -U safetensors

# Install visualization tools
pip install -U matplotlib
```

### GPU Acceleration (Optional)
If you have an NVIDIA GPU, you can speed up text processing by installing PyTorch with CUDA support:
```bash
conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Running the App
Once installed, navigate to the project folder and run:
```bash
python main.py
```

---

## Data Organization

The application relies on a strict folder structure to automatically detect dates and IDs. 

**Important:** The application manages its own data in an `archive/` directory. 

### File Naming Convention
When organizing your source images before upload:
-   **Encounters** must be folders named with the date: `MM_DD_YY` (e.g., `09_08_25`).
-   You can add a suffix for uniqueness: `09_08_25_dive1`.

### The Archive Structure
The app creates and manages this structure automatically. You usually do not need to touch this manually, but understanding it helps.

```text
archive/
├── gallery/                    # The Known Population
│   ├── <gallery_id>/           # e.g., "stella_the_star"
│   │   ├── 01_15_25/           # Encounter folder
│   │   │   └── image.jpg
│   │   └── _merge_history.csv  # History of merged queries
│   └── ...
├── queries/                    # The Unknowns (New Encounters)
│   ├── <query_id>/             # e.g., "unknown_001"
│   │   ├── 12_05_24/           # Encounter folder
│   │   └── _second_order_labels.csv  # Your decisions (Yes/No/Maybe)
│   └── ...
└── reports/                    # Generated CSV exports and summaries
```

---

## User Guide

### 1. Setup Tab
*   **Single Upload:** Use this to create a new ID or add to an existing one.
*   **Batch Upload:** Point to a folder containing multiple ID subfolders to import them all at once.
*   **Metadata Editor:** Essential for the ranking engine. Ensure you fill out fields like `arm_color`, `stripe_order`, and `tip_to_tip_size_cm`. The more data you enter, the better the ranking.

### 2. First-order Tab (Ranking)
This is your "broad search."
*   **Select a Query:** Choose the unknown individual on the left.
*   **Filters:** Toggle which metadata fields (Color, Pattern, Text) contribute to the score.
*   **Results:** The gallery list on the right sorts by probability. High scores indicate a close match in metadata.
*   **Action:** Click the **Pin** icon on likely matches to send them to the next stage.

### 3. Second-order Tab (Verification)
This is your "fine-tooth comb."
*   **Compare:** View the Query images alongside the pinned Gallery candidate.
*   **Tools:** Pan, zoom, and annotate images to verify specific morphological features.
*   **Decision:** Record your judgment.
    *   **YES:** Confirmed match.
    *   **MAYBE:** Needs more review.
    *   **NO:** Confirmed different individual.

### 4. Past Matches Tab
*   **Review:** See a timeline of all decisions.
*   **Merge:** "Merge YES's" will move the Query's images and data into the Gallery folder. This is how your population database grows.
*   **Revert:** If you make a mistake, you can undo a merge here.

---

## Metadata Reference

The ranking engine uses these fields to calculate similarity.

### Morphological & Color Fields

| Field | Type | Description |
|-------|------|-------------|
| **Numeric** | | |
| `num_apparent_arms` | Numeric | Count of visible arms. |
| `num_total_arms` | Numeric | Total arms including small/hidden ones. |
| `tip_to_tip_size_cm` | Numeric | Approximate diameter (tip to tip). |
| **Arms & Stripes** | | |
| `short_arm_code` | Set/Code | Specific notation for growing and regenerating arms (e.g., `tiny(2)`). |
| `arm_color` | Color | Primary color of the arms. |
| `arm_thickness` | Ordinal | Relative thickness (Thin → Thick). |
| `stripe_color` | Color | General color of arm stripes. |
| `stripe_order` | Ordinal | Regularity of stripes (None → Regular). |
| `stripe_prominence` | Ordinal | Contrast of stripes (Weak → Strongest). |
| `stripe_extent` | Ordinal | How far stripes extend (None → Full). |
| **Central Disc & Features** | | |
| `central_disc_color` | Color | Color of the central disc background. |
| `papillae_central_disc_color` | Color | Color of papillae on the central disc. |
| `madreporite_color` | Color | Color of the madreporite. |
| `rosette_color` | Color | General color of rosettes. |
| `rosette_prominence` | Ordinal | Visibility of rosettes (Weak → Strong). |
| **Other** | | |
| `papillae_stripe_color` | Color | Color of papillae in stripe regions. |
| `reticulation_order` | Ordinal | Pattern of reticulation (None → Train tracks). |
| `overall_color` | Color | Overall color impression. |

### Text Descriptions (AI-Enhanced)
Fields like `location`, `unusual_observation`, and `health_observation` are analyzed using **sentence-transformers**. This means "large gash on arm" and "big cut on limb" will be recognized as similar.

---

## Technical Details

### Scoring System
The app calculates a similarity score `[0, 1]` for each field:
-   **Numeric/Ordinal:** Gaussian decay based on deviation.
-   **Color:** Perceptual distance in LAB space.
-   **Text:** Cosine similarity of embeddings (`BAAI/bge-small-en-v1.5`).
-   **Short Arm Code:** Position-aware fuzzy Jaccard index.

### Environment Variables
Advanced configuration for deployment or power users.

| Variable | Default | Description |
|----------|---------|-------------|
| `STARBOARD_ARCHIVE_DIR` | `./archive` | Location of the data database. |
| `STARBOARD_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Hugging Face model for text embeddings. |
| `STARBOARD_LOG_LEVEL` | `INFO` | Logging verbosity. |

---

## Development

### Project Structure
```text
starBoard/
├── main.py                 # Entry point
├── src/
│   ├── data/               # I/O, CSV parsers, Merge logic
│   ├── search/             # Ranking engine, Embeddings, Scorers
│   └── ui/                 # PySide6 Widgets and Tabs
├── analysis/               # Accuracy evaluation scripts
└── archive/                # Local data storage (git-ignored)
```

### Troubleshooting
*   **"Embedding backend unavailable":** Ensure `safetensors` is installed.
*   **Blank Visualizations:** Ensure `matplotlib` is installed.
*   **Ranking is weird:** Try "Rebuild Index" in the Setup tab to refresh the embeddings cache.
