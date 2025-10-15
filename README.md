# starBoard

A lightweight, Qt‑based desktop app for organizing a Gallery of IDs, loading Queries, and ranking likely matches using a mix of metadata (numeric, categorical, set codes) plus text embeddings. It also lets you label pairs (“yes/maybe/no”) and export past matches.

---

## Core ideas

- Keep your data in an `archive/` folder (or any folder you choose).
- Each **Gallery** / **Query** ID has an **image folder**.
- Metadata lives in two CSVs: `gallery_metadata.csv` and `queries_metadata.csv`.
- Ranking uses field‑specific scorers + **sentence‑transformer** embeddings (default model: `BAAI/bge-m3`).

---

## Quickstart (conda)

> Works with conda or mamba. Replace `conda` with `mamba` if you prefer.

1) **Create and activate an environment (Python 3.10+)**

    conda create -n starboard -y python=3.11
    conda activate starboard

2) **Install runtime dependencies** (pick ONE of the following patterns)

   **All‑pip inside conda (simple and reliable):**

    pip install -U PySide6 sentence-transformers numpy pillow

   **Or conda‑first + pip (reduces build time on some OSes):**

    conda install -y -c conda-forge pyside6 numpy pillow
    pip install -U sentence-transformers

3) **(Optional but recommended for embeddings)**

   Fast, safe model weights:

    conda install -y -c conda-forge safetensors

   *Alternative:* a recent PyTorch (>= 2.6). CPU is fine; GPU is optional.

   **GPU (optional):** install CUDA‑enabled PyTorch that matches your system:

    conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

4) **(Optional) Choose a data location** (defaults to `./archive` next to `main.py`)

   **Bash / zsh**

    export STARBOARD_ARCHIVE_DIR="/path/to/archive"

   **Windows PowerShell**

    $env:STARBOARD_ARCHIVE_DIR="D:\archive"

5) **Run the app**

    python main.py

> If the embedding backend complains, install **safetensors** (recommended) or upgrade to a recent **PyTorch (>= 2.6)**.

---

## Installation (details)

### Requirements

- **Python 3.10+** (uses modern type syntax like `float | None`)
- **OS:** Windows / macOS / Linux

### Packages

- **Required:** `PySide6`, `sentence-transformers`, `numpy`, `pillow`
- **Optional (recommended):** `safetensors` (or recent `torch`) for loading embedding models
- **Optional (for in‑app plots):** `matplotlib` (the app runs without it)

### One‑liners

**Everything via pip (inside conda env):**

    pip install -U PySide6 sentence-transformers numpy pillow
    pip install -U safetensors        # optional but recommended for embeddings
    pip install -U matplotlib         # optional (plots)

**Conda‑first (conda‑forge) + pip:**

    conda install -y -c conda-forge pyside6 numpy pillow matplotlib
    conda install -y -c conda-forge safetensors
    pip install -U sentence-transformers

**GPU note:** If you want GPU acceleration for embeddings, install a CUDA‑enabled PyTorch that matches your system:

    conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

This is optional—CPU works, just slower.

---

## Folder layout & data model

`starBoard` reads and writes under a single **archive root**.

- **Default:** `./archive` (created next to `main.py`)
- **Or set via env var:** `STARBOARD_ARCHIVE_DIR`

**Within the archive:**

archive/
├─ gallery/
│  ├─ gallery_metadata.csv
│  ├─ <gallery_id>/
│  │  ├─ 01_15_25/              # encounter folder (MM_DD_YY[_...])
│  │  │  └─ *.jpg|*.png|...
│  │  └─ 03_02_25_A/            # additional encounters allowed
│
├─ queries/
│  ├─ queries_metadata.csv
│  ├─ <query_id>/
│  │  ├─ 12_05_24/
│  │  │  └─ *.jpg|*.png|...
│  │  └─ _second_order_labels.csv   # created when labeling pairs
│
├─ logs/
│  └─ first_order_<query>_<timestamp>.csv   # if STARBOARD_DUMP_RANK_CSV is set
│
├─ reports/
│  └─ past_matches_master.csv              # export from Past Matches tab
│
└─ starboard.log                           # rotating app log


markdown
Always show details

Copy code

### Encounter folders

Image folders must be named like `MM_DD_YY` with an optional suffix (e.g., `01_15_25`, `01_15_25_A`). The app validates this format.

### Metadata CSVs

- **Gallery:** `archive/gallery/gallery_metadata.csv`
- **Queries:** `archive/queries/queries_metadata.csv`

Use the exact header names expected by the app (case & spacing matter).  
The most relevant fields used by the ranking engine are:

- **Numeric:** `diameter_cm`, `volume_ml`, `num_apparent_arms`, `num_arms`
- **Categorical:** `sex`, `disk color`, `arm color`
- **Codes (set/jaccard):** `short_arm_codes`
- **Text/Location (embedded):** `Last location`, `stripe_descriptions`, `reticulation_descriptions`, `rosette_descriptions`, `madreporite_descriptions`, `Other_descriptions`

Each CSV must include the **ID column**:

- `gallery_metadata.csv` → `gallery_id`
- `queries_metadata.csv` → `query_id`

**Minimal example (Gallery):**

gallery_id,Last location,sex,num_arms,short_arm_codes
G001,North Reef,f,5,AB;CD
G002,South Bay,m,5,EF

java
Always show details

Copy code

**Minimal example (Queries):**

query_id,Last location,sex,num_arms,short_arm_codes
Q001,North Reef,f,5,AB;CD

yaml
Always show details

Copy code

*Tip:* Save CSVs as UTF‑8 (BOM is fine). The app will create CSVs with headers if they don’t exist yet.

---


## Running the app

From the repo root:

    python main.py

The log file is at:

    archive/starboard.log

The first time you **rank**, the app builds an index and caches metadata embeddings to:

archive/gallery/metadata_embeddings.json
archive/queries/metadata_embeddings.json

pgsql
Always show details

Copy code

---

## Using the app (brief tour)

### Setup tab
- Create/verify your archive location.
- Single or batch ingest: copy/move images into the correct per‑ID encounter folders.
- The app enforces encounter folder names (`MM_DD_YY[_suffix]`).

### First‑order tab
- Choose a Query and tune which fields to include (numeric / categorical / codes / text).
- Click **Rebuild index** after changing metadata or models; otherwise it rebuilds as needed.
- Adjust weights or numeric offsets; request top‑K results (default: 50).
- (Optional) Set `STARBOARD_DUMP_RANK_CSV=1` to emit a CSV of the top‑K to `archive/logs/`.

### Second‑order tab
- Inspect recommended pairs for a chosen Query.
- Label each `(query, gallery)` as **yes / maybe / no** and add notes.
- Labels are stored per query at `queries/<query_id>/_second_order_labels.csv`.

### Past Matches
- Build an export that joins your latest decisions with metadata.
- **Output:** `archive/reports/past_matches_master.csv` (Excel‑friendly UTF‑8 with BOM).

### Silencing Queries
You can hide a query from interactive tabs without deleting it. The app writes a
`queries/<query_id>/_SILENT.flag` sidecar; remove it (or use the app) to un‑silence.

---

## Configuration (environment variables)

| Name                    | What it controls                                         | Default     |
|-------------------------|-----------------------------------------------------------|-------------|
| `STARBOARD_ARCHIVE_DIR` | Where all data, logs, embeddings, and reports go         | `./archive` |
| `STARBOARD_EMBED_MODEL` | HF model ID for text embeddings                           | `BAAI/bge-m3` |
| `STARBOARD_LOG_LEVEL`   | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)   | `INFO`      |
| `STARBOARD_SESSION_ID`  | Tag included in log lines (auto‑generated if not set)     | `auto`      |
| `STARBOARD_DUMP_RANK_CSV` | If set (any value), saves top‑K ranking CSVs under `archive/logs/` | unset (off) |

**Examples**

**Bash / zsh:**

export STARBOARD_ARCHIVE_DIR="$HOME/data/starboard-archive"
export STARBOARD_EMBED_MODEL="BAAI/bge-small-en"
export STARBOARD_LOG_LEVEL="DEBUG"
export STARBOARD_DUMP_RANK_CSV=1
python main.py

makefile
Always show details

Copy code

**PowerShell:**

$env:STARBOARD_ARCHIVE_DIR="D:\starboard\archive"
$env:STARBOARD_EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
$env:STARBOARD_LOG_LEVEL="DEBUG"
$env:STARBOARD_DUMP_RANK_CSV="1"
python .\main.py

yaml
Always show details

Copy code

---


## Troubleshooting

**“Embedding backend unavailable …”**  
Install `safetensors` (`conda install -c conda-forge safetensors`) or use a recent PyTorch (>= 2.6).

**Blank plots or no charts**  
Install `matplotlib` (optional feature).

**Nothing shows up under IDs**  
Check your archive path and that images are under `gallery/<id>/<MM_DD_YY...>/` (same for `queries/`).

**Ranking looks wrong**  
Click **Rebuild index**, confirm metadata headers match exactly, and verify the chosen fields/weights.

**Conda cannot find a package**  
Try the `conda-forge` channel or fall back to `pip install` inside the activated env.

---


## Development notes

The app runs directly from source; `main.py` adds the project root to `sys.path` so `src/` is importable.

**Key modules:**

- `src/search/engine.py`: scoring logic (numeric, categorical, set‑Jaccard, text embeddings).
- `src/search/embed_store.py`: caching & storage of metadata embeddings.
- `src/data/*`: I/O utilities, path helpers, ingest, CSV helpers.
- `src/ui/*`: Qt widgets and tabs.

Logs rotate automatically and live at `archive/starboard.log`.

---


## License

Specify your project’s license here (e.g., MIT, Apache‑2.0).
"""

path = "/mnt/data/starBoard_README_conda.md"
with open(path, "w", encoding="utf-8") as f:
    f.write(content)

path
Result
'/mnt/data/starBoard_README_conda.md'
I rewrote your README to use a conda-based environment and packaged it as a single, copy‑paste safe Markdown file.

Download: starBoard_README_conda.md

For convenience, here’s the exact content as one continuous, copy‑friendly block (all code sections are indented so Markdown renders correctly when you paste it into GitHub):

markdown
Copy code
# starBoard

A lightweight, Qt‑based desktop app for organizing a Gallery of IDs, loading Queries, and ranking likely matches using a mix of metadata (numeric, categorical, set codes) plus text embeddings. It also lets you label pairs (“yes/maybe/no”) and export past matches.

---

## Core ideas

- Keep your data in an `archive/` folder (or any folder you choose).
- Each **Gallery** / **Query** ID has an **image folder**.
- Metadata lives in two CSVs: `gallery_metadata.csv` and `queries_metadata.csv`.
- Ranking uses field‑specific scorers + **sentence‑transformer** embeddings (default model: `BAAI/bge-m3`).

---

## Quickstart (conda)

> Works with conda or mamba. Replace `conda` with `mamba` if you prefer.

1) **Create and activate an environment (Python 3.10+)**

    conda create -n starboard -y python=3.11
    conda activate starboard

2) **Install runtime dependencies** (pick ONE of the following patterns)

   **All‑pip inside conda (simple and reliable):**

    pip install -U PySide6 sentence-transformers numpy pillow

   **Or conda‑first + pip (reduces build time on some OSes):**

    conda install -y -c conda-forge pyside6 numpy pillow
    pip install -U sentence-transformers

3) **(Optional but recommended for embeddings)**

   Fast, safe model weights:

    conda install -y -c conda-forge safetensors

   *Alternative:* a recent PyTorch (>= 2.6). CPU is fine; GPU is optional.

   **GPU (optional):** install CUDA‑enabled PyTorch that matches your system:

    conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

4) **(Optional) Choose a data location** (defaults to `./archive` next to `main.py`)

   **Bash / zsh**

    export STARBOARD_ARCHIVE_DIR="/path/to/archive"

   **Windows PowerShell**

    $env:STARBOARD_ARCHIVE_DIR="D:\archive"

5) **Run the app**

    python main.py

> If the embedding backend complains, install **safetensors** (recommended) or upgrade to a recent **PyTorch (>= 2.6)**.

---

## Installation (details)

### Requirements

- **Python 3.10+** (uses modern type syntax like `float | None`)
- **OS:** Windows / macOS / Linux

### Packages

- **Required:** `PySide6`, `sentence-transformers`, `numpy`, `pillow`
- **Optional (recommended):** `safetensors` (or recent `torch`) for loading embedding models
- **Optional (for in‑app plots):** `matplotlib` (the app runs without it)

### One‑liners

**Everything via pip (inside conda env):**

    pip install -U PySide6 sentence-transformers numpy pillow
    pip install -U safetensors        # optional but recommended for embeddings
    pip install -U matplotlib         # optional (plots)

**Conda‑first (conda‑forge) + pip:**

    conda install -y -c conda-forge pyside6 numpy pillow matplotlib
    conda install -y -c conda-forge safetensors
    pip install -U sentence-transformers

**GPU note:** If you want GPU acceleration for embeddings, install a CUDA‑enabled PyTorch that matches your system:

    conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

This is optional—CPU works, just slower.

---

## Folder layout & data model

`starBoard` reads and writes under a single **archive root**.

- **Default:** `./archive` (created next to `main.py`)
- **Or set via env var:** `STARBOARD_ARCHIVE_DIR`

**Within the archive:**

archive/
gallery/
gallery_metadata.csv
<gallery_id>/
01_15_25/ # encounter folder (MM_DD_YY[...])
.jpg|.png|...
03_02_25_A/ # more encounters allowed
queries/
queries_metadata.csv
<query_id>/
12_05_24/
.jpg|.png|...
<query_id>/second_order_labels.csv # created by the app when you label pairs
logs/
first_order<query><timestamp>.csv # only if STARBOARD_DUMP_RANK_CSV is set
reports/
past_matches_master.csv # created by the Past Matches export
starboard.log # rotating app log

markdown
Copy code

### Encounter folders

Image folders must be named like `MM_DD_YY` with an optional suffix (e.g., `01_15_25`, `01_15_25_A`). The app validates this format.

### Metadata CSVs

- **Gallery:** `archive/gallery/gallery_metadata.csv`
- **Queries:** `archive/queries/queries_metadata.csv`

Use the exact header names expected by the app (case & spacing matter).  
The most relevant fields used by the ranking engine are:

- **Numeric:** `diameter_cm`, `volume_ml`, `num_apparent_arms`, `num_arms`
- **Categorical:** `sex`, `disk color`, `arm color`
- **Codes (set/jaccard):** `short_arm_codes`
- **Text/Location (embedded):** `Last location`, `stripe_descriptions`, `reticulation_descriptions`, `rosette_descriptions`, `madreporite_descriptions`, `Other_descriptions`

Each CSV must include the **ID column**:

- `gallery_metadata.csv` → `gallery_id`
- `queries_metadata.csv` → `query_id`

**Minimal example (Gallery):**

gallery_id,Last location,sex,num_arms,short_arm_codes
G001,North Reef,f,5,AB;CD
G002,South Bay,m,5,EF

java
Copy code

**Minimal example (Queries):**

query_id,Last location,sex,num_arms,short_arm_codes
Q001,North Reef,f,5,AB;CD

yaml
Copy code

*Tip:* Save CSVs as UTF‑8 (BOM is fine). The app will create CSVs with headers if they don’t exist yet.

---

## Running the app

From the repo root:

    python main.py

The log file is at:

    archive/starboard.log

The first time you **rank**, the app builds an index and caches metadata embeddings to:

archive/gallery/metadata_embeddings.json
archive/queries/metadata_embeddings.json

pgsql
Copy code

---

## Using the app (brief tour)

### Setup tab
- Create/verify your archive location.
- Single or batch ingest: copy/move images into the correct per‑ID encounter folders.
- The app enforces encounter folder names (`MM_DD_YY[_suffix]`).

### First‑order tab
- Choose a Query and tune which fields to include (numeric / categorical / codes / text).
- Click **Rebuild index** after changing metadata or models; otherwise it rebuilds as needed.
- Adjust weights or numeric offsets; request top‑K results (default: 50).
- (Optional) Set `STARBOARD_DUMP_RANK_CSV=1` to emit a CSV of the top‑K to `archive/logs/`.

### Second‑order tab
- Inspect recommended pairs for a chosen Query.
- Label each `(query, gallery)` as **yes / maybe / no** and add notes.
- Labels are stored per query at `queries/<query_id>/_second_order_labels.csv`.

### Past Matches
- Build an export that joins your latest decisions with metadata.
- **Output:** `archive/reports/past_matches_master.csv` (Excel‑friendly UTF‑8 with BOM).

### Silencing Queries
You can hide a query from interactive tabs without deleting it. The app writes a
`queries/<query_id>/_SILENT.flag` sidecar; remove it (or use the app) to un‑silence.

---

## Configuration (environment variables)

| Name                    | What it controls                                         | Default     |
|-------------------------|-----------------------------------------------------------|-------------|
| `STARBOARD_ARCHIVE_DIR` | Where all data, logs, embeddings, and reports go         | `./archive` |
| `STARBOARD_EMBED_MODEL` | HF model ID for text embeddings                           | `BAAI/bge-m3` |
| `STARBOARD_LOG_LEVEL`   | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)   | `INFO`      |
| `STARBOARD_SESSION_ID`  | Tag included in log lines (auto‑generated if not set)     | `auto`      |
| `STARBOARD_DUMP_RANK_CSV` | If set (any value), saves top‑K ranking CSVs under `archive/logs/` | unset (off) |

**Examples**

**Bash / zsh:**

export STARBOARD_ARCHIVE_DIR="$HOME/data/starboard-archive"
export STARBOARD_EMBED_MODEL="BAAI/bge-small-en"
export STARBOARD_LOG_LEVEL="DEBUG"
export STARBOARD_DUMP_RANK_CSV=1
python main.py

makefile
Copy code

**PowerShell:**

$env:STARBOARD_ARCHIVE_DIR="D:\starboard\archive"
$env:STARBOARD_EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
$env:STARBOARD_LOG_LEVEL="DEBUG"
$env:STARBOARD_DUMP_RANK_CSV="1"
python .\main.py

yaml
Copy code

---

## Troubleshooting

**“Embedding backend unavailable …”**  
Install `safetensors` (`conda install -c conda-forge safetensors`) or use a recent PyTorch (>= 2.6).

**Blank plots or no charts**  
Install `matplotlib` (optional feature).

**Nothing shows up under IDs**  
Check your archive path and that images are under `gallery/<id>/<MM_DD_YY...>/` (same for `queries/`).

**Ranking looks wrong**  
Click **Rebuild index**, confirm metadata headers match exactly, and verify the chosen fields/weights.

**Conda cannot find a package**  
Try the `conda-forge` channel or fall back to `pip install` inside the activated env.

---

## Development notes

The app runs directly from source; `main.py` adds the project root to `sys.path` so `src/` is importable.

**Key modules:**

- `src/search/engine.py`: scoring logic (numeric, categorical, set‑Jaccard, text embeddings).
- `src/search/embed_store.py`: caching & storage of metadata embeddings.
- `src/data/*`: I/O utilities, path helpers, ingest, CSV helpers.
- `src/ui/*`: Qt widgets and tabs.

Logs rotate automatically and live at `archive/starboard.log`.

---

## License

Specify your project’s license here (e.g., MIT, Apache‑2.0).
