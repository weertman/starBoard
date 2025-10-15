# First‑order Search (metadata ranking)

This package powers the **First‑order** tab: given a selected **Query ID**, it ranks **Gallery IDs** by metadata similarity and shows a fast visual “line‑up” so you can decide which candidates to compare in detail next.

**Key behavior**

- You choose the **fields to include** (checkboxes).  
- The rank score is the **average of the included fields that are present** for both query and candidate (equal weights by default).  
- Missing fields never penalize: they’re simply excluded from the denominator.  
- Each candidate shows a **per‑field breakdown** and how many fields actually contributed.

The UI for this lives in `src/ui/tab_first_order.py`, which calls the engine in this folder. :contentReference[oaicite:0]{index=0}

---

## What lives here

search/
engine.py # FirstOrderSearchEngine façade: loads data, computes ranks
interfaces.py # FieldScorer protocol
fields_numeric.py # NumericGaussianScorer (robust med/MAD + exp decay)
fields_categorical.py# CategoricalMatchScorer (exact match)
fields_set_jaccard.py# SetJaccardScorer (token sets)
fields_text_ngrams.py# TextNgramScorer (char 3–5 gram Jaccard)

markdown
Copy code

- **`interfaces.py`** defines the `FieldScorer` protocol that all scorers implement:
  `build_gallery(...)`, `prepare_query(...)`, `has_query_signal(...)`, and  
  `score_pair(q_state, gallery_id) -> (score∈[0,1], present_mask:bool)`.  
  The `present_mask` is `True` only when both the query and candidate have that field populated. :contentReference[oaicite:1]{index=1}

- **`engine.py`** loads “latest rows per ID” for Gallery and Queries, constructs per‑field scorers, and returns a sorted list of `RankItem(gallery_id, score, field_breakdown, k_contrib)`. The UI uses this to build the line‑up. :contentReference[oaicite:2]{index=2}

- **Scorers**:
  - `NumericGaussianScorer` (e.g., `diameter_cm`, `volume_ml`, `num_*arms`):  
    uses robust **median/MAD** across the gallery and  
    \( s = \exp\!\big(-|x-q|/(k\cdot \text{MAD})\big) \) with an epsilon when MAD=0. :contentReference[oaicite:3]{index=3}
  - `CategoricalMatchScorer` (e.g., `sex`): exact match = 1.0, else 0.0 (contributes only if both present). :contentReference[oaicite:4]{index=4}
  - `SetJaccardScorer` (e.g., `short_arm_codes`): tokenizes the field to an **uppercase alphanumeric set** and computes Jaccard similarity. :contentReference[oaicite:5]{index=5}
  - `TextNgramScorer` (e.g., `stripe_descriptions`, `Last location`, etc.):  
    **character 3–5‑gram Jaccard** on normalized text; tolerant to typos and short descriptors. :contentReference[oaicite:6]{index=6}

---

## Data sources & normalization

- **Archive layout** and canonical **CSV headers** for Gallery/Queries are declared in `src/data/archive_paths.py`. Queries support both **`queries`** (new) and the legacy **`querries`** folder; readers check both. :contentReference[oaicite:7]{index=7}

- **CSV ingestion** is **append‑only**. We reduce to the **latest row per ID** via `last_row_per_id(...)`, which also **ignores “pure‑ID” rows** (rows that fill only the ID and no payload) so an accidental save doesn’t wipe your data. Normalization utilities also accept legacy ID header spellings (e.g., `querries_id`). See `src/data/csv_io.py`. :contentReference[oaicite:8]{index=8}

- The engine uses those helpers to build `{id -> latest_row}` maps for Gallery and Queries before ranking. :contentReference[oaicite:9]{index=9}

---

## Ranking pipeline (how scores are computed)

1. **Pick a Query ID** in the UI.  
   The engine fetches its latest metadata row.

2. **Field selection** (UI → engine):  
   - `include_fields`: the set you checked; default is “all”.  
   - For each included field, we `prepare_query(...)`. If the query has no signal for that field, it’s dropped from this run (“active fields”).

3. **Per‑field scoring** for every candidate Gallery ID:  
   - Each scorer returns `(score, present_mask)`.  
   - Only fields with `present_mask=True` (query **and** candidate both non‑empty) contribute.

4. **Fusion (default = average across included + present):**
   \[
   \text{Score}(x)=\frac{\sum_{f\in\mathcal{F}_\text{ui}} m_f(q,x)\, s_f(x)}
                        {\sum_{f\in\mathcal{F}_\text{ui}} m_f(q,x)}
   \]
   where \(m_f\) is the present mask and \(s_f\) the per‑field similarity.  
   The engine also reports **`k_contrib`** (how many fields actually contributed) and a `field_breakdown` dict for each candidate. :contentReference[oaicite:10]{index=10}

5. **Sort & return Top‑K** to the UI.  
   (Candidates with no contributing fields fall to the bottom since they score 0.)

---

## UI wiring (where to click)

- The **First‑order** tab (`src/ui/tab_first_order.py`) provides:
  - **Query selector**
  - **Presets** (e.g., “Average (All)”, “Size only”, “Text only”)
  - **Field checkboxes** (what to include in the average)
  - **Top‑K** control and **Refresh**
  - A left **Query panel** + right **candidate line‑up** with mini‑viewers, per‑field badges, score, and “📌 Pin for Compare”. :contentReference[oaicite:11]{index=11}

- The mini‑viewers use:
  - `src/ui/image_strip.py` → scaled decoding with `QImageReader.setScaledSize(...)` and a small in‑memory LRU cache for **fast 62 MP image previews** (zoom/pan/rotate supported). :contentReference[oaicite:12]{index=12}
  - `src/data/image_index.py` → lists images for an ID deterministically across archive roots. :contentReference[oaicite:13]{index=13}

---

## Logging & diagnostics

- The app initializes logging to `<archive_root>/starboard.log` using a rotating file handler at **INFO** by default (`main.py`). :contentReference[oaicite:14]{index=14}

- With the current first‑order implementation, the engine logs:
  - index coverage per field, numeric med/MAD, and
  - for each rank: selected/active fields, a `k_contrib` histogram, and a compact **Top‑N with per‑field contributions**.  
  UI actions (query change, preset/field toggles, rebuild, refresh, pin) are also logged.

- Optional environment toggles (if present in your build):
  ```bash
  # Increase verbosity temporarily
  STARBOARD_LOG_LEVEL=DEBUG

  # Dump each ranking to CSV under <archive_root>/logs/
  STARBOARD_DUMP_RANK_CSV=1

  # Provide a custom session id in logs (handy for multi-run analysis)
  STARBOARD_SESSION_ID=my-session-001
Tip: grep for rank_top and k_contrib in the log to quickly assess sort quality for a session.

Programmatic usage
You can use the engine directly (outside the UI) for experiments or tests:

python
Copy code
from src.search.engine import FirstOrderSearchEngine

eng = FirstOrderSearchEngine()
eng.rebuild()

fields = {"num_arms", "short_arm_codes", "stripe_descriptions"}  # choose any subset
results = eng.rank(
    query_id="Q_2024_017",
    include_fields=fields,
    equalize_weights=True,   # average across the included fields
    top_k=20,
)

for r in results:
    print(r.gallery_id, f"{r.score:.3f}", r.k_contrib, r.field_breakdown)
See engine.py for the RankItem structure and the full list of supported fields. python_files

Extending the system
Add a new metadata scorer
Implement FieldScorer in a new file (e.g., fields_custom.py):

python
Copy code
from src.search.interfaces import FieldScorer

class MyFieldScorer:
    name = "my_field"

    def build_gallery(self, gallery_rows_by_id): ...
    def prepare_query(self, q_row): ...
    def has_query_signal(self, q_state) -> bool: ...
    def score_pair(self, q_state, gallery_id) -> tuple[float, bool]: ...
Register it in FirstOrderSearchEngine.__init__ (alongside the others).
It will automatically appear in the field list used for fusion. python_files

Add image embeddings later (recommended path)
Treat the embedding as just another FieldScorer (e.g., ImageEmbeddingScorer(name="clip_v1")) that returns a normalized similarity in [0,1]. The UI and fusion logic remain unchanged.

Field details (current set)
Numeric: diameter_cm, volume_ml, num_apparent_arms, num_arms
Robust median/MAD scaling, epsilon fallback when MAD=0. python_files

Categorical: sex (exact match). python_files

Codes (set): short_arm_codes (uppercase token set; Jaccard). python_files

Text/Location: Last location, stripe_descriptions, reticulation_descriptions,
rosette_descriptions, madreporite_descriptions, Other_descriptions (char 3–5 n‑gram Jaccard). python_files

CSV headers are defined centrally to keep Gallery and Queries aligned. python_files

Performance notes
The default text scorer is dependency‑free and fast for thousands of IDs.

For larger galleries, consider:

swapping char‑Jaccard for TF‑IDF cosine with hashing, or

adding an ANN index (e.g., HNSW) per text field behind the FieldScorer interface.
The engine/API won’t need to change for either approach. python_files

Known edge cases
No included fields → ranking is disabled (UI shows empty line‑up).

Query has none of the included fields → engine returns nothing; enable more fields. python_files

Numeric field with zero MAD → epsilon fallback makes equal values ≈1.0; non‑equal decay sharply. python_files

Legacy Queries folder → both queries and querries are read; the new format “wins” on collisions. python_files

Where it is used in the UI
src/ui/tab_first_order.py → constructs the controls, calls engine.rank(...), renders the line‑up.

src/ui/image_strip.py and src/data/image_index.py → the fast image strip (scaled decode + LRU) and deterministic image listing. python_files

Troubleshooting
“Results look noisy.” Try adding more fields (e.g., include one or more text descriptors plus a size or code field). Watch k_contrib in the log to ensure multiple fields are contributing.

“Large images stutter.” The strip uses scaled decode; if your workstation is older, reduce the long_edge parameter in ImageStrip. python_files

“I don’t see my query/gallery IDs.” Confirm folder names and that a metadata row exists with the correct ID header; the reader tolerates legacy headers, but IDs must not be empty after normalization. 