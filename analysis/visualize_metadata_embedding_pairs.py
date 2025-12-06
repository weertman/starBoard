#!/usr/bin/env python
"""
analysis/visualize_metadata_pair_embeddings.py

Visualize Query–Gallery positive pairs in a 2D projection of the
metadata embedding space.

- Each gallery ID is a blue dot, labeled with its name.
- Each query ID is an orange "x".
- For every positive (query, gallery) pair (verdict == "yes"),
  draw a light line from the gallery point to the query point.

Embeddings:
    - Text: BGE metadata embeddings (via src.search.embed_store),
      aggregated across selected TEXT_FIELDS.
    - Numeric: selected NUMERIC_FIELDS (e.g. num_apparent_arms),
      standardized and appended to the text embedding vector.

Dimensionality reduction:
    - PHATE, UMAP, t‑SNE, MDS, PCA (configurable).

Run from project root:

    python analysis/visualize_metadata_pair_embeddings.py

There are **no CLI arguments**. Edit the configuration section below.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Set, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Make "src" importable when running as a standalone script
# ----------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# starBoard imports
from src.data.csv_io import normalize_id_value
from src.data.archive_paths import archive_root
from src.data.past_matches import (
    _latest_label_rows_all_pairs,
    _load_latest_metadata_maps,
)
from src.search.engine import TEXT_FIELDS, NUMERIC_FIELDS
from src.search.embed_store import (
    ensure_metadata_embeddings,
    load_vectors_for_field,
    embedding_backend_ready,
    DEFAULT_MODEL_ID,
)

log = logging.getLogger("starBoard.analysis.metadata_pair_embeddings")


# ----------------------------------------------------------------------
# User configuration (EDIT THESE)
# ----------------------------------------------------------------------

# Which dimensionality‑reduction methods to run.
# Any subset of {"phate", "umap", "tsne", "mds", "pca"}.
PROJECTION_METHODS: List[str] = ["tsne"]  # e.g. ["phate", "umap", "tsne", "mds", "pca"]

# Fields you NEVER want to use in the embedding.
# (Names must match canonical headers from src.search.engine.)
# Default: drop lab‑only numeric fields and "Last location".
EXCLUDED_FIELDS: Set[str] = {
    "Last location",
    "diameter_cm",
    "volume_ml",
    "num_arms",
    "sex",
}

# Fields that should be kept EVEN IF they appear in EXCLUDED_FIELDS.
# Default: keep num_apparent_arms as the only numeric.
ALWAYS_INCLUDE_FIELDS: Set[str] = {
    "num_apparent_arms",
}

# If True: only plot pairs where BOTH the query and gallery have
# non‑empty values for *all* selected fields (text + numeric).
# If False: also include pairs with missing fields; missing numeric
# values are filled with the global mean (0 after standardization).
REQUIRE_ALL_FIELDS_FOR_PAIR: bool = True

# Which verdicts from second‑order labels count as "positive" links.
# Usually just {"yes"}.
INCLUDE_VERDICTS: Set[str] = {"yes"}

# Random seed for reproducible projections
RANDOM_STATE: int = 42

# Optional throttle for debugging: set to e.g. 50 to use at most 50 queries.
MAX_QUERIES: Optional[int] = None  # or an int

# Subdirectory under analysis/ for output PNGs
OUTPUT_SUBDIR_NAME: str = "visualize_metadata_pair_embeddings"


# ----------------------------------------------------------------------
# Helper: select fields from config
# ----------------------------------------------------------------------

def select_fields_from_config() -> Tuple[List[str], List[str]]:
    """
    Decide which TEXT_FIELDS and NUMERIC_FIELDS to use based on
    EXCLUDED_FIELDS and ALWAYS_INCLUDE_FIELDS.
    """
    excluded = set(EXCLUDED_FIELDS)
    forced = set(ALWAYS_INCLUDE_FIELDS)

    text_fields = [f for f in TEXT_FIELDS if (f not in excluded) or (f in forced)]
    numeric_fields = [f for f in NUMERIC_FIELDS if (f not in excluded) or (f in forced)]

    if not text_fields:
        raise RuntimeError(
            "No text fields selected. At least one TEXT field is required.\n"
            f"Current TEXT_FIELDS: {TEXT_FIELDS}\n"
            f"EXCLUDED_FIELDS: {sorted(excluded)}\n"
            f"ALWAYS_INCLUDE_FIELDS: {sorted(forced)}"
        )

    return text_fields, numeric_fields


# ----------------------------------------------------------------------
# Text embedding aggregation (copied/adapted from identity_metrics script)
# ----------------------------------------------------------------------

def build_text_embeddings_for_target(
    target: str,
    fields: Sequence[str],
    model_id: str = DEFAULT_MODEL_ID,
) -> Tuple[List[str], np.ndarray]:
    """
    Aggregate per‑field text embeddings into one vector per ID by
    averaging across all available `fields`.

    Parameters
    ----------
    target : "Gallery" or "Queries"
    fields : sequence of text fields (subset of TEXT_FIELDS)
    model_id : embedding model ID (BGE)

    Returns
    -------
    item_ids : list of normalized IDs
    X_raw : np.ndarray, shape (N, D)
        Aggregated embedding per ID.
    """
    from src.data.csv_io import normalize_id_value as _norm  # local alias

    per_id_vectors: Dict[str, List[np.ndarray]] = {}
    total_vectors = 0

    for field in fields:
        try:
            vecs = load_vectors_for_field(target, field, expected_model_id=model_id)
        except FileNotFoundError:
            vecs = {}
        except Exception as e:
            log.warning("Failed to load embeddings for %s/%s: %s", target, field, e)
            vecs = {}

        if not vecs:
            continue

        for raw_id, v in vecs.items():
            _id = _norm(raw_id)
            if not _id:
                continue
            per_id_vectors.setdefault(_id, []).append(np.asarray(v, dtype=np.float32))
            total_vectors += 1

    if not per_id_vectors:
        raise RuntimeError(
            f"No metadata embeddings found for target={target}. "
            "Make sure text fields are populated and embeddings have been built."
        )

    item_ids: List[str] = []
    emb_list: List[np.ndarray] = []

    for _id in sorted(per_id_vectors.keys()):
        vs = per_id_vectors[_id]
        stacked = np.stack(vs, axis=0)
        mean_vec = stacked.mean(axis=0).astype(np.float32)
        item_ids.append(_id)
        emb_list.append(mean_vec)

    X_raw = np.stack(emb_list, axis=0)

    log.info(
        "Aggregated %d embeddings for target=%s from %d field‑vectors "
        "across %d text fields (dim=%d).",
        len(item_ids),
        target,
        total_vectors,
        len(fields),
        X_raw.shape[1],
    )

    return item_ids, X_raw


# ----------------------------------------------------------------------
# Numeric feature helpers
# ----------------------------------------------------------------------

def _collect_numeric_raw(
    rows_by_id: Dict[str, Dict[str, str]],
    numeric_fields: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    Extract raw numeric vectors (possibly containing NaNs) per ID.
    """
    out: Dict[str, np.ndarray] = {}
    if not numeric_fields:
        return out

    for _id, row in rows_by_id.items():
        vals: List[float] = []
        for f in numeric_fields:
            raw = row.get(f, "")
            s = "" if raw is None else str(raw).strip()
            if not s:
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(s))
                except Exception:
                    vals.append(np.nan)
        out[_id] = np.asarray(vals, dtype=np.float32)

    return out


def _standardize_numeric_maps(
    num_q_raw: Dict[str, np.ndarray],
    num_g_raw: Dict[str, np.ndarray],
    numeric_fields: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Standardize numeric features across Queries+Gallery jointly:
        x' = (x - mean) / std
    Missing values stay at 0 (mean after z‑scoring).
    """
    if not numeric_fields:
        return {}, {}

    all_vecs: List[np.ndarray] = list(num_q_raw.values()) + list(num_g_raw.values())
    if not all_vecs:
        return {}, {}

    stacked = np.stack(all_vecs, axis=0)

    with np.errstate(invalid="ignore"):
        means = np.nanmean(stacked, axis=0)
        stds = np.nanstd(stacked, axis=0)

    means = np.where(np.isfinite(means), means, 0.0)
    stds = np.where((stds > 0) & np.isfinite(stds), stds, 1.0)

    def _scale(raw_map: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for _id, vec in raw_map.items():
            arr = (vec - means) / stds
            arr = np.where(np.isfinite(arr), arr, 0.0)
            out[_id] = arr.astype(np.float32)
        return out

    return _scale(num_q_raw), _scale(num_g_raw)


def compute_field_presence(
    rows_by_id: Dict[str, Dict[str, str]],
    text_fields: Sequence[str],
    numeric_fields: Sequence[str],
) -> Dict[str, Set[str]]:
    """
    For each ID, record which of the requested fields are non‑empty.
    """
    present: Dict[str, Set[str]] = {}
    for _id, row in rows_by_id.items():
        s: Set[str] = set()
        for f in text_fields:
            val = row.get(f, "")
            if str(val or "").strip():
                s.add(f)
        for f in numeric_fields:
            raw = row.get(f, "")
            t = "" if raw is None else str(raw).strip()
            if not t:
                continue
            try:
                float(t)
                s.add(f)
            except Exception:
                continue
        present[_id] = s
    return present


def has_all_fields(
    presence_map: Dict[str, Set[str]],
    _id: str,
    required_fields: Sequence[str],
) -> bool:
    fields = presence_map.get(_id)
    if not fields:
        return False
    for f in required_fields:
        if f not in fields:
            return False
    return True


# ----------------------------------------------------------------------
# Positive pairs from second‑order labels
# ----------------------------------------------------------------------

def load_positive_pairs(
    include_verdicts: Set[str],
) -> List[Tuple[str, str]]:
    """
    Load (query_id, gallery_id) pairs with verdict in include_verdicts.
    """
    include = {v.strip().lower() for v in include_verdicts}
    pairs: List[Tuple[str, str]] = []

    for r in _latest_label_rows_all_pairs():
        qid = normalize_id_value(r.get("query_id", ""))
        gid = normalize_id_value(r.get("gallery_id", ""))
        if not qid or not gid:
            continue
        v = (r.get("verdict", "") or "").strip().lower()
        if v not in include:
            continue
        pairs.append((qid, gid))

    # Deduplicate while preserving order
    seen: Set[Tuple[str, str]] = set()
    uniq: List[Tuple[str, str]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


# ----------------------------------------------------------------------
# Dimensionality reduction
# ----------------------------------------------------------------------

def project_embeddings(
    X: np.ndarray,
    method: str,
    random_state: int = 42,
) -> np.ndarray:
    """
    Project high‑dim embeddings X (N, D) → (N, 2) using the requested method.
    """
    method = method.lower()
    n_samples = X.shape[0]

    if n_samples < 2:
        raise ValueError(f"Need at least 2 points for projection (got {n_samples}).")

    if method == "tsne":
        from sklearn.manifold import TSNE

        # Keep perplexity valid: < n_samples, not too tiny
        perplexity = max(5.0, min(30.0, (n_samples - 1) / 3.0))
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        )
        return tsne.fit_transform(X)

    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=random_state)
        return pca.fit_transform(X)

    if method == "mds":
        from sklearn.manifold import MDS

        mds = MDS(
            n_components=2,
            random_state=random_state,
            dissimilarity="euclidean",
            n_init=4,
            max_iter=300,
        )
        return mds.fit_transform(X)

    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "UMAP is not installed. Install with `pip install umap-learn`."
            ) from e

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        return reducer.fit_transform(X)

    if method == "phate":
        try:
            import phate  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "PHATE is not installed. Install with `pip install phate`."
            ) from e

        reducer = phate.PHATE(n_components=2, random_state=random_state)
        return reducer.fit_transform(X)

    raise ValueError(
        f"Unknown projection method '{method}'. "
        "Use one of: 'phate', 'umap', 'tsne', 'mds', 'pca'."
    )


# ----------------------------------------------------------------------
# Data prep: build embeddings + filtered pairs
# ----------------------------------------------------------------------

def prepare_embeddings_and_pairs(
    text_fields: Sequence[str],
    numeric_fields: Sequence[str],
) -> Tuple[
    List[str], np.ndarray,  # gallery_ids, G
    List[str], np.ndarray,  # query_ids,   Q
    List[Tuple[str, str]],  # valid_pairs (qid, gid)
    List[str],              # all_fields_used (for title)
]:
    # Load latest metadata
    q_by_id, g_by_id = _load_latest_metadata_maps()
    log.info(
        "Loaded metadata rows: %d queries, %d gallery IDs.",
        len(q_by_id),
        len(g_by_id),
    )

    # Ensure BGE embeddings exist for selected text fields
    ok, reason = embedding_backend_ready()
    if not ok:
        raise RuntimeError(
            f"Embedding backend is not available: {reason}. "
            "Cannot build metadata text embeddings."
        )

    ensure_metadata_embeddings("Gallery", g_by_id, fields=text_fields)
    ensure_metadata_embeddings("Queries", q_by_id, fields=text_fields)

    # Aggregate text embeddings
    g_ids_all, G_text = build_text_embeddings_for_target("Gallery", text_fields)
    q_ids_all, Q_text = build_text_embeddings_for_target("Queries", text_fields)

    g_ids_all = list(g_ids_all)
    q_ids_all = list(q_ids_all)

    g_text_by_id = {gid: G_text[i] for i, gid in enumerate(g_ids_all)}
    q_text_by_id = {qid: Q_text[i] for i, qid in enumerate(q_ids_all)}

    # Numeric features (standardized)
    num_q_raw = _collect_numeric_raw(q_by_id, numeric_fields)
    num_g_raw = _collect_numeric_raw(g_by_id, numeric_fields)
    num_q_scaled, num_g_scaled = _standardize_numeric_maps(
        num_q_raw, num_g_raw, numeric_fields
    )

    # Field presence for "require all fields" filter
    presence_q = compute_field_presence(q_by_id, text_fields, numeric_fields)
    presence_g = compute_field_presence(g_by_id, text_fields, numeric_fields)
    all_fields_used: List[str] = list(text_fields) + list(numeric_fields)

    # Helper: combined embedding vector (text + numeric)
    def _combined_vec(
        _id: str,
        text_map: Dict[str, np.ndarray],
        num_map: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        text_vec = text_map.get(_id)
        if text_vec is None:
            return None
        if numeric_fields:
            num_vec = num_map.get(_id)
            if num_vec is None:
                num_vec = np.zeros(len(numeric_fields), dtype=np.float32)
            return np.concatenate([text_vec, num_vec.astype(np.float32)], axis=0)
        return text_vec

    g_vecs_all: Dict[str, np.ndarray] = {}
    q_vecs_all: Dict[str, np.ndarray] = {}

    for gid in g_ids_all:
        v = _combined_vec(gid, g_text_by_id, num_g_scaled)
        if v is not None:
            g_vecs_all[gid] = v

    for qid in q_ids_all:
        v = _combined_vec(qid, q_text_by_id, num_q_scaled)
        if v is not None:
            q_vecs_all[qid] = v

    log.info(
        "Combined embeddings: %d gallery IDs, %d query IDs.",
        len(g_vecs_all),
        len(q_vecs_all),
    )

    # Load positive pairs and filter
    raw_pairs = load_positive_pairs(INCLUDE_VERDICTS)
    log.info("Loaded %d positive pairs (before filtering).", len(raw_pairs))

    valid_pairs: List[Tuple[str, str]] = []
    used_g_ids: Set[str] = set()
    used_q_ids: Set[str] = set()

    for qid, gid in raw_pairs:
        if MAX_QUERIES is not None and len(used_q_ids) >= MAX_QUERIES and qid not in used_q_ids:
            # Optional throttle: stop adding *new* query IDs after limit
            continue

        if qid not in q_vecs_all or gid not in g_vecs_all:
            continue

        if REQUIRE_ALL_FIELDS_FOR_PAIR:
            if not (
                has_all_fields(presence_q, qid, all_fields_used)
                and has_all_fields(presence_g, gid, all_fields_used)
            ):
                continue

        valid_pairs.append((qid, gid))
        used_q_ids.add(qid)
        used_g_ids.add(gid)

    if not valid_pairs:
        raise RuntimeError("No valid pairs after filtering; nothing to visualize.")

    gallery_ids = sorted(used_g_ids)
    query_ids = sorted(used_q_ids)

    G = np.stack([g_vecs_all[gid] for gid in gallery_ids], axis=0)
    Q = np.stack([q_vecs_all[qid] for qid in query_ids], axis=0)

    log.info(
        "Final data for plotting: %d gallery IDs, %d query IDs, %d pairs.",
        len(gallery_ids),
        len(query_ids),
        len(valid_pairs),
    )

    return gallery_ids, G, query_ids, Q, valid_pairs, all_fields_used


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_projection(
    method: str,
    gallery_ids: List[str],
    G: np.ndarray,
    query_ids: List[str],
    Q: np.ndarray,
    pairs: List[Tuple[str, str]],
    fields_used: Sequence[str],
    output_dir: Path,
) -> None:
    """
    Create and save a 2D scatter + lines plot for a given projection method.
    """
    # Build joint embedding then split back
    X_all = np.concatenate([G, Q], axis=0)
    coords_all = project_embeddings(X_all, method=method, random_state=RANDOM_STATE)

    coords_g = coords_all[: len(gallery_ids), :]
    coords_q = coords_all[len(gallery_ids) :, :]

    pos_g: Dict[str, np.ndarray] = {
        gid: coords_g[i] for i, gid in enumerate(gallery_ids)
    }
    pos_q: Dict[str, np.ndarray] = {
        qid: coords_q[i] for i, qid in enumerate(query_ids)
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    # Gallery points
    gx = coords_g[:, 0]
    gy = coords_g[:, 1]
    ax.scatter(gx, gy, marker="o", label="Gallery")

    # Query points
    qx = coords_q[:, 0]
    qy = coords_q[:, 1]
    ax.scatter(qx, qy, marker="x", label="Queries")

    # Lines per positive pair
    for qid, gid in pairs:
        g_pt = pos_g.get(gid)
        q_pt = pos_q.get(qid)
        if g_pt is None or q_pt is None:
            continue
        ax.plot(
            [g_pt[0], q_pt[0]],
            [g_pt[1], q_pt[1]],
            color="0.8",
            linewidth=0.8,
            alpha=0.7,
        )

    # Label each gallery point with its ID, slightly above
    if len(coords_all) > 0:
        y_min = float(coords_all[:, 1].min())
        y_max = float(coords_all[:, 1].max())
        y_range = y_max - y_min if y_max > y_min else 1.0
        offset = 0.02 * y_range
    else:
        offset = 0.1

    for gid, (x, y) in pos_g.items():
        ax.text(
            x,
            y + offset,
            gid,
            fontsize=9,
            ha="center",
            va="bottom",
        )

    fields_str = ", ".join(fields_used) if fields_used else "(no fields)"
    mode_str = (
        "require all fields"
        if REQUIRE_ALL_FIELDS_FOR_PAIR
        else "allow missing fields"
    )

    ax.set_title(
        f"Query–gallery pairs in {method.upper()} space\n"
        f"Fields: {fields_str}  |  {mode_str}",
        fontsize=12,
    )
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    out_path = output_dir / f"pairs_{method.lower()}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    log.info("Saved %s projection to: %s", method.upper(), out_path)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    log.info("Project root: %s", PROJECT_ROOT)
    log.info("Archive root: %s", archive_root())

    text_fields, numeric_fields = select_fields_from_config()
    log.info("Text fields used: %s", ", ".join(text_fields))
    log.info("Numeric fields used: %s", ", ".join(numeric_fields) or "(none)")
    log.info("Require all fields per pair: %s", REQUIRE_ALL_FIELDS_FOR_PAIR)

    gallery_ids, G, query_ids, Q, pairs, fields_used = prepare_embeddings_and_pairs(
        text_fields=text_fields,
        numeric_fields=numeric_fields,
    )

    output_dir = THIS_FILE.parent / OUTPUT_SUBDIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    for method in PROJECTION_METHODS:
        try:
            plot_projection(
                method=method,
                gallery_ids=gallery_ids,
                G=G,
                query_ids=query_ids,
                Q=Q,
                pairs=pairs,
                fields_used=fields_used,
                output_dir=output_dir,
            )
        except Exception as e:
            log.error("Projection '%s' failed: %s", method, e)

    log.info("Done.")


if __name__ == "__main__":
    main()
