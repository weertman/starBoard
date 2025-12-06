#!/usr/bin/env python
"""
analysis/plot_query_gallery_pair_embeddings.py

Visualize query–gallery positive pairs in a low-dimensional embedding space.

What this script does
---------------------
* Uses metadata text embeddings + selected numeric fields to build a single
  embedding vector per ID (Gallery and Queries).
* Allows you to EXCLUDE certain metadata fields entirely.
* By default:
    - uses all TEXT_FIELDS except those you exclude
    - uses ONLY the numeric field 'num_apparent_arms' (others ignored)
* Loads human "second-order" labels and keeps only verdict == "yes" pairs.
* By default (REQUIRE_ALL_FIELDS = False), allows partially filled metadata:
    - each ID just needs:
        - a metadata row
        - text embeddings for the selected text fields
        - valid values for the selected numeric fields (if any)
  If you set REQUIRE_ALL_FIELDS = True, it will instead restrict to IDs with
  non-empty values for ALL chosen fields.
* Projects the combined embedding space (Gallery + Queries) down to 2D using:
    - PCA
    - MDS
    - t-SNE
    - UMAP        (if the umap-learn package is installed)
    - PHATE       (if the phate package is installed)
* For each method, makes a scatter plot:
    - blue circles: Gallery
    - orange crosses: Queries
    - light gray lines: positive query–gallery pairs
    - text labels above each Gallery point (gallery_id)

Outputs
-------
One PNG per method in the same directory as this script:

    pairs_pca.png
    pairs_mds.png
    pairs_tsne.png
    pairs_umap.png      (if available)
    pairs_phate.png     (if available)

Usage
-----
From project root:

    python analysis/plot_query_gallery_pair_embeddings.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Set, Optional

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS

# Optional reducers
try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None

try:
    import phate  # type: ignore
except Exception:  # pragma: no cover
    phate = None

# ----------------------------------------------------------------------
# Project import setup
# ----------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.past_matches import _load_latest_metadata_maps  # type: ignore
from src.data.archive_paths import archive_root  # type: ignore
from src.data.csv_io import normalize_id_value  # type: ignore
from src.search.engine import (  # type: ignore
    TEXT_FIELDS,
    NUMERIC_FIELDS,
)
from src.search.embed_store import (  # type: ignore
    ensure_metadata_embeddings,
    embedding_backend_ready,
    load_vectors_for_field,
    DEFAULT_MODEL_ID,
)
from evaluate_metadata_identity_metrics import (  # type: ignore
    build_metadata_embeddings_for_target,
    load_positive_annotations_from_labels,
)

log = logging.getLogger("starBoard.analysis.plot_pair_embeddings")

# ----------------------------------------------------------------------
# Config – tweak these
# ----------------------------------------------------------------------

#: Metadata fields to drop completely from consideration.
# Fields to completely exclude from consideration in this script.
# These can be:
# - fields only measured in the lab (not in the field),
# - fields you never want to use for plotting / pairing.
EXCLUDED_FIELDS: Set[str] = {
    "Last location",    # explicitly requested to be excluded
    "diameter_cm",      # lab-only numeric fields you mentioned elsewhere
    "volume_ml",
    "num_arms",
    "short_arm_codes",
    # Add any other fields you decide to skip entirely:
    "Other_descriptions",
    "sex",
}

#: Numeric fields allowed to contribute to the embedding.
#: By default we *only* keep 'num_apparent_arms'.
NUMERIC_FIELDS_TO_USE: Set[str] = {
    "num_apparent_arms",
}

#: Dimensionality reduction methods to run.
REDUCTION_METHODS: Sequence[str] = ("pca", "mds", "tsne", "umap", "phate")

#: Random seed for all stochastic reducers
RANDOM_STATE: int = 42

#: If True, require that every selected text + numeric field is present
#: (non-empty / parseable) for an ID to be used.
#: If False (default), allow partially populated metadata; we only require:
#:   - a metadata row,
#:   - text embeddings,
#:   - valid numeric values for the chosen numeric fields (if any).
REQUIRE_ALL_FIELDS: bool = False

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _safe_float(s: object) -> Optional[float]:
    try:
        return float(s)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _row_has_all_fields(
    row: Dict[str, str],
    text_fields: Sequence[str],
    numeric_fields: Sequence[str],
) -> bool:
    # All text fields must be non-empty
    for f in text_fields:
        if not (row.get(f, "") or "").strip():
            return False
    # All numeric fields must parse to a float
    for f in numeric_fields:
        if _safe_float(row.get(f, "")) is None:
            return False
    return True


def _build_numeric_by_id(
    rows_by_id: Dict[str, Dict[str, str]],
    numeric_fields: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    Build per-ID numeric vectors for the given fields.
    Only IDs where *all* numeric_fields parse cleanly are kept.
    """
    out: Dict[str, np.ndarray] = {}
    if not numeric_fields:
        return out

    for _id, row in rows_by_id.items():
        vals: List[float] = []
        ok = True
        for f in numeric_fields:
            v = _safe_float(row.get(f, ""))
            if v is None:
                ok = False
                break
            vals.append(float(v))
        if ok:
            out[_id] = np.asarray(vals, dtype=np.float32)
    return out


def _run_reducer(X: np.ndarray, method: str) -> Optional[np.ndarray]:
    method = method.lower()
    log.info("Running reducer: %s", method)

    if method == "pca":
        model = PCA(n_components=2, random_state=RANDOM_STATE)
        return model.fit_transform(X)

    if method == "mds":
        model = MDS(
            n_components=2,
            random_state=RANDOM_STATE,
            n_init=4,
            max_iter=300,
            n_jobs=None,
        )
        return model.fit_transform(X)

    if method == "tsne":
        model = TSNE(
            n_components=2,
            random_state=RANDOM_STATE,
            init="random",
            learning_rate="auto",
        )
        return model.fit_transform(X)

    if method == "umap":
        if umap is None:
            log.warning("UMAP is not installed; skipping.")
            return None
        model = umap.UMAP(
            n_components=2,
            random_state=RANDOM_STATE,
        )
        return model.fit_transform(X)

    if method == "phate":
        if phate is None:
            log.warning("PHATE is not installed; skipping.")
            return None
        op = phate.PHATE(
            n_components=2,
            random_state=RANDOM_STATE,
        )
        return op.fit_transform(X)

    raise ValueError(f"Unknown reduction method: {method}")


def _plot_pairs_2d(
    coords: np.ndarray,
    method_name: str,
    outdir: Path,
    gallery_indices: Sequence[int],
    gallery_ids: Sequence[str],
    query_indices: Sequence[int],
    pair_indices: Sequence[Tuple[int, int]],  # (g_idx, q_idx)
    text_fields: Sequence[str],
    numeric_fields: Sequence[str],
) -> Path:
    xs = coords[:, 0]
    ys = coords[:, 1]

    fig, ax = plt.subplots(figsize=(9, 7))

    # Scatter points
    ax.scatter(
        xs[gallery_indices],
        ys[gallery_indices],
        marker="o",
        label="Gallery",
        zorder=3,
    )
    ax.scatter(
        xs[query_indices],
        ys[query_indices],
        marker="x",
        label="Queries",
        zorder=3,
    )

    # Lines between paired points
    for g_idx, q_idx in pair_indices:
        ax.plot(
            [xs[g_idx], xs[q_idx]],
            [ys[g_idx], ys[q_idx]],
            color="lightgray",
            linewidth=0.5,
            alpha=0.6,
            zorder=1,
        )

    # Text labels above gallery points
    for gid, g_idx in zip(gallery_ids, gallery_indices):
        ax.text(
            xs[g_idx],
            ys[g_idx] + 0.01,
            gid,
            fontsize=8,
            ha="center",
            va="bottom",
            zorder=4,
        )

    # Cosmetics
    txt_fields = ", ".join(text_fields) if text_fields else "(none)"
    num_fields = ", ".join(numeric_fields) if numeric_fields else "(none)"

    title = (
        f"{method_name.upper()} metadata text embedding space"
    )
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    fig.tight_layout()
    outpath = outdir / f"pairs_{method_name.lower()}.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    log.info("Wrote %s", outpath)
    return outpath


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

    # Choose which fields are active
    text_fields: List[str] = [
        f for f in TEXT_FIELDS if f not in EXCLUDED_FIELDS
    ]
    numeric_fields: List[str] = [
        f
        for f in NUMERIC_FIELDS
        if (f in NUMERIC_FIELDS_TO_USE) and (f not in EXCLUDED_FIELDS)
    ]

    log.info("Using text fields:   %s", ", ".join(text_fields) or "(none)")
    log.info("Using numeric fields:%s", ", ".join(numeric_fields) or "(none)")

    if not text_fields and not numeric_fields:
        log.error("No fields selected after exclusions; nothing to do.")
        return

    # 1) Load positive labels
    positives_by_query, n_rows_total, n_positive = load_positive_annotations_from_labels()
    if not positives_by_query:
        log.warning(
            "No positive matches (verdict='yes') found in labels. Nothing to plot."
        )
        return
    log.info(
        "Labels summary: %d total rows, %d positive rows (yes).",
        n_rows_total,
        n_positive,
    )

    # 2) Load latest metadata for Queries + Gallery
    q_by_id, g_by_id = _load_latest_metadata_maps()
    log.info(
        "Loaded latest metadata rows: %d queries, %d gallery IDs.",
        len(q_by_id),
        len(g_by_id),
    )

    # IDs that are allowed to participate
    if REQUIRE_ALL_FIELDS:
        # Strict: all selected fields must be present/non-empty/parseable
        q_ids_ok: Set[str] = {
            _id
            for _id, row in q_by_id.items()
            if _row_has_all_fields(row, text_fields, numeric_fields)
        }
        g_ids_ok: Set[str] = {
            _id
            for _id, row in g_by_id.items()
            if _row_has_all_fields(row, text_fields, numeric_fields)
        }
        log.info(
            "Coverage (strict): %d queries and %d gallery IDs have all selected fields.",
            len(q_ids_ok),
            len(g_ids_ok),
        )
    else:
        # Relaxed: as long as a metadata row exists, we consider the ID OK.
        q_ids_ok = set(q_by_id.keys())
        g_ids_ok = set(g_by_id.keys())
        log.info(
            "Coverage (relaxed): %d queries and %d gallery IDs have metadata rows "
            "(fields may be partially missing).",
            len(q_ids_ok),
            len(g_ids_ok),
        )

    # 3) Ensure metadata text embeddings exist (for chosen text fields only)
    ok, reason = embedding_backend_ready()
    if not ok:
        log.error("Embedding backend is not available: %s", reason)
        return

    try:
        g_done, g_cand = ensure_metadata_embeddings(
            "Gallery", g_by_id, fields=text_fields
        )
        q_done, q_cand = ensure_metadata_embeddings(
            "Queries", q_by_id, fields=text_fields
        )
        log.info(
            "ensure_metadata_embeddings: Gallery %d/%d updated; "
            "Queries %d/%d updated.",
            g_done,
            g_cand,
            q_done,
            q_cand,
        )
    except Exception as e:
        log.error("Failed to build/ensure metadata embeddings: %s", e)
        return

    # 4) Load aggregated text embeddings (mean over active text_fields)
    try:
        gallery_ids_all, G_text_all = build_metadata_embeddings_for_target(
            "Gallery", text_fields
        )
        query_ids_all, Q_text_all = build_metadata_embeddings_for_target(
            "Queries", text_fields
        )
    except Exception as e:
        log.error("Failed to load/aggregate metadata embeddings: %s", e)
        return

    emb_g_text: Dict[str, np.ndarray] = {
        str(gid): G_text_all[i] for i, gid in enumerate(gallery_ids_all)
    }
    emb_q_text: Dict[str, np.ndarray] = {
        str(qid): Q_text_all[i] for i, qid in enumerate(query_ids_all)
    }

    # 5) Numeric features
    use_numeric = bool(numeric_fields)
    numeric_g_by_id = _build_numeric_by_id(g_by_id, numeric_fields) if use_numeric else {}
    numeric_q_by_id = _build_numeric_by_id(q_by_id, numeric_fields) if use_numeric else {}

    # 6) Build the filtered list of positive pairs
    pairs: List[Tuple[str, str]] = []  # (query_id, gallery_id)
    query_ids_in_pairs: Set[str] = set()
    gallery_ids_in_pairs: Set[str] = set()

    for qid_raw, pos_gids in positives_by_query.items():
        qid = normalize_id_value(qid_raw)
        if not qid:
            continue

        # Query must have a metadata row + embeddings (+ numeric if used)
        if qid not in q_ids_ok:
            continue
        if qid not in emb_q_text:
            continue
        if use_numeric and (qid not in numeric_q_by_id):
            continue

        for gid_raw in pos_gids:
            gid = normalize_id_value(gid_raw)
            if not gid:
                continue

            # Gallery must have a metadata row + embeddings (+ numeric if used)
            if gid not in g_ids_ok:
                continue
            if gid not in emb_g_text:
                continue
            if use_numeric and (gid not in numeric_g_by_id):
                continue

            pairs.append((qid, gid))
            query_ids_in_pairs.add(qid)
            gallery_ids_in_pairs.add(gid)

    if not pairs:
        log.warning(
            "After filtering for metadata availability, embeddings, and numeric "
            "coverage, no positive query–gallery pairs remain."
        )
        return

    log.info(
        "Using %d positive pairs across %d queries and %d gallery IDs.",
        len(pairs),
        len(query_ids_in_pairs),
        len(gallery_ids_in_pairs),
    )

    # 7) Build the combined embedding matrix (Gallery + Queries)
    gallery_ids_sorted = sorted(gallery_ids_in_pairs)
    query_ids_sorted = sorted(query_ids_in_pairs)

    all_ids: List[str] = []
    X_text_list: List[np.ndarray] = []
    X_num_list: List[np.ndarray] = []

    for gid in gallery_ids_sorted:
        all_ids.append(f"G:{gid}")
        X_text_list.append(emb_g_text[gid])
        if use_numeric:
            X_num_list.append(numeric_g_by_id[gid])

    for qid in query_ids_sorted:
        all_ids.append(f"Q:{qid}")
        X_text_list.append(emb_q_text[qid])
        if use_numeric:
            X_num_list.append(numeric_q_by_id[qid])

    X_text = np.vstack(X_text_list).astype(np.float32)

    if use_numeric:
        X_num = np.vstack(X_num_list).astype(np.float32)
        # Standardize numeric features so they don't dominate
        mean = X_num.mean(axis=0, keepdims=True)
        std = X_num.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        X_num_z = (X_num - mean) / std
        X_full = np.hstack([X_text, X_num_z])
    else:
        X_full = X_text

    # Index maps
    index_by_full_id: Dict[str, int] = {
        fid: i for i, fid in enumerate(all_ids)
    }
    gallery_indices: List[int] = [
        index_by_full_id[f"G:{gid}"] for gid in gallery_ids_sorted
    ]
    query_indices: List[int] = [
        index_by_full_id[f"Q:{qid}"] for qid in query_ids_sorted
    ]
    pair_indices: List[Tuple[int, int]] = [
        (index_by_full_id[f"G:{gid}"], index_by_full_id[f"Q:{qid}"])
        for (qid, gid) in pairs
    ]

    outdir = THIS_FILE.parent
    outdir.mkdir(parents=True, exist_ok=True)

    # 8) Run each reducer and plot
    for method in REDUCTION_METHODS:
        try:
            coords = _run_reducer(X_full, method)
        except Exception as e:
            log.error("Reducer '%s' failed: %s", method, e)
            continue
        if coords is None:
            continue

        _plot_pairs_2d(
            coords=coords,
            method_name=method,
            outdir=outdir,
            gallery_indices=gallery_indices,
            gallery_ids=gallery_ids_sorted,
            query_indices=query_indices,
            pair_indices=pair_indices,
            text_fields=text_fields,
            numeric_fields=numeric_fields,
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
