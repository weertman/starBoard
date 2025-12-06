#!/usr/bin/env python
"""
analysis/plot_metadata_pairs_embeddings.py

Visualize positive Query–Gallery matches in a 2D embedding space
(PHATE, UMAP, t-SNE, MDS, PCA).

Features
--------
- Uses metadata text embeddings (BGE backend via embed_store) plus
  selected numeric fields (e.g. num_apparent_arms) to build a single
  joint vector per ID for Queries and Gallery.

- Connects each positive (query_id, gallery_id) pair with a faint line.

- Labels each Gallery point with its gallery_id.

- Field selection:
    * By default, uses all TEXT_FIELDS except "Last location"
      plus the numeric field "num_apparent_arms".
    * You can override via --include-fields / --exclude-fields.

- Pair filtering:
    * By default, keeps only pairs where *all* selected fields are present
      (non-empty / numeric-parsable) on both Query and Gallery.
    * Pass --allow-incomplete-pairs to include pairs that do NOT have all
      fields, as long as they share at least ONE selected field with data
      on both sides.

Outputs
-------
PNG files under:
    analysis/metadata_pairs_embeddings/pairs_<METHOD>.png
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Set

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

log = logging.getLogger("starBoard.analysis.metadata_pairs_embeddings")

# ----------------------------------------------------------------------
# Helpers: field parsing / selection
# ----------------------------------------------------------------------


def _parse_fields_arg(arg: str | None) -> List[str]:
    """Parse a comma- or plus-separated list of field names."""
    if not arg:
        return []
    out: List[str] = []
    tmp = arg.replace("+", ",")
    for part in tmp.split(","):
        name = part.strip()
        if name:
            out.append(name)
    return out


def choose_fields(include_arg: str | None, exclude_arg: str | None) -> List[str]:
    """
    Decide which metadata fields to use.

    Default:
      - All TEXT_FIELDS except "Last location"
      - plus "num_apparent_arms"
    Then apply --include-fields / --exclude-fields overrides.
    """
    all_known = set(TEXT_FIELDS) | set(NUMERIC_FIELDS)

    # --- default include ---
    default_text = [f for f in TEXT_FIELDS if f != "Last location"]
    default_numeric = ["num_apparent_arms"]
    default_fields = default_text + default_numeric

    include_fields = _parse_fields_arg(include_arg) or default_fields
    exclude_fields = set(_parse_fields_arg(exclude_arg))

    selected: List[str] = []
    for f in include_fields:
        if f in exclude_fields:
            continue
        if f not in all_known:
            log.warning("Ignoring unknown field '%s' in include-fields.", f)
            continue
        selected.append(f)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    final_fields: List[str] = []
    for f in selected:
        if f not in seen:
            seen.add(f)
            final_fields.append(f)

    if not final_fields:
        raise SystemExit(
            "After applying include/exclude, no valid fields remain. "
            "Check --include-fields/--exclude-fields."
        )

    log.info("Selected fields for embedding/filtering: %s", ", ".join(final_fields))
    return final_fields


# ----------------------------------------------------------------------
# Embedding construction
# ----------------------------------------------------------------------


def build_metadata_embeddings_for_target(
    target: str,
    fields: Sequence[str],
    model_id: str = DEFAULT_MODEL_ID,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Copy of the identity-metrics helper:

    Aggregate per-field text embeddings into one vector per ID by
    averaging across all available `fields`.
    """
    from src.data.csv_io import normalize_id_value  # local import to avoid cycles

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
            _id = normalize_id_value(raw_id)
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
    item_ids_arr = np.array(item_ids, dtype=object)

    log.info(
        "Aggregated %d embeddings for target=%s from %d field-vectors "
        "across %d text fields (dim=%d).",
        len(item_ids_arr),
        target,
        total_vectors,
        len(fields),
        X_raw.shape[1],
    )

    return item_ids_arr, X_raw


def build_joint_embeddings(
    target: str,
    text_fields: Sequence[str],
    numeric_fields: Sequence[str],
    rows_by_id: Dict[str, Dict[str, str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build one joint embedding per ID:
      [ aggregated_text_embedding , standardized_numeric_fields... ]
    """
    if not text_fields:
        raise SystemExit(
            "At least one text field must be selected; numeric-only embeddings "
            "are not implemented in this helper."
        )

    item_ids, X_text = build_metadata_embeddings_for_target(target, text_fields)
    N = X_text.shape[0]

    if not numeric_fields:
        return item_ids, X_text

    num_dims = len(numeric_fields)
    X_num = np.zeros((N, num_dims), dtype=np.float32)
    mask = np.zeros((N, num_dims), dtype=bool)

    for i, _id in enumerate(item_ids):
        row = rows_by_id.get(_id, {}) or {}
        for j, field in enumerate(numeric_fields):
            s = (row.get(field, "") or "").strip()
            if not s:
                continue
            try:
                val = float(s)
            except Exception:
                continue
            X_num[i, j] = val
            mask[i, j] = True

    # Standardize each numeric dimension; encode missing as 0 (mean)
    for j in range(num_dims):
        col_mask = mask[:, j]
        if not np.any(col_mask):
            continue
        vals = X_num[col_mask, j]
        mean = float(vals.mean())
        std = float(vals.std()) or 1.0
        col = np.zeros(N, dtype=np.float32)
        col[col_mask] = (vals - mean) / std
        X_num[:, j] = col

    X_joint = np.concatenate([X_text, X_num], axis=1)
    log.info(
        "Joint embeddings for %s: N=%d, dim_text=%d, dim_numeric=%d, dim_total=%d",
        target,
        N,
        X_text.shape[1],
        num_dims,
        X_joint.shape[1],
    )
    return item_ids, X_joint


# ----------------------------------------------------------------------
# Pair loading & filtering
# ----------------------------------------------------------------------


def _field_present(row: Dict[str, str], field: str) -> bool:
    """Return True if this row has a usable value for the given field."""
    val = (row.get(field, "") or "").strip()
    if not val:
        return False
    if field in NUMERIC_FIELDS:
        try:
            float(val)
            return True
        except Exception:
            return False
    # Text-ish
    return True


def load_positive_pairs() -> List[Tuple[str, str]]:
    """Load latest 'yes' verdicts as (query_id, gallery_id) pairs."""
    out: List[Tuple[str, str]] = []
    for r in _latest_label_rows_all_pairs():
        verdict = (r.get("verdict", "") or "").strip().lower()
        if verdict != "yes":
            continue
        qid = normalize_id_value(r.get("query_id", ""))
        gid = normalize_id_value(r.get("gallery_id", ""))
        if not qid or not gid:
            continue
        out.append((qid, gid))
    return out


def filter_pairs_by_fields(
    pairs: List[Tuple[str, str]],
    q_rows: Dict[str, Dict[str, str]],
    g_rows: Dict[str, Dict[str, str]],
    fields: Sequence[str],
    allow_incomplete: bool,
) -> List[Tuple[str, str]]:
    """
    Filter pairs based on metadata presence for selected fields.

    If allow_incomplete == False:
        keep only pairs where ALL fields are present on BOTH sides.

    If allow_incomplete == True:
        keep pairs where AT LEAST ONE field is present on BOTH sides.
    """
    kept: List[Tuple[str, str]] = []
    dropped_missing_meta = 0
    dropped_field_criteria = 0

    for qid, gid in pairs:
        q_row = q_rows.get(qid)
        g_row = g_rows.get(gid)
        if not q_row or not g_row:
            dropped_missing_meta += 1
            continue

        all_present = True
        any_shared = False
        for f in fields:
            q_ok = _field_present(q_row, f)
            g_ok = _field_present(g_row, f)
            if q_ok and g_ok:
                any_shared = True
            if not (q_ok and g_ok):
                all_present = False

        if allow_incomplete:
            if not any_shared:
                dropped_field_criteria += 1
                continue
        else:
            if not all_present:
                dropped_field_criteria += 1
                continue

        kept.append((qid, gid))

    log.info(
        "Field filter: kept=%d, dropped_missing_meta=%d, dropped_field_criteria=%d",
        len(kept),
        dropped_missing_meta,
        dropped_field_criteria,
    )
    return kept


# ----------------------------------------------------------------------
# Dimensionality reduction
# ----------------------------------------------------------------------


def reduce_dimensionality(X: np.ndarray, method: str) -> np.ndarray:
    method = method.lower()
    n = X.shape[0]
    if n < 2:
        raise RuntimeError("Need at least 2 points to run dimensionality reduction.")

    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30.0, max(5.0, (n - 1) / 3.0))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        return tsne.fit_transform(X)

    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "UMAP is not installed. Install with `pip install umap-learn` "
                "or remove 'umap' from --methods."
            ) from e

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
        )
        return reducer.fit_transform(X)

    if method == "phate":
        try:
            import phate  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "PHATE is not installed. Install with `pip install phate` "
                "or remove 'phate' from --methods."
            ) from e

        op = phate.PHATE(n_components=2, random_state=42)
        return op.fit_transform(X)

    if method == "mds":
        from sklearn.manifold import MDS

        mds = MDS(n_components=2, random_state=42, dissimilarity="euclidean")
        return mds.fit_transform(X)

    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(X)

    raise ValueError(f"Unknown method '{method}'.")


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def plot_pairs(
    coords: np.ndarray,
    gallery_ids: List[str],
    query_ids: List[str],
    pairs: List[Tuple[str, str]],
    fields: Sequence[str],
    method: str,
    out_dir: Path,
    allow_incomplete: bool,
) -> Path:
    n_g = len(gallery_ids)
    g_coords = coords[:n_g]
    q_coords = coords[n_g:]

    idx_g = {gid: i for i, gid in enumerate(gallery_ids)}
    idx_q = {qid: n_g + i for i, qid in enumerate(query_ids)}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Grey lines for each pair
    for qid, gid in pairs:
        ig = idx_g.get(gid)
        iq = idx_q.get(qid)
        if ig is None or iq is None:
            continue
        x = [coords[ig, 0], coords[iq, 0]]
        y = [coords[ig, 1], coords[iq, 1]]
        ax.plot(x, y, color="0.8", linewidth=0.8, alpha=0.7, zorder=1)

    # Scatter points
    ax.scatter(
        g_coords[:, 0],
        g_coords[:, 1],
        s=40,
        marker="o",
        label="Gallery",
        zorder=3,
    )
    ax.scatter(
        q_coords[:, 0],
        q_coords[:, 1],
        s=35,
        marker="x",
        label="Queries",
        zorder=2,
    )

    # Gallery labels
    for i, gid in enumerate(gallery_ids):
        ax.text(
            g_coords[i, 0],
            g_coords[i, 1] + 0.05,
            gid,
            fontsize=8,
            ha="center",
            va="bottom",
        )

    fields_str = ", ".join(fields)
    mode_str = (
        "complete pairs only"
        if not allow_incomplete
        else "allowing missing fields"
    )

    ax.set_title(
        f"Query–gallery pairs in {method.upper()} space\n"
        f"({fields_str}  |  {mode_str})"
    )
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pairs_{method.lower()}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    log.info("Saved %s projection plot to: %s", method.upper(), out_path)
    return out_path


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Query–Gallery positive pairs in metadata embedding space."
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="phate,umap,tsne,mds,pca",
        help=(
            "Comma-separated list of methods to run. "
            "Options: phate, umap, tsne, mds, pca. "
            "Default: phate,umap,tsne,mds,pca"
        ),
    )
    parser.add_argument(
        "--include-fields",
        type=str,
        default=None,
        help=(
            "Comma/plus separated list of metadata fields to include. "
            "If omitted, uses TEXT_FIELDS except 'Last location' plus 'num_apparent_arms'."
        ),
    )
    parser.add_argument(
        "--exclude-fields",
        type=str,
        default=None,
        help="Comma/plus separated list of fields to exclude after inclusion.",
    )
    parser.add_argument(
        "--allow-incomplete-pairs",
        action="store_true",
        help=(
            "If set, include pairs that do not have ALL selected fields present "
            "on both sides; require only at least one shared field. "
            "Default behavior is to require all fields to be present."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for PNGs. "
            "Default: analysis/metadata_pairs_embeddings (next to this script)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    methods_raw = _parse_fields_arg(args.methods)
    methods = [m.lower() for m in methods_raw] or ["tsne"]

    out_dir = args.output_dir or (THIS_FILE.parent / "metadata_pairs_embeddings")

    # 1) Decide fields
    selected_fields = choose_fields(args.include_fields, args.exclude_fields)
    text_fields = [f for f in selected_fields if f in TEXT_FIELDS]
    numeric_fields = [f for f in selected_fields if f in NUMERIC_FIELDS]

    # 2) Load metadata rows
    q_by_id, g_by_id = _load_latest_metadata_maps()
    log.info(
        "Loaded latest metadata rows: %d queries, %d gallery IDs.",
        len(q_by_id),
        len(g_by_id),
    )

    # 3) Ensure embeddings for selected text fields
    ok, reason = embedding_backend_ready()
    if not ok:
        raise SystemExit(
            f"Embedding backend is not available: {reason}. "
            "Cannot build metadata embeddings."
        )

    if text_fields:
        g_done, g_cand = ensure_metadata_embeddings(
            "Gallery", g_by_id, fields=list(text_fields)
        )
        q_done, q_cand = ensure_metadata_embeddings(
            "Queries", q_by_id, fields=list(text_fields)
        )
        log.info(
            "ensure_metadata_embeddings: Gallery %d/%d updated; "
            "Queries %d/%d updated.",
            g_done,
            g_cand,
            q_done,
            q_cand,
        )

    # 4) Build joint embeddings
    gallery_ids_all, G_joint = build_joint_embeddings(
        "Gallery", text_fields, numeric_fields, g_by_id
    )
    query_ids_all, Q_joint = build_joint_embeddings(
        "Queries", text_fields, numeric_fields, q_by_id
    )

    emb_g = {gid: G_joint[i] for i, gid in enumerate(gallery_ids_all)}
    emb_q = {qid: Q_joint[i] for i, qid in enumerate(query_ids_all)}

    # 5) Load positive pairs
    all_pairs = load_positive_pairs()
    log.info("Loaded %d positive (yes) pairs from labels.", len(all_pairs))

    # 6) Filter to pairs that have embeddings on both sides
    pairs_with_emb: List[Tuple[str, str]] = []
    missing_emb = 0
    for qid, gid in all_pairs:
        if qid in emb_q and gid in emb_g:
            pairs_with_emb.append((qid, gid))
        else:
            missing_emb += 1
    log.info(
        "Pairs after requiring embeddings on both sides: kept=%d, dropped_missing_emb=%d",
        len(pairs_with_emb),
        missing_emb,
    )

    # 7) Apply field presence filter (complete vs incomplete)
    pairs_filtered = filter_pairs_by_fields(
        pairs_with_emb,
        q_rows=q_by_id,
        g_rows=g_by_id,
        fields=selected_fields,
        allow_incomplete=args.allow_incomplete_pairs,
    )
    if not pairs_filtered:
        raise SystemExit("No pairs left after filtering; nothing to plot.")

    # IDs actually involved in the filtered pairs
    gallery_ids_used = sorted({gid for _, gid in pairs_filtered})
    query_ids_used = sorted({qid for qid, _ in pairs_filtered})

    G_used = np.stack([emb_g[gid] for gid in gallery_ids_used], axis=0)
    Q_used = np.stack([emb_q[qid] for qid in query_ids_used], axis=0)
    X_all = np.vstack([G_used, Q_used])

    log.info(
        "Final dataset for DR: %d gallery, %d queries, dim=%d",
        G_used.shape[0],
        Q_used.shape[0],
        X_all.shape[1],
    )

    # 8) Run each DR method + plot
    for method in methods:
        coords = reduce_dimensionality(X_all, method=method)
        plot_pairs(
            coords=coords,
            gallery_ids=gallery_ids_used,
            query_ids=query_ids_used,
            pairs=pairs_filtered,
            fields=selected_fields,
            method=method,
            out_dir=out_dir,
            allow_incomplete=args.allow_incomplete_pairs,
        )


if __name__ == "__main__":
    main()
