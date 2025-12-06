#!/usr/bin/env python
"""
analysis/evaluate_metadata_identity_metrics.py

Standalone analysis script for starBoard.

What it does
------------
1. Loads the latest human "second-order" labels from all
   `_second_order_labels.csv` files (via src.data.past_matches) and
   treats rows with verdict == "yes" as **positive identity matches**
   between a Query ID and a Gallery ID.

2. Ensures metadata text embeddings exist for both Gallery and Queries
   using the BGE backend (via src.search.embed_store and TEXT_FIELDS).

3. Aggregates text embeddings across all TEXT_FIELDS into a single
   embedding per ID (mean over available fields), separately for:
     - Queries (query_id)
     - Gallery (gallery_id)

4. Sweeps several simple metric configurations:
     - preprocess ∈ { "raw", "l2", "standard" }
     - similarity ∈ { "dot", "neg_l2", "neg_l1" }

   and computes mAP@K for K ∈ {1, 5, 10, 20, 50} using:
     - each Query with at least one positive Gallery in the embeddings
     - all Gallery embeddings as the candidate set

5. Picks the "best" metric configuration by mAP@max(K)
   (i.e., highest mAP@50 by default). :contentReference[oaicite:2]{index=2}

6. Writes outputs under:
       /analysis/evaluate_metadata_identity_metrics/
   including:
       - metrics_results.json
       - metrics_results.csv
       - best_metric_config.json
       - map_at_k_best_metric.png

7. EXTRA: Using the best metric config from (5), it then:
     - evaluates EACH individual TEXT field separately
       (Gallery+Queries embedded only for that field),
     - ranks single fields by mAP@max(K),
     - takes the top 4 single fields (or fewer if <4 exist),
     - evaluates ALL 2-, 3-, and 4-way combinations among those top fields,
     - ranks combos by mAP@max(K),
     - saves results to:
           per_field_and_combos.json
           per_field_and_combos.csv
       and logs a readable summary.

Usage
-----
From the project root:

    python analysis/evaluate_metadata_identity_metrics.py

No CLI arguments are required; everything is discovered from the
standard starBoard archive layout.
"""

from __future__ import annotations

import sys
import csv
import json
import logging
import itertools
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional, Set

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
from src.search.engine import TEXT_FIELDS
from src.search.embed_store import (
    ensure_metadata_embeddings,
    load_vectors_for_field,
    embedding_backend_ready,
    DEFAULT_MODEL_ID,
)

log = logging.getLogger("starBoard.analysis.metadata_identity")


# ----------------------------------------------------------------------
# Metric configuration
# ----------------------------------------------------------------------


@dataclass
class MetricConfig:
    """
    Defines one similarity configuration for metadata embeddings.

    Attributes
    ----------
    name : str
        Human-readable name, used as key in results and filenames.
    preprocess : str
        One of: "raw", "l2", "standard"
    similarity : str
        One of: "dot", "neg_l2", "neg_l1"
    """

    name: str
    preprocess: str
    similarity: str


def build_default_metric_configs() -> List[MetricConfig]:
    """
    Define a small set of metric configurations to sweep over.
    """
    configs = [
        # Cosine-like similarity: L2-normalized + dot
        MetricConfig(name="cosine_l2", preprocess="l2", similarity="dot"),

        # Raw dot-product
        MetricConfig(name="dot_raw", preprocess="raw", similarity="dot"),

        # Negative squared L2 distance
        MetricConfig(name="neg_l2_raw", preprocess="raw", similarity="neg_l2"),
        MetricConfig(name="neg_l2_standard", preprocess="standard", similarity="neg_l2"),

        # Negative L1 distance
        MetricConfig(name="neg_l1_raw", preprocess="raw", similarity="neg_l1"),
        MetricConfig(name="neg_l1_standard", preprocess="standard", similarity="neg_l1"),
    ]
    return configs


# ----------------------------------------------------------------------
# mAP@K computation
# ----------------------------------------------------------------------


def average_precision_at_k(
    ranked_ids: Sequence[str],
    positive_ids: Set[str],
    k: int,
) -> float:
    """
    Compute Average Precision at K for a single query.

    Parameters
    ----------
    ranked_ids : sequence of str
        Gallery IDs ranked from most to least similar.
    positive_ids : set[str]
        IDs considered relevant (positive matches) for this query.
    k : int
        Cutoff rank.

    Returns
    -------
    ap_k : float
        Average Precision at K (in [0, 1]). Returns 0.0 if there are
        no positives or no hits before rank K.
    """
    if not positive_ids:
        return 0.0

    max_rank = min(int(k), len(ranked_ids))
    hits = 0
    sum_precisions = 0.0

    for i in range(max_rank):
        gid = ranked_ids[i]
        if gid in positive_ids:
            hits += 1
            precision_at_i = hits / float(i + 1)
            sum_precisions += precision_at_i

    denom = min(len(positive_ids), int(k))
    if denom == 0:
        return 0.0

    return sum_precisions / float(denom)


def average_precision_vector_at_ks(
    ranked_ids: Sequence[str],
    positive_ids: Set[str],
    k_values: Sequence[int],
) -> np.ndarray:
    """
    Compute AP@K for multiple K values for a single query.
    """
    return np.array(
        [average_precision_at_k(ranked_ids, positive_ids, k) for k in k_values],
        dtype=np.float64,
    )


# ----------------------------------------------------------------------
# Similarity computation
# ----------------------------------------------------------------------


def compute_similarity_matrix(
    Q: np.ndarray,
    G: np.ndarray,
    similarity: str,
) -> np.ndarray:
    """
    Compute similarity matrix between query embeddings Q and gallery
    embeddings G under a given similarity rule.

    Parameters
    ----------
    Q : np.ndarray, shape (n_queries_batch, D)
    G : np.ndarray, shape (n_gallery, D)
    similarity : str
        One of "dot", "neg_l2", "neg_l1".

    Returns
    -------
    sims : np.ndarray, shape (n_queries_batch, n_gallery)
        Similarity scores (higher = more similar).
    """
    if similarity == "dot":
        return Q @ G.T

    if similarity == "neg_l2":
        # Negative squared L2 distance
        Q2 = np.sum(Q * Q, axis=1, keepdims=True)  # (b, 1)
        G2 = np.sum(G * G, axis=1, keepdims=True)  # (g, 1)
        cross = Q @ G.T  # (b, g)
        d2 = Q2 + G2.T - 2.0 * cross
        return -d2

    if similarity == "neg_l1":
        # Negative L1 distance
        Q_exp = Q[:, None, :]      # (b, 1, D)
        G_exp = G[None, :, :]      # (1, g, D)
        d1 = np.abs(Q_exp - G_exp).sum(axis=2)  # (b, g)
        return -d1

    raise ValueError(f"Unknown similarity '{similarity}'. Use 'dot', 'neg_l2', or 'neg_l1'.")


# ----------------------------------------------------------------------
# Embedding aggregation & preprocessing
# ----------------------------------------------------------------------


def build_metadata_embeddings_for_target(
    target: str,
    fields: Sequence[str],
    model_id: str = DEFAULT_MODEL_ID,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate per-field text embeddings into one vector per ID by
    averaging across all available `fields`.

    Parameters
    ----------
    target : str
        "Gallery" or "Queries".
    fields : sequence of str
        Text fields to aggregate (subset of TEXT_FIELDS).
    model_id : str
        Expected embedding model ID.

    Returns
    -------
    item_ids : np.ndarray, shape (N,)
        Normalized IDs for this target.
    X_raw : np.ndarray, shape (N, D)
        Raw aggregated embeddings.
    """
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


def build_preprocessed_embeddings_pair(
    Q_raw: np.ndarray,
    G_raw: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Build preprocessed versions of (Query, Gallery) embeddings:

      - "raw":      as-is
      - "l2":       row-wise L2 normalization
      - "standard": z-score per dimension (fit on Gallery, applied to both)
    """
    embeddings_q: Dict[str, np.ndarray] = {}
    embeddings_g: Dict[str, np.ndarray] = {}

    Q_raw = np.asarray(Q_raw, dtype=np.float32)
    G_raw = np.asarray(G_raw, dtype=np.float32)

    embeddings_q["raw"] = Q_raw.copy()
    embeddings_g["raw"] = G_raw.copy()

    eps = np.finfo(np.float32).eps

    def _l2_normalize(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
        norms = np.maximum(norms, eps)
        return (X / norms).astype(np.float32)

    embeddings_q["l2"] = _l2_normalize(Q_raw)
    embeddings_g["l2"] = _l2_normalize(G_raw)

    mean = G_raw.mean(axis=0, keepdims=True).astype(np.float32)
    std = G_raw.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)

    embeddings_q["standard"] = ((Q_raw - mean) / std).astype(np.float32)
    embeddings_g["standard"] = ((G_raw - mean) / std).astype(np.float32)

    log.info("Built preprocessed variants (raw, l2, standard) using gallery stats.")
    return embeddings_q, embeddings_g


# ----------------------------------------------------------------------
# Labels: positive identity matches from second-order labels
# ----------------------------------------------------------------------


def load_positive_annotations_from_labels() -> Tuple[Dict[str, Set[str]], int, int]:
    """
    Load positive matches from all _second_order_labels.csv files.

    We use src.data.past_matches._latest_label_rows_all_pairs() to get the
    latest row per (query_id, gallery_id) where verdict ∈ {"yes","maybe","no"},
    then keep only verdict == "yes" as positives.

    Returns
    -------
    positives_by_query : dict[str, set[str]]
        Mapping from query_id -> set of positive gallery_ids.
    n_rows_total : int
        Number of (query, gallery) pairs after latest-row reduction.
    n_positive_rows : int
        Number of rows with verdict == "yes".
    """
    rows = _latest_label_rows_all_pairs()
    n_rows_total = len(rows)
    positives_by_query: Dict[str, Set[str]] = {}
    n_positive = 0

    for r in rows:
        verdict = (r.get("verdict", "") or "").strip().lower()
        if verdict != "yes":
            continue
        qid = normalize_id_value(r.get("query_id", ""))
        gid = normalize_id_value(r.get("gallery_id", ""))
        if not qid or not gid:
            continue
        positives_by_query.setdefault(qid, set()).add(gid)
        n_positive += 1

    log.info(
        "Loaded %d latest label rows; %d rows with verdict='yes'; "
        "%d queries with ≥1 positive.",
        n_rows_total,
        n_positive,
        len(positives_by_query),
    )
    return positives_by_query, n_rows_total, n_positive


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------


def evaluate_metric_configs(
    metric_configs: Sequence[MetricConfig],
    embeddings_q_by_preprocess: Dict[str, np.ndarray],
    embeddings_g_by_preprocess: Dict[str, np.ndarray],
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    positives_by_query: Dict[str, Set[str]],
    k_values: Sequence[int],
    batch_size: int = 128,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all metric configurations and compute mAP@K for each.

    Parameters
    ----------
    metric_configs : sequence[MetricConfig]
    embeddings_q_by_preprocess : dict[str, np.ndarray]
        Map preprocess name -> Query embeddings matrix.
    embeddings_g_by_preprocess : dict[str, np.ndarray]
        Map preprocess name -> Gallery embeddings matrix.
    query_ids : np.ndarray
        Query ID array, same ordering as Q_raw.
    gallery_ids : np.ndarray
        Gallery ID array, same ordering as G_raw.
    positives_by_query : dict[str, set[str]]
        Positive gallery IDs per query.
    k_values : sequence[int]
        Ks at which to evaluate mAP.
    batch_size : int
        Batch size for similarity computation.

    Returns
    -------
    results : dict[str, dict]
        name -> { "config": MetricConfig, "n_valid_queries": int, "mAP": {k: float} }
    """
    k_values = sorted(int(k) for k in k_values)
    max_k = max(k_values)
    n_gallery = int(len(gallery_ids))

    if max_k > n_gallery:
        log.warning(
            "max K (%d) is larger than gallery size (%d). Using K up to %d where applicable.",
            max_k,
            n_gallery,
            n_gallery,
        )

    # ID -> index maps
    q_id_to_index: Dict[str, int] = {str(qid): i for i, qid in enumerate(query_ids)}
    g_id_to_index: Dict[str, int] = {str(gid): i for i, gid in enumerate(gallery_ids)}

    # Filter to queries with embeddings and at least one positive gallery with embeddings
    filtered_positives_by_query: Dict[str, Set[str]] = {}
    for qid, pos_set in positives_by_query.items():
        if qid not in q_id_to_index:
            continue
        filtered_pos = {gid for gid in pos_set if gid in g_id_to_index}
        if filtered_pos:
            filtered_positives_by_query[qid] = filtered_pos

    eval_query_ids = sorted(filtered_positives_by_query.keys())
    if not eval_query_ids:
        raise RuntimeError(
            "After filtering, no queries have both embeddings and at least one "
            "positive gallery with embeddings. Cannot compute mAP."
        )

    q_indices = np.array([q_id_to_index[qid] for qid in eval_query_ids], dtype=np.int64)
    log.info(
        "Using %d queries with at least one positive gallery in embeddings; gallery size=%d.",
        len(eval_query_ids),
        n_gallery,
    )

    results: Dict[str, Dict[str, Any]] = {}

    for config in metric_configs:
        log.info(
            "Evaluating metric config '%s' (preprocess=%s, similarity=%s).",
            config.name,
            config.preprocess,
            config.similarity,
        )

        if config.preprocess not in embeddings_q_by_preprocess:
            raise KeyError(
                f"Preprocess '{config.preprocess}' not found in embeddings_q_by_preprocess."
            )
        if config.preprocess not in embeddings_g_by_preprocess:
            raise KeyError(
                f"Preprocess '{config.preprocess}' not found in embeddings_g_by_preprocess."
            )

        Q_all = embeddings_q_by_preprocess[config.preprocess]
        G_all = embeddings_g_by_preprocess[config.preprocess]

        sum_ap_by_k = np.zeros(len(k_values), dtype=np.float64)
        n_valid_queries = 0

        n_queries = len(eval_query_ids)
        for start in range(0, n_queries, batch_size):
            end = min(start + batch_size, n_queries)
            batch_query_ids = eval_query_ids[start:end]
            batch_indices = q_indices[start:end]

            Q_batch = Q_all[batch_indices]  # (b, D)
            sims_batch = compute_similarity_matrix(Q_batch, G_all, config.similarity)

            for row_idx, qid in enumerate(batch_query_ids):
                positive_set = filtered_positives_by_query.get(qid)
                if not positive_set:
                    continue

                sims = sims_batch[row_idx]
                ranked_idx = np.argsort(-sims, kind="mergesort")
                ranked_gallery_ids = gallery_ids[ranked_idx]

                ap_vec = average_precision_vector_at_ks(
                    ranked_gallery_ids,
                    positive_set,
                    k_values,
                )

                sum_ap_by_k += ap_vec
                n_valid_queries += 1

        if n_valid_queries == 0:
            log.warning(
                "No valid queries evaluated for metric '%s'; mAP will be NaN.",
                config.name,
            )
            map_by_k = {int(k): float("nan") for k in k_values}
        else:
            map_values = sum_ap_by_k / float(n_valid_queries)
            map_by_k = {int(k): float(v) for k, v in zip(k_values, map_values)}

        results[config.name] = {
            "config": config,
            "n_valid_queries": int(n_valid_queries),
            "mAP": map_by_k,
        }

        log.info(
            "Done with '%s': %s",
            config.name,
            ", ".join(f"mAP@{k}={map_by_k[int(k)]:.4f}" for k in k_values),
        )

    return results


# ----------------------------------------------------------------------
# Selection & output
# ----------------------------------------------------------------------


def select_best_config(
    results: Dict[str, Dict[str, Any]],
    k_values: Sequence[int],
    selection_k: Optional[int] = None,
) -> Tuple[str, Dict[str, Any], int]:
    """
    Select the best metric configuration based on mAP@selection_k.

    If selection_k is None, uses max(K) from k_values.

    Returns
    -------
    best_name : str
        Name of the best metric configuration.
    best_result : dict
        Result dict for the best configuration.
    selection_k_final : int
        The K value actually used for selection.
    """
    if not results:
        raise ValueError("No results to select from.")

    k_values = sorted(int(k) for k in k_values)
    if selection_k is None:
        selection_k_final = int(k_values[-1])
    else:
        selection_k_final = int(selection_k)
        if selection_k_final not in k_values:
            raise ValueError(
                f"selection_k={selection_k_final} not in K values {k_values}."
            )

    best_name: Optional[str] = None
    best_score: float = float("-inf")

    for name, res in results.items():
        score = float(res["mAP"].get(selection_k_final, float("nan")))
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_name = name

    if best_name is None:
        raise RuntimeError(
            "No metric configuration had a finite mAP at selection_k. "
            "Check your data and K values."
        )

    best_result = results[best_name]

    log.info(
        "Best metric at K=%d: '%s' (mAP@%d=%.4f, %d valid queries).",
        selection_k_final,
        best_name,
        selection_k_final,
        best_score,
        best_result["n_valid_queries"],
    )

    return best_name, best_result, selection_k_final


def save_results(
    results: Dict[str, Dict[str, Any]],
    k_values: Sequence[int],
    best_name: str,
    selection_k: int,
    output_dir: Path,
) -> Tuple[Path, Path, Path]:
    """
    Save JSON + CSV summaries and the best config JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values_int = [int(k) for k in sorted(k_values)]

    # Full JSON
    json_payload: Dict[str, Any] = {
        "k_values": k_values_int,
        "selection_k": int(selection_k),
        "best_metric_name": best_name,
        "results": {},
    }

    for name, res in results.items():
        cfg: MetricConfig = res["config"]
        json_payload["results"][name] = {
            "config": asdict(cfg),
            "n_valid_queries": res["n_valid_queries"],
            "mAP": res["mAP"],
        }

    json_path = output_dir / "metrics_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    log.info("Wrote JSON results to: %s", json_path)

    # CSV summary
    header = ["metric_name", "preprocess", "similarity", "n_valid_queries"] + [
        f"mAP@{k}" for k in k_values_int
    ]
    csv_rows: List[Dict[str, Any]] = []
    for name, res in results.items():
        cfg: MetricConfig = res["config"]
        row: Dict[str, Any] = {
            "metric_name": name,
            "preprocess": cfg.preprocess,
            "similarity": cfg.similarity,
            "n_valid_queries": res["n_valid_queries"],
        }
        for k in k_values_int:
            row[f"mAP@{k}"] = res["mAP"].get(int(k), float("nan"))
        csv_rows.append(row)

    csv_path = output_dir / "metrics_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    log.info("Wrote CSV results to: %s", csv_path)

    # Best config JSON
    best_res = results[best_name]
    best_cfg: MetricConfig = best_res["config"]
    best_payload = {
        "best_metric_name": best_name,
        "selection_k": int(selection_k),
        "config": asdict(best_cfg),
        "n_valid_queries": best_res["n_valid_queries"],
        "mAP": best_res["mAP"],
    }
    best_config_path = output_dir / "best_metric_config.json"
    with best_config_path.open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)
    log.info("Wrote best config JSON to: %s", best_config_path)

    return json_path, csv_path, best_config_path


def plot_map_at_k_for_best(
    best_metric_name: str,
    best_result: Dict[str, Any],
    k_values: Sequence[int],
    output_dir: Path,
) -> Path:
    """
    Plot mAP@K for the best metric configuration and save as PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values_sorted = sorted(int(k) for k in k_values)
    map_values = [best_result["mAP"][int(k)] for k in k_values_sorted]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values_sorted, map_values, marker="o")
    ax.set_xlabel("K")
    ax.set_ylabel("mAP@K")
    ax.set_title(f"mAP@K for best metric: {best_metric_name}")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    fig_path = output_dir / "map_at_k_best_metric.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    log.info("Wrote mAP@K plot for best metric to: %s", fig_path)
    return fig_path


# ----------------------------------------------------------------------
# EXTRA: per-field & combination analysis using best metric
# ----------------------------------------------------------------------


def _evaluate_fields_with_fixed_metric(
    fields: Sequence[str],
    positives_by_query: Dict[str, Set[str]],
    k_values: Sequence[int],
    batch_size: int,
    metric_config: MetricConfig,
) -> Optional[Dict[str, Any]]:
    """
    Build embeddings for a given set of fields and evaluate a single
    metric configuration. Returns the result dict for that config.name,
    or None on hard failure.
    """
    try:
        gallery_ids, G_raw = build_metadata_embeddings_for_target("Gallery", fields)
        query_ids, Q_raw = build_metadata_embeddings_for_target("Queries", fields)
    except Exception as e:
        log.error(
            "Failed to build embeddings for fields=%s: %s",
            ", ".join(fields),
            e,
        )
        return None

    if Q_raw.shape[1] != G_raw.shape[1]:
        log.warning(
            "Skipping fields=%s: query and gallery dims differ (%d vs %d).",
            ", ".join(fields),
            Q_raw.shape[1],
            G_raw.shape[1],
        )
        return None

    emb_q, emb_g = build_preprocessed_embeddings_pair(Q_raw, G_raw)

    # Evaluate just this metric
    results = evaluate_metric_configs(
        metric_configs=[metric_config],
        embeddings_q_by_preprocess=emb_q,
        embeddings_g_by_preprocess=emb_g,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        positives_by_query=positives_by_query,
        k_values=k_values,
        batch_size=batch_size,
    )

    return results.get(metric_config.name)


def analyze_top_fields_and_combinations(
    positives_by_query: Dict[str, Set[str]],
    k_values: Sequence[int],
    batch_size: int,
    best_metric_config: MetricConfig,
    output_dir: Path,
    top_n_single_fields: int = 4,
    top_n_combos_per_size: int = 5,
) -> None:
    """
    Using the supplied best_metric_config, evaluate:

      1) Each individual TEXT field.
      2) Take the top `top_n_single_fields` by mAP@max(K).
      3) Evaluate all 2/3/4-way combinations among those top fields.
      4) Save results to JSON/CSV and log a human-readable summary.
    """
    k_values = sorted(int(k) for k in k_values)
    selection_k = int(k_values[-1])

    log.info(
        "=== Per-field and combination analysis using metric '%s' "
        "(preprocess=%s, similarity=%s) ===",
        best_metric_config.name,
        best_metric_config.preprocess,
        best_metric_config.similarity,
    )

    # ---- 1) single fields ----
    single_results: List[Dict[str, Any]] = []

    for field in TEXT_FIELDS:
        log.info("Evaluating single field '%s' under fixed metric.", field)
        res = _evaluate_fields_with_fixed_metric(
            fields=[field],
            positives_by_query=positives_by_query,
            k_values=k_values,
            batch_size=batch_size,
            metric_config=best_metric_config,
        )
        if res is None:
            continue

        map_by_k = res["mAP"]
        score = float(map_by_k.get(selection_k, float("nan")))
        single_results.append(
            {
                "kind": "single",
                "name": field,
                "size": 1,
                "fields": [field],
                "config": asdict(best_metric_config),
                "n_valid_queries": res["n_valid_queries"],
                "mAP": map_by_k,
                "selection_k": selection_k,
                "selection_map": score,
            }
        )

    if not single_results:
        log.warning("No per-field results were produced; skipping combo analysis.")
        return

    # Rank single fields by mAP@selection_k (descending)
    single_results_sorted = sorted(
        single_results,
        key=lambda r: (
            -(r["selection_map"] if not np.isnan(r["selection_map"]) else -1e9),
            r["name"],
        ),
    )

    log.info("Single-field ranking (by mAP@%d):", selection_k)
    for r in single_results_sorted:
        log.info(
            "  field=%s  mAP@%d=%.4f  (n_valid=%d)",
            r["name"],
            selection_k,
            r["selection_map"],
            r["n_valid_queries"],
        )

    top_fields = [r["name"] for r in single_results_sorted[:top_n_single_fields]]
    log.info("Top %d single TEXT fields: %s", len(top_fields), ", ".join(top_fields))

    # ---- 2) combinations among top fields ----
    combo_results: List[Dict[str, Any]] = []

    max_combo_size = min(4, len(top_fields))
    for size in range(2, max_combo_size + 1):
        for combo in itertools.combinations(top_fields, size):
            fields = list(combo)
            combo_name = "+".join(fields)
            log.info(
                "Evaluating combo size=%d fields=%s under fixed metric.",
                size,
                combo_name,
            )
            res = _evaluate_fields_with_fixed_metric(
                fields=fields,
                positives_by_query=positives_by_query,
                k_values=k_values,
                batch_size=batch_size,
                metric_config=best_metric_config,
            )
            if res is None:
                continue

            map_by_k = res["mAP"]
            score = float(map_by_k.get(selection_k, float("nan")))
            combo_results.append(
                {
                    "kind": f"combo_{size}",
                    "name": combo_name,
                    "size": size,
                    "fields": fields,
                    "config": asdict(best_metric_config),
                    "n_valid_queries": res["n_valid_queries"],
                    "mAP": map_by_k,
                    "selection_k": selection_k,
                    "selection_map": score,
                }
            )

    if not combo_results:
        log.warning("No combination results produced.")
    else:
        for size in range(2, max_combo_size + 1):
            of_size = [r for r in combo_results if r["size"] == size]
            if not of_size:
                continue
            of_size_sorted = sorted(
                of_size,
                key=lambda r: (
                    -(r["selection_map"] if not np.isnan(r["selection_map"]) else -1e9),
                    r["name"],
                ),
            )
            top_subset = of_size_sorted[:top_n_combos_per_size]
            log.info(
                "Top %d combos of size %d (by mAP@%d):",
                len(top_subset),
                size,
                selection_k,
            )
            for r in top_subset:
                log.info(
                    "  fields=%s  mAP@%d=%.4f  (n_valid=%d)",
                    "+".join(r["fields"]),
                    selection_k,
                    r["selection_map"],
                    r["n_valid_queries"],
                )

    # ---- 3) Save singles + combos to JSON / CSV ----
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = single_results_sorted + combo_results
    json_payload = {
        "k_values": k_values,
        "selection_k": selection_k,
        "metric_config": asdict(best_metric_config),
        "results": all_rows,
    }

    json_path = output_dir / "per_field_and_combos.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    log.info("Wrote per-field/combo JSON to: %s", json_path)

    csv_header = [
        "kind",
        "name",
        "size",
        "fields",
        "n_valid_queries",
        "selection_k",
        "selection_map",
    ] + [f"mAP@{k}" for k in k_values]

    csv_path = output_dir / "per_field_and_combos.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        writer.writeheader()
        for r in all_rows:
            row: Dict[str, Any] = {
                "kind": r["kind"],
                "name": r["name"],
                "size": r["size"],
                "fields": " + ".join(r["fields"]),
                "n_valid_queries": r["n_valid_queries"],
                "selection_k": r["selection_k"],
                "selection_map": r["selection_map"],
            }
            m_map = r["mAP"]
            for k in k_values:
                row[f"mAP@{k}"] = float(m_map.get(int(k), float("nan")))
            writer.writerow(row)
    log.info("Wrote per-field/combo CSV to: %s", csv_path)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    script_dir = THIS_FILE.parent
    output_dir = script_dir / "evaluate_metadata_identity_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Output directory: %s", output_dir)
    log.info("Project root: %s", PROJECT_ROOT)
    log.info("Archive root: %s", archive_root())

    # Hyperparameters
    k_values: List[int] = [1, 5, 10, 20, 50]
    batch_size: int = 128

    # 1) Load human labels (verdict == "yes")
    positives_by_query, n_rows_total, n_positive = load_positive_annotations_from_labels()
    if not positives_by_query:
        log.warning(
            "No positive matches (verdict='yes') found in second-order labels. "
            "Nothing to evaluate."
        )
        return
    log.info(
        "Labels summary: %d total rows, %d positive rows (yes).",
        n_rows_total,
        n_positive,
    )

    # 2) Load metadata for Queries and Gallery (for embedding ensure)
    q_by_id, g_by_id = _load_latest_metadata_maps()
    log.info(
        "Loaded latest metadata rows: %d queries, %d gallery IDs.",
        len(q_by_id),
        len(g_by_id),
    )

    # 3) Ensure metadata text embeddings exist for all TEXT_FIELDS
    ok, reason = embedding_backend_ready()
    if not ok:
        log.error(
            "Embedding backend is not available: %s\n"
            "Cannot compute metadata embedding identity metrics.",
            reason,
        )
        return

    try:
        g_done, g_cand = ensure_metadata_embeddings(
            "Gallery", g_by_id, fields=TEXT_FIELDS
        )
        q_done, q_cand = ensure_metadata_embeddings(
            "Queries", q_by_id, fields=TEXT_FIELDS
        )
        log.info(
            "ensure_metadata_embeddings: Gallery %d/%d updated; Queries %d/%d updated.",
            g_done,
            g_cand,
            q_done,
            q_cand,
        )
    except Exception as e:
        log.error("Failed to build/ensure metadata embeddings: %s", e)
        return

    # 4) Build aggregated embeddings across all TEXT_FIELDS
    try:
        gallery_ids, G_raw = build_metadata_embeddings_for_target(
            "Gallery", TEXT_FIELDS
        )
        query_ids, Q_raw = build_metadata_embeddings_for_target(
            "Queries", TEXT_FIELDS
        )
    except Exception as e:
        log.error("Failed to load/aggregate metadata embeddings: %s", e)
        return

    if Q_raw.shape[1] != G_raw.shape[1]:
        log.error(
            "Query and gallery embeddings have different dimensionality: %d vs %d.",
            Q_raw.shape[1],
            G_raw.shape[1],
        )
        return

    # 5) Preprocess embeddings
    embeddings_q_by_preprocess, embeddings_g_by_preprocess = (
        build_preprocessed_embeddings_pair(Q_raw, G_raw)
    )

    # 6) Metric configs
    metric_configs = build_default_metric_configs()
    log.info(
        "Metric configurations to evaluate: %s",
        ", ".join(cfg.name for cfg in metric_configs),
    )

    # 7) Evaluate
    try:
        results = evaluate_metric_configs(
            metric_configs=metric_configs,
            embeddings_q_by_preprocess=embeddings_q_by_preprocess,
            embeddings_g_by_preprocess=embeddings_g_by_preprocess,
            query_ids=query_ids,
            gallery_ids=gallery_ids,
            positives_by_query=positives_by_query,
            k_values=k_values,
            batch_size=batch_size,
        )
    except Exception as e:
        log.error("Evaluation failed: %s", e)
        return

    # 8) Select best config (by highest mAP@max(K))
    best_name, best_result, selection_k = select_best_config(
        results=results,
        k_values=k_values,
        selection_k=None,
    )
    best_metric_config: MetricConfig = results[best_name]["config"]

    # 9) Save results and plot
    json_path, csv_path, best_config_path = save_results(
        results=results,
        k_values=k_values,
        best_name=best_name,
        selection_k=selection_k,
        output_dir=output_dir,
    )

    _ = plot_map_at_k_for_best(
        best_metric_name=best_name,
        best_result=best_result,
        k_values=k_values,
        output_dir=output_dir,
    )

    log.info("Global metric sweep complete.")
    log.info("All results JSON : %s", json_path)
    log.info("All results CSV  : %s", csv_path)
    log.info("Best config JSON : %s", best_config_path)

    # 10) EXTRA: per-field + combo analysis under the best metric
    analyze_top_fields_and_combinations(
        positives_by_query=positives_by_query,
        k_values=k_values,
        batch_size=batch_size,
        best_metric_config=best_metric_config,
        output_dir=output_dir,
        top_n_single_fields=4,        # "top 4" single fields
        top_n_combos_per_size=5,      # log top 5 combos per size
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
