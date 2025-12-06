#!/usr/bin/env python
"""
evaluate_metadata_field_sets.py

Re‑ID‑style evaluation of starBoard's metadata‑based First‑order search
for different subsets of fields.

Key points:

- Ground truth comes from your existing second‑order labels
  (verdict == "yes"), via src.data.past_matches.build_past_matches_dataset().

- Retrieval engine is src.search.engine.FirstOrderSearchEngine, using
  the actual field scorers you use in the First‑order tab.

- Metrics are *photo‑reID style*:
    * CMC / Rank‑k accuracy (probability that at least one true match
      appears in top‑k).
    * mAP@K (average precision per query, truncated at K, then averaged).

- Field sets include:
    * Single fields (e.g. "[single] num_apparent_arms")
    * Grouped sets (numeric_only, text_only, numeric+codes, etc.)
    * Full ALL_FIELDS

- Output is written to:
    analysis/evaluate_metadata_field_sets/
      - field_sets_results.json
      - field_sets_results.csv
      - best_field_set.json
      - map_at_k_best_field_set.png
      - cmc_at_k_best_field_set.png

HOW TO RUN (your preferred way):

    1. Open this file in PyCharm (or similar).
    2. Hit "Run" with NO arguments.

It will auto‑detect the starBoard root from this file's location and
use the default archive / metadata.

OPTIONAL: CLI MODE
------------------
If you *do* supply command‑line arguments, the script will switch to a
more configurable mode (different K values, output dir, etc.). If you
never touch CLI, you can ignore this section.
"""

from __future__ import annotations

import sys
import argparse
import logging
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import math

import matplotlib.pyplot as plt

# ------------------- project import setup -------------------

# This file lives at: <repo_root>/analysis/evaluate_metadata_field_sets.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.search.engine import (  # type: ignore
    FirstOrderSearchEngine,
    ALL_FIELDS,
    NUMERIC_FIELDS,
    CATEGORICAL_FIELDS,
    SET_FIELDS,
    TEXT_FIELDS,
)
from src.data.past_matches import (  # type: ignore
    build_past_matches_dataset,
    PastMatchesDataset,
)
from src.data.archive_paths import archive_root  # type: ignore

logger = logging.getLogger("starBoard.analysis.metadata_field_sets")


# ---------------------- field set definitions ----------------------


def build_field_sets() -> Dict[str, List[str]]:
    """
    Define the metadata field subsets to evaluate.

    You can tweak this if you want different groupings.
    """
    field_sets: Dict[str, List[str]] = {}

    # Singles for introspection
    for f in NUMERIC_FIELDS + CATEGORICAL_FIELDS + SET_FIELDS + TEXT_FIELDS:
        field_sets[f"[single] {f}"] = [f]

    # Grouped sets
    field_sets["numeric_only"] = list(NUMERIC_FIELDS)
    field_sets["categorical_only"] = list(CATEGORICAL_FIELDS)
    field_sets["codes_only"] = list(SET_FIELDS)
    field_sets["text_only"] = list(TEXT_FIELDS)

    field_sets["numeric+categorical"] = list(NUMERIC_FIELDS + CATEGORICAL_FIELDS)
    field_sets["numeric+codes"] = list(NUMERIC_FIELDS + SET_FIELDS)
    field_sets["categorical+codes"] = list(CATEGORICAL_FIELDS + SET_FIELDS)
    field_sets["non_text_all"] = list(NUMERIC_FIELDS + CATEGORICAL_FIELDS + SET_FIELDS)
    field_sets["all_fields"] = list(ALL_FIELDS)

    return field_sets


# ---------------------- ground truth from labels ----------------------


def build_ground_truth(ds: PastMatchesDataset) -> Dict[str, Set[str]]:
    """
    Build {query_id -> set of gallery_ids} where verdict == 'yes'.
    """
    positives_by_query: Dict[str, Set[str]] = {}
    for rec in ds.records:
        if (rec.verdict or "").strip().lower() != "yes":
            continue
        positives_by_query.setdefault(rec.query_id, set()).add(rec.gallery_id)
    return positives_by_query


# ---------------------- Re‑ID metrics ----------------------


@dataclass
class FieldSetMetrics:
    name: str
    fields: List[str]

    # Query counts
    num_queries_with_pos: int          # has at least one positive in gallery universe
    num_eval_queries: int              # actually evaluated (rank() returned non‑empty)

    # mAP@K: photo re‑ID style (mean AP across eval queries)
    map_at_k: Dict[int, float]         # K -> mAP@K

    # CMC / Rank‑K accuracy: P(positive appears in top‑K)
    cmc_at_k: Dict[int, float]         # K -> CMC(K)

    # Convenience: selection K & its scores (for "best" ranking)
    selection_k: int
    selection_map: float
    selection_cmc: float


def _compute_ap_at_k(relevant_flags: List[int], k: int) -> float:
    """
    Average Precision truncated at K.
    relevant_flags: list of 0/1 booleans in ranking order.
    """
    n = min(k, len(relevant_flags))
    if n <= 0:
        return 0.0

    num_rel = 0
    prec_sum = 0.0
    for i in range(n):
        if relevant_flags[i]:
            num_rel += 1
            prec_sum += num_rel / float(i + 1)

    if num_rel == 0:
        return 0.0
    return prec_sum / float(num_rel)


def evaluate_field_set(
    engine: FirstOrderSearchEngine,
    positives_by_query: Dict[str, Set[str]],
    gallery_ids: Set[str],
    field_set_name: str,
    fields: List[str],
    k_values: List[int],
) -> FieldSetMetrics:
    """
    Evaluate a single field set with Re‑ID style metrics.
    """
    k_values = sorted(set(k_values))
    max_k = max(k_values)

    include_fields = set(fields)
    if not include_fields:
        logger.warning("Field set '%s' is empty; skipping.", field_set_name)

    num_queries_with_pos = 0
    num_eval_queries = 0

    # accumulators
    map_sums: Dict[int, float] = {k: 0.0 for k in k_values}
    cmc_hits: Dict[int, int] = {k: 0 for k in k_values}

    for qid, pos_all in positives_by_query.items():
        pos = pos_all & gallery_ids
        if not pos:
            continue  # no positives in gallery
        num_queries_with_pos += 1

        # Rank gallery for this query under this field set
        results = engine.rank(
            qid,
            include_fields=include_fields,
            equalize_weights=True,
            top_k=max_k,
            numeric_offsets=None,
        )
        if not results:
            # No usable signal for this query / field set (e.g. all fields empty)
            continue

        num_eval_queries += 1
        ranked_gids = [it.gallery_id for it in results]
        rel_flags = [1 if gid in pos else 0 for gid in ranked_gids]

        # CMC: does any positive appear in top‑K?
        for k in k_values:
            n = min(k, len(rel_flags))
            if n > 0 and any(rel_flags[:n]):
                cmc_hits[k] += 1

        # mAP@K: AP per query, then mean later
        for k in k_values:
            ap_k = _compute_ap_at_k(rel_flags, k)
            map_sums[k] += ap_k

    if num_eval_queries == 0:
        logger.warning(
            "Field set '%s': no evaluable queries (all had zero signal).", field_set_name
        )

    map_at_k = {
        k: (map_sums[k] / float(num_eval_queries) if num_eval_queries > 0 else 0.0)
        for k in k_values
    }
    cmc_at_k = {
        k: (cmc_hits[k] / float(num_eval_queries) if num_eval_queries > 0 else 0.0)
        for k in k_values
    }

    # For "best" decision we use the highest K (more stable)
    selection_k = max_k
    selection_map = map_at_k.get(selection_k, 0.0)
    selection_cmc = cmc_at_k.get(selection_k, 0.0)

    logger.info(
        "Field set '%s': queries_with_pos=%d, eval_queries=%d, mAP@%d=%.4f, CMC@%d=%.4f",
        field_set_name,
        num_queries_with_pos,
        num_eval_queries,
        selection_k,
        selection_map,
        selection_k,
        selection_cmc,
    )

    return FieldSetMetrics(
        name=field_set_name,
        fields=list(fields),
        num_queries_with_pos=num_queries_with_pos,
        num_eval_queries=num_eval_queries,
        map_at_k=map_at_k,
        cmc_at_k=cmc_at_k,
        selection_k=selection_k,
        selection_map=selection_map,
        selection_cmc=selection_cmc,
    )


# ---------------------- I/O helpers ----------------------


def _ensure_output_dir(output_dir: Optional[Path]) -> Path:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "evaluate_metadata_field_sets"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_results(
    results: List[FieldSetMetrics],
    k_values: List[int],
    output_dir: Path,
    best: FieldSetMetrics,
) -> None:
    # JSON dump
    json_path = output_dir / "field_sets_results.json"
    payload = {
        "k_values": k_values,
        "results": [asdict(r) for r in results],
        "best": asdict(best),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote JSON results to %s", json_path)

    # CSV dump (one row per field set)
    import csv

    csv_path = output_dir / "field_sets_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        header = [
            "name",
            "fields",
            "num_queries_with_pos",
            "num_eval_queries",
            "selection_k",
            "selection_map",
            "selection_cmc",
        ]
        for k in k_values:
            header.append(f"map@{k}")
        for k in k_values:
            header.append(f"cmc@{k}")
        w.writerow(header)

        for r in results:
            row = [
                r.name,
                "|".join(r.fields),
                r.num_queries_with_pos,
                r.num_eval_queries,
                r.selection_k,
                f"{r.selection_map:.6f}",
                f"{r.selection_cmc:.6f}",
            ]
            for k in k_values:
                row.append(f"{r.map_at_k.get(k, 0.0):.6f}")
            for k in k_values:
                row.append(f"{r.cmc_at_k.get(k, 0.0):.6f}")
            w.writerow(row)

    logger.info("Wrote CSV results to %s", csv_path)

    # Best field set JSON
    best_path = output_dir / "best_field_set.json"
    best_path.write_text(json.dumps(asdict(best), indent=2), encoding="utf-8")
    logger.info("Wrote best field set JSON to %s", best_path)


def _plot_best(best: FieldSetMetrics, k_values: List[int], output_dir: Path) -> None:
    ks = sorted(k_values)

    # mAP@K
    fig, ax = plt.subplots()
    ys = [best.map_at_k.get(k, 0.0) for k in ks]
    ax.plot(ks, ys, marker="o")
    ax.set_xlabel("K")
    ax.set_ylabel("mAP@K")
    ax.set_title(f"mAP@K for best field set: {best.name}")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    map_path = output_dir / "map_at_k_best_field_set.png"
    fig.savefig(map_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote mAP@K plot for best field set to %s", map_path)

    # CMC / Rank‑K
    fig, ax = plt.subplots()
    ys_cmc = [best.cmc_at_k.get(k, 0.0) for k in ks]
    ax.plot(ks, ys_cmc, marker="o")
    ax.set_xlabel("Rank K")
    ax.set_ylabel("CMC(K)")
    ax.set_title(f"CMC for best field set: {best.name}")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    cmc_path = output_dir / "cmc_at_k_best_field_set.png"
    fig.savefig(cmc_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote CMC plot for best field set to %s", cmc_path)


# ---------------------- default (no‑CLI) runner ----------------------


DEFAULT_K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def run_default() -> None:
    """
    Main entry point when running with *no* CLI args.
    Uses starBoard's default archive & metadata layout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = _ensure_output_dir(None)
    logger.info("Output directory: %s", out_dir)

    # Just to make it obvious in logs where we're reading from
    logger.info("archive_root: %s", archive_root())

    # Load dataset / ground truth
    ds = build_past_matches_dataset()
    positives_by_query = build_ground_truth(ds)
    logger.info(
        "Ground truth: %d queries with at least one 'yes' label.",
        len(positives_by_query),
    )

    # Build First‑order engine once
    engine = FirstOrderSearchEngine()
    engine.rebuild()  # uses default text backend (BGE if available, else n‑grams)

    gallery_ids = set(engine._gallery_rows_by_id.keys())
    logger.info("Gallery size for engine: %d IDs.", len(gallery_ids))

    # Evaluate all field sets
    field_sets = build_field_sets()
    k_values = list(DEFAULT_K_VALUES)

    # If gallery is very small, note that large K will saturate
    max_possible_k = max(ks for ks in k_values)
    if len(gallery_ids) < max_possible_k:
        logger.info(
            "Gallery has only %d IDs; metrics for K > %d will effectively saturate.",
            len(gallery_ids),
            len(gallery_ids),
        )

    results: List[FieldSetMetrics] = []
    for name, fields in field_sets.items():
        r = evaluate_field_set(
            engine=engine,
            positives_by_query=positives_by_query,
            gallery_ids=gallery_ids,
            field_set_name=name,
            fields=fields,
            k_values=k_values,
        )
        results.append(r)

    if not results:
        logger.warning("No field sets evaluated. Nothing to write.")
        return

    # Pick best by mAP at the largest K
    results_sorted = sorted(
        results, key=lambda r: (r.selection_map, r.num_eval_queries), reverse=True
    )
    best = results_sorted[0]

    logger.info(
        "\n=== Best field set by mAP@K (K = %d) ===\n"
        "Name: %s\n"
        "Fields (%d): %s\n"
        "mAP@K:\n%s\n"
        "CMC(K):\n%s\n"
        "Queries with positives: %d\n"
        "Evaluated queries:      %d\n",
        best.selection_k,
        best.name,
        len(best.fields),
        ", ".join(best.fields),
        "".join(
            f"  K={k:3d}: {best.map_at_k.get(k, 0.0):.4f}\n"
            for k in sorted(best.map_at_k.keys())
        ),
        "".join(
            f"  K={k:3d}: {best.cmc_at_k.get(k, 0.0):.4f}\n"
            for k in sorted(best.cmc_at_k.keys())
        ),
        best.num_queries_with_pos,
        best.num_eval_queries,
    )

    # Persist
    _save_results(results, k_values, out_dir, best)
    _plot_best(best, k_values, out_dir)

    print("\nDone.")
    print(f"- Results JSON/CSV are in: {out_dir}")
    print(f"- Best field set: {best.name}")
    print(f"- Plots: {out_dir / 'map_at_k_best_field_set.png'}")
    print(f"        {out_dir / 'cmc_at_k_best_field_set.png'}")
    print(f"- Engine logs (if any) are under: {archive_root() / 'logs'}")


# ---------------------- optional CLI mode ----------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate metadata field sets for identity retrieval (photo Re‑ID metrics). "
            "If you run with no arguments, defaults are used."
        )
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,5,10,20,50",
        help="Comma‑separated K values for mAP@K / CMC (default: 1,5,10,20,50).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for output directory.",
    )
    return parser


def parse_k_values(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    ks = sorted({int(p) for p in parts})
    ks = [k for k in ks if k > 0]
    if not ks:
        raise ValueError("At least one positive integer K is required.")
    return ks


def run_cli(argv: List[str]) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = _ensure_output_dir(args.output_dir)
    logger.info("Output directory: %s", out_dir)
    logger.info("archive_root: %s", archive_root())

    ds = build_past_matches_dataset()
    positives_by_query = build_ground_truth(ds)
    logger.info(
        "Ground truth: %d queries with at least one 'yes' label.",
        len(positives_by_query),
    )

    engine = FirstOrderSearchEngine()
    engine.rebuild()

    gallery_ids = set(engine._gallery_rows_by_id.keys())
    logger.info("Gallery size for engine: %d IDs.", len(gallery_ids))

    k_values = parse_k_values(args.k_values)
    field_sets = build_field_sets()

    results: List[FieldSetMetrics] = []
    for name, fields in field_sets.items():
        r = evaluate_field_set(
            engine=engine,
            positives_by_query=positives_by_query,
            gallery_ids=gallery_ids,
            field_set_name=name,
            fields=fields,
            k_values=k_values,
        )
        results.append(r)

    if not results:
        logger.warning("No field sets evaluated. Nothing to write.")
        return

    results_sorted = sorted(
        results, key=lambda r: (r.selection_map, r.num_eval_queries), reverse=True
    )
    best = results_sorted[0]

    logger.info(
        "\n=== Best field set by mAP@K (K = %d) ===\n"
        "Name: %s\n"
        "Fields (%d): %s\n"
        "mAP@K:\n%s\n"
        "CMC(K):\n%s\n"
        "Queries with positives: %d\n"
        "Evaluated queries:      %d\n",
        best.selection_k,
        best.name,
        len(best.fields),
        ", ".join(best.fields),
        "".join(
            f"  K={k:3d}: {best.map_at_k.get(k, 0.0):.4f}\n"
            for k in sorted(best.map_at_k.keys())
        ),
        "".join(
            f"  K={k:3d}: {best.cmc_at_k.get(k, 0.0):.4f}\n"
            for k in sorted(best.cmc_at_k.keys())
        ),
        best.num_queries_with_pos,
        best.num_eval_queries,
    )

    _save_results(results, k_values, out_dir, best)
    _plot_best(best, k_values, out_dir)

    print("\nDone (CLI mode).")
    print(f"- Results JSON/CSV are in: {out_dir}")
    print(f"- Best field set: {best.name}")


# ---------------------- main ----------------------


def main() -> None:
    if len(sys.argv) == 1:
        # No arguments: "just run it" mode
        run_default()
    else:
        # Advanced: CLI mode
        run_cli(sys.argv[1:])


if __name__ == "__main__":
    main()

