#!/usr/bin/env python
"""
evaluate_numeric_text_combos_field_sets.py

Evaluate metadata field-set combinations for identity, using
starBoard's FirstOrderSearchEngine and past matches.

High-level behavior
-------------------
1) Load per-field-set results from:
       analysis/evaluate_metadata_field_sets/field_sets_results.csv
   and pick out SINGLE-field rows.

2) Apply field-level exclusions:
       EXCLUDE_FIELDS = {
           "Last location", "diameter_cm", "volume_ml", "num_arms",
           "short_arm_codes", "Other_descriptions", "sex"
       }
   - Ignore any single-field result whose field is in EXCLUDE_FIELDS.
   - Only allow numeric field 'num_apparent_arms' as the numeric base.

3) Among remaining single fields:
   - Identify 'num_apparent_arms' as NUMERIC_BASE_FIELD.
   - Define "text-ish" candidates as those in
         CATEGORICAL_FIELDS ∪ SET_FIELDS ∪ TEXT_FIELDS
     and not in EXCLUDE_FIELDS and not numeric.

   - Rank those text-ish fields by selection_cmc (CMC@selection_k) and
     keep the top TOP_N_TEXT_FIELDS.

4) Build field-set combinations from the candidate pool:
       candidate_fields = [NUMERIC_BASE_FIELD] + top_text_fields

   For each size in COMBO_SIZES (e.g. [2, 3, 4]), evaluate *all*
   combinations of that size drawn from candidate_fields. This means:

     - some combos include 'num_apparent_arms' + text fields
     - some combos are pure text/categorical sets (no numeric field)

5) For each combo, use FirstOrderSearchEngine.rank() with:
       include_fields = combo_fields
       equalize_weights = True

   and compute Re-ID style metrics over queries that have at least one
   'yes' match in the gallery universe:
       - CMC(K): fraction of queries whose positive appears in top-K
       - mAP@K : mean average precision at K

   K values are set to:
       1, 2, ..., max_K
   where max_K = min(MAX_K, gallery_size) and MAX_K is configurable.

6) Save results to:
       analysis/evaluate_numeric_text_combos_field_sets/
           numeric_text_combos_results.json
           numeric_text_combos_results.csv
           cmc_numeric_text_combos.png

Usage
-----
From project root:

    python analysis/evaluate_numeric_text_combos_field_sets.py

No command-line arguments required.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.search.engine import (  # type: ignore
    FirstOrderSearchEngine,
    NUMERIC_FIELDS,
    CATEGORICAL_FIELDS,
    SET_FIELDS,
    TEXT_FIELDS,
)
from src.data.past_matches import build_past_matches_dataset  # type: ignore
from src.data.archive_paths import archive_root  # type: ignore

log = logging.getLogger("starBoard.analysis.numeric_text_combos")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Fields that must NEVER appear in any combo
EXCLUDE_FIELDS: Set[str] = {
    "Last location",    # explicitly requested to be excluded
    "diameter_cm",      # lab-only numeric fields
    "volume_ml",
    "num_arms",
    "short_arm_codes",
    # Add any other fields you decide to skip entirely:
    "Other_descriptions",
    "sex",
}

# Numeric base field (the only numeric we allow in combos)
NUMERIC_BASE_FIELD = "num_apparent_arms"

# How many top single text-ish fields to use when forming combos
TOP_N_TEXT_FIELDS = 6

# Combo sizes: sizes of combinations drawn from candidate_fields
# (candidate_fields = [NUMERIC_BASE_FIELD] + top text-ish fields)
COMBO_SIZES = [2, 3, 4]  # up to 4 fields total

# Maximum K to evaluate in CMC/mAP (truncated by gallery size)
MAX_K = 50

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "evaluate_numeric_text_combos_field_sets"


# ---------------------------------------------------------------------
# Dataclasses & helpers
# ---------------------------------------------------------------------


@dataclass
class ComboMetrics:
    name: str
    fields: List[str]
    n_fields: int
    num_queries_with_pos: int
    num_eval_queries: int
    selection_k: int
    selection_cmc: float
    cmc_at_k: Dict[int, float]
    map_at_k: Dict[int, float]


def parse_fields_str(s: str) -> List[str]:
    """
    Parse the 'fields' column from field_sets_results.csv into a list.
    Handles delimiters: '|' and ';'.
    """
    if s is None:
        return []
    text = str(s).replace("|", ";")
    return [p.strip() for p in text.split(";") if p.strip()]


def build_ground_truth() -> Dict[str, Set[str]]:
    """
    Build {query_id -> set of positive gallery_ids} from past matches
    where verdict == 'yes'.
    """
    ds = build_past_matches_dataset()
    positives: Dict[str, Set[str]] = {}
    for rec in ds.records:
        verdict = (rec.verdict or "").strip().lower()
        if verdict != "yes":
            continue
        positives.setdefault(rec.query_id, set()).add(rec.gallery_id)
    log.info(
        "Ground truth: %d queries with at least one 'yes' match.",
        len(positives),
    )
    return positives


def _compute_ap_at_k(relevant_flags: List[int], k: int) -> float:
    """
    AP@K for a single query, given a relevance flag list.
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


def evaluate_combo(
    engine: FirstOrderSearchEngine,
    positives_by_query: Dict[str, Set[str]],
    gallery_ids: Set[str],
    fields: List[str],
    combo_name: str,
    k_values: Sequence[int],
) -> ComboMetrics:
    """
    Evaluate a single combo (set of fields) with CMC(K) and mAP@K.
    """
    include_fields = set(fields)
    k_values = sorted(set(int(k) for k in k_values))
    max_k = max(k_values)

    num_queries_with_pos = 0
    num_eval_queries = 0
    map_sums: Dict[int, float] = {k: 0.0 for k in k_values}
    cmc_hits: Dict[int, int] = {k: 0 for k in k_values}

    for qid, pos_all in positives_by_query.items():
        pos = pos_all & gallery_ids
        if not pos:
            continue
        num_queries_with_pos += 1

        try:
            results = engine.rank(
                qid,
                include_fields=include_fields,
                equalize_weights=True,
                top_k=max_k,
                numeric_offsets=None,
            )
        except Exception as e:
            log.warning("engine.rank failed for combo '%s', qid=%s: %s", combo_name, qid, e)
            continue

        if not results:
            # No usable signal for this query / combo
            continue

        num_eval_queries += 1
        ranked_ids = [r.gallery_id for r in results]
        rel_flags = [1 if gid in pos else 0 for gid in ranked_ids]

        # CMC: any positive in top-K?
        for k in k_values:
            n = min(k, len(rel_flags))
            if n > 0 and any(rel_flags[:n]):
                cmc_hits[k] += 1

        # mAP
        for k in k_values:
            map_sums[k] += _compute_ap_at_k(rel_flags, k)

    if num_eval_queries == 0:
        cmc_at_k = {k: 0.0 for k in k_values}
        map_at_k = {k: 0.0 for k in k_values}
    else:
        cmc_at_k = {k: cmc_hits[k] / num_eval_queries for k in k_values}
        map_at_k = {k: map_sums[k] / num_eval_queries for k in k_values}

    selection_k = max(k_values)
    selection_cmc = cmc_at_k[selection_k]

    return ComboMetrics(
        name=combo_name,
        fields=list(fields),
        n_fields=len(fields),
        num_queries_with_pos=num_queries_with_pos,
        num_eval_queries=num_eval_queries,
        selection_k=selection_k,
        selection_cmc=selection_cmc,
        cmc_at_k=cmc_at_k,
        map_at_k=map_at_k,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Project root  : %s", PROJECT_ROOT)
    log.info("Archive root  : %s", archive_root())
    log.info("Output dir    : %s", OUTPUT_DIR)

    # 1) Load field_sets_results.csv (from evaluate_metadata_field_sets)
    fs_dir = PROJECT_ROOT / "analysis" / "evaluate_metadata_field_sets"
    fs_csv = fs_dir / "field_sets_results.csv"
    if not fs_csv.exists():
        raise FileNotFoundError(
            f"Expected field_sets_results.csv at {fs_csv}. "
            "Run evaluate_metadata_field_sets.py first."
        )

    fs_df = pd.read_csv(fs_csv)
    log.info("Loaded %d field-set rows from %s", len(fs_df), fs_csv)

    fs_df["parsed_fields"] = fs_df["fields"].apply(parse_fields_str)
    fs_df["n_fields"] = fs_df["parsed_fields"].apply(len)

    # 2) Singles: n_fields == 1
    singles = fs_df[fs_df["n_fields"] == 1].copy()
    if singles.empty:
        raise RuntimeError("No single-field rows in field_sets_results.csv.")

    if "selection_cmc" not in singles.columns:
        raise ValueError("field_sets_results.csv has no 'selection_cmc' column.")

    singles["field_name"] = singles["parsed_fields"].apply(
        lambda xs: xs[0] if xs else None
    )

    # Filter out excluded fields and unwanted numeric fields
    numeric_set = set(NUMERIC_FIELDS)

    def single_allowed(field: str | None) -> bool:
        if not field:
            return False
        if field in EXCLUDE_FIELDS:
            return False
        # If it's numeric and not the numeric base, drop it
        if field in numeric_set and field != NUMERIC_BASE_FIELD:
            return False
        return True

    singles_allowed = singles[singles["field_name"].apply(single_allowed)].copy()

    log.info(
        "After filtering singles (EXCLUDE_FIELDS & non-base numerics), "
        "%d singles remain.",
        len(singles_allowed),
    )

    # Numeric base row
    base_rows = singles_allowed[singles_allowed["field_name"] == NUMERIC_BASE_FIELD]
    if base_rows.empty:
        raise RuntimeError(
            f"Numeric base '{NUMERIC_BASE_FIELD}' not found among filtered singles."
        )

    # Text-ish candidate pool: categorical + set + text fields, minus exclusions
    qual_field_pool = (
        set(CATEGORICAL_FIELDS) | set(SET_FIELDS) | set(TEXT_FIELDS)
    ) - EXCLUDE_FIELDS
    # Remove numeric fields from that pool
    qual_field_pool -= set(NUMERIC_FIELDS)

    text_singles = singles_allowed[
        singles_allowed["field_name"].apply(lambda f: f in qual_field_pool)
    ].copy()
    if text_singles.empty:
        raise RuntimeError(
            "No qualifying text-ish singles found after filtering. "
            "Check EXCLUDE_FIELDS or field definitions."
        )

    # Rank text-ish singles by selection_cmc (desc)
    text_singles_sorted = text_singles.sort_values(
        "selection_cmc", ascending=False
    ).reset_index(drop=True)
    top_text = text_singles_sorted.head(
        min(TOP_N_TEXT_FIELDS, len(text_singles_sorted))
    )

    text_field_names = top_text["field_name"].tolist()
    selection_k = int(top_text["selection_k"].iloc[0])

    log.info(
        "Top %d text-ish fields by CMC@%d:",
        len(text_field_names),
        selection_k,
    )
    for _, row in top_text.iterrows():
        log.info(
            "  %s: selection_cmc=%.4f (n_eval=%d)",
            row["field_name"],
            row["selection_cmc"],
            row["num_eval_queries"],
        )

    # 3) Build combos: any subset of candidate_fields of the given sizes
    candidate_fields: List[str] = [NUMERIC_BASE_FIELD] + text_field_names
    combos: List[Tuple[str, List[str]]] = []

    for size in COMBO_SIZES:
        if size < 2:
            continue  # keep them as true combos (>=2 fields)
        if size > len(candidate_fields):
            continue
        for subset in combinations(candidate_fields, size):
            fields = list(subset)
            name = " + ".join(fields)
            combos.append((name, fields))

    if not combos:
        raise RuntimeError("No combos were generated; check COMBO_SIZES or candidate fields.")

    log.info("Built %d combos:", len(combos))
    for i, (name, fields) in enumerate(combos, start=1):
        log.info("  %2d. %s  (fields: %s)", i, name, ", ".join(fields))

    # 4) Ground truth
    positives_by_query = build_ground_truth()
    if not positives_by_query:
        log.warning("No positive matches found. Nothing to evaluate.")
        return

    # 5) FirstOrderSearchEngine
    engine = FirstOrderSearchEngine()
    engine.rebuild()
    gallery_ids = set(engine._gallery_rows_by_id.keys())
    log.info("Gallery size: %d IDs.", len(gallery_ids))

    # 6) K values
    max_k = min(MAX_K, len(gallery_ids))
    k_values = list(range(1, max_k + 1))
    log.info("Evaluating CMC/mAP at K = %s", k_values)

    # 7) Evaluate combos
    combo_metrics: List[ComboMetrics] = []
    for name, fields in combos:
        log.info("Evaluating combo: %s", name)
        m = evaluate_combo(
            engine=engine,
            positives_by_query=positives_by_query,
            gallery_ids=gallery_ids,
            fields=fields,
            combo_name=name,
            k_values=k_values,
        )
        combo_metrics.append(m)
        log.info(
            "  -> CMC@%d=%.4f (num_eval_queries=%d)",
            m.selection_k,
            m.selection_cmc,
            m.num_eval_queries,
        )

    if not combo_metrics:
        log.warning("No combos successfully evaluated.")
        return

    # 8) Save JSON/CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "k_values": k_values,
        "numeric_base_field": NUMERIC_BASE_FIELD,
        "exclude_fields": sorted(EXCLUDE_FIELDS),
        "results": [asdict(m) for m in combo_metrics],
    }
    json_path = OUTPUT_DIR / "numeric_text_combos_results.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    log.info("Wrote JSON combo results to: %s", json_path)

    csv_header = [
        "name",
        "fields",
        "n_fields",
        "num_queries_with_pos",
        "num_eval_queries",
        "selection_k",
        "selection_cmc",
    ] + [f"cmc@{k}" for k in k_values] + [f"map@{k}" for k in k_values]

    csv_path = OUTPUT_DIR / "numeric_text_combos_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        writer.writeheader()
        for m in combo_metrics:
            row: Dict[str, Any] = {
                "name": m.name,
                "fields": " + ".join(m.fields),
                "n_fields": m.n_fields,
                "num_queries_with_pos": m.num_queries_with_pos,
                "num_eval_queries": m.num_eval_queries,
                "selection_k": m.selection_k,
                "selection_cmc": m.selection_cmc,
            }
            for k in k_values:
                row[f"cmc@{k}"] = m.cmc_at_k[k]
                row[f"map@{k}"] = m.map_at_k[k]
            writer.writerow(row)
    log.info("Wrote CSV combo results to: %s", csv_path)

    # 9) Plot CMC swarm
    fig, ax = plt.subplots(figsize=(9, 7))
    for m in combo_metrics:
        ys = [m.cmc_at_k[k] for k in k_values]
        ax.plot(
            k_values,
            ys,
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=m.name,
        )

    ax.set_xlabel("Rank K")
    ax.set_ylabel("CMC(K)")
    ax.set_title(
        "CMC curves for numeric/text combos\n"
        f"(numeric base allowed = '{NUMERIC_BASE_FIELD}', "
        f"excluded = {sorted(EXCLUDE_FIELDS)})"
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    if len(k_values) <= 20:
        ax.set_xticks(k_values)

    ax.legend(
        title="Field set (combo)",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    png_path = OUTPUT_DIR / "cmc_numeric_text_combos.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    log.info("Wrote CMC swarm plot to: %s", png_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
