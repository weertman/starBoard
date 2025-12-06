#!/usr/bin/env python
"""
plot_numeric_text_combos_with_singles_cmc.py

Plot CMC(K) vs K for the TOP field sets drawn from BOTH:

  1) numeric+text combos evaluated in:
         analysis/evaluate_numeric_text_combos_field_sets/
             numeric_text_combos_results.csv

  2) single-field sets (non-excluded) from:
         analysis/evaluate_metadata_field_sets/
             field_sets_results.csv

We:
  - Exclude any field set (single or combo) whose field list contains
    any member of EXCLUDE_FIELDS.
  - Treat all rows in numeric_text_combos_results.csv as "combo" rows.
  - Rank ALL remaining rows together by CMC@K_rank, where:
        K_rank = 5 if cmc@5 is available for both singles and combos,
        otherwise K_rank = max(common K values).
  - Plot CMC(K) vs K for the top N overall, with a legend that indicates
    which ones are singles vs combos.

Outputs:
  analysis/evaluate_numeric_text_combos_field_sets/
      cmc_numeric_text_combos_with_singles.png

Usage:
  python analysis/plot_numeric_text_combos_with_singles_cmc.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Set
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fields that should never appear in plotted sets (single or combo)
EXCLUDE_FIELDS: Set[str] = {
    "Last location",
    "diameter_cm",
    "volume_ml",
    "num_arms",
    "short_arm_codes",
    "Other_descriptions",   # ensure this one really gets dropped
    "other_descriptions",
    "sex",
    'madreporite_descriptions',
}

# How many total curves (singles + combos) to plot
TOP_N_FIELDSETS = 32


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def eval_dir_numeric_text() -> Path:
    return PROJECT_ROOT / "analysis" / "evaluate_numeric_text_combos_field_sets"


def eval_dir_field_sets() -> Path:
    return PROJECT_ROOT / "analysis" / "evaluate_metadata_field_sets"


def parse_fields_str(s: str) -> List[str]:
    """
    Parse the 'fields' column into a list of field names.

    Handles:
      - 'a|b|c'
      - 'a;b;c'
      - 'a + b + c'

    by normalizing all delimiters to ';' and splitting.
    """
    if s is None:
        return []
    text = str(s)
    # Normalize known delimiters to ';'
    for delim in ("|", "+"):
        text = text.replace(delim, ";")
    parts = [p.strip() for p in text.split(";") if p.strip()]
    return parts


def discover_cmc_k_values(df: pd.DataFrame) -> List[int]:
    """Find cmc@K columns and return sorted K values."""
    ks: List[int] = []
    for col in df.columns:
        if col.startswith("cmc@"):
            try:
                k = int(col.split("@", 1)[1])
                ks.append(k)
            except Exception:
                continue
    if not ks:
        raise ValueError(
            "No 'cmc@K' columns found.\n"
            f"Available columns: {list(df.columns)}"
        )
    return sorted(set(ks))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    # --- Load combo results ---
    combos_csv = eval_dir_numeric_text() / "numeric_text_combos_results.csv"
    if not combos_csv.exists():
        raise FileNotFoundError(
            f"Could not find numeric_text_combos_results.csv at {combos_csv}.\n"
            "Run evaluate_numeric_text_combos_field_sets.py first."
        )
    combos_df = pd.read_csv(combos_csv)
    combos_df["kind"] = "combo"
    combos_df["parsed_fields"] = combos_df["fields"].apply(parse_fields_str)

    print(f"[Info] Loaded {len(combos_df)} combo rows from {combos_csv}")

    # Filter combos: drop any that contain excluded fields
    def combo_allowed(fields: List[str]) -> bool:
        return not any(f in EXCLUDE_FIELDS for f in fields)

    combos_df = combos_df[combos_df["parsed_fields"].apply(combo_allowed)].copy()
    print(
        f"[Info] After excluding combos with any of {EXCLUDE_FIELDS}, "
        f"{len(combos_df)} combo rows remain."
    )

    # --- Load single-field results from original field_sets_results.csv ---
    fs_csv = eval_dir_field_sets() / "field_sets_results.csv"
    if not fs_csv.exists():
        raise FileNotFoundError(
            f"Could not find field_sets_results.csv at {fs_csv}.\n"
            "Run evaluate_metadata_field_sets.py first."
        )
    fs_df = pd.read_csv(fs_csv)
    print(f"[Info] Loaded {len(fs_df)} field-set rows from {fs_csv}")

    fs_df["parsed_fields"] = fs_df["fields"].apply(parse_fields_str)
    fs_df["n_fields"] = fs_df["parsed_fields"].apply(len)

    # singles = rows where there is exactly one field
    singles = fs_df[fs_df["n_fields"] == 1].copy()
    if "selection_cmc" not in singles.columns:
        raise ValueError("field_sets_results.csv has no 'selection_cmc' column.")

    singles["field_name"] = singles["parsed_fields"].apply(
        lambda xs: xs[0] if xs else None
    )

    # filter singles by EXCLUDE_FIELDS
    singles_allowed = singles[~singles["field_name"].isin(EXCLUDE_FIELDS)].copy()
    singles_allowed["kind"] = "single"

    print(
        f"[Info] After excluding singles with any of {EXCLUDE_FIELDS}, "
        f"{len(singles_allowed)} single-field rows remain."
    )

    # --- Align K values between combos & singles (intersection of cmc@K columns) ---
    ks_combos = set(discover_cmc_k_values(combos_df))
    ks_singles = set(discover_cmc_k_values(singles_allowed))
    ks_common = sorted(ks_combos & ks_singles)
    if not ks_common:
        raise RuntimeError(
            "No overlapping CMC@K columns between combos and singles.\n"
            f"Combos K:   {sorted(ks_combos)}\n"
            f"Singles K:  {sorted(ks_singles)}"
        )

    print(f"[Info] Using common K values for CMC: {ks_common}")
    cmc_cols = [f"cmc@{k}" for k in ks_common]

    # Ensure selection_cmc / selection_k exist (for info only)
    if "selection_cmc" not in combos_df.columns:
        raise ValueError(
            "numeric_text_combos_results.csv has no 'selection_cmc' column."
        )
    if "selection_k" not in combos_df.columns:
        raise ValueError(
            "numeric_text_combos_results.csv has no 'selection_k' column."
        )
    if "selection_cmc" not in singles_allowed.columns:
        raise ValueError("field_sets_results.csv has no 'selection_cmc' column.")
    if "selection_k" not in singles_allowed.columns:
        raise ValueError("field_sets_results.csv has no 'selection_k' column.")

    # For plotting names, make them compact and label singles vs combos
    def make_plot_name_single(row: pd.Series) -> str:
        field = row["field_name"]
        return f"single: {field}"

    def make_plot_name_combo(row: pd.Series) -> str:
        return str(row["name"])

    singles_allowed["plot_name"] = singles_allowed.apply(make_plot_name_single, axis=1)
    combos_df["plot_name"] = combos_df.apply(make_plot_name_combo, axis=1)

    # For info: selection_k of combos (not used for ranking anymore)
    selection_k_info = int(combos_df["selection_k"].iloc[0])
    print(f"[Info] selection_k stored in combos (for info): {selection_k_info}")

    # --- Determine ranking K (prefer 5, else largest common K) ---
    if 5 in ks_common:
        rank_k = 5
    else:
        rank_k = ks_common[-1]
    rank_col = f"cmc@{rank_k}"
    print(f"[Info] Ranking field sets by {rank_col} (CMC@{rank_k}).")

    # --- Merge singles + combos and rank them by CMC@rank_k ---
    common_cols = [
        "name",
        "fields",
        "selection_k",
        "selection_cmc",
        "kind",
        "plot_name",
    ] + cmc_cols

    singles_for_merge = singles_allowed[common_cols].copy()
    combos_for_merge = combos_df[common_cols].copy()

    all_df = pd.concat([singles_for_merge, combos_for_merge], ignore_index=True)

    all_df_sorted = all_df.sort_values(rank_col, ascending=False).reset_index(drop=True)

    top_df = all_df_sorted.head(min(TOP_N_FIELDSETS, len(all_df_sorted)))

    print(
        f"\n[Info] Top {len(top_df)} field sets by {rank_col} "
        f"(singles + combos):"
    )
    for i, (_, row) in enumerate(top_df.iterrows(), start=1):
        print(
            f"{i:2d}. {row['kind']:6s}  "
            f"{rank_col}={row[rank_col]:.4f}  "
            f"(selection_cmc={row['selection_cmc']:.4f})  "
            f"name='{row['plot_name']}'  fields={row['fields']}"
        )

    # --- Plot CMC(K) curves for top field sets ---
    fig, ax = plt.subplots(figsize=(14, 7))

    # Choose a colormap and sample as many colors as we have curves
    cmap = plt.colormaps.get_cmap("nipy_spectral")  # newer API
    n_curves = len(top_df)
    colors = cmap(np.linspace(0, 1, n_curves))

    for idx, (_, row) in enumerate(top_df.iterrows()):
        ys = [row[f"cmc@{k}"] for k in ks_common]
        label = row["plot_name"]
        ax.plot(
            ks_common,
            ys,
            marker="o",
            linewidth=1.5,
            markersize=4,
            color=colors[idx],
            label=label,
        )

    ax.set_xlabel("Comparison Rank K")
    ax.set_ylabel("Cumulative Matching Characteristics@K")
    ax.grid(True, linestyle=":", alpha=0.5)

    if len(ks_common) <= 40:
        ax.set_xticks(ks_common[::4])

    ax.legend(
        title=f"Top {TOP_N_FIELDSETS} Metadata Combinations (ranked by CMC@{rank_k})",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    out_path = eval_dir_numeric_text() / "cmc_numeric_text_combos_with_singles.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n[Plot] Wrote combined CMC plot to: {out_path}")


if __name__ == "__main__":
    main()
