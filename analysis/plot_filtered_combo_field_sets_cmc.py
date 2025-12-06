#!/usr/bin/env python
"""
plot_filtered_combo_field_sets_cmc.py

Visualize CMC(K) vs K for the *top* metadata field-set combinations
under biologically motivated constraints:

  - Exclude any field set that contains 'Last location'.
  - Exclude any field set that contains a numeric field other than
    'num_apparent_arms'.

Then:

  - Keep only "combos" (field sets with at least 2 fields).
  - Rank those combos by selection_cmc (CMC@selection_k).
  - Plot CMC(K) vs K for the top N combos on a single figure, with
    a legend listing which combos are plotted.
  - Print the combos and their fields to the console.

Input (must already exist, produced by evaluate_metadata_field_sets.py):
    analysis/evaluate_metadata_field_sets/field_sets_results.csv

Output:
    analysis/evaluate_metadata_field_sets/cmc_filtered_combos.png

Usage:
    python analysis/plot_filtered_combo_field_sets_cmc.py

No command-line arguments needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# -------------------- locate project root & NUMERIC_FIELDS --------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.search.engine import NUMERIC_FIELDS  # type: ignore


# ------------------------- configuration knobs -------------------------

# Minimum number of fields for a "combo".
# Set to 2 to ignore single-field sets ([single] ...).
MIN_FIELDS = 2

# How many best combos (by selection_cmc) to plot.
TOP_N_COMBOS = 10


# ----------------------------- helpers -----------------------------


def eval_dir() -> Path:
    """Directory where evaluate_metadata_field_sets outputs live."""
    return PROJECT_ROOT / "analysis" / "evaluate_metadata_field_sets"


def results_csv_path() -> Path:
    """Full path to field_sets_results.csv."""
    return eval_dir() / "field_sets_results.csv"


def parse_fields_str(s: str) -> List[str]:
    """
    Parse a field list string into a list of field names.

    We accept both 'a|b|c' and 'a;b;c' style delimiters, strip whitespace.
    """
    if s is None:
        return []
    text = str(s)
    text = text.replace("|", ";")
    parts = [p.strip() for p in text.split(";") if p.strip()]
    return parts


def discover_cmc_k_values(df: pd.DataFrame) -> List[int]:
    """
    Find all cmc@K columns and return the sorted Ks.
    """
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
            "No 'cmc@K' columns found in field_sets_results.csv.\n"
            f"Available columns: {list(df.columns)}"
        )
    return sorted(set(ks))


# ---------------------------- main logic ----------------------------


def main() -> None:
    out_dir = eval_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_csv_path()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find field_sets_results.csv at {csv_path}.\n"
            "Run evaluate_metadata_field_sets.py first."
        )

    df = pd.read_csv(csv_path)
    print(f"[Info] Loaded {len(df)} field-set rows from {csv_path}")

    # Parse fields into lists & count
    df["parsed_fields"] = df["fields"].apply(parse_fields_str)
    df["n_fields"] = df["parsed_fields"].apply(len)

    # Exclude numeric fields except num_apparent_arms
    excluded_numeric = {f for f in NUMERIC_FIELDS if f != "num_apparent_arms"}
    print(f"[Info] Numeric fields in engine: {NUMERIC_FIELDS}")
    print(f"[Info] Excluding numeric fields (except num_apparent_arms): {sorted(excluded_numeric)}")

    def allow_row(fields: List[str]) -> bool:
        # Exclude if 'Last location' present
        if "Last location" in fields:
            return False
        # Exclude if any numeric field (except num_apparent_arms) present
        for f in fields:
            if f in excluded_numeric:
                return False
        return True

    mask_allowed = df["parsed_fields"].apply(allow_row)
    df_allowed = df[mask_allowed].copy()

    # Keep only combos: at least MIN_FIELDS fields.
    df_allowed = df_allowed[df_allowed["n_fields"] >= MIN_FIELDS]

    print(
        f"[Info] After exclusions + MIN_FIELDS={MIN_FIELDS}, "
        f"{len(df_allowed)} combo field sets remain."
    )

    if df_allowed.empty:
        raise RuntimeError(
            "No combo field sets left after applying exclusions. "
            "Check your constraints or lower MIN_FIELDS."
        )

    # Must have selection_cmc (CMC@selection_k summary)
    if "selection_cmc" not in df_allowed.columns:
        raise ValueError(
            "field_sets_results.csv has no 'selection_cmc' column. "
            "You may be using an older version of evaluate_metadata_field_sets.py."
        )

    # Sort combos by selection_cmc (descending) and take top N
    df_ranked = df_allowed.sort_values(
        "selection_cmc", ascending=False
    ).reset_index(drop=True)

    top = df_ranked.head(min(TOP_N_COMBOS, len(df_ranked))).copy()
    selection_k = int(top["selection_k"].iloc[0])

    print(f"\n=== Top {len(top)} combo field sets (by CMC@{selection_k}) ===")
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        name = str(row["name"])
        fields = row["parsed_fields"]
        sel_cmc = float(row["selection_cmc"])
        print(
            f"{i:2d}. {name:25s}  CMC@{selection_k:>2d}={sel_cmc:.4f}  "
            f"fields ({len(fields)}): {', '.join(fields)}"
        )
    print("")

    # Build CMC(K) curves for these combos
    k_values = discover_cmc_k_values(df_ranked)
    cmc_cols = [f"cmc@{k}" for k in k_values]

    missing = [c for c in cmc_cols if c not in df_ranked.columns]
    if missing:
        raise ValueError(
            f"Missing expected CMC columns {missing} in field_sets_results.csv."
        )

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))

    for _, row in top.iterrows():
        name = str(row["name"])
        ys = [float(row[c]) for c in cmc_cols]
        ax.plot(k_values, ys, marker="o", linewidth=1.5, markersize=4, label=name)

    ax.set_xlabel("Rank K")
    ax.set_ylabel("CMC(K)")
    ax.set_title(
        "CMC curves for top combo field sets\n"
        "(no 'Last location'; numeric fields restricted to 'num_apparent_arms')"
    )
    ax.grid(True, linestyle=":", alpha=0.5)

    if len(k_values) <= 20:
        ax.set_xticks(k_values)

    # Legend on the right
    ax.legend(
        title="Field set (combo)",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    out_path = out_dir / "cmc_filtered_combos.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[Plot] Wrote filtered combo CMC plot to: {out_path}")


if __name__ == "__main__":
    main()
