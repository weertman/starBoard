#!/usr/bin/env python
"""
plot_best_filtered_field_set_cmc.py

Make a CMC(K) vs K plot for **all** metadata field sets that pass
certain biological/metadata constraints:

  - 'Last location' must be excluded
  - all numeric fields except 'num_apparent_arms' must be excluded

This reads the results from:

    analysis/evaluate_metadata_field_sets/field_sets_results.csv

(which was produced by evaluate_metadata_field_sets.py) and then:

  1. Parses the "fields" column into a list of field names.
  2. Discards any field sets that contain:
       - "Last location", or
       - any numeric field in NUMERIC_FIELDS except "num_apparent_arms".
  3. Optionally discards sets with too few fields (MIN_FIELDS).
  4. Among the remaining rows, identifies the "best" by selection_cmc
     (CMC@selection_k) for reporting.
  5. Plots CMC(K) vs K for **all remaining field sets** on one figure,
     similar to cmc_all_field_sets.png but restricted to the filtered set.

Output:
    analysis/evaluate_metadata_field_sets/cmc_filtered_field_sets.png

Usage:
    python analysis/plot_best_filtered_field_set_cmc.py

No command-line arguments needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# ------------- locate project root & import NUMERIC_FIELDS -------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.search.engine import NUMERIC_FIELDS  # type: ignore


# ------------- configuration -------------

# If you ONLY want "combinations" (not single fields), set this to 2.
# If you're OK with single-field sets as well, leave as 1.
MIN_FIELDS = 1


# ------------- helpers -------------


def eval_dir() -> Path:
    """Directory where evaluate_metadata_field_sets outputs live."""
    return PROJECT_ROOT / "analysis" / "evaluate_metadata_field_sets"


def results_csv_path() -> Path:
    """Full path to field_sets_results.csv."""
    return eval_dir() / "field_sets_results.csv"


def parse_fields_str(s: str) -> List[str]:
    """
    Parse a field list string into a list of field names.

    We accept both 'a|b|c' and 'a;b;c' style delimiters, and strip whitespace.
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


# ------------- main plotting logic -------------


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

    # Parse fields into lists
    df["parsed_fields"] = df["fields"].apply(parse_fields_str)
    df["n_fields"] = df["parsed_fields"].apply(len)

    # Set of numeric fields that must be excluded, except num_apparent_arms
    excluded_numeric = {f for f in NUMERIC_FIELDS if f != "num_apparent_arms"}

    def allow_row(fields: List[str]) -> bool:
        # Exclude if "Last location" present
        if "Last location" in fields:
            return False
        # Exclude if any numeric field (except num_apparent_arms) present
        for f in fields:
            if f in excluded_numeric:
                return False
        return True

    mask_allowed = df["parsed_fields"].apply(allow_row)
    df_allowed = df[mask_allowed].copy()

    # Require minimum number of fields (if you truly only want "combos")
    df_allowed = df_allowed[df_allowed["n_fields"] >= MIN_FIELDS]

    print(
        f"[Info] After exclusions + MIN_FIELDS={MIN_FIELDS}, "
        f"{len(df_allowed)} field sets remain."
    )

    if df_allowed.empty:
        raise RuntimeError(
            "No field sets left after applying exclusions. "
            "Check your constraints or MIN_FIELDS."
        )

    # Pick best by selection_cmc (CMC at selection_k) just for reporting
    if "selection_cmc" not in df_allowed.columns:
        raise ValueError(
            "field_sets_results.csv has no 'selection_cmc' column. "
            "You may be using an older version of evaluate_metadata_field_sets.py."
        )

    best_idx = df_allowed["selection_cmc"].idxmax()
    best_row = df_allowed.loc[best_idx]

    best_name = str(best_row["name"])
    best_fields = best_row["parsed_fields"]
    selection_k = int(best_row["selection_k"])
    best_cmc = float(best_row["selection_cmc"])

    print("\n=== Best allowed field set (by selection_cmc) ===")
    print(f"Name        : {best_name}")
    print(f"Fields ({len(best_fields)}): {', '.join(best_fields)}")
    print(f"selection_k : {selection_k}")
    print(f"CMC@{selection_k:>2d}   : {best_cmc:.4f}\n")

    # Build CMC(K) curves for ALL allowed field sets
    k_values = discover_cmc_k_values(df_allowed)
    cmc_cols = [f"cmc@{k}" for k in k_values]

    # Sanity check
    missing_cols = [c for c in cmc_cols if c not in df_allowed.columns]
    if missing_cols:
        raise ValueError(
            f"Missing expected CMC columns {missing_cols} in field_sets_results.csv."
        )

    fig, ax = plt.subplots(figsize=(9, 7))

    for _, row in df_allowed.iterrows():
        name = str(row["name"])
        ys = [float(row[c]) for c in cmc_cols]
        ax.plot(k_values, ys, marker="o", linewidth=1.5, markersize=4, label=name)

    ax.set_xlabel("Rank K")
    ax.set_ylabel("CMC(K)")
    ax.set_title(
        "CMC curves for filtered metadata field sets\n"
        "(no 'Last location', numeric fields only 'num_apparent_arms')"
    )
    ax.grid(True, linestyle=":", alpha=0.5)

    # Ticks
    if len(k_values) <= 20:
        ax.set_xticks(k_values)

    # Legend on the right to keep plot area clear
    ax.legend(
        title="Field set",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    out_path = out_dir / "cmc_filtered_field_sets.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[Plot] Wrote filtered multi-curve CMC plot to: {out_path}")


if __name__ == "__main__":
    main()

