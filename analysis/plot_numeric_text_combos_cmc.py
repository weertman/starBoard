#!/usr/bin/env python
"""
plot_numeric_text_combos_cmc.py

Visualize CMC(K) vs K for all metadata field-set *combinations* that:

  - include at least one numeric field (only 'num_apparent_arms' allowed),
  - include at least one text field (from TEXT_FIELDS),
  - do NOT include 'Last location',
  - do NOT include any other numeric field besides 'num_apparent_arms',
  - have at least 2 fields (true "combos").

Input (must already exist, produced by evaluate_metadata_field_sets.py):
    analysis/evaluate_metadata_field_sets/field_sets_results.csv

Output:
    analysis/evaluate_metadata_field_sets/cmc_numeric_text_combos.png

Usage:
    python analysis/plot_numeric_text_combos_cmc.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import sys

# -------------------- locate project root & import field lists --------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.search.engine import NUMERIC_FIELDS, TEXT_FIELDS  # type: ignore


# ----------------------------- configuration -----------------------------

# Require combos (not singles)
MIN_FIELDS = 2

# If you want to cap how many combos you plot, set this to an int.
# If None, plot ALL matching numeric+text combos.
TOP_N_COMBOS: int | None = None


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

    Accept both 'a|b|c' and 'a;b;c' style delimiters; strip whitespace.
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


# ----------------------------- main logic -----------------------------


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

    # Parse fields and count
    df["parsed_fields"] = df["fields"].apply(parse_fields_str)
    df["n_fields"] = df["parsed_fields"].apply(len)

    # Numeric filtering
    allowed_numeric = {"num_apparent_arms"}
    excluded_numeric = {f for f in NUMERIC_FIELDS if f not in allowed_numeric}

    print(f"[Info] NUMERIC_FIELDS from engine: {NUMERIC_FIELDS}")
    print(f"[Info] Allowed numeric: {sorted(allowed_numeric)}")
    print(f"[Info] Excluded numeric: {sorted(excluded_numeric)}")
    print(f"[Info] TEXT_FIELDS from engine: {TEXT_FIELDS}")

    def allow_row(fields: List[str]) -> bool:
        # Exclude if 'Last location' present
        if "Last location" in fields:
            return False

        # Must have at least one allowed numeric field (num_apparent_arms)
        has_allowed_numeric = any(f in allowed_numeric for f in fields)

        # Exclude if any *other* numeric field is present
        has_excluded_numeric = any(f in excluded_numeric for f in fields)
        if has_excluded_numeric:
            return False

        # Must have at least one text field (from TEXT_FIELDS), excluding 'Last location'
        has_text = any((f in TEXT_FIELDS) and (f != "Last location") for f in fields)

        # Enforce numeric+text combo
        if not has_allowed_numeric or not has_text:
            return False

        # Enforce combo size
        if len(fields) < MIN_FIELDS:
            return False

        return True

    mask_allowed = df["parsed_fields"].apply(allow_row)
    df_allowed = df[mask_allowed].copy()

    print(
        f"[Info] After filtering for numeric+text combos (and exclusions), "
        f"{len(df_allowed)} field sets remain."
    )

    if df_allowed.empty:
        raise RuntimeError(
            "No numeric+text combos left after filtering.\n"
            "Check that evaluate_metadata_field_sets.py actually created "
            "field sets that mix num_apparent_arms with text fields."
        )

    if "selection_cmc" not in df_allowed.columns:
        raise ValueError(
            "field_sets_results.csv has no 'selection_cmc' column. "
            "You may be using an older version of evaluate_metadata_field_sets.py."
        )

    # Rank by selection_cmc (CMC@selection_k)
    df_ranked = df_allowed.sort_values(
        "selection_cmc", ascending=False
    ).reset_index(drop=True)

    if TOP_N_COMBOS is not None:
        df_ranked = df_ranked.head(min(TOP_N_COMBOS, len(df_ranked)))

    selection_k = int(df_ranked["selection_k"].iloc[0])

    print(
        f"\n=== Numeric+text combos (sorted by CMC@{selection_k}) "
        f"[showing {len(df_ranked)}] ==="
    )
    for i, (_, row) in enumerate(df_ranked.iterrows(), start=1):
        name = str(row["name"])
        fields = row["parsed_fields"]
        sel_cmc = float(row["selection_cmc"])
        print(
            f"{i:2d}. {name:25s}  CMC@{selection_k:>2d}={sel_cmc:.4f}  "
            f"fields ({len(fields)}): {', '.join(fields)}"
        )
    print("")

    # Build CMC(K) curves
    k_values = discover_cmc_k_values(df_ranked)
    cmc_cols = [f"cmc@{k}" for k in k_values]

    missing = [c for c in cmc_cols if c not in df_ranked.columns]
    if missing:
        raise ValueError(
            f"Missing expected CMC columns {missing} in field_sets_results.csv."
        )

    fig, ax = plt.subplots(figsize=(9, 7))

    for _, row in df_ranked.iterrows():
        name = str(row["name"])
        ys = [float(row[c]) for c in cmc_cols]
        ax.plot(k_values, ys, marker="o", linewidth=1.5, markersize=4, label=name)

    ax.set_xlabel("Rank K")
    ax.set_ylabel("CMC(K)")
    ax.set_title(
        "CMC curves for numeric+text combo field sets\n"
        "(no 'Last location'; numeric fields restricted to 'num_apparent_arms')"
    )
    ax.grid(True, linestyle=":", alpha=0.5)

    if len(k_values) <= 20:
        ax.set_xticks(k_values)

    ax.legend(
        title="Field set (numeric+text combo)",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    out_path = out_dir / "cmc_numeric_text_combos.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[Plot] Wrote numeric+text combo CMC plot to: {out_path}")


if __name__ == "__main__":
    main()
