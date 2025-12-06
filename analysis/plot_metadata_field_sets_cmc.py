#!/usr/bin/env python
"""
plot_metadata_field_sets_cmc.py

Read analysis/evaluate_metadata_field_sets/field_sets_results.csv and
produce a multi-curve CMC plot, one curve per metadata field-set
configuration that was evaluated.

- X axis: rank K (taken from columns like cmc@1, cmc@2, ...).
- Y axis: CMC(K) = fraction of evaluated queries whose correct ID
  appears within the top-K ranking for that field set.

Styling:
- Field sets whose name starts with "[single]" are drawn with a dashed
  line; combined field sets use solid lines.
- The "[single] num_apparent_arms" curve is highlighted with a much
  thicker line and larger markers so it stands out clearly.

Outputs:
    analysis/evaluate_metadata_field_sets/cmc_all_field_sets.png

This script is self-contained and requires no command-line arguments.
Just run it from anywhere:

    python analysis/plot_metadata_field_sets_cmc.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths / I/O
# ---------------------------------------------------------------------


def project_root() -> Path:
    """
    Assume this file lives at <project_root>/analysis/plot_metadata_field_sets_cmc.py
    so project_root is two levels up from __file__.
    """
    return Path(__file__).resolve().parents[1]


def results_dir() -> Path:
    return project_root() / "analysis" / "evaluate_metadata_field_sets"


def results_csv_path() -> Path:
    return results_dir() / "field_sets_results.csv"


# ---------------------------------------------------------------------
# Load and interpret results
# ---------------------------------------------------------------------


def _extract_k_from_col(col: str, prefix: str) -> int | None:
    """
    Given something like 'cmc@10' or 'map@5', return 10 or 5 for the
    matching prefix, else None.
    """
    if not col.startswith(prefix):
        return None
    try:
        return int(col.split("@", 1)[1])
    except Exception:
        return None


def load_results() -> tuple[pd.DataFrame, List[int]]:
    """
    Load the field_sets_results.csv and discover all cmc@K columns.

    Returns
    -------
    df : DataFrame
        The full results table.
    k_values : list[int]
        Sorted list of K values for which CMC is available.
    """
    csv_path = results_csv_path()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected results CSV at {csv_path}, "
            "but it does not exist. Run evaluate_metadata_field_sets.py first."
        )

    df = pd.read_csv(csv_path)

    # Find all cmc@K columns
    cmc_cols: List[Tuple[int, str]] = []
    for col in df.columns:
        k = _extract_k_from_col(col, "cmc@")
        if k is not None:
            cmc_cols.append((k, col))

    if not cmc_cols:
        raise ValueError(
            f"Results CSV at {csv_path} does not contain any 'cmc@K' columns.\n"
            f"Found columns: {list(df.columns)}"
        )

    # Sort by K and return the K list
    cmc_cols.sort(key=lambda t: t[0])
    k_values = [k for k, _ in cmc_cols]

    return df, k_values


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def plot_all_cmc(df: pd.DataFrame, k_values: List[int]) -> Path:
    """
    Plot CMC(K) for all field-set configurations.

    Each row in df is a field set; we expect columns 'name' and 'cmc@K'.
    """
    out_dir = results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cmc_all_field_sets.png"

    # Only keep rows that actually evaluated at least one query
    if "num_eval_queries" in df.columns:
        df_plot = df[df["num_eval_queries"] > 0].copy()
    else:
        df_plot = df.copy()

    if df_plot.empty:
        raise RuntimeError("No rows with num_eval_queries > 0 to plot CMC for.")

    # Build list of cmc@K column names in K order
    cmc_cols = [f"cmc@{k}" for k in k_values]

    # Sanity check: all cmc@K columns should exist
    missing = [c for c in cmc_cols if c not in df_plot.columns]
    if missing:
        raise ValueError(
            f"Missing expected CMC columns in results CSV: {missing}\n"
            f"Available columns: {list(df_plot.columns)}"
        )

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot one curve per field set
    for _, row in df_plot.iterrows():
        name = str(row.get("name", "unnamed")).strip()
        y_vals = [row[c] for c in cmc_cols]

        # Skip completely NaN series (matplotlib would give a legend entry but no line)
        if all(pd.isna(v) for v in y_vals):
            continue

        # Style logic
        is_single = name.startswith("[single]")
        is_target = "num_apparent_arms" in name  # e.g. "[single] num_apparent_arms"

        # Line style: dashed for singles, solid otherwise
        linestyle = "--" if is_single else "-"

        # Base widths / marker sizes
        base_lw = 1.8 if is_single else 1.4
        base_ms = 4.0

        # Make target curve very bold & on top
        if is_target:
            linewidth = 3.2
            markersize = 7.0
            zorder = 10
        else:
            linewidth = base_lw
            markersize = base_ms
            zorder = 5 if is_single else 4

        ax.plot(
            k_values,
            y_vals,
            marker="o",
            linewidth=linewidth,
            markersize=markersize,
            linestyle=linestyle,
            label=name,
            zorder=zorder,
        )

    ax.set_xlabel("Rank K")
    ax.set_ylabel("CMC(K)")
    ax.set_title("CMC curves for all metadata field sets")

    ax.set_xticks(k_values[::2])
    ax.grid(True, linestyle="--", alpha=0.4)

    # Put the legend outside the plot so long names fit
    ax.legend(
        title="Field set",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])  # leave space on the right for legend
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[Plot] Wrote CMC plot for all field sets to: {out_path}")
    return out_path


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def main() -> None:
    df, k_values = load_results()
    print(
        f"[Info] Loaded {len(df)} field-set configurations with "
        f"CMC evaluated at K = {k_values}"
    )
    plot_all_cmc(df, k_values)


if __name__ == "__main__":
    main()
