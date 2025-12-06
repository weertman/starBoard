#!/usr/bin/env python
"""
plot_metadata_identity_results.py

Helper script to visualize the outputs from:

    analysis/evaluate_metadata_identity_metrics.py

Assumes you've already run that script and it has created:

    analysis/evaluate_metadata_identity_metrics/per_field_and_combos.csv
    analysis/evaluate_metadata_identity_metrics/metrics_results.csv

This script produces several plots:

1. single_fields_selection_map.png
   - Bar plot of all single TEXT fields, sorted by mAP@K (K = selection_k).

2. top_combos_size2_selection_map.png
   top_combos_size3_selection_map.png
   top_combos_size4_selection_map.png
   - Bar plots of top N combos (N configurable) for each combo size (2/3/4),
     ranked by mAP@K.

3. single_vs_combo_map_at_k.png
   - Line plot of mAP@K curves for:
       * top 4 single fields (by selection_map)
       * best size-2 combo
       * best size-3 combo
       * best size-4 combo

Usage
-----
Just run from your project root or directly in your IDE:

    python analysis/plot_metadata_identity_results.py

No command-line arguments are required.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths / I/O
# ---------------------------------------------------------------------


def _project_root() -> Path:
    # This script lives at <root>/analysis/plot_metadata_identity_results.py
    return Path(__file__).resolve().parents[1]


def _eval_dir() -> Path:
    return _project_root() / "analysis" / "evaluate_metadata_identity_metrics"


def _load_per_field_and_combos() -> pd.DataFrame:
    csv_path = _eval_dir() / "per_field_and_combos.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find per_field_and_combos.csv at {csv_path}\n"
            "Run evaluate_metadata_identity_metrics.py first."
        )
    df = pd.read_csv(csv_path)
    return df


def _discover_k_values_from_columns(df: pd.DataFrame) -> List[int]:
    """
    Look for columns named like 'mAP@1', 'mAP@5', ... and return sorted Ks.
    """
    ks: List[int] = []
    for col in df.columns:
        if col.startswith("mAP@"):
            try:
                k = int(col.split("@", 1)[1])
                ks.append(k)
            except Exception:
                pass
    if not ks:
        raise ValueError(
            "No mAP@K columns found in per_field_and_combos.csv; "
            f"got columns: {list(df.columns)}"
        )
    return sorted(set(ks))


# ---------------------------------------------------------------------
# Plot 1: Single fields bar plot (selection_map)
# ---------------------------------------------------------------------


def plot_single_fields(df: pd.DataFrame, outdir: Path) -> Path:
    """
    Make a bar plot of single fields, sorted by selection_map (mAP@selection_k).
    """
    singles = df[df["kind"] == "single"].copy()
    if singles.empty:
        raise RuntimeError("No 'single' rows found in per_field_and_combos.csv.")

    singles_sorted = singles.sort_values(
        "selection_map", ascending=False
    ).reset_index(drop=True)

    # Prepare plotting data
    labels = singles_sorted["name"].tolist()
    scores = singles_sorted["selection_map"].tolist()
    selection_k = int(singles_sorted["selection_k"].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), scores, color="tab:blue")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(f"mAP@{selection_k}")
    ax.set_title(f"Single fields ranked by mAP@{selection_k}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    outpath = outdir / "single_fields_selection_map.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    print(f"[Plot] Wrote single-fields bar plot to: {outpath}")
    return outpath


# ---------------------------------------------------------------------
# Plot 2: Top combos per size (2, 3, 4) bar plots
# ---------------------------------------------------------------------


def plot_top_combos_by_size(
    df: pd.DataFrame,
    outdir: Path,
    max_size: int = 4,
    top_n: int = 5,
) -> Dict[int, Path]:
    """
    For each combo size 2..max_size, plot top 'top_n' combos by selection_map.
    """
    paths: Dict[int, Path] = {}
    selection_k = int(df["selection_k"].iloc[0])

    for size in range(2, max_size + 1):
        combos = df[(df["kind"] == f"combo_{size}")].copy()
        if combos.empty:
            print(f"[Plot] No combos of size {size} found, skipping.")
            continue

        combos_sorted = combos.sort_values(
            "selection_map", ascending=False
        ).reset_index(drop=True)

        top = combos_sorted.head(top_n)
        labels = top["name"].tolist()
        scores = top["selection_map"].tolist()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(labels)), scores, color="tab:green")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(f"mAP@{selection_k}")
        ax.set_title(f"Top {len(top)} combos of size {size} by mAP@{selection_k}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

        outpath = outdir / f"top_combos_size{size}_selection_map.png"
        fig.savefig(outpath, dpi=150)
        plt.close(fig)

        print(f"[Plot] Wrote top combos (size={size}) bar plot to: {outpath}")
        paths[size] = outpath

    return paths


# ---------------------------------------------------------------------
# Plot 3: mAP@K curves for top 4 singles + best combos (size 2/3/4)
# ---------------------------------------------------------------------


def plot_single_vs_combo_curves(
    df: pd.DataFrame,
    k_values: List[int],
    outdir: Path,
    top_n_singles: int = 4,
) -> Path:
    """
    Plot mAP@K curves for:
      - top_n_singles single fields
      - best size-2 combo
      - best size-3 combo
      - best size-4 combo
    """
    singles = df[df["kind"] == "single"].copy()
    if singles.empty:
        raise RuntimeError("No 'single' rows found in per_field_and_combos.csv.")

    selection_k = int(df["selection_k"].iloc[0])

    # Rank singles and take top_n
    singles_sorted = singles.sort_values(
        "selection_map", ascending=False
    ).reset_index(drop=True)
    top_singles = singles_sorted.head(top_n_singles)

    # Helper to extract best combo of given size
    def best_combo_of_size(size: int) -> pd.Series | None:
        combos = df[df["kind"] == f"combo_{size}"]
        if combos.empty:
            return None
        return combos.sort_values(
            "selection_map", ascending=False
        ).iloc[0]

    best_combo2 = best_combo_of_size(2)
    best_combo3 = best_combo_of_size(3)
    best_combo4 = best_combo_of_size(4)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot singles
    for _, row in top_singles.iterrows():
        label = f"single: {row['name']}"
        ys = [row[f"mAP@{k}"] for k in k_values]
        ax.plot(k_values, ys, marker="o", linestyle="-", label=label)

    # Plot best combos
    for size, row in [(2, best_combo2), (3, best_combo3), (4, best_combo4)]:
        if row is None:
            continue
        label = f"combo{size}: {row['name']}"
        ys = [row[f"mAP@{k}"] for k in k_values]
        ax.plot(k_values, ys, marker="o", linestyle="--", label=label)

    ax.set_xlabel("K")
    ax.set_ylabel("mAP@K")
    ax.set_title(
        f"Top {len(top_singles)} singles and best combos (sizes 2–4) by mAP@{selection_k}"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    outpath = outdir / "single_vs_combo_map_at_k.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    print(f"[Plot] Wrote single-vs-combo mAP@K curves to: {outpath}")
    return outpath


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    eval_dir = _eval_dir()
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Using evaluation directory: {eval_dir}")

    df = _load_per_field_and_combos()
    k_values = _discover_k_values_from_columns(df)

    print(
        f"[Info] Loaded per_field_and_combos.csv with {len(df)} rows, "
        f"mAP evaluated at K = {k_values}"
    )

    # 1. Single-field bar plot
    plot_single_fields(df, eval_dir)

    # 2. Top combos per size (2,3,4)
    plot_top_combos_by_size(df, eval_dir, max_size=4, top_n=5)

    # 3. Single vs combo mAP@K curves
    plot_single_vs_combo_curves(df, k_values, eval_dir, top_n_singles=4)


if __name__ == "__main__":
    main()
