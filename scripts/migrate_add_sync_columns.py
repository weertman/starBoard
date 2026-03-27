#!/usr/bin/env python3
"""
One-time migration: add sync columns to existing metadata CSVs.

What it does:
  1. Backs up gallery_metadata.csv and queries_metadata.csv
  2. Adds the 3 sync columns (last_modified_utc, modified_by_lab, source_lab)
     to the CSV headers with empty values for existing rows
  3. Uses the existing ensure_header() mechanism — same code path
     that runs on normal app startup

Safe to run multiple times (idempotent).
Does NOT populate existing rows with sync data — only adds the columns.

Usage:
    ./scripts/python scripts/migrate_add_sync_columns.py
    ./scripts/python scripts/migrate_add_sync_columns.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.data.archive_paths import (
    archive_root,
    gallery_root,
    queries_root,
    GALLERY_HEADER,
    QUERIES_HEADER,
)
from src.data.csv_io import ensure_header


def _read_header(csv_path: Path) -> list[str]:
    """Read just the header row from a CSV."""
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def _backup(csv_path: Path, dry_run: bool) -> Path | None:
    """Create a timestamped backup of a CSV file."""
    if not csv_path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = csv_path.with_suffix(f".pre_sync_migration_{ts}.csv.bak")
    if dry_run:
        print(f"  [DRY RUN] Would back up: {csv_path.name} -> {backup.name}")
        return backup
    shutil.copy2(csv_path, backup)
    print(f"  Backed up: {csv_path.name} -> {backup.name}")
    return backup


def migrate_csv(csv_path: Path, header: list[str], label: str, dry_run: bool) -> None:
    """Migrate a single CSV to include sync columns."""
    print(f"\n--- {label}: {csv_path} ---")

    if not csv_path.exists():
        print(f"  File does not exist, skipping.")
        return

    current = _read_header(csv_path)
    current_clean = [h.replace("\ufeff", "").strip() for h in current]

    new_cols = [c for c in header if c not in current_clean]
    if not new_cols:
        print(f"  Already up to date ({len(current_clean)} columns).")
        return

    print(f"  Current columns: {len(current_clean)}")
    print(f"  Target columns:  {len(header)}")
    print(f"  New columns to add: {new_cols}")

    # Count rows
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        row_count = sum(1 for _ in csv.reader(f)) - 1  # subtract header
    print(f"  Existing data rows: {row_count}")

    _backup(csv_path, dry_run)

    if dry_run:
        print(f"  [DRY RUN] Would upgrade header (new columns get empty values)")
        return

    # Use the same mechanism the app uses
    ensure_header(csv_path, header)

    # Verify
    after = _read_header(csv_path)
    after_clean = [h.replace("\ufeff", "").strip() for h in after]
    print(f"  Upgraded: {len(current_clean)} -> {len(after_clean)} columns")

    # Verify row count preserved
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        after_rows = sum(1 for _ in csv.reader(f)) - 1
    if after_rows != row_count:
        print(f"  WARNING: Row count changed! {row_count} -> {after_rows}")
    else:
        print(f"  Row count preserved: {after_rows}")


def main():
    parser = argparse.ArgumentParser(description="Add sync columns to starBoard CSVs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    print("starBoard Sync Column Migration")
    print(f"Archive root: {archive_root()}")
    print(f"Dry run: {args.dry_run}")

    # Gallery
    gallery_csv = gallery_root() / "gallery_metadata.csv"
    migrate_csv(gallery_csv, GALLERY_HEADER, "Gallery", args.dry_run)

    # Queries (handle both spellings)
    for name in ["queries", "querries"]:
        q_dir = archive_root() / name
        if not q_dir.exists():
            continue
        q_csv = q_dir / f"{name}_metadata.csv"
        if q_csv.exists():
            migrate_csv(q_csv, QUERIES_HEADER, f"Queries ({name})", args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
