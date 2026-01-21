# src/data/rename_id.py
"""
Rename a Gallery or Query ID (folder + CSV rows).

This module provides functionality to rename an ID by:
  - Moving the folder archive/<target>/<old_id> to archive/<target>/<new_id>
  - Rewriting all CSV rows to replace old_id with new_id
  - Invalidating relevant caches
"""
from __future__ import annotations

import csv
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List

from .archive_paths import root_for, metadata_csv_for, id_column_name
from .id_registry import id_exists, invalidate_id_cache
from .image_index import invalidate_image_cache
from .validators import validate_id
from .csv_io import normalize_id_value

log = logging.getLogger("starBoard.data.rename_id")


@dataclass
class RenameReport:
    """Result of a rename_id() operation."""
    old_id: str
    new_id: str
    success: bool
    folder_renamed: bool
    csv_rows_updated: int
    errors: List[str]


def rename_id(target: str, old_id: str, new_id: str) -> RenameReport:
    """
    Rename a Gallery or Query ID.
    
    Steps:
      1. Validate new_id
      2. Check old_id exists and new_id doesn't
      3. Rename folder: archive/<target>/<old_id> → <new_id>
      4. Rewrite CSV: replace old_id → new_id in all rows
      5. Invalidate caches
    
    Parameters
    ----------
    target : str
        "Gallery" or "Queries"
    old_id : str
        The current ID to rename
    new_id : str
        The new ID name
    
    Returns
    -------
    RenameReport
        Contains details about the operation including any errors.
    """
    errors: List[str] = []
    folder_renamed = False
    csv_rows_updated = 0
    
    old_id = normalize_id_value(old_id)
    new_id = normalize_id_value(new_id)
    
    # Validate new ID
    v = validate_id(new_id)
    if not v.ok:
        return RenameReport(old_id, new_id, False, False, 0, [v.message])
    
    # Check old_id exists
    if not id_exists(target, old_id):
        return RenameReport(old_id, new_id, False, False, 0, [f"'{old_id}' does not exist."])
    
    # Check new_id doesn't exist
    if id_exists(target, new_id):
        return RenameReport(old_id, new_id, False, False, 0, [f"'{new_id}' already exists."])
    
    # Rename folder
    root = root_for(target)
    old_folder = root / old_id
    new_folder = root / new_id
    
    if old_folder.exists():
        try:
            shutil.move(str(old_folder), str(new_folder))
            folder_renamed = True
            log.info("Renamed folder %s → %s", old_folder, new_folder)
        except Exception as e:
            errors.append(f"Failed to rename folder: {e}")
            return RenameReport(old_id, new_id, False, False, 0, errors)
    
    # Rewrite CSV (replace old_id with new_id in all rows)
    csv_path, header = metadata_csv_for(target)
    id_col = id_column_name(target)
    
    if csv_path.exists():
        try:
            # Read all rows
            with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or header
                rows = list(reader)
            
            # Update rows where ID matches
            for row in rows:
                if normalize_id_value(row.get(id_col, "")) == old_id:
                    row[id_col] = new_id
                    csv_rows_updated += 1
            
            # Write back atomically via temp file
            tmp = csv_path.with_suffix(".csv.tmp")
            with tmp.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            tmp.replace(csv_path)
            
            log.info("Updated %d CSV rows for %s → %s", csv_rows_updated, old_id, new_id)
        except Exception as e:
            errors.append(f"Failed to update CSV: {e}")
    
    # Invalidate caches
    invalidate_id_cache()
    invalidate_image_cache()
    
    success = folder_renamed and len(errors) == 0
    
    log.info(
        "rename_id target=%s old=%s new=%s folder=%s rows=%d errors=%d success=%s",
        target, old_id, new_id, folder_renamed, csv_rows_updated, len(errors), success
    )
    
    return RenameReport(old_id, new_id, success, folder_renamed, csv_rows_updated, errors)
