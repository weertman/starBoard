# src/data/promote_query.py
"""
Promote a Query to a new Gallery identity.

Use this when you discover that a Query represents a unique individual
not yet in the Gallery. This module:
  - Copies encounter folders from queries/<query_id>/ to gallery/<new_id>/
  - Copies metadata row from queries_metadata.csv to gallery_metadata.csv
  - Marks the original query as silent (hidden from interactive tabs)
  - Invalidates relevant caches
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import shutil
import logging

from .archive_paths import (
    gallery_root, queries_root, roots_for_read,
    metadata_csv_paths_for_read, metadata_csv_for,
    GALLERY_HEADER, QUERIES_HEADER,
)
from .csv_io import (
    read_rows_multi, last_row_per_id, append_row, normalize_id_value
)
from .id_registry import id_exists, invalidate_id_cache
from .image_index import invalidate_image_cache
from .validators import validate_id, validate_mmddyy_string
from .silence import set_silent_query

log = logging.getLogger("starBoard.data.promote_query")


@dataclass
class PromoteReport:
    """Result of a promote_query_to_gallery() operation."""
    query_id: str
    gallery_id: str
    num_encounter_dirs: int
    metadata_copied: bool
    errors: List[str]
    success: bool


def _list_encounter_dirs(query_id: str) -> List[Path]:
    """
    List encounter directories (MM_DD_YY*) under any query root for the given query_id.
    Reuses the same logic as merge_yes.py.
    """
    out: List[Path] = []
    seen: set[str] = set()
    for root in roots_for_read("Queries"):
        qroot = root / query_id
        if not qroot.exists():
            continue
        for child in sorted(p for p in qroot.iterdir() if p.is_dir()):
            name = child.name
            if validate_mmddyy_string(name).ok:
                key = str(child.resolve()) if child.exists() else str(child)
                if key not in seen:
                    out.append(child)
                    seen.add(key)
    return out


def _ensure_unique_dir(dest: Path) -> Path:
    """Return a unique directory path by suffixing ' (n)' if needed."""
    if not dest.exists():
        return dest
    base = dest.name
    parent = dest.parent
    n = 1
    while True:
        cand = parent / f"{base} ({n})"
        if not cand.exists():
            return cand
        n += 1


def promote_query_to_gallery(
    query_id: str,
    new_gallery_id: Optional[str] = None,
    copy_metadata: bool = True
) -> PromoteReport:
    """
    Promote a Query to a new Gallery identity.
    
    Parameters
    ----------
    query_id : str
        The query ID to promote.
    new_gallery_id : str, optional
        The new gallery ID. If None, uses query_id as the gallery ID.
    copy_metadata : bool
        If True, copy metadata row from queries CSV to gallery CSV.
    
    Returns
    -------
    PromoteReport
        Contains details about the operation including any errors.
    """
    query_id = normalize_id_value(query_id)
    gallery_id = normalize_id_value(new_gallery_id) if new_gallery_id else query_id
    
    errors: List[str] = []
    num_encounter_dirs = 0
    metadata_copied = False
    
    # Validate inputs
    if not query_id:
        errors.append("Query ID cannot be empty.")
        return PromoteReport(
            query_id=query_id, gallery_id=gallery_id,
            num_encounter_dirs=0, metadata_copied=False,
            errors=errors, success=False
        )
    
    v = validate_id(gallery_id)
    if not v.ok:
        errors.append(f"Invalid gallery ID: {v.message}")
        return PromoteReport(
            query_id=query_id, gallery_id=gallery_id,
            num_encounter_dirs=0, metadata_copied=False,
            errors=errors, success=False
        )
    
    # Check query exists
    if not id_exists("Queries", query_id):
        errors.append(f"Query '{query_id}' does not exist.")
        return PromoteReport(
            query_id=query_id, gallery_id=gallery_id,
            num_encounter_dirs=0, metadata_copied=False,
            errors=errors, success=False
        )
    
    # Check gallery ID doesn't already exist
    if id_exists("Gallery", gallery_id):
        errors.append(f"Gallery ID '{gallery_id}' already exists. Choose a different name.")
        return PromoteReport(
            query_id=query_id, gallery_id=gallery_id,
            num_encounter_dirs=0, metadata_copied=False,
            errors=errors, success=False
        )
    
    # Get encounter directories to copy
    encounter_dirs = _list_encounter_dirs(query_id)
    if not encounter_dirs:
        errors.append(f"Query '{query_id}' has no encounter directories to copy.")
        return PromoteReport(
            query_id=query_id, gallery_id=gallery_id,
            num_encounter_dirs=0, metadata_copied=False,
            errors=errors, success=False
        )
    
    # Create gallery folder and copy encounters
    dest_base = gallery_root() / gallery_id
    try:
        dest_base.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Failed to create gallery folder: {e}")
        return PromoteReport(
            query_id=query_id, gallery_id=gallery_id,
            num_encounter_dirs=0, metadata_copied=False,
            errors=errors, success=False
        )
    
    for enc_dir in encounter_dirs:
        dest_dir = _ensure_unique_dir(dest_base / enc_dir.name)
        try:
            shutil.copytree(enc_dir, dest_dir)
            num_encounter_dirs += 1
            log.info("Copied %s -> %s", enc_dir, dest_dir)
        except Exception as e:
            errors.append(f"Failed to copy {enc_dir.name}: {e}")
    
    # Copy metadata if requested
    if copy_metadata:
        try:
            # Read query metadata
            q_csv_paths = metadata_csv_paths_for_read("Queries")
            q_rows = read_rows_multi(q_csv_paths)
            q_by_id = last_row_per_id(q_rows, "query_id")
            q_meta = q_by_id.get(normalize_id_value(query_id), {})
            
            if q_meta:
                # Build gallery row from query row
                g_csv_path, g_header = metadata_csv_for("Gallery")
                g_row = {}
                
                # Copy all fields except the ID column
                for col in g_header:
                    if col == "gallery_id":
                        g_row[col] = gallery_id
                    elif col in q_meta:
                        g_row[col] = q_meta[col]
                    elif col == "query_id":
                        # Skip query_id in gallery
                        continue
                    else:
                        # Copy from query if field name matches
                        g_row[col] = q_meta.get(col, "")
                
                append_row(g_csv_path, g_header, g_row)
                metadata_copied = True
                log.info("Copied metadata for %s -> %s", query_id, gallery_id)
            else:
                log.warning("No metadata found for query %s", query_id)
        except Exception as e:
            errors.append(f"Failed to copy metadata: {e}")
    
    # Mark query as silent
    try:
        set_silent_query(
            query_id,
            reason="promoted-to-gallery",
            notes=f"Promoted to gallery ID: {gallery_id}"
        )
        log.info("Marked query %s as silent", query_id)
    except Exception as e:
        errors.append(f"Failed to mark query as silent: {e}")
    
    # Invalidate caches
    invalidate_id_cache()
    invalidate_image_cache()
    
    success = num_encounter_dirs > 0 and len(errors) == 0
    
    log.info(
        "promote_query_to_gallery query_id=%s gallery_id=%s dirs=%d metadata=%s errors=%d success=%s",
        query_id, gallery_id, num_encounter_dirs, metadata_copied, len(errors), success
    )
    
    return PromoteReport(
        query_id=query_id,
        gallery_id=gallery_id,
        num_encounter_dirs=num_encounter_dirs,
        metadata_copied=metadata_copied,
        errors=errors,
        success=success
    )
