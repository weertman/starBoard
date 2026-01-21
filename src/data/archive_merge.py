# src/data/archive_merge.py
"""
Archive Merge: Import identities from an external archive into the current archive.

Supports two conflict resolution strategies:
  - "merge": Combine encounters for matching IDs (union mode)
  - "offset": Rename all imported IDs with prefix/suffix (namespace mode)
"""
from __future__ import annotations

import csv
import shutil
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

from .archive_paths import (
    archive_root, gallery_root, queries_root, root_for,
    metadata_csv_for, id_column_name, GALLERY_HEADER, QUERIES_HEADER,
)
from .csv_io import read_rows, append_row, normalize_id_value
from .id_registry import id_exists, invalidate_id_cache
from .image_index import invalidate_image_cache
from .validators import validate_mmddyy_string

log = logging.getLogger("starBoard.data.archive_merge")

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}


@dataclass
class MergeItem:
    """Represents a single identity to be merged."""
    source_id: str              # ID in external archive
    target_id: str              # ID after merge (may be renamed if offsetting)
    source_folder: Path         # Full path to ID folder in external archive
    encounters: List[str]       # Encounter folder names (MM_DD_YY format)
    image_count: int            # Total images across all encounters
    conflict: bool              # True if target_id already exists in destination
    action: str                 # "create" | "merge_into"
    metadata: Dict[str, str] = field(default_factory=dict)  # CSV row data if available


@dataclass
class MergePlan:
    """Complete plan for merging an external archive."""
    source_archive: Path        # Root of external archive
    target: str                 # "Gallery" | "Queries"
    strategy: str               # "merge" | "offset"
    prefix: str                 # Prefix for offset strategy
    suffix: str                 # Suffix for offset strategy
    items: List[MergeItem]      # All items to merge
    
    @property
    def new_count(self) -> int:
        """Number of new IDs to create."""
        return sum(1 for item in self.items if item.action == "create")
    
    @property
    def merge_count(self) -> int:
        """Number of existing IDs to merge into."""
        return sum(1 for item in self.items if item.action == "merge_into")
    
    @property
    def total_encounters(self) -> int:
        """Total encounter folders to copy."""
        return sum(len(item.encounters) for item in self.items)
    
    @property
    def total_images(self) -> int:
        """Total images to copy."""
        return sum(item.image_count for item in self.items)


@dataclass
class MergeReport:
    """Result of a merge operation."""
    batch_id: str
    items_processed: int
    encounters_copied: int
    csv_rows_added: int
    errors: List[str]
    dry_run: bool = False
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0


def _get_external_archive_paths(external_root: Path, target: str) -> Tuple[List[Path], List[Path]]:
    """
    Get ID root folders and CSV paths for an external archive.
    Handles both 'queries' and legacy 'querries' spelling.
    
    Returns:
        (id_roots, csv_paths) - Lists of paths that exist
    """
    id_roots: List[Path] = []
    csv_paths: List[Path] = []
    
    if target.lower() == "gallery":
        gallery_dir = external_root / "gallery"
        if gallery_dir.exists():
            id_roots.append(gallery_dir)
            csv_path = gallery_dir / "gallery_metadata.csv"
            if csv_path.exists():
                csv_paths.append(csv_path)
    else:
        # Check both spellings for queries
        for spelling in ["queries", "querries"]:
            q_dir = external_root / spelling
            if q_dir.exists():
                id_roots.append(q_dir)
                for csv_name in [f"{spelling}_metadata.csv", "queries_metadata.csv"]:
                    csv_path = q_dir / csv_name
                    if csv_path.exists():
                        csv_paths.append(csv_path)
                        break
    
    return id_roots, csv_paths


def _count_images_in_folder(folder: Path) -> int:
    """Count image files recursively in a folder."""
    count = 0
    if folder.exists():
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                count += 1
    return count


def _list_encounter_folders(id_folder: Path) -> List[str]:
    """List valid encounter folders (MM_DD_YY format) under an ID folder."""
    encounters = []
    if not id_folder.exists():
        return encounters
    for child in sorted(id_folder.iterdir()):
        if child.is_dir():
            # Validate folder name matches MM_DD_YY pattern
            v = validate_mmddyy_string(child.name)
            if v.ok:
                encounters.append(child.name)
    return encounters


def scan_external_archive(external_root: Path, target: str) -> List[MergeItem]:
    """
    Scan an external archive directory and discover IDs with their encounters.
    
    Parameters
    ----------
    external_root : Path
        Root directory of the external archive (should contain gallery/ and/or queries/)
    target : str
        "Gallery" or "Queries" - which type of IDs to scan
    
    Returns
    -------
    List[MergeItem]
        List of discovered IDs with metadata, encounters, and image counts.
        Items have source_id == target_id initially (no renaming applied yet).
    """
    items: List[MergeItem] = []
    seen_ids: Set[str] = set()
    
    id_roots, csv_paths = _get_external_archive_paths(external_root, target)
    
    if not id_roots:
        log.warning("No %s directory found in external archive: %s", target.lower(), external_root)
        return items
    
    # Load metadata from CSV if available
    id_col = id_column_name(target)
    metadata_by_id: Dict[str, Dict[str, str]] = {}
    for csv_path in csv_paths:
        rows = read_rows(csv_path)
        for row in rows:
            raw_id = row.get(id_col, "").strip()
            if raw_id:
                norm_id = normalize_id_value(raw_id)
                # Keep last row per ID (append-only semantics)
                metadata_by_id[norm_id] = row
    
    # Scan ID folders
    for id_root in id_roots:
        for child in sorted(id_root.iterdir()):
            if not child.is_dir():
                continue
            # Skip system folders
            if child.name.startswith("_"):
                continue
            
            source_id = child.name
            norm_id = normalize_id_value(source_id)
            
            # Avoid duplicates if same ID appears in multiple roots
            if norm_id in seen_ids:
                continue
            seen_ids.add(norm_id)
            
            encounters = _list_encounter_folders(child)
            image_count = _count_images_in_folder(child)
            
            # Check if this ID exists in destination
            conflict = id_exists(target, source_id)
            
            item = MergeItem(
                source_id=source_id,
                target_id=source_id,  # Will be updated by build_merge_plan if offsetting
                source_folder=child,
                encounters=encounters,
                image_count=image_count,
                conflict=conflict,
                action="merge_into" if conflict else "create",
                metadata=metadata_by_id.get(norm_id, {}),
            )
            items.append(item)
    
    log.info(
        "Scanned external archive %s for %s: found %d IDs, %d with conflicts",
        external_root, target, len(items), sum(1 for i in items if i.conflict)
    )
    return items


def build_merge_plan(
    external_root: Path,
    target: str,
    items: List[MergeItem],
    strategy: str,
    prefix: str = "",
    suffix: str = "",
) -> MergePlan:
    """
    Build a merge plan from scanned items, applying the chosen strategy.
    
    Parameters
    ----------
    external_root : Path
        Root of the external archive
    target : str
        "Gallery" or "Queries"
    items : List[MergeItem]
        Items from scan_external_archive()
    strategy : str
        "merge" - combine encounters for matching IDs
        "offset" - rename all IDs with prefix/suffix
    prefix : str
        Prefix to add to IDs (only used with "offset" strategy)
    suffix : str
        Suffix to add to IDs (only used with "offset" strategy)
    
    Returns
    -------
    MergePlan
        Complete plan ready for execution
    """
    plan_items: List[MergeItem] = []
    
    for item in items:
        if strategy == "offset":
            # Apply prefix/suffix renaming
            new_id = f"{prefix}{item.source_id}{suffix}"
            conflict = id_exists(target, new_id)
            plan_items.append(MergeItem(
                source_id=item.source_id,
                target_id=new_id,
                source_folder=item.source_folder,
                encounters=item.encounters,
                image_count=item.image_count,
                conflict=conflict,
                action="merge_into" if conflict else "create",
                metadata=item.metadata.copy(),
            ))
        else:
            # Merge strategy: keep original ID
            plan_items.append(MergeItem(
                source_id=item.source_id,
                target_id=item.source_id,
                source_folder=item.source_folder,
                encounters=item.encounters,
                image_count=item.image_count,
                conflict=item.conflict,
                action=item.action,
                metadata=item.metadata.copy(),
            ))
    
    plan = MergePlan(
        source_archive=external_root,
        target=target,
        strategy=strategy,
        prefix=prefix,
        suffix=suffix,
        items=plan_items,
    )
    
    log.info(
        "Built merge plan: strategy=%s, items=%d, new=%d, merge=%d, encounters=%d",
        strategy, len(plan_items), plan.new_count, plan.merge_count, plan.total_encounters
    )
    return plan


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


def execute_merge(plan: MergePlan, dry_run: bool = False) -> MergeReport:
    """
    Execute a merge plan, copying folders and merging CSV data.
    
    Parameters
    ----------
    plan : MergePlan
        Plan from build_merge_plan()
    dry_run : bool
        If True, don't actually copy files or modify CSVs
    
    Returns
    -------
    MergeReport
        Summary of operations performed
    """
    batch_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    errors: List[str] = []
    encounters_copied = 0
    csv_rows_added = 0
    
    dest_root = root_for(plan.target)
    csv_path, header = metadata_csv_for(plan.target)
    id_col = id_column_name(plan.target)
    
    log.info(
        "Executing merge: batch=%s target=%s items=%d dry_run=%s",
        batch_id, plan.target, len(plan.items), dry_run
    )
    
    for item in plan.items:
        target_folder = dest_root / item.target_id
        
        # Create target folder if new ID
        if not dry_run and item.action == "create":
            try:
                target_folder.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Failed to create folder {target_folder}: {e}")
                continue
        
        # Copy encounter folders
        for enc_name in item.encounters:
            src_enc = item.source_folder / enc_name
            dest_enc = target_folder / enc_name
            
            # Handle collision with existing encounter folder
            if dest_enc.exists():
                dest_enc = _ensure_unique_dir(dest_enc)
            
            if not dry_run:
                try:
                    shutil.copytree(src_enc, dest_enc)
                    encounters_copied += 1
                    log.debug("Copied %s -> %s", src_enc, dest_enc)
                except Exception as e:
                    errors.append(f"Failed to copy {src_enc} -> {dest_enc}: {e}")
            else:
                encounters_copied += 1  # Count for dry run preview
        
        # Append metadata row to CSV
        if item.metadata or item.action == "create":
            row = item.metadata.copy() if item.metadata else {}
            # Update ID column to target_id (handles renaming for offset strategy)
            row[id_col] = item.target_id
            
            if not dry_run:
                try:
                    append_row(csv_path, header, row)
                    csv_rows_added += 1
                except Exception as e:
                    errors.append(f"Failed to append CSV row for {item.target_id}: {e}")
            else:
                csv_rows_added += 1  # Count for dry run preview
    
    # Invalidate caches after merge
    if not dry_run:
        invalidate_id_cache()
        invalidate_image_cache()
        
        # Mark new IDs as pending for DL precomputation
        try:
            from src.dl.registry import DLRegistry
            registry = DLRegistry.load()
            for item in plan.items:
                registry.add_pending_id(plan.target, item.target_id)
        except Exception as e:
            log.debug("Could not mark IDs as pending for DL: %s", e)
    
    report = MergeReport(
        batch_id=batch_id,
        items_processed=len(plan.items),
        encounters_copied=encounters_copied,
        csv_rows_added=csv_rows_added,
        errors=errors,
        dry_run=dry_run,
    )
    
    log.info(
        "Merge complete: batch=%s processed=%d encounters=%d rows=%d errors=%d",
        batch_id, report.items_processed, report.encounters_copied,
        report.csv_rows_added, len(report.errors)
    )
    
    return report
