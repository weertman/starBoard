# src/data/metadata_history.py
"""
Metadata change history tracking for gallery identities.

Records all metadata actions (creates, updates, imports) with timestamps
to maintain a complete audit trail of metadata changes for each gallery ID.

Similar pattern to location_history.py and merge_yes.py history tracking.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from .archive_paths import gallery_root, metadata_csv_for
from .csv_io import read_rows, last_row_per_id

log = logging.getLogger("starBoard.data.metadata_history")


# =============================================================================
# CONSTANTS
# =============================================================================

METADATA_HISTORY_FILENAME = "_metadata_history.csv"

METADATA_HISTORY_HEADER = [
    "timestamp_utc",    # ISO timestamp when change occurred
    "action",           # create | update | bulk_update | morphometric_import | merge_import | revert
    "field_name",       # Field that changed (or "multiple" for bulk)
    "old_value",        # Previous value (empty for create)
    "new_value",        # New value
    "source",           # ui | morphometric_tool | merge | batch_upload
    "source_ref",       # Optional reference (query_id, mFolder path, etc.)
    "snapshot_json",    # Optional: full state snapshot for bulk updates
]

# Actions that can be recorded
ACTION_CREATE = "create"
ACTION_UPDATE = "update"
ACTION_BULK_UPDATE = "bulk_update"
ACTION_MORPHOMETRIC_IMPORT = "morphometric_import"
ACTION_MERGE_IMPORT = "merge_import"
ACTION_REVERT = "revert"

# Sources of metadata changes
SOURCE_UI = "ui"
SOURCE_MORPHOMETRIC_TOOL = "morphometric_tool"
SOURCE_MERGE = "merge"
SOURCE_BATCH_UPLOAD = "batch_upload"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MetadataChange:
    """A single metadata change record."""
    timestamp_utc: str
    action: str
    field_name: str
    old_value: str
    new_value: str
    source: str
    source_ref: str = ""
    snapshot_json: str = ""

    def to_row(self) -> Dict[str, str]:
        """Convert to CSV row dict."""
        return {
            "timestamp_utc": self.timestamp_utc,
            "action": self.action,
            "field_name": self.field_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "source": self.source,
            "source_ref": self.source_ref,
            "snapshot_json": self.snapshot_json,
        }

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "MetadataChange":
        """Create from CSV row dict."""
        return cls(
            timestamp_utc=(row.get("timestamp_utc") or "").strip(),
            action=(row.get("action") or "").strip(),
            field_name=(row.get("field_name") or "").strip(),
            old_value=(row.get("old_value") or "").strip(),
            new_value=(row.get("new_value") or "").strip(),
            source=(row.get("source") or "").strip(),
            source_ref=(row.get("source_ref") or "").strip(),
            snapshot_json=(row.get("snapshot_json") or "").strip(),
        )


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def _history_path(gallery_id: str) -> Path:
    """Get the metadata history CSV path for a gallery individual."""
    return gallery_root() / gallery_id / METADATA_HISTORY_FILENAME


def _ensure_header(path: Path) -> None:
    """Ensure CSV file exists with header."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_HISTORY_HEADER)
        writer.writeheader()


def _append_row(path: Path, row: Dict[str, str]) -> None:
    """Append a single row to CSV."""
    _ensure_header(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_HISTORY_HEADER)
        writer.writerow(row)


def _append_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    """Append multiple rows to CSV."""
    if not rows:
        return
    _ensure_header(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_HISTORY_HEADER)
        for row in rows:
            writer.writerow(row)


def _utc_now() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


# =============================================================================
# HELPER: GET CURRENT METADATA STATE
# =============================================================================

def get_current_metadata_for_gallery(gallery_id: str) -> Dict[str, str]:
    """
    Get current metadata state for a gallery ID from the main CSV.
    
    Args:
        gallery_id: The gallery individual ID
        
    Returns:
        Dict of field_name -> value for the latest metadata row,
        or empty dict if no metadata exists.
    """
    csv_path, _ = metadata_csv_for("Gallery")
    if not csv_path.exists():
        return {}
    rows = read_rows(csv_path)
    latest = last_row_per_id(rows, "gallery_id")
    return latest.get(gallery_id, {})


# =============================================================================
# PUBLIC API: RECORDING CHANGES
# =============================================================================

def record_metadata_change(
    gallery_id: str,
    action: str,
    field_name: str,
    old_value: str,
    new_value: str,
    source: str,
    source_ref: str = "",
) -> None:
    """
    Record a single metadata field change.
    
    Args:
        gallery_id: The gallery individual ID
        action: Type of action (create, update, etc.)
        field_name: Name of the field that changed
        old_value: Previous value
        new_value: New value
        source: Source of the change (ui, morphometric_tool, etc.)
        source_ref: Optional reference info
    """
    change = MetadataChange(
        timestamp_utc=_utc_now(),
        action=action,
        field_name=field_name,
        old_value=old_value or "",
        new_value=new_value or "",
        source=source,
        source_ref=source_ref,
    )
    
    path = _history_path(gallery_id)
    _append_row(path, change.to_row())
    log.debug("Recorded metadata change: %s.%s = '%s' -> '%s' (%s)",
              gallery_id, field_name, old_value, new_value, action)


def record_bulk_update(
    gallery_id: str,
    old_values: Dict[str, str],
    new_values: Dict[str, str],
    source: str,
    source_ref: str = "",
) -> int:
    """
    Record a bulk metadata update, tracking each changed field.
    
    Compares old_values and new_values, recording individual change rows
    for each field that differs. Also stores a full snapshot.
    
    Args:
        gallery_id: The gallery individual ID
        old_values: Previous metadata state (field_name -> value)
        new_values: New metadata state (field_name -> value)
        source: Source of the change
        source_ref: Optional reference info
        
    Returns:
        Number of fields that changed
    """
    timestamp = _utc_now()
    
    # Find all fields that changed
    all_fields = set(old_values.keys()) | set(new_values.keys())
    changed_rows: List[Dict[str, str]] = []
    
    # Skip the ID column from change tracking
    skip_fields = {"gallery_id", "query_id"}
    
    for field in all_fields:
        if field in skip_fields:
            continue
            
        old_val = (old_values.get(field) or "").strip()
        new_val = (new_values.get(field) or "").strip()
        
        if old_val != new_val:
            change = MetadataChange(
                timestamp_utc=timestamp,
                action=ACTION_UPDATE,
                field_name=field,
                old_value=old_val,
                new_value=new_val,
                source=source,
                source_ref=source_ref,
            )
            changed_rows.append(change.to_row())
    
    if not changed_rows:
        log.debug("record_bulk_update: No changes detected for %s", gallery_id)
        return 0
    
    # Add a snapshot row for bulk updates (for state reconstruction)
    # Only include non-empty values in snapshot
    snapshot_data = {k: v for k, v in new_values.items() if v and k not in skip_fields}
    snapshot_row = MetadataChange(
        timestamp_utc=timestamp,
        action=ACTION_BULK_UPDATE,
        field_name="multiple",
        old_value="",
        new_value=f"{len(changed_rows)} fields changed",
        source=source,
        source_ref=source_ref,
        snapshot_json=json.dumps(snapshot_data, ensure_ascii=False),
    ).to_row()
    
    # Write snapshot first, then individual changes
    path = _history_path(gallery_id)
    _append_row(path, snapshot_row)
    _append_rows(path, changed_rows)
    
    log.info("Recorded bulk update for %s: %d fields changed (%s)",
             gallery_id, len(changed_rows), source)
    
    return len(changed_rows)


def record_morphometric_import(
    gallery_id: str,
    old_values: Dict[str, str],
    new_values: Dict[str, str],
    mfolder_path: str = "",
) -> int:
    """
    Record metadata changes from morphometric tool import.
    
    Convenience wrapper around record_bulk_update with morphometric-specific
    action type and source.
    
    Args:
        gallery_id: The gallery individual ID
        old_values: Previous metadata state
        new_values: New metadata state (with morph_* fields populated)
        mfolder_path: Path to the source mFolder
        
    Returns:
        Number of fields that changed
    """
    timestamp = _utc_now()
    
    # Filter to only morph_* fields for this action type
    skip_fields = {"gallery_id", "query_id"}
    changed_rows: List[Dict[str, str]] = []
    
    all_fields = set(old_values.keys()) | set(new_values.keys())
    
    for field in all_fields:
        if field in skip_fields:
            continue
            
        old_val = (old_values.get(field) or "").strip()
        new_val = (new_values.get(field) or "").strip()
        
        if old_val != new_val:
            # Use morphometric_import action for morph_* fields
            action = ACTION_MORPHOMETRIC_IMPORT if field.startswith("morph_") else ACTION_UPDATE
            change = MetadataChange(
                timestamp_utc=timestamp,
                action=action,
                field_name=field,
                old_value=old_val,
                new_value=new_val,
                source=SOURCE_MORPHOMETRIC_TOOL,
                source_ref=mfolder_path,
            )
            changed_rows.append(change.to_row())
    
    if not changed_rows:
        return 0
    
    # Snapshot with all values
    snapshot_data = {k: v for k, v in new_values.items() if v and k not in skip_fields}
    snapshot_row = MetadataChange(
        timestamp_utc=timestamp,
        action=ACTION_MORPHOMETRIC_IMPORT,
        field_name="multiple",
        old_value="",
        new_value=f"{len(changed_rows)} fields changed",
        source=SOURCE_MORPHOMETRIC_TOOL,
        source_ref=mfolder_path,
        snapshot_json=json.dumps(snapshot_data, ensure_ascii=False),
    ).to_row()
    
    path = _history_path(gallery_id)
    _append_row(path, snapshot_row)
    _append_rows(path, changed_rows)
    
    log.info("Recorded morphometric import for %s: %d fields from %s",
             gallery_id, len(changed_rows), mfolder_path)
    
    return len(changed_rows)


def record_merge_import(
    gallery_id: str,
    query_id: str,
    merged_values: Dict[str, str],
) -> None:
    """
    Record metadata import from a merged query.
    
    Args:
        gallery_id: The gallery individual ID receiving the merge
        query_id: The query ID being merged
        merged_values: Metadata values from the query
    """
    timestamp = _utc_now()
    
    skip_fields = {"gallery_id", "query_id"}
    non_empty = {k: v for k, v in merged_values.items() 
                 if v and v.strip() and k not in skip_fields}
    
    if not non_empty:
        log.debug("record_merge_import: No non-empty values to record for %s <- %s",
                  gallery_id, query_id)
        return
    
    # Record snapshot of merged values
    snapshot_row = MetadataChange(
        timestamp_utc=timestamp,
        action=ACTION_MERGE_IMPORT,
        field_name="multiple",
        old_value="",
        new_value=f"Merged from {query_id}: {len(non_empty)} fields",
        source=SOURCE_MERGE,
        source_ref=query_id,
        snapshot_json=json.dumps(non_empty, ensure_ascii=False),
    ).to_row()
    
    path = _history_path(gallery_id)
    _append_row(path, snapshot_row)
    
    log.info("Recorded merge import for %s <- %s: %d fields",
             gallery_id, query_id, len(non_empty))


# =============================================================================
# PUBLIC API: READING HISTORY
# =============================================================================

def get_metadata_history(gallery_id: str) -> List[MetadataChange]:
    """
    Load all metadata change records for a gallery individual.
    
    Args:
        gallery_id: The gallery individual ID
        
    Returns:
        List of MetadataChange objects, ordered chronologically (oldest first)
    """
    path = _history_path(gallery_id)
    if not path.exists():
        return []
    
    rows = read_rows(path)
    changes = [MetadataChange.from_row(r) for r in rows]
    
    # Sort by timestamp (should already be in order, but ensure it)
    changes.sort(key=lambda c: c.timestamp_utc)
    return changes


def get_field_history(gallery_id: str, field_name: str) -> List[MetadataChange]:
    """
    Get history for a specific field.
    
    Args:
        gallery_id: The gallery individual ID
        field_name: Name of the field to get history for
        
    Returns:
        List of MetadataChange objects for that field, chronologically ordered
    """
    all_changes = get_metadata_history(gallery_id)
    return [c for c in all_changes if c.field_name == field_name]


def get_latest_snapshot(gallery_id: str) -> Optional[Dict[str, str]]:
    """
    Get the most recent full state snapshot for a gallery individual.
    
    Looks for the most recent bulk_update or morphometric_import entry
    that contains a snapshot_json.
    
    Args:
        gallery_id: The gallery individual ID
        
    Returns:
        Dict of field values from the snapshot, or None if no snapshot exists
    """
    changes = get_metadata_history(gallery_id)
    
    # Find the most recent entry with a snapshot
    for change in reversed(changes):
        if change.snapshot_json:
            try:
                return json.loads(change.snapshot_json)
            except json.JSONDecodeError:
                log.warning("Failed to parse snapshot JSON for %s", gallery_id)
                continue
    
    return None


def get_all_snapshots(gallery_id: str) -> List[tuple]:
    """
    Get all snapshots with their timestamps.
    
    Args:
        gallery_id: The gallery individual ID
        
    Returns:
        List of (timestamp_utc, snapshot_dict) tuples, chronologically ordered
    """
    changes = get_metadata_history(gallery_id)
    snapshots = []
    
    for change in changes:
        if change.snapshot_json:
            try:
                data = json.loads(change.snapshot_json)
                snapshots.append((change.timestamp_utc, data))
            except json.JSONDecodeError:
                continue
    
    return snapshots


def has_metadata_history(gallery_id: str) -> bool:
    """Check if a gallery individual has any metadata history."""
    path = _history_path(gallery_id)
    return path.exists()


def list_galleries_with_metadata_history() -> List[str]:
    """
    Get all gallery IDs that have metadata history files.
    
    Returns:
        List of gallery IDs with history, sorted alphabetically
    """
    root = gallery_root()
    if not root.exists():
        return []
    
    galleries = []
    for p in root.iterdir():
        if p.is_dir() and (p / METADATA_HISTORY_FILENAME).exists():
            galleries.append(p.name)
    
    return sorted(galleries)
