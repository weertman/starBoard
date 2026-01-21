# src/data/batch_undo.py
"""
Batch upload undo/redo functionality.

Tracks batch uploads and allows undoing (deleting copied files) and
redoing (re-copying from original sources).
"""
from __future__ import annotations

import csv
import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .archive_paths import archive_root, root_for, metadata_csv_for, id_column_name
from .csv_io import read_rows, append_row

log = logging.getLogger("starBoard.data.batch_undo")

HISTORY_FILENAME = "_batch_upload_history.csv"
METADATA_DIR = "_batch_metadata"

HISTORY_HEADER = [
    "batch_id",
    "timestamp_utc",
    "operation",  # upload, undo, redo, purge
    "id_str",
    "encounter_name",
    "kind",  # file, csv_row
    "src_path",
    "dest_rel",
    "id_was_new",
    "status",  # ok, removed, restored, missing, purged, error
    "note",
]


@dataclass
class BatchInfo:
    """Summary info for a batch upload."""
    batch_id: str
    timestamp: datetime
    target: str
    encounter_name: str
    id_count: int
    file_count: int
    new_ids: List[str]
    state: str  # "active", "undone", "purged"


@dataclass
class UndoReport:
    """Result of an undo operation."""
    batch_id: str
    files_removed: int = 0
    files_missing: int = 0
    csv_rows_removed: int = 0
    ids_affected: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class RedoReport:
    """Result of a redo operation."""
    batch_id: str
    files_restored: int = 0
    files_failed: int = 0
    csv_rows_restored: int = 0
    ids_affected: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _history_path(target: str) -> Path:
    """Get path to batch history CSV for a target."""
    return root_for(target) / HISTORY_FILENAME


def _metadata_dir() -> Path:
    """Get path to metadata backup directory."""
    return archive_root() / METADATA_DIR


def _metadata_path(batch_id: str) -> Path:
    """Get path to metadata JSON for a batch."""
    return _metadata_dir() / f"{batch_id}.json"


def generate_batch_id() -> str:
    """Generate unique batch ID: upload_YYYYMMDD_HHMMSS_<random>"""
    now = datetime.utcnow()
    rand = uuid.uuid4().hex[:4]
    return f"upload_{now.strftime('%Y%m%d_%H%M%S')}_{rand}"


def _read_history(target: str) -> List[Dict[str, str]]:
    """Read all history rows for a target."""
    path = _history_path(target)
    if not path.exists():
        return []
    try:
        return read_rows(path)
    except Exception as e:
        log.warning("Failed to read batch history: %s", e)
        return []


def _append_history(target: str, rows: List[Dict[str, str]]) -> None:
    """Append rows to history CSV."""
    path = _history_path(target)
    
    # Ensure file exists with header
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_HEADER)
            writer.writeheader()
    
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_HEADER)
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in HISTORY_HEADER})


def record_batch_upload(
    target: str,
    batch_id: str,
    file_ops: List[Tuple[Path, Path]],  # (src, dest) pairs
    new_ids: Set[str],
    encounter_name: str,
) -> None:
    """
    Record a batch upload to history.
    
    Args:
        target: "Gallery" or "Queries"
        batch_id: Unique batch identifier
        file_ops: List of (source_path, dest_path) tuples
        new_ids: Set of IDs that were newly created
        encounter_name: The encounter folder name
    """
    now = datetime.utcnow().isoformat() + "Z"
    target_root = root_for(target)
    rows: List[Dict[str, str]] = []
    
    # Group files by ID
    ids_seen: Set[str] = set()
    for src, dest in file_ops:
        # Extract ID from dest path (target_root / id_str / encounter / filename)
        try:
            rel = dest.relative_to(target_root)
            id_str = rel.parts[0] if rel.parts else ""
        except ValueError:
            id_str = ""
        
        ids_seen.add(id_str)
        
        rows.append({
            "batch_id": batch_id,
            "timestamp_utc": now,
            "operation": "upload",
            "id_str": id_str,
            "encounter_name": encounter_name,
            "kind": "file",
            "src_path": str(src),
            "dest_rel": str(dest.relative_to(target_root)) if dest.is_relative_to(target_root) else str(dest),
            "id_was_new": "true" if id_str in new_ids else "false",
            "status": "ok",
            "note": "",
        })
    
    # Record CSV rows for new IDs
    for id_str in new_ids:
        rows.append({
            "batch_id": batch_id,
            "timestamp_utc": now,
            "operation": "upload",
            "id_str": id_str,
            "encounter_name": encounter_name,
            "kind": "csv_row",
            "src_path": "",
            "dest_rel": "",
            "id_was_new": "true",
            "status": "ok",
            "note": "",
        })
    
    _append_history(target, rows)
    log.info("Recorded batch %s: %d files, %d new IDs", batch_id, len(file_ops), len(new_ids))


def list_batches(target: str) -> List[BatchInfo]:
    """
    List all batches for a target, most recent first.
    
    Aggregates history to determine current state of each batch.
    """
    rows = _read_history(target)
    if not rows:
        return []
    
    # Group by batch_id
    batches: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        bid = row.get("batch_id", "")
        if bid:
            batches.setdefault(bid, []).append(row)
    
    result: List[BatchInfo] = []
    for batch_id, batch_rows in batches.items():
        # Find earliest timestamp for this batch
        timestamps = [r.get("timestamp_utc", "") for r in batch_rows if r.get("timestamp_utc")]
        if not timestamps:
            continue
        try:
            ts = datetime.fromisoformat(timestamps[0].rstrip("Z"))
        except ValueError:
            ts = datetime.utcnow()
        
        # Get encounter name
        encounter = ""
        for r in batch_rows:
            if r.get("encounter_name"):
                encounter = r["encounter_name"]
                break
        
        # Count files and IDs
        file_rows = [r for r in batch_rows if r.get("kind") == "file" and r.get("operation") == "upload"]
        new_id_rows = [r for r in batch_rows if r.get("kind") == "csv_row" and r.get("id_was_new") == "true"]
        
        ids_in_batch = set(r.get("id_str", "") for r in file_rows if r.get("id_str"))
        new_ids = [r.get("id_str", "") for r in new_id_rows if r.get("id_str")]
        
        # Determine state from most recent operation
        operations = [(r.get("timestamp_utc", ""), r.get("operation", "")) for r in batch_rows]
        operations.sort(reverse=True)
        last_op = operations[0][1] if operations else "upload"
        
        if last_op == "purge":
            state = "purged"
        elif last_op == "undo":
            state = "undone"
        else:
            state = "active"
        
        result.append(BatchInfo(
            batch_id=batch_id,
            timestamp=ts,
            target=target,
            encounter_name=encounter,
            id_count=len(ids_in_batch),
            file_count=len(file_rows),
            new_ids=new_ids,
            state=state,
        ))
    
    # Sort by timestamp, most recent first
    result.sort(key=lambda b: b.timestamp, reverse=True)
    return result


def get_batch_state(target: str, batch_id: str) -> str:
    """Return 'active', 'undone', or 'purged'."""
    batches = list_batches(target)
    for b in batches:
        if b.batch_id == batch_id:
            return b.state
    return "unknown"


def _get_batch_file_entries(target: str, batch_id: str) -> List[Dict[str, str]]:
    """Get all file entries for a batch (from upload or redo operations)."""
    rows = _read_history(target)
    return [
        r for r in rows
        if r.get("batch_id") == batch_id
        and r.get("kind") == "file"
        and r.get("operation") in ("upload", "redo")
    ]


def _get_batch_new_ids(target: str, batch_id: str) -> List[str]:
    """Get list of IDs that were newly created by this batch."""
    rows = _read_history(target)
    return list(set(
        r.get("id_str", "")
        for r in rows
        if r.get("batch_id") == batch_id
        and r.get("id_was_new") == "true"
        and r.get("id_str")
    ))


def _backup_csv_rows(target: str, batch_id: str, id_list: List[str]) -> List[Dict[str, str]]:
    """
    Read and backup CSV rows for the given IDs.
    Returns the rows that were backed up.
    """
    if not id_list:
        return []
    
    csv_path, _ = metadata_csv_for(target)
    if not csv_path.exists():
        return []
    
    id_col = id_column_name(target)
    all_rows = read_rows(csv_path)
    
    # Find rows for our IDs
    backup_rows = [r for r in all_rows if r.get(id_col, "") in id_list]
    
    if backup_rows:
        # Save to metadata JSON
        meta_dir = _metadata_dir()
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = _metadata_path(batch_id)
        
        data = {
            "batch_id": batch_id,
            "target": target,
            "removed_rows": backup_rows,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        log.info("Backed up %d CSV rows for batch %s", len(backup_rows), batch_id)
    
    return backup_rows


def _remove_csv_rows(target: str, id_list: List[str]) -> int:
    """Remove rows for the given IDs from the CSV. Returns count removed."""
    if not id_list:
        return 0
    
    csv_path, header = metadata_csv_for(target)
    if not csv_path.exists():
        return 0
    
    id_col = id_column_name(target)
    all_rows = read_rows(csv_path)
    
    id_set = set(id_list)
    remaining = [r for r in all_rows if r.get(id_col, "") not in id_set]
    removed_count = len(all_rows) - len(remaining)
    
    if removed_count > 0:
        # Rewrite CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(remaining)
        log.info("Removed %d CSV rows for IDs: %s", removed_count, id_list)
    
    return removed_count


def _restore_csv_rows(target: str, batch_id: str) -> int:
    """Restore CSV rows from backup. Returns count restored."""
    meta_path = _metadata_path(batch_id)
    if not meta_path.exists():
        log.warning("No metadata backup found for batch %s", batch_id)
        return 0
    
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.error("Failed to read metadata backup: %s", e)
        return 0
    
    rows_to_restore = data.get("removed_rows", [])
    if not rows_to_restore:
        return 0
    
    csv_path, header = metadata_csv_for(target)
    
    # Append rows back
    for row in rows_to_restore:
        append_row(csv_path, header, row)
    
    log.info("Restored %d CSV rows for batch %s", len(rows_to_restore), batch_id)
    return len(rows_to_restore)


def undo_batch(target: str, batch_id: str, permanent: bool = False) -> UndoReport:
    """
    Undo a batch upload.
    
    Args:
        target: "Gallery" or "Queries"
        batch_id: The batch to undo
        permanent: If True, purge completely (no redo possible)
    
    Returns:
        UndoReport with results
    """
    report = UndoReport(batch_id=batch_id)
    
    # Check state
    state = get_batch_state(target, batch_id)
    if state == "undone":
        report.errors.append("Batch is already undone")
        return report
    if state == "purged":
        report.errors.append("Batch was permanently deleted")
        return report
    
    target_root = root_for(target)
    file_entries = _get_batch_file_entries(target, batch_id)
    new_ids = _get_batch_new_ids(target, batch_id)
    
    # Track affected IDs
    ids_affected: Set[str] = set()
    
    # Delete files
    for entry in file_entries:
        dest_rel = entry.get("dest_rel", "")
        if not dest_rel:
            continue
        
        dest_path = target_root / dest_rel
        id_str = entry.get("id_str", "")
        if id_str:
            ids_affected.add(id_str)
        
        if dest_path.exists():
            try:
                dest_path.unlink()
                report.files_removed += 1
            except Exception as e:
                report.errors.append(f"Failed to delete {dest_rel}: {e}")
        else:
            report.files_missing += 1
    
    # Backup and remove CSV rows for new IDs
    if new_ids:
        if not permanent:
            _backup_csv_rows(target, batch_id, new_ids)
        report.csv_rows_removed = _remove_csv_rows(target, new_ids)
    
    # Clean up empty directories
    for id_str in ids_affected:
        id_dir = target_root / id_str
        if id_dir.exists():
            # Check encounter directories
            for enc_dir in id_dir.iterdir():
                if enc_dir.is_dir() and not any(enc_dir.iterdir()):
                    try:
                        enc_dir.rmdir()
                        log.debug("Removed empty encounter dir: %s", enc_dir)
                    except Exception:
                        pass
            # Check if ID dir is now empty
            if id_dir.is_dir() and not any(id_dir.iterdir()):
                try:
                    id_dir.rmdir()
                    log.debug("Removed empty ID dir: %s", id_dir)
                except Exception:
                    pass
    
    report.ids_affected = list(ids_affected)
    
    # Record operation
    now = datetime.utcnow().isoformat() + "Z"
    op = "purge" if permanent else "undo"
    history_rows = [{
        "batch_id": batch_id,
        "timestamp_utc": now,
        "operation": op,
        "id_str": "",
        "encounter_name": "",
        "kind": "batch_op",
        "src_path": "",
        "dest_rel": "",
        "id_was_new": "",
        "status": "ok",
        "note": f"files_removed={report.files_removed}, csv_rows={report.csv_rows_removed}",
    }]
    _append_history(target, history_rows)
    
    # If permanent, delete metadata backup
    if permanent:
        meta_path = _metadata_path(batch_id)
        if meta_path.exists():
            try:
                meta_path.unlink()
            except Exception:
                pass
    
    log.info("Undo batch %s: %d files removed, %d missing, %d CSV rows removed",
             batch_id, report.files_removed, report.files_missing, report.csv_rows_removed)
    
    return report


def check_redo_sources(target: str, batch_id: str) -> Tuple[int, int, List[str]]:
    """
    Check how many source files still exist for redo.
    
    Returns:
        (available_count, missing_count, list_of_missing_paths)
    """
    file_entries = _get_batch_file_entries(target, batch_id)
    
    available = 0
    missing = 0
    missing_paths: List[str] = []
    
    for entry in file_entries:
        src = entry.get("src_path", "")
        if src and Path(src).exists():
            available += 1
        else:
            missing += 1
            if src:
                missing_paths.append(src)
    
    return available, missing, missing_paths


def redo_batch(target: str, batch_id: str) -> RedoReport:
    """
    Redo a previously undone batch.
    
    Re-copies files from original source paths and restores CSV rows.
    
    Args:
        target: "Gallery" or "Queries"
        batch_id: The batch to redo
    
    Returns:
        RedoReport with results
    """
    report = RedoReport(batch_id=batch_id)
    
    # Check state
    state = get_batch_state(target, batch_id)
    if state == "active":
        report.errors.append("Batch is already active")
        return report
    if state == "purged":
        report.errors.append("Batch was permanently deleted and cannot be redone")
        return report
    
    target_root = root_for(target)
    file_entries = _get_batch_file_entries(target, batch_id)
    
    ids_affected: Set[str] = set()
    
    # Re-copy files
    for entry in file_entries:
        src_path = entry.get("src_path", "")
        dest_rel = entry.get("dest_rel", "")
        id_str = entry.get("id_str", "")
        
        if not src_path or not dest_rel:
            continue
        
        src = Path(src_path)
        dest = target_root / dest_rel
        
        if id_str:
            ids_affected.add(id_str)
        
        if not src.exists():
            report.files_failed += 1
            report.errors.append(f"Source missing: {src_path}")
            continue
        
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dest))
            report.files_restored += 1
        except Exception as e:
            report.files_failed += 1
            report.errors.append(f"Failed to copy {src.name}: {e}")
    
    # Restore CSV rows
    report.csv_rows_restored = _restore_csv_rows(target, batch_id)
    report.ids_affected = list(ids_affected)
    
    # Record operation
    now = datetime.utcnow().isoformat() + "Z"
    history_rows = [{
        "batch_id": batch_id,
        "timestamp_utc": now,
        "operation": "redo",
        "id_str": "",
        "encounter_name": "",
        "kind": "batch_op",
        "src_path": "",
        "dest_rel": "",
        "id_was_new": "",
        "status": "ok",
        "note": f"files_restored={report.files_restored}, failed={report.files_failed}",
    }]
    _append_history(target, history_rows)
    
    log.info("Redo batch %s: %d files restored, %d failed, %d CSV rows restored",
             batch_id, report.files_restored, report.files_failed, report.csv_rows_restored)
    
    return report









