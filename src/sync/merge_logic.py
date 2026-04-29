"""
Merge logic for the starBoard sync system.

Handles three types of incoming data from field machines:
  1. Encounter folders (images) — copy + dedup by hash
  2. Metadata rows (CSV) — row-level merge by ID + timestamp
  3. Match decisions — dedup append to master log
"""
from __future__ import annotations

import csv
import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.data.image_formats import ARCHIVE_IMAGE_EXTS, is_archive_image

log = logging.getLogger("starBoard.sync.merge")

IMAGE_EXTENSIONS = ARCHIVE_IMAGE_EXTS


# ─── Result types ───────────────────────────────────────────────────────────

@dataclass
class EncounterIngestReport:
    entity_type: str = ""
    entity_id: str = ""
    encounter_folder: str = ""
    accepted_images: int = 0
    skipped_duplicates: int = 0
    new_encounter_created: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class MetadataMergeReport:
    updated_count: int = 0
    skipped_count: int = 0
    conflicts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DecisionMergeReport:
    appended_count: int = 0
    duplicate_count: int = 0


# ─── Image hashing ──────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_image(path: Path) -> bool:
    return is_archive_image(path)


# ─── Encounter ingest ───────────────────────────────────────────────────────

def ingest_encounter_files(
    archive_root: Path,
    entity_type: str,
    entity_id: str,
    encounter_folder: str,
    files: List[Tuple[str, bytes]],
    source_lab: str = "",
    existing_hashes: Optional[set] = None,
) -> EncounterIngestReport:
    """
    Ingest uploaded image files into the archive.

    Args:
        archive_root: Path to archive/
        entity_type: 'gallery' or 'query'
        entity_id: e.g. 'anchovy' or 'Q_2026_001'
        encounter_folder: e.g. '03_15_26'
        files: List of (filename, file_bytes) tuples
        source_lab: Lab that created these files
        existing_hashes: Set of SHA-256 hashes already in the archive
                         (for dedup). Pass None to skip dedup.

    Returns:
        EncounterIngestReport
    """
    report = EncounterIngestReport(
        entity_type=entity_type,
        entity_id=entity_id,
        encounter_folder=encounter_folder,
    )

    # Determine target directory
    if entity_type == "gallery":
        target_root = archive_root / "gallery"
    else:
        # Prefer 'queries' over legacy 'querries'
        target_root = archive_root / "queries"
        if not target_root.exists():
            legacy = archive_root / "querries"
            if legacy.exists():
                target_root = legacy

    entity_dir = target_root / entity_id
    encounter_dir = entity_dir / encounter_folder

    if not encounter_dir.exists():
        encounter_dir.mkdir(parents=True, exist_ok=True)
        report.new_encounter_created = True

    for filename, data in files:
        # Only accept archive-visible image files
        if not is_archive_image(filename):
            log.debug("Skipping non-image file: %s", filename)
            continue

        # Dedup by hash
        file_hash = _sha256_bytes(data)
        if existing_hashes is not None and file_hash in existing_hashes:
            report.skipped_duplicates += 1
            log.debug("Duplicate skipped (hash match): %s", filename)
            continue

        # Write file, handling name collisions
        dest = encounter_dir / filename
        if dest.exists():
            # Check if it's the same content
            if _sha256_file(dest) == file_hash:
                report.skipped_duplicates += 1
                continue
            # Different content, same name — rename with suffix
            stem = dest.stem
            suffix_ext = dest.suffix
            i = 1
            while dest.exists():
                dest = encounter_dir / f"{stem}_{i}{suffix_ext}"
                i += 1

        dest.write_bytes(data)
        report.accepted_images += 1

        # Track hash for future dedup within this batch
        if existing_hashes is not None:
            existing_hashes.add(file_hash)

    log.info(
        "Ingested encounter %s/%s/%s: %d accepted, %d dupes",
        entity_type, entity_id, encounter_folder,
        report.accepted_images, report.skipped_duplicates,
    )
    return report


# ─── Metadata merge ─────────────────────────────────────────────────────────

def merge_metadata_rows(
    archive_root: Path,
    target: str,
    client_rows: List[Dict[str, str]],
) -> MetadataMergeReport:
    """
    Merge metadata rows from a client into the server's CSV.

    Rules:
      - Compare by entity ID + last_modified_utc
      - If client row is newer: append to CSV (append-only semantics)
      - If server row is newer or equal: skip
      - source_lab is preserved (set-once)

    Args:
        archive_root: Path to archive/
        target: 'gallery' or 'queries'
        client_rows: List of row dicts from the client
    """
    import sys
    sys.path.insert(0, str(archive_root.parent))
    from src.data.archive_paths import metadata_csv_for, id_column_name
    from src.data.csv_io import read_rows, last_row_per_id, append_row

    report = MetadataMergeReport()

    csv_path, header = metadata_csv_for(target)
    id_col = id_column_name(target)

    # Load current server state
    server_rows = read_rows(csv_path)
    server_latest = last_row_per_id(server_rows, id_col)

    for client_row in client_rows:
        client_id = client_row.get(id_col, "").strip()
        if not client_id:
            continue

        client_ts = client_row.get("last_modified_utc", "")
        server_row = server_latest.get(client_id)

        if server_row:
            server_ts = server_row.get("last_modified_utc", "")

            # Both have timestamps — compare
            if server_ts and client_ts:
                if client_ts <= server_ts:
                    report.skipped_count += 1
                    continue
            elif server_ts and not client_ts:
                # Server has timestamp, client doesn't — server wins
                report.skipped_count += 1
                continue
            # If client has timestamp but server doesn't, or neither does,
            # let the client row through

        # Preserve source_lab from server if client doesn't have one
        if server_row and not client_row.get("source_lab", "").strip():
            client_row["source_lab"] = server_row.get("source_lab", "")

        # Append (csv_io.append_row stamps last_modified_utc and modified_by_lab)
        # But we want to keep the CLIENT's timestamp, not re-stamp.
        # So we write directly instead of using append_row.
        from src.data.csv_io import ensure_header
        ensure_header(csv_path, header)
        ordered = [client_row.get(col, "") for col in header]
        with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(ordered)

        report.updated_count += 1

    log.info(
        "Metadata merge for %s: %d updated, %d skipped",
        target, report.updated_count, report.skipped_count,
    )
    return report


# ─── Decision merge ─────────────────────────────────────────────────────────

def merge_decisions(
    archive_root: Path,
    client_decisions: List[Dict[str, str]],
) -> DecisionMergeReport:
    """
    Append match decisions to the master log, deduplicating by
    (query_id, gallery_id, timestamp).

    Decisions are stored in archive/reports/past_matches_master.csv.
    The existing CSV may have many columns (74+) from the starBoard app;
    we detect the existing header and write rows that match it.
    The dedup key uses 'updated_utc' (app format) or 'timestamp' (sync format).
    """
    report = DecisionMergeReport()

    master_csv = archive_root / "reports" / "past_matches_master.csv"
    master_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load existing header and decisions for dedup
    existing_header: List[str] = []
    existing_keys: set = set()

    if master_csv.exists():
        with master_csv.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                existing_header = list(reader.fieldnames)
            for row in reader:
                # The app uses 'updated_utc', sync uses 'timestamp'
                ts = (row.get("updated_utc", "") or row.get("timestamp", "")).strip()
                key = (
                    row.get("query_id", "").strip(),
                    row.get("gallery_id", "").strip(),
                    ts,
                )
                existing_keys.add(key)

    # If no existing file, create with the sync-compatible header
    if not existing_header:
        existing_header = [
            "query_id", "gallery_id", "verdict", "updated_utc",
            "notes", "lab_id", "user",
        ]
        with master_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(existing_header)

    # Map client field names to existing header field names
    # Client sends: query_id, gallery_id, decision, timestamp, lab_id, user, notes
    # App format:   query_id, gallery_id, verdict, updated_utc, notes, ...
    FIELD_MAP = {
        "decision": "verdict",
        "timestamp": "updated_utc",
    }

    # Append new decisions
    with master_csv.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for d in client_decisions:
            ts = (d.get("timestamp", "") or d.get("updated_utc", "")).strip()
            key = (
                d.get("query_id", "").strip(),
                d.get("gallery_id", "").strip(),
                ts,
            )
            if key in existing_keys:
                report.duplicate_count += 1
                continue

            # Build row matching existing header
            mapped = {}
            for k, v in d.items():
                mapped_key = FIELD_MAP.get(k, k)
                mapped[mapped_key] = v

            row_out = [mapped.get(col, "") for col in existing_header]
            writer.writerow(row_out)
            existing_keys.add(key)
            report.appended_count += 1

    log.info(
        "Decision merge: %d appended, %d duplicates",
        report.appended_count, report.duplicate_count,
    )
    return report
