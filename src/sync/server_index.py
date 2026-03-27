"""
Server-side SQLite index for the starBoard sync system.

Provides a fast queryable index of the archive for catalog browsing,
filtered pulls, and deduplication. Lives at archive/_sync/sync_index.db.

The index is a DERIVED cache — the filesystem archive is always the
source of truth. The index can be fully rebuilt at any time.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field as dc_field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("starBoard.sync.index")

# Image file extensions (case-insensitive matching)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Encounter folder pattern: MM_DD_YY with optional suffix
_MMDDYY_RE = re.compile(r"^(\d{2})_(\d{2})_(\d{2})(?:_.*)?$")

# Directories/files to skip when scanning entity folders
_SKIP_NAMES = {
    "_embeddings", "_dl_precompute", "_batch_metadata",
    "_sync", "vocabularies", "reports", "logs",
    "__pycache__",
}

# ─── Schema ─────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS gallery_ids (
    id TEXT PRIMARY KEY,
    source_lab TEXT DEFAULT '',
    last_modified_utc TEXT DEFAULT '',
    encounter_count INTEGER DEFAULT 0,
    image_count INTEGER DEFAULT 0,
    locations TEXT DEFAULT '[]',   -- JSON array
    date_range_start TEXT DEFAULT '',
    date_range_end TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS encounters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,       -- 'gallery' or 'query'
    entity_id TEXT NOT NULL,
    encounter_folder TEXT NOT NULL,
    date TEXT DEFAULT '',
    location TEXT DEFAULT '',
    source_lab TEXT DEFAULT '',
    image_count INTEGER DEFAULT 0,
    total_bytes INTEGER DEFAULT 0,
    UNIQUE(entity_type, entity_id, encounter_folder)
);

CREATE TABLE IF NOT EXISTS query_ids (
    id TEXT PRIMARY KEY,
    source_lab TEXT DEFAULT '',
    last_modified_utc TEXT DEFAULT '',
    encounter_count INTEGER DEFAULT 0,
    image_count INTEGER DEFAULT 0,
    location TEXT DEFAULT '',
    date TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS images (
    path TEXT PRIMARY KEY,           -- relative to archive root
    sha256 TEXT DEFAULT '',          -- empty until computed (lazy)
    size_bytes INTEGER DEFAULT 0,
    entity_type TEXT NOT NULL,       -- 'gallery' or 'query'
    entity_id TEXT NOT NULL,
    encounter_folder TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS sync_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,            -- 'push', 'pull', 'rebuild'
    lab_id TEXT DEFAULT '',
    user_email TEXT DEFAULT '',
    details TEXT DEFAULT '{}'        -- JSON
);

CREATE INDEX IF NOT EXISTS idx_encounters_entity
    ON encounters(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_images_entity
    ON images(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_images_sha256
    ON images(sha256) WHERE sha256 != '';
"""


# ─── Helpers ────────────────────────────────────────────────────────────────

def _parse_encounter_date(folder_name: str) -> Optional[date]:
    """Parse MM_DD_YY folder name to a date, or None."""
    m = _MMDDYY_RE.match(folder_name)
    if not m:
        return None
    mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(2000 + yy, mm, dd)
    except ValueError:
        return None


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _is_entity_dir(name: str) -> bool:
    """True if this directory name looks like an entity (not a system dir)."""
    if name.startswith("_") or name.startswith("."):
        return False
    if name in _SKIP_NAMES:
        return False
    return True


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Index class ────────────────────────────────────────────────────────────

class SyncIndex:
    """SQLite-backed index of the starBoard archive."""

    def __init__(self, archive_root: Path):
        self.archive_root = archive_root.resolve()
        self.db_path = self.archive_root / "_sync" / "sync_index.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA_SQL)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Full rebuild ────────────────────────────────────────────────────

    def rebuild_index(self, compute_hashes: bool = False) -> Dict[str, Any]:
        """
        Full scan of the archive. Drops and repopulates all tables
        (except sync_log which is append-only).

        Args:
            compute_hashes: If True, compute SHA-256 for every image.
                            Slow on large archives (~4k images = minutes).
                            Default False — hashes computed lazily on push.

        Returns summary stats dict.
        """
        log.info("Rebuilding sync index for %s", self.archive_root)

        # Load CSV metadata for source_lab / last_modified_utc / location
        gallery_meta = self._load_csv_metadata("gallery")
        query_meta = self._load_csv_metadata("queries")

        c = self.conn
        c.execute("DELETE FROM gallery_ids")
        c.execute("DELETE FROM query_ids")
        c.execute("DELETE FROM encounters")
        c.execute("DELETE FROM images")

        stats = {
            "gallery_ids": 0, "query_ids": 0,
            "encounters": 0, "images": 0,
            "total_bytes": 0,
        }

        # Scan gallery
        gallery_root = self.archive_root / "gallery"
        if gallery_root.exists():
            for entity_dir in sorted(gallery_root.iterdir()):
                if entity_dir.is_dir() and _is_entity_dir(entity_dir.name):
                    self._index_entity(
                        "gallery", entity_dir, gallery_meta,
                        compute_hashes, stats,
                    )

        # Scan queries (handle both spellings)
        for q_name in ("queries", "querries"):
            q_root = self.archive_root / q_name
            if q_root.exists():
                for entity_dir in sorted(q_root.iterdir()):
                    if entity_dir.is_dir() and _is_entity_dir(entity_dir.name):
                        self._index_entity(
                            "query", entity_dir, query_meta,
                            compute_hashes, stats,
                        )

        c.commit()

        # Log the rebuild
        self._log_action("rebuild", details=stats)

        log.info(
            "Index rebuilt: %d gallery, %d query, %d encounters, %d images (%.1f GB)",
            stats["gallery_ids"], stats["query_ids"],
            stats["encounters"], stats["images"],
            stats["total_bytes"] / (1024**3),
        )
        return stats

    def _index_entity(
        self,
        entity_type: str,
        entity_dir: Path,
        csv_meta: Dict[str, Dict[str, str]],
        compute_hashes: bool,
        stats: Dict[str, Any],
    ) -> None:
        """Index a single gallery or query entity."""
        entity_id = entity_dir.name
        meta = csv_meta.get(entity_id, {})

        source_lab = meta.get("source_lab", "")
        last_modified = meta.get("last_modified_utc", "")
        location = meta.get("location", "")

        total_images = 0
        total_bytes = 0
        encounter_count = 0
        dates: List[date] = []
        locations: List[str] = []

        if location:
            locations.append(location)

        # Scan encounter subfolders
        for child in sorted(entity_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name.startswith("_") or child.name.startswith("."):
                continue

            enc_date = _parse_encounter_date(child.name)
            enc_images = 0
            enc_bytes = 0

            for img_path in child.iterdir():
                if img_path.is_file() and _is_image(img_path):
                    size = img_path.stat().st_size
                    enc_images += 1
                    enc_bytes += size

                    sha = ""
                    if compute_hashes:
                        sha = _sha256_file(img_path)

                    rel_path = str(img_path.relative_to(self.archive_root))
                    self.conn.execute(
                        """INSERT OR REPLACE INTO images
                           (path, sha256, size_bytes, entity_type, entity_id, encounter_folder)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (rel_path, sha, size, entity_type, entity_id, child.name),
                    )

            if enc_images > 0:
                encounter_count += 1
                total_images += enc_images
                total_bytes += enc_bytes
                if enc_date:
                    dates.append(enc_date)

                self.conn.execute(
                    """INSERT OR REPLACE INTO encounters
                       (entity_type, entity_id, encounter_folder, date, location, source_lab,
                        image_count, total_bytes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (entity_type, entity_id, child.name,
                     enc_date.isoformat() if enc_date else "",
                     location, source_lab, enc_images, enc_bytes),
                )

        # Insert entity summary
        date_start = min(dates).isoformat() if dates else ""
        date_end = max(dates).isoformat() if dates else ""

        if entity_type == "gallery":
            self.conn.execute(
                """INSERT OR REPLACE INTO gallery_ids
                   (id, source_lab, last_modified_utc, encounter_count,
                    image_count, locations, date_range_start, date_range_end)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (entity_id, source_lab, last_modified, encounter_count,
                 total_images, json.dumps(locations), date_start, date_end),
            )
            stats["gallery_ids"] += 1
        else:
            self.conn.execute(
                """INSERT OR REPLACE INTO query_ids
                   (id, source_lab, last_modified_utc, encounter_count,
                    image_count, location, date)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (entity_id, source_lab, last_modified, encounter_count,
                 total_images, location, date_start),
            )
            stats["query_ids"] += 1

        stats["encounters"] += encounter_count
        stats["images"] += total_images
        stats["total_bytes"] += total_bytes

    def _load_csv_metadata(self, target: str) -> Dict[str, Dict[str, str]]:
        """Load metadata CSV into a dict keyed by entity ID."""
        try:
            import sys
            sys.path.insert(0, str(self.archive_root.parent))
            from src.data.archive_paths import metadata_csv_paths_for_read, id_column_name
            from src.data.csv_io import read_rows_multi, last_row_per_id

            paths = metadata_csv_paths_for_read(target)
            rows = read_rows_multi(paths)
            id_col = id_column_name(target)
            return last_row_per_id(rows, id_col)
        except Exception as e:
            log.warning("Could not load CSV metadata for %s: %s", target, e)
            return {}

    # ── Incremental update ──────────────────────────────────────────────

    def update_index_for_entity(
        self, entity_type: str, entity_id: str,
        compute_hashes: bool = False,
    ) -> Dict[str, Any]:
        """
        Re-index a single entity (after push or local change).
        Deletes old rows for this entity and re-scans its directory.
        """
        c = self.conn

        # Remove old data for this entity
        c.execute(
            "DELETE FROM encounters WHERE entity_type=? AND entity_id=?",
            (entity_type, entity_id),
        )
        c.execute(
            "DELETE FROM images WHERE entity_type=? AND entity_id=?",
            (entity_type, entity_id),
        )
        if entity_type == "gallery":
            c.execute("DELETE FROM gallery_ids WHERE id=?", (entity_id,))
        else:
            c.execute("DELETE FROM query_ids WHERE id=?", (entity_id,))

        # Find the entity directory
        if entity_type == "gallery":
            entity_dir = self.archive_root / "gallery" / entity_id
            csv_meta = self._load_csv_metadata("gallery")
        else:
            # Check both query dir spellings
            entity_dir = self.archive_root / "queries" / entity_id
            if not entity_dir.exists():
                entity_dir = self.archive_root / "querries" / entity_id
            csv_meta = self._load_csv_metadata("queries")

        stats = {
            "gallery_ids": 0, "query_ids": 0,
            "encounters": 0, "images": 0,
            "total_bytes": 0,
        }

        if entity_dir.exists():
            self._index_entity(entity_type, entity_dir, csv_meta, compute_hashes, stats)

        c.commit()
        return stats

    # ── Catalog query ───────────────────────────────────────────────────

    def query_catalog(
        self,
        locations: Optional[List[str]] = None,
        labs: Optional[List[str]] = None,
        date_after: Optional[str] = None,
        date_before: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return a browsable catalog summary, optionally filtered.

        Returns:
            {
                "gallery": [{id, encounter_count, image_count, locations, date_range, source_lab}, ...],
                "queries": [{id, image_count, location, date, source_lab}, ...],
                "all_locations": [...],
                "all_labs": [...],
                "date_range": {earliest, latest},
                "totals": {gallery_ids, query_ids, images, encounters}
            }
        """
        c = self.conn

        # ── Gallery ──
        gallery_sql = "SELECT * FROM gallery_ids WHERE 1=1"
        params: List[Any] = []

        if labs:
            placeholders = ",".join("?" * len(labs))
            gallery_sql += f" AND source_lab IN ({placeholders})"
            params.extend(labs)
        if date_after:
            gallery_sql += " AND date_range_end >= ?"
            params.append(date_after)
        if date_before:
            gallery_sql += " AND date_range_start <= ?"
            params.append(date_before)

        gallery_sql += " ORDER BY id"
        gallery_rows = c.execute(gallery_sql, params).fetchall()

        # Filter by location (locations column is JSON array)
        gallery_results = []
        for row in gallery_rows:
            row_locations = json.loads(row["locations"]) if row["locations"] else []
            if locations:
                if not any(loc in row_locations for loc in locations):
                    continue
            gallery_results.append({
                "id": row["id"],
                "source_lab": row["source_lab"],
                "encounter_count": row["encounter_count"],
                "image_count": row["image_count"],
                "locations": row_locations,
                "date_range_start": row["date_range_start"],
                "date_range_end": row["date_range_end"],
            })

        # ── Queries ──
        query_sql = "SELECT * FROM query_ids WHERE 1=1"
        params = []

        if labs:
            placeholders = ",".join("?" * len(labs))
            query_sql += f" AND source_lab IN ({placeholders})"
            params.extend(labs)
        if locations:
            placeholders = ",".join("?" * len(locations))
            query_sql += f" AND location IN ({placeholders})"
            params.extend(locations)
        if date_after:
            query_sql += " AND date >= ?"
            params.append(date_after)
        if date_before:
            query_sql += " AND date <= ?"
            params.append(date_before)

        query_sql += " ORDER BY id"
        query_rows = c.execute(query_sql, params).fetchall()
        query_results = [{
            "id": row["id"],
            "source_lab": row["source_lab"],
            "encounter_count": row["encounter_count"],
            "image_count": row["image_count"],
            "location": row["location"],
            "date": row["date"],
        } for row in query_rows]

        # ── Aggregates ──
        all_locations = set()
        all_labs = set()
        all_dates = []

        for g in gallery_results:
            all_locations.update(g["locations"])
            if g["source_lab"]:
                all_labs.add(g["source_lab"])
            if g["date_range_start"]:
                all_dates.append(g["date_range_start"])
            if g["date_range_end"]:
                all_dates.append(g["date_range_end"])

        for q in query_results:
            if q["location"]:
                all_locations.add(q["location"])
            if q["source_lab"]:
                all_labs.add(q["source_lab"])
            if q["date"]:
                all_dates.append(q["date"])

        total_images = (
            sum(g["image_count"] for g in gallery_results)
            + sum(q["image_count"] for q in query_results)
        )
        total_encounters = (
            sum(g["encounter_count"] for g in gallery_results)
            + sum(q["encounter_count"] for q in query_results)
        )

        return {
            "gallery": gallery_results,
            "queries": query_results,
            "all_locations": sorted(all_locations - {""}),
            "all_labs": sorted(all_labs - {""}),
            "date_range": {
                "earliest": min(all_dates) if all_dates else "",
                "latest": max(all_dates) if all_dates else "",
            },
            "totals": {
                "gallery_ids": len(gallery_results),
                "query_ids": len(query_results),
                "images": total_images,
                "encounters": total_encounters,
            },
        }

    # ── Package manifest ────────────────────────────────────────────────

    def build_package_manifest(
        self,
        gallery_ids: Optional[List[str]] = None,
        query_ids: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        labs: Optional[List[str]] = None,
        date_after: Optional[str] = None,
        date_before: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a manifest of files matching the given filters.
        Used by the pull endpoint to package files for download.

        Returns:
            {
                "files": [{"path": ..., "size_bytes": ...}, ...],
                "file_count": N,
                "total_bytes": N,
                "entities": {"gallery": [...], "queries": [...]},
            }
        """
        c = self.conn

        # Determine which entities match the filters
        matched_gallery: List[str] = []
        matched_queries: List[str] = []

        if gallery_ids:
            matched_gallery.extend(gallery_ids)
        if query_ids:
            matched_queries.extend(query_ids)

        # If no explicit IDs, use the catalog filters to find matching entities
        if not gallery_ids and not query_ids:
            catalog = self.query_catalog(
                locations=locations, labs=labs,
                date_after=date_after, date_before=date_before,
            )
            matched_gallery = [g["id"] for g in catalog["gallery"]]
            matched_queries = [q["id"] for q in catalog["queries"]]

        # Collect image files for matched entities
        files = []
        total_bytes = 0

        for eid in matched_gallery:
            enc_filter = self._encounter_filter(
                "gallery", eid, date_after, date_before,
            )
            for enc_folder in enc_filter:
                rows = c.execute(
                    """SELECT path, size_bytes FROM images
                       WHERE entity_type='gallery' AND entity_id=?
                       AND encounter_folder=?""",
                    (eid, enc_folder),
                ).fetchall()
                for r in rows:
                    files.append({"path": r["path"], "size_bytes": r["size_bytes"]})
                    total_bytes += r["size_bytes"]

        for eid in matched_queries:
            enc_filter = self._encounter_filter(
                "query", eid, date_after, date_before,
            )
            for enc_folder in enc_filter:
                rows = c.execute(
                    """SELECT path, size_bytes FROM images
                       WHERE entity_type='query' AND entity_id=?
                       AND encounter_folder=?""",
                    (eid, enc_folder),
                ).fetchall()
                for r in rows:
                    files.append({"path": r["path"], "size_bytes": r["size_bytes"]})
                    total_bytes += r["size_bytes"]

        return {
            "files": files,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "entities": {
                "gallery": matched_gallery,
                "queries": matched_queries,
            },
        }

    def _encounter_filter(
        self, entity_type: str, entity_id: str,
        date_after: Optional[str], date_before: Optional[str],
    ) -> List[str]:
        """Return encounter folder names for an entity, optionally date-filtered."""
        sql = """SELECT encounter_folder, date FROM encounters
                 WHERE entity_type=? AND entity_id=?"""
        params: List[Any] = [entity_type, entity_id]

        if date_after:
            sql += " AND date >= ?"
            params.append(date_after)
        if date_before:
            sql += " AND date <= ?"
            params.append(date_before)

        rows = self.conn.execute(sql, params).fetchall()
        return [r["encounter_folder"] for r in rows]

    # ── Hash computation (lazy) ─────────────────────────────────────────

    def compute_hash_for_image(self, rel_path: str) -> str:
        """Compute and store SHA-256 for a single image."""
        abs_path = self.archive_root / rel_path
        if not abs_path.exists():
            return ""
        sha = _sha256_file(abs_path)
        self.conn.execute(
            "UPDATE images SET sha256=? WHERE path=?", (sha, rel_path),
        )
        self.conn.commit()
        return sha

    def find_image_by_hash(self, sha256: str) -> Optional[str]:
        """Return the relative path of an image with this hash, or None."""
        row = self.conn.execute(
            "SELECT path FROM images WHERE sha256=? LIMIT 1", (sha256,),
        ).fetchone()
        return row["path"] if row else None

    # ── Sync log ────────────────────────────────────────────────────────

    def _log_action(
        self, action: str,
        lab_id: str = "",
        user_email: str = "",
        details: Any = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO sync_log (timestamp, action, lab_id, user_email, details)
               VALUES (?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), action,
             lab_id, user_email, json.dumps(details or {})),
        )
        self.conn.commit()

    def get_sync_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent sync log entries."""
        rows = self.conn.execute(
            "SELECT * FROM sync_log ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
