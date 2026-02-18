from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Iterable, Tuple
import csv
import logging

log = logging.getLogger("starBoard.data.csv")

# Lazy import to avoid circular dependency
def _invalidate_id_cache() -> None:
    """Invalidate the ID registry cache (lazy import to avoid circular dependency)."""
    try:
        from .id_registry import invalidate_id_cache
        invalidate_id_cache()
    except ImportError:
        pass

# -------- normalization helpers --------
def normalize_key(s: str) -> str:
    """Normalize CSV header keys: strip whitespace and remove BOM."""
    if s is None:
        return ""
    return s.replace("\ufeff", "").strip()

def normalize_id_value(s: str) -> str:
    """Normalize ID values for robust matching."""
    if s is None:
        return ""
    return s.replace("\ufeff", "").strip()

# Accept legacy/misspelled ID column names on READ
ID_SYNONYMS = {
    "gallery_id": ["gallery_id"],
    "query_id": ["query_id", "queries_id", "querries_id"],  # tolerate legacy spellings
}

# ---------------------------------------
def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def ensure_header(csv_path: Path, header: List[str]) -> None:
    """
    Ensure the CSV exists **and** its header is at least the provided header.
    - If file missing: create with `header`.
    - If file exists and header matches exactly: do nothing.
    - If file exists and is a **subset** of `header` (same relative order):
        rewrite the CSV with the new header, filling missing columns with "".
    - If file exists and has **unknown/extra** columns or columns in a conflicting order:
        raise ValueError (we don't drop or shuffle existing columns automatically).
    Always uses 'utf-8-sig' for Excel tolerance.
    """
    _ensure_parent_dir(csv_path)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        log.info("Created metadata CSV with header: %s", csv_path)
        return

    # Read existing header row (may be empty file)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        try:
            existing: List[str] = next(r)
        except StopIteration:
            existing = []

    if existing == header:
        return

    # Compare in normalized space (strip BOM/whitespace)
    ex_norm = [normalize_key(x) for x in existing]
    want_norm = [normalize_key(x) for x in header]

    # Legacy/extra columns in file — strip them and rewrite
    extras = [x for x in ex_norm if x and x not in want_norm]
    if extras:
        log.warning(
            "Stripping legacy columns from %s: %s", csv_path, extras
        )
        _rewrite_with_upgraded_header(csv_path, header, existing)
        return

    # If existing is empty, treat as subset (upgrade)
    if not ex_norm:
        _rewrite_with_upgraded_header(csv_path, header, existing)
        log.info("Upgraded empty CSV header: %s", csv_path)
        return

    # Check that existing columns appear in the same relative order in the desired header
    pos = []
    try:
        for col in ex_norm:
            pos.append(want_norm.index(col))
    except ValueError:
        # existing col not in desired header -> already handled above, but keep safe
        raise ValueError(
            f"CSV header mismatch in {csv_path}: existing column not in canonical list.\n"
            f"Existing: {existing}\nCanonical: {header}"
        )
    if any(pos[i] >= pos[i+1] for i in range(len(pos)-1)):
        raise ValueError(
            "CSV header order conflict — existing columns are not in the same relative order.\n"
            f"File: {csv_path}\nExisting: {existing}\nCanonical: {header}"
        )

    # Subset & compatible order → upgrade by appending the missing columns
    _rewrite_with_upgraded_header(csv_path, header, existing)
    log.info(
        "Upgraded CSV header by appending %d column(s): %s",
        len(want_norm) - len(ex_norm),
        csv_path,
    )

def _rewrite_with_upgraded_header(csv_path: Path, new_header: List[str], old_header: List[str]) -> None:
    """Rewrite CSV with `new_header`, preserving data in existing columns."""
    # Map normalized old -> real old key
    old_norm_to_real = {normalize_key(k): k for k in old_header}
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    # Read all rows with DictReader under the old header
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f_in, \
         tmp.open("w", newline="", encoding="utf-8-sig") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(new_header)
        for row in reader:
            # Pull each value by normalized key if present; else blank
            out = []
            for col in new_header:
                real_key = old_norm_to_real.get(normalize_key(col))
                out.append(row.get(real_key, "") if real_key else "")
            writer.writerow(out)
    tmp.replace(csv_path)

def append_row(csv_path: Path, header: List[str], row: Dict[str, str]) -> None:
    """Append a row; missing columns -> empty strings; extra keys ignored."""
    ensure_header(csv_path, header)
    ordered = [row.get(col, "") for col in header]
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(ordered)
    log.info("Appended row to %s for ID=%s", csv_path, ordered[0] if ordered else "")
    # New IDs may have been added; invalidate the cache
    _invalidate_id_cache()

def _normalize_row_keys(row: Dict[str, str]) -> Dict[str, str]:
    """Return a copy of row with normalized keys; values left as-is."""
    return {normalize_key(k): v for k, v in row.items()}

def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Read rows (BOM-tolerant). Return [] if file missing. Keys normalized."""
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [_normalize_row_keys(dict(row)) for row in reader]

def read_rows_multi(csv_paths: List[Path]) -> List[Dict[str, str]]:
    """Concatenate rows from multiple CSVs (BOM/whitespace tolerant)."""
    rows: List[Dict[str, str]] = []
    for p in csv_paths:
        rows.extend(read_rows(p))
    return rows

def _find_id_key(row: Dict[str, str], id_col: str) -> str | None:
    """Find the ID column in a row dict, tolerant to BOM/whitespace/casing and legacy names."""
    id_lower = normalize_key(id_col).lower()
    candidates = {id_lower}
    candidates.update(normalize_key(s).lower() for s in ID_SYNONYMS.get(id_lower, []))
    # Exact/synonym match first
    for k in row.keys():
        if normalize_key(k).lower() in candidates:
            return k
    # Heuristic fallback: any header ending with "_id"
    for k in row.keys():
        if normalize_key(k).lower().endswith("_id"):
            return k
    return None

def _has_any_payload(row: Dict[str, str], id_key_norm: str) -> bool:
    """True if any non-ID field is non-empty (after stripping)."""
    for k, v in row.items():
        if normalize_key(k).lower() == id_key_norm:
            continue
        if (v or "").strip():
            return True
    return False

def last_row_per_id(rows: Iterable[Dict[str, str]], id_col: str) -> Dict[str, Dict[str, str]]:
    """
    Reduce to the last row for each ID (append-only semantics).
    Uses normalized ID values as keys so lookups are robust.

    Safety: ignore trailing "pure-ID" rows (only the ID is filled) so accidental
    empty saves do not wipe previously saved metadata for that ID.
    """
    latest: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = _find_id_key(row, id_col)
        if not key:
            continue
        _id_raw = row.get(key, "")
        _id = normalize_id_value(_id_raw)
        if not _id:
            continue
        # If this row has no payload beyond the ID and we already have data for this ID, skip it
        if _id in latest and not _has_any_payload(row, normalize_key(key).lower()):
            continue
        latest[_id] = row
    return latest
