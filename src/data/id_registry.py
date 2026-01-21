# src/data/id_registry.py
from __future__ import annotations

"""
ID registry utilities.

- list_ids(target, *, exclude_silent=False) → List[str]
    Returns the union of folder names under all plausible roots and IDs
    present in all plausible metadata CSVs (new + legacy). Sorted.

    If target == "Queries" and exclude_silent=True, any Query folder that
    contains a silence flag (see src/data/silence.py) is filtered out.
    This allows interactive tabs (First/Second order) to hide "merged"
    queries while other data surfaces (reports, past matches, etc.)
    continue to see the full registry by default.

- id_exists(target, id_str) → bool
    True if the ID exists either as a folder under any plausible root
    OR in any of the plausible metadata CSVs.

Notes:
- This module remains tolerant to the legacy "querries" directory and
  misspelled "querries_metadata.csv" header variants via csv_io helpers.
"""

from functools import lru_cache
from typing import Dict, List, Set, Tuple
import logging

from .archive_paths import (
    roots_for_read,
    metadata_csv_paths_for_read,
    id_column_name,
)
from .csv_io import read_rows_multi, normalize_id_value, _find_id_key

# Optional silence filtering for Queries
try:
    from .silence import is_silent_query
except Exception:  # pragma: no cover
    def is_silent_query(_qid: str) -> bool:  # type: ignore
        return False

log = logging.getLogger("starBoard.data.id_registry")

# Never treat these system folders as real IDs
RESERVED_ID_NAMES: Set[str] = {"_embeddings" , }

def _csv_ids(target: str) -> Set[str]:
    """
    IDs discovered in metadata CSVs (latest rows only are not required here;
    we only need the presence of an ID in any row).
    """
    ids: Set[str] = set()
    id_col = id_column_name(target)
    rows = read_rows_multi(metadata_csv_paths_for_read(target))
    for row in rows:
        real_key = _find_id_key(row, id_col)
        if real_key:
            val = normalize_id_value(row.get(real_key, ""))
            if val:
                ids.add(val)
    ids.discard("")
    return ids


def _folder_ids(target: str) -> Set[str]:
    """
    IDs discovered as folder names under all plausible roots.
    """
    ids: Set[str] = set()
    for root in roots_for_read(target):
        if not root or not root.exists():
            continue
        for p in root.iterdir():
            if not p.is_dir():
                continue
            name = p.name
            # Exclude system folders (e.g., the embedding store lives under "_embeddings")
            if name in RESERVED_ID_NAMES:
                continue
            ids.add(name)
    ids.discard("")
    return ids


@lru_cache(maxsize=4)
def _list_ids_cached(target: str, exclude_silent: bool) -> Tuple[str, ...]:
    """
    Internal cached implementation of list_ids.
    Returns a tuple (hashable) for caching; outer function converts to list.
    """
    # Collect candidates
    folder = _folder_ids(target)
    csv = _csv_ids(target)
    # Guard against reserved names leaking in from either source
    ids = {x for x in (set(folder) | set(csv)) if x not in RESERVED_ID_NAMES}

    if target.lower().startswith("q") and exclude_silent:
        before = len(ids)
        ids = {q for q in ids if not is_silent_query(q)}
        removed = before - len(ids)
        if removed:
            log.info("list_ids exclude_silent=True removed=%d", removed)

    out = tuple(sorted(ids))
    log.info("list_ids target=%s count=%d (exclude_silent=%s)", target, len(out), exclude_silent)
    return out


def list_ids(target: str, *, exclude_silent: bool = False) -> List[str]:
    """
    Union of:
      - folder names under all plausible roots
      - IDs present in all plausible metadata CSVs

    Sorted alphabetically.

    Parameters
    ----------
    target : str
        "Gallery" or "Queries" (case-insensitive).
    exclude_silent : bool (default False)
        Only applies when target is "Queries". If True, any query that has a
        `_SILENT.flag` in its folder is omitted.
    """
    return list(_list_ids_cached(target, exclude_silent))


def invalidate_id_cache() -> None:
    """Clear the list_ids cache. Call after adding/removing IDs or modifying silence flags."""
    _list_ids_cached.cache_clear()
    log.debug("Invalidated list_ids cache")


def id_exists(target: str, id_str: str) -> bool:
    """
    True if the ID exists as a folder under any plausible root or appears
    in any plausible metadata CSV for the target.
    """
    if not id_str or id_str in RESERVED_ID_NAMES:
        return False
    # Folder existence in any root
    for root in roots_for_read(target):
        if (root / id_str).exists():
            return True
    # CSV presence (tolerant to legacy ID headers)
    id_col = id_column_name(target)
    rows = read_rows_multi(metadata_csv_paths_for_read(target))
    needle = normalize_id_value(id_str)
    for row in rows:
        real_key = _find_id_key(row, id_col)
        if real_key and normalize_id_value(row.get(real_key, "")) == needle:
            return True
    return False


# ---------------- convenience helpers ----------------

def list_queries_with_silence_flag() -> List[Dict[str, str]]:
    """
    Return a small table for UI/debugging:

    [
      {"query_id": "...", "silent": "1"|"0"}
      ...
    ]
    """
    from .silence import is_silent_query  # local import to avoid cycles in tools
    out: List[Dict[str, str]] = []
    for qid in list_ids("Queries"):
        out.append({
            "query_id": qid,
            "silent": "1" if is_silent_query(qid) else "0",
        })
    return out
