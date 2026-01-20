# src/data/encounter_info.py
"""
Encounter information utilities for dynamic display of:
- Encounter date from image path
- Associated queries for gallery members
- Editable encounter date overrides
"""
from __future__ import annotations

import csv
import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.data.archive_paths import archive_root, roots_for_read
from src.data.csv_io import read_rows, normalize_id_value
from src.data.observation_dates import last_observation_date

# Reuse the same date pattern from observation_dates
_MMDDYY = re.compile(r"^(\d{2})_(\d{2})_(\d{2})(?:_|$)")

# ---- Encounter date overrides ----

ENCOUNTER_DATES_HEADER = ["target", "id_str", "encounter_name", "encounter_date"]
_encounter_dates_cache: Optional[Dict[Tuple[str, str, str], date]] = None


def _encounter_dates_csv_path() -> Path:
    """Path to the encounter dates override CSV."""
    return archive_root() / "encounter_dates.csv"


def _load_encounter_dates() -> Dict[Tuple[str, str, str], date]:
    """Load encounter date overrides from CSV into cache."""
    global _encounter_dates_cache
    if _encounter_dates_cache is not None:
        return _encounter_dates_cache
    
    _encounter_dates_cache = {}
    csv_path = _encounter_dates_csv_path()
    if not csv_path.exists():
        return _encounter_dates_cache
    
    try:
        for row in read_rows(csv_path):
            target = row.get("target", "").strip()
            id_str = row.get("id_str", "").strip()
            enc_name = row.get("encounter_name", "").strip()
            date_str = row.get("encounter_date", "").strip()
            if target and id_str and enc_name and date_str:
                try:
                    d = date.fromisoformat(date_str)
                    _encounter_dates_cache[(target, id_str, enc_name)] = d
                except ValueError:
                    pass
    except Exception:
        pass
    return _encounter_dates_cache


def invalidate_encounter_dates_cache() -> None:
    """Clear cache to force reload on next access."""
    global _encounter_dates_cache
    _encounter_dates_cache = None


def get_encounter_date_override(target: str, id_str: str, encounter_name: str) -> Optional[date]:
    """Get the override date for an encounter, or None if not set."""
    cache = _load_encounter_dates()
    return cache.get((target, id_str, encounter_name))


def set_encounter_date(target: str, id_str: str, encounter_name: str, d: date) -> None:
    """Set or update the encounter date override."""
    csv_path = _encounter_dates_csv_path()
    
    # Load existing rows
    rows: List[Dict[str, str]] = []
    if csv_path.exists():
        try:
            rows = list(read_rows(csv_path))
        except Exception:
            pass
    
    # Update or add entry
    found = False
    for row in rows:
        if (row.get("target") == target and 
            row.get("id_str") == id_str and 
            row.get("encounter_name") == encounter_name):
            row["encounter_date"] = d.isoformat()
            found = True
            break
    
    if not found:
        rows.append({
            "target": target,
            "id_str": id_str,
            "encounter_name": encounter_name,
            "encounter_date": d.isoformat(),
        })
    
    # Write back
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ENCOUNTER_DATES_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    
    invalidate_encounter_dates_cache()


def list_encounters_for_id(target: str, id_str: str) -> List[str]:
    """List encounter folder names for an ID."""
    encounters = []
    for root in roots_for_read(target):
        id_dir = root / id_str
        if not id_dir.exists():
            continue
        for child in id_dir.iterdir():
            if child.is_dir() and child.name not in encounters:
                encounters.append(child.name)
    encounters.sort()
    return encounters


def get_encounter_date(target: str, id_str: str, encounter_name: str) -> Optional[date]:
    """Get encounter date: check override first, then parse folder name."""
    override = get_encounter_date_override(target, id_str, encounter_name)
    if override is not None:
        return override
    return _parse_mmddyy(encounter_name)


def _parse_mmddyy(s: str) -> Optional[date]:
    """Parse MM_DD_YY folder name to date object."""
    m = _MMDDYY.match(s or "")
    if not m:
        return None
    mm, dd, yy = map(int, (m.group(1), m.group(2), m.group(3)))
    yy = 2000 + yy
    try:
        return date(yy, mm, dd)
    except Exception:
        return None


def get_encounter_date_from_path(image_path: Path) -> Optional[date]:
    """
    Extract encounter date from an image path.
    
    Expected path structure: .../Gallery/<id>/<MM_DD_YY...>/image.jpg
    or: .../Queries/<id>/<MM_DD_YY...>/image.jpg
    
    Checks for date override first, then parses folder name.
    """
    if not image_path:
        return None
    
    try:
        # Extract target, id_str, encounter_name from path
        parts = image_path.parts
        target, id_str, encounter_name = None, None, None
        for i, part in enumerate(parts):
            if part.lower() in ("gallery", "queries"):
                target = "Gallery" if part.lower() == "gallery" else "Queries"
                if i + 2 < len(parts):
                    id_str = parts[i + 1]
                    encounter_name = parts[i + 2]
                break
        
        if target and id_str and encounter_name:
            return get_encounter_date(target, id_str, encounter_name)
        
        # Fallback: walk up path looking for MM_DD_YY pattern
        current = image_path.parent
        for _ in range(3):
            if not current or current == current.parent:
                break
            d = _parse_mmddyy(current.name)
            if d is not None:
                return d
            current = current.parent
        return None
    except Exception:
        return None


def get_observation_date_for_query(query_id: str) -> Optional[date]:
    """
    Get the observation date for a query from its encounter folder structure.
    
    Queries have encounter folders with MM_DD_YY names, just like gallery members.
    """
    return last_observation_date("Queries", query_id)


def format_encounter_date(d: Optional[date]) -> str:
    """Format date for display, or return empty string if None."""
    if d is None:
        return ""
    return d.strftime("%Y-%m-%d")


# ---- Gallery to Queries lookup (with caching) ----

_gallery_to_queries_cache: Optional[Dict[str, List[Tuple[str, str]]]] = None


def _iter_all_label_files():
    """Iterate over all _second_order_labels.csv files in Queries."""
    for root in roots_for_read("Queries"):
        if not root.exists():
            continue
        for query_folder in root.iterdir():
            if not query_folder.is_dir():
                continue
            labels_csv = query_folder / "_second_order_labels.csv"
            if labels_csv.exists():
                yield labels_csv, query_folder.name


def build_gallery_to_queries_map(force_reload: bool = False) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build a mapping from gallery_id to list of (query_id, verdict) tuples.
    
    This scans all _second_order_labels.csv files and aggregates by gallery_id.
    Results are cached for performance.
    
    Returns:
        Dict mapping gallery_id -> [(query_id, verdict), ...]
        Only includes pairs with verdict in {"yes", "maybe", "no"}
    """
    global _gallery_to_queries_cache
    
    if _gallery_to_queries_cache is not None and not force_reload:
        return _gallery_to_queries_cache
    
    result: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    
    # Track latest verdict per (query_id, gallery_id) pair
    latest_by_pair: Dict[Tuple[str, str], Tuple[str, str]] = {}  # (qid, gid) -> (verdict, updated_utc)
    
    for labels_csv, query_id in _iter_all_label_files():
        try:
            rows = read_rows(labels_csv)
            for row in rows:
                qid = normalize_id_value(row.get("query_id", ""))
                gid = normalize_id_value(row.get("gallery_id", ""))
                verdict = (row.get("verdict", "") or "").strip().lower()
                updated = row.get("updated_utc", "") or ""
                
                if not qid or not gid:
                    continue
                if verdict not in {"yes", "maybe", "no"}:
                    continue
                
                pair_key = (qid, gid)
                existing = latest_by_pair.get(pair_key)
                
                # Keep the latest entry (by updated_utc)
                if existing is None or updated > existing[1]:
                    latest_by_pair[pair_key] = (verdict, updated)
        except Exception:
            continue
    
    # Build the gallery -> queries mapping
    for (qid, gid), (verdict, _) in latest_by_pair.items():
        result[gid].append((qid, verdict))
    
    # Sort queries by query_id for consistent display
    for gid in result:
        result[gid].sort(key=lambda x: x[0])
    
    _gallery_to_queries_cache = dict(result)
    return _gallery_to_queries_cache


def invalidate_gallery_queries_cache() -> None:
    """Clear the cache, forcing a rebuild on next access."""
    global _gallery_to_queries_cache
    _gallery_to_queries_cache = None


def get_matched_queries_for_gallery(gallery_id: str) -> List[str]:
    """
    Get list of query IDs that have a "yes" verdict for this gallery member.
    
    Args:
        gallery_id: The gallery member ID
        
    Returns:
        List of query IDs with "yes" match verdicts
    """
    mapping = build_gallery_to_queries_map()
    pairs = mapping.get(normalize_id_value(gallery_id), [])
    return [qid for qid, verdict in pairs if verdict == "yes"]


def get_queries_for_encounter(gallery_id: str, encounter_date: Optional[date]) -> List[str]:
    """
    Get query IDs that match this gallery member AND have the same observation date
    as the given encounter date.
    
    This is useful for showing which query led to a specific encounter being added.
    
    Args:
        gallery_id: The gallery member ID
        encounter_date: The date of the current encounter/image
        
    Returns:
        List of query IDs with "yes" verdict whose observation date matches
    """
    if encounter_date is None:
        return []
    
    matched = get_matched_queries_for_gallery(gallery_id)
    
    # Filter to queries whose observation date matches the encounter date
    result = []
    for qid in matched:
        qid_date = get_observation_date_for_query(qid)
        if qid_date == encounter_date:
            result.append(qid)
    
    return result


def get_all_queries_for_gallery(gallery_id: str) -> List[Tuple[str, str]]:
    """
    Get all (query_id, verdict) pairs for this gallery member.
    
    Args:
        gallery_id: The gallery member ID
        
    Returns:
        List of (query_id, verdict) tuples
    """
    mapping = build_gallery_to_queries_map()
    return mapping.get(normalize_id_value(gallery_id), [])


def format_queries_for_display(query_ids: List[str], max_display: int = 3) -> str:
    """
    Format query IDs for compact display.
    
    Args:
        query_ids: List of query IDs
        max_display: Maximum number to show before truncating
        
    Returns:
        Formatted string like "Q1, Q2" or "Q1, Q2, +3 more"
    """
    if not query_ids:
        return ""
    
    if len(query_ids) <= max_display:
        return ", ".join(query_ids)
    
    shown = query_ids[:max_display]
    remaining = len(query_ids) - max_display
    return f"{', '.join(shown)}, +{remaining} more"

