# src/data/observation_dates.py
"""
Observation date utilities for Gallery and Queries.

This module provides functions to get first/last observation dates for IDs.
Dates are sourced from:
1. encounter_dates.csv overrides (primary - set during upload)
2. Folder name parsing MM_DD_YY (fallback for legacy data)
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
from datetime import date
import re
from .archive_paths import roots_for_read

_MMDDYY = re.compile(r"^(\d{2})_(\d{2})_(\d{2})(?:_|$)")


def _parse_mmddyy(s: str) -> Optional[date]:
    """Parse MM_DD_YY folder name to date. Used as fallback when no override exists."""
    m = _MMDDYY.match(s or "")
    if not m:
        return None
    mm, dd, yy = map(int, (m.group(1), m.group(2), m.group(3)))
    yy = 2000 + yy
    try:
        return date(yy, mm, dd)
    except Exception:
        return None


def _get_encounter_date_safe(target: str, id_str: str, encounter_name: str) -> Optional[date]:
    """
    Get encounter date, checking override CSV first then falling back to folder parsing.
    
    Uses lazy import to avoid circular dependency with encounter_info.py.
    """
    try:
        from .encounter_info import get_encounter_date
        return get_encounter_date(target, id_str, encounter_name)
    except ImportError:
        # Fallback if encounter_info not available
        return _parse_mmddyy(encounter_name)


def last_observation_date(target: str, id_str: str) -> Optional[date]:
    """
    Return the most recent observation date from encounter subfolders of <target>/<id_str>.
    
    Checks encounter_dates.csv overrides first, then falls back to folder name parsing.
    """
    latest: Optional[date] = None
    for root in roots_for_read(target):
        base = root / id_str
        if not base.exists():
            continue
        for child in base.iterdir():
            if child.is_dir():
                d = _get_encounter_date_safe(target, id_str, child.name)
                if d and (latest is None or d > latest):
                    latest = d
    return latest

def last_observation_for_all(target: str) -> Dict[str, Optional[date]]:
    """
    Scan all IDs under all plausible roots for *target* ("Gallery" or "Queries")
    and return a mapping {id_str -> last_observation_date(...)}.
    """
    out: Dict[str, Optional[date]] = {}
    seen: set[str] = set()
    for root in roots_for_read(target):
        if not root.exists():
            continue
        for p in root.iterdir():
            if not p.is_dir():
                continue
            _id = p.name
            if _id in seen:
                continue
            seen.add(_id)
            out[_id] = last_observation_date(target, _id)
    return out


def first_observation_date(target: str, id_str: str) -> Optional[date]:
    """
    Return the earliest observation date from encounter subfolders of <target>/<id_str>.
    
    Checks encounter_dates.csv overrides first, then falls back to folder name parsing.
    """
    earliest: Optional[date] = None
    for root in roots_for_read(target):
        base = root / id_str
        if not base.exists():
            continue
        for child in base.iterdir():
            if child.is_dir():
                d = _get_encounter_date_safe(target, id_str, child.name)
                if d and (earliest is None or d < earliest):
                    earliest = d
    return earliest


def first_observation_for_all(target: str) -> Dict[str, Optional[date]]:
    """
    Scan all IDs under all plausible roots for *target* ("Gallery" or "Queries")
    and return a mapping {id_str -> first_observation_date(...)}.
    """
    out: Dict[str, Optional[date]] = {}
    seen: set[str] = set()
    for root in roots_for_read(target):
        if not root.exists():
            continue
        for p in root.iterdir():
            if not p.is_dir():
                continue
            _id = p.name
            if _id in seen:
                continue
            seen.add(_id)
            out[_id] = first_observation_date(target, _id)
    return out