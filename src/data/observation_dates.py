# src/data/observation_dates.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
from datetime import date
import re
from .archive_paths import roots_for_read

_MMDDYY = re.compile(r"^(\d{2})_(\d{2})_(\d{2})(?:_|$)")

def _parse_mmddyy(s: str) -> Optional[date]:
    m = _MMDDYY.match(s or "");
    if not m: return None
    mm, dd, yy = map(int, (m.group(1), m.group(2), m.group(3)))
    yy = 2000 + yy
    try: return date(yy, mm, dd)
    except Exception: return None

def last_observation_date(target: str, id_str: str) -> Optional[date]:
    latest: Optional[date] = None
    for root in roots_for_read(target):
        base = root / id_str
        if not base.exists(): continue
        for child in base.iterdir():
            if child.is_dir():
                d = _parse_mmddyy(child.name)
                if d and (latest is None or d > latest): latest = d
    return latest

def last_observation_for_all(target: str) -> Dict[str, Optional[date]]:
    out: Dict[str, Optional[date]] = {}; seen: set[str] = set()
    for root in roots_for_read(target):
        if not root.exists(): continue
        for p in root.iterdir():
            if not p.is_dir(): continue
            _id = p.name
            if _id in seen: continue
            seen.add(_id)
            out[_id] = last_observation_date(target, _id)
    return out


def first_observation_date(target: str, id_str: str) -> Optional[date]:
    """Return the earliest parsed MM_DD_YY date from encounter subfolders of <target>/<id_str>."""
    earliest: Optional[date] = None
    for root in roots_for_read(target):
        base = root / id_str
        if not base.exists(): continue
        for child in base.iterdir():
            if child.is_dir():
                d = _parse_mmddyy(child.name)
                if d and (earliest is None or d < earliest): earliest = d
    return earliest


def first_observation_for_all(target: str) -> Dict[str, Optional[date]]:
    """
    Scan all IDs under all plausible roots for *target* ("Gallery" or "Queries")
    and return a mapping {id_str -> first_observation_date(...)}.
    """
    out: Dict[str, Optional[date]] = {}
    seen: set[str] = set()
    for root in roots_for_read(target):
        if not root.exists(): continue
        for p in root.iterdir():
            if not p.is_dir(): continue
            _id = p.name
            if _id in seen: continue
            seen.add(_id)
            out[_id] = first_observation_date(target, _id)
    return out