# src/data/matches_matrix.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from datetime import date
import re

from src.data.id_registry import list_ids
from src.data.compare_labels import load_latest_map_for_query
from src.data.observation_dates import last_observation_for_all

Verdict = str  # "yes" | "maybe" | "no"

@dataclass
class MatchMatrixData:
    query_ids: List[str]
    gallery_ids: List[str]
    last_obs_by_query: Dict[str, Optional[date]]
    verdict_by_pair: Dict[Tuple[str, str], Verdict]
    notes_by_pair: Dict[Tuple[str, str], str]
    updated_by_pair: Dict[Tuple[str, str], str]

_NUM_SPLIT = re.compile(r"(\d+)")
def _natural_key(s: str) -> tuple:
    parts = _NUM_SPLIT.split(s or "")
    out: List[object] = []
    for t in parts: out.append(int(t) if t.isdigit() else t.lower())
    return tuple(out)

def load_match_matrix() -> MatchMatrixData:
    galleries = list_ids("Gallery")
    queries   = list_ids("Queries")
    gallery_ids = sorted(galleries, key=_natural_key)
    last_obs = last_observation_for_all("Queries")

    def _row_key(qid: str):
        d = last_obs.get(qid); secondary = -(d.toordinal()) if d is not None else 10**12
        return (_natural_key(qid), secondary)

    query_ids = sorted(queries, key=_row_key)

    verdict_by_pair: Dict[Tuple[str, str], Verdict] = {}
    notes_by_pair: Dict[Tuple[str, str], str] = {}
    updated_by_pair: Dict[Tuple[str, str], str] = {}

    for qid in queries:
        latest = load_latest_map_for_query(qid)  # {gallery_id -> row}
        for gid, row in latest.items():
            v = (row.get("verdict", "") or "").strip().lower()
            if v in ("yes", "maybe", "no"):
                key = (qid, gid)
                verdict_by_pair[key] = v
                notes_by_pair[key] = row.get("notes", "") or ""
                updated_by_pair[key] = row.get("updated_utc", "") or ""

    return MatchMatrixData(
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        last_obs_by_query=last_obs,
        verdict_by_pair=verdict_by_pair,
        notes_by_pair=notes_by_pair,
        updated_by_pair=updated_by_pair,
    )
