# src/search/fields_set_jaccard.py
from __future__ import annotations
from typing import Dict, Set, Any
import re
import logging

Row = Dict[str, str]
_SPLIT = re.compile(r"[^A-Za-z0-9]+")
_log = logging.getLogger("starBoard.search.field.set")

def _to_set(s: str) -> Set[str]:
    toks = [t for t in _SPLIT.split((s or "").upper()) if t]
    return set(toks)

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    if inter == 0: return 0.0
    return inter / float(len(a | b))

class SetJaccardScorer:
    def __init__(self, field: str):
        self.name = field
        self.sets_by_id: Dict[str, Set[str]] = {}

    def build_gallery(self, gallery_rows_by_id: Dict[str, Row]) -> None:
        self.sets_by_id = {gid: _to_set(r.get(self.name, "")) for gid, r in gallery_rows_by_id.items()}
        nonempty = [s for s in self.sets_by_id.values() if s]
        avg_sz = (sum(len(s) for s in nonempty) / len(nonempty)) if nonempty else 0.0
        _log.info("build field=%s n_nonempty=%d avg_set_size=%.2f", self.name, len(nonempty), avg_sz)

    def prepare_query(self, q_row: Row) -> Any:
        s = _to_set(q_row.get(self.name, ""))
        return s if s else None

    def has_query_signal(self, q_state: Any) -> bool:
        return bool(q_state)

    def score_pair(self, q_state: Any, gallery_id: str):
        s = self.sets_by_id.get(gallery_id, set())
        if not q_state or not s:
            return 0.0, False
        return _jaccard(q_state, s), True
