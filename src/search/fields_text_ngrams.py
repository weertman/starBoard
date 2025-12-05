# src/search/fields_text_ngrams.py
from __future__ import annotations
from typing import Dict, Set, Any, Iterable, Tuple
import logging

Row = Dict[str, str]
_log = logging.getLogger("starBoard.search.field.text")

def _normalize_text(s: str) -> str:
    # Keep it simple; lowercased; strip; collapse spaces
    return " ".join((s or "").lower().strip().split())

def _char_ngrams(s: str, n_low=3, n_high=5) -> Set[str]:
    if not s: return set()
    grams: Set[str] = set()
    for n in range(n_low, n_high + 1):
        if len(s) < n: continue
        for i in range(len(s) - n + 1):
            grams.add(s[i:i+n])
    return grams

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    if inter == 0: return 0.0
    return inter / float(len(a | b))

class TextNgramScorer:
    """
    Dependency-free text scorer: character n-gram Jaccard (3..5).
    Robust to typos and short descriptors; fast enough for thousands of IDs.
    """
    def __init__(self, field: str, n_low: int = 3, n_high: int = 5):
        self.name = field
        self.n_low = n_low
        self.n_high = n_high
        self.grams_by_id: Dict[str, Set[str]] = {}

    def build_gallery(self, gallery_rows_by_id: Dict[str, Row]) -> None:
        self.grams_by_id.clear()
        for gid, r in gallery_rows_by_id.items():
            s = _normalize_text(r.get(self.name, ""))
            self.grams_by_id[gid] = _char_ngrams(s, self.n_low, self.n_high)
        nonempty = [g for g in self.grams_by_id.values() if g]
        avg = int(sum(len(g) for g in nonempty) / len(nonempty)) if nonempty else 0
        _log.info("build field=%s n_nonempty=%d avg_ngrams=%d", self.name, len(nonempty), avg)

    def prepare_query(self, q_row: Row) -> Any:
        s = _normalize_text(q_row.get(self.name, ""))
        grams = _char_ngrams(s, self.n_low, self.n_high)
        return grams if grams else None

    def has_query_signal(self, q_state: Any) -> bool:
        return bool(q_state)

    def score_pair(self, q_state: Any, gallery_id: str):
        g = self.grams_by_id.get(gallery_id, set())
        if not q_state or not g:
            return 0.0, False
        return _jaccard(q_state, g), True
