# src/search/fields_categorical.py
from __future__ import annotations
from typing import Dict, Any
import logging

Row = Dict[str, str]
_log = logging.getLogger("starBoard.search.field.categorical")

def _norm(s: str) -> str:
    return (s or "").strip().lower()

class CategoricalMatchScorer:
    """Exact match = 1.0, mismatch = 0.0 (only contributes if both present)."""
    def __init__(self, field: str):
        self.name = field
        self.values_by_id: Dict[str, str] = {}

    def build_gallery(self, gallery_rows_by_id: Dict[str, Row]) -> None:
        self.values_by_id = {gid: _norm(r.get(self.name, "")) for gid, r in gallery_rows_by_id.items()}
        n = sum(1 for v in self.values_by_id.values() if v)
        _log.info("build field=%s n_nonempty=%d", self.name, n)

    def prepare_query(self, q_row: Row) -> Any:
        v = _norm(q_row.get(self.name, ""))
        return v if v else None

    def has_query_signal(self, q_state: Any) -> bool:
        return q_state is not None

    def score_pair(self, q_state: Any, gallery_id: str):
        v = self.values_by_id.get(gallery_id, "")
        if q_state is None or not v:
            return 0.0, False
        return (1.0 if v == q_state else 0.0), True
