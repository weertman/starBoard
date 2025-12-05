# src/search/interfaces.py
from __future__ import annotations
from typing import Protocol, Dict, Any, Iterable

Row = Dict[str, str]   # normalized CSV row (keys are canonical headers)

class FieldScorer(Protocol):
    """
    A pluggable scorer for one metadata field.
    Build once on the Gallery; then for each query, prepare a query state and
    score all Gallery candidates on demand.
    """
    name: str  # canonical field name (matches CSV header)

    def build_gallery(self, gallery_rows_by_id: Dict[str, Row]) -> None: ...
    def prepare_query(self, q_row: Row) -> Any: ...
    def has_query_signal(self, q_state: Any) -> bool: ...
    def score_pair(self, q_state: Any, gallery_id: str) -> tuple[float, bool]:
        """
        Returns (score in [0,1], present_mask). present_mask==True only if the field
        is present for BOTH the query and this gallery_id.
        """
        ...
