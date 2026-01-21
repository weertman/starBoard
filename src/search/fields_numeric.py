# src/search/fields_numeric.py
from __future__ import annotations
from typing import Dict, Tuple, Any
import math
import logging

Row = Dict[str, str]
_log = logging.getLogger("starBoard.search.field.numeric")

def _parse_float(s: str) -> float | None:
    try:
        v = float((s or "").strip())
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0: return 0.0
    m = n // 2
    return float(xs[m]) if n % 2 else 0.5 * (xs[m-1] + xs[m])

def _mad(xs, med):
    abs_dev = [abs(x - med) for x in xs]
    return _median(abs_dev)

class NumericGaussianScorer:
    """
    Similarity = exp( -|x - q| / (k * MAD) ), clamped with epsilon when MAD==0.
    """
    def __init__(self, field: str, k: float = 2.0, eps: float = 1e-6):
        self.name = field
        self.k = k
        self.eps = eps
        self.values_by_id: Dict[str, float] = {}
        self.med = 0.0
        self.mad = 1.0

    def build_gallery(self, gallery_rows_by_id: Dict[str, Row]) -> None:
        vals = []
        self.values_by_id.clear()
        for gid, row in gallery_rows_by_id.items():
            v = _parse_float(row.get(self.name, ""))
            if v is not None:
                self.values_by_id[gid] = v
                vals.append(v)
        self.med = _median(vals)
        self.mad = max(_mad(vals, self.med), self.eps)
        _log.info("build field=%s n=%d med=%.3f mad=%.3f", self.name, len(self.values_by_id), self.med, self.mad)

    def prepare_query(self, q_row: Row) -> Any:
        return _parse_float(q_row.get(self.name, ""))

    def has_query_signal(self, q_state: Any) -> bool:
        return q_state is not None

    def score_pair(self, q_state: Any, gallery_id: str) -> tuple[float, bool]:
        x = self.values_by_id.get(gallery_id, None)
        if q_state is None or x is None:
            return 0.0, False
        d = abs(x - float(q_state))
        s = math.exp(-d / (self.k * self.mad))
        return float(s), True
