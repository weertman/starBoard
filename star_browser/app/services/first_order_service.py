from __future__ import annotations

from src.search.engine import FirstOrderSearchEngine

from ..models.search_api import FirstOrderCandidate, FirstOrderSearchResponse

_ENGINE: FirstOrderSearchEngine | None = None


def _get_engine() -> FirstOrderSearchEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = FirstOrderSearchEngine()
    return _ENGINE


def run_first_order_search(query_id: str, top_k: int = 10) -> FirstOrderSearchResponse:
    engine = _get_engine()
    items = engine.rank(query_id, top_k=top_k)
    return FirstOrderSearchResponse(
        query_id=query_id,
        candidates=[
            FirstOrderCandidate(
                entity_id=item.gallery_id,
                score=item.score,
                k_contrib=item.k_contrib,
                field_breakdown=item.field_breakdown,
            )
            for item in items
        ],
    )
