from __future__ import annotations

from src.search.engine import FirstOrderSearchEngine
from src.search.field_sets import COLOR_FIELDS, ORDINAL_FIELDS, SET_FIELDS, TEXT_FIELDS, ALL_FIELDS

from ..models.search_api import FirstOrderCandidate, FirstOrderSearchResponse

_ENGINE: FirstOrderSearchEngine | None = None
_PRESETS = {
    'all': set(ALL_FIELDS),
    'colors': set(COLOR_FIELDS),
    'text': set(TEXT_FIELDS),
    'arms_patterns': set(['num_apparent_arms', 'num_total_arms', 'short_arm_code', *ORDINAL_FIELDS]),
}


def _get_engine() -> FirstOrderSearchEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = FirstOrderSearchEngine()
    return _ENGINE


def run_first_order_search(query_id: str, top_k: int = 10, preset: str = 'all') -> FirstOrderSearchResponse:
    engine = _get_engine()
    include_fields = _PRESETS.get(preset, _PRESETS['all'])
    items = engine.rank(query_id, top_k=top_k, include_fields=include_fields)
    return FirstOrderSearchResponse(
        query_id=query_id,
        preset=preset,  # type: ignore[arg-type]
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
