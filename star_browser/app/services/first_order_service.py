from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.data import archive_paths as ap
from src.data.compare_labels import load_latest_map_for_query
from src.data.csv_io import last_row_per_id, normalize_id_value, read_rows_multi
from src.data.id_registry import list_ids
from src.search.engine import FirstOrderSearchEngine
from src.search.field_sets import COLOR_FIELDS, ORDINAL_FIELDS, SET_FIELDS, TEXT_FIELDS, ALL_FIELDS

from ..models.search_api import (
    FirstOrderCandidate,
    FirstOrderQueryOption,
    FirstOrderQueryOptionsResponse,
    FirstOrderSearchResponse,
)
from .first_order_media_service import first_order_image_id_for_path, resolve_first_order_media_path

_ENGINE: FirstOrderSearchEngine | None = None
_PRESETS = {
    'all': set(ALL_FIELDS),
    'colors': set(COLOR_FIELDS),
    'text': set(TEXT_FIELDS),
    'arms_patterns': set(['num_apparent_arms', 'num_total_arms', 'short_arm_code', *ORDINAL_FIELDS]),
    'megastar': {'megastar'},
}


@dataclass(frozen=True)
class _MegaStarScoreStore:
    query_ids: tuple[str, ...]
    gallery_ids: tuple[str, ...]
    scores: np.ndarray


def _get_engine() -> FirstOrderSearchEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = FirstOrderSearchEngine()
    return _ENGINE


QUALITY_FIELDS = ('madreporite_visibility', 'anus_visibility', 'postural_visibility')
QUALITY_MAX_VALUES = {
    'madreporite_visibility': 3.0,
    'anus_visibility': 3.0,
    'postural_visibility': 4.0,
}


def _has_first_order_pins(query_id: str) -> bool:
    for root in ap.roots_for_read('Queries'):
        path = root / query_id / '_pins_first_order.json'
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get('pinned'):
            return True
    return False


def _query_state(query_id: str) -> str:
    labels = load_latest_map_for_query(query_id)
    if not labels:
        return 'pinned' if _has_first_order_pins(query_id) else 'not_attempted'
    for row in labels.values():
        if (row.get('verdict') or '').strip().lower() == 'yes':
            return 'matched'
    return 'attempted'


def _query_quality_rows() -> dict[str, dict[str, str]]:
    rows = read_rows_multi(ap.metadata_csv_paths_for_read('Queries'))
    normalized_rows = []
    for row in rows:
        query_id = normalize_id_value(row.get('query_id') or row.get('queries_id') or row.get('querries_id') or '')
        if query_id:
            row = dict(row)
            row['query_id'] = query_id
            normalized_rows.append(row)
    return last_row_per_id(normalized_rows, 'query_id')


def _normalized_quality(row: dict[str, str]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for field in QUALITY_FIELDS:
        raw = (row.get(field) or '').strip()
        if not raw:
            values[field] = None
            continue
        try:
            value = float(raw)
        except ValueError:
            values[field] = None
            continue
        values[field] = max(0.0, min(1.0, value / QUALITY_MAX_VALUES[field]))
    return values


def _easy_match_score(state: str) -> float:
    # Lightweight selector ordering score from existing workflow evidence.
    # This avoids running the full first-order ranking engine for every query on list load.
    return {
        'matched': 1.0,
        'pinned': 0.75,
        'attempted': 0.5,
        'not_attempted': 0.0,
    }.get(state, 0.0)


def list_first_order_query_options() -> FirstOrderQueryOptionsResponse:
    ids = list_ids('Queries', exclude_silent=True)
    last_obs = ap.last_observation_for_all('Queries')
    metadata_rows = _query_quality_rows()

    unmatched: list[FirstOrderQueryOption] = []
    matched: list[FirstOrderQueryOption] = []
    for query_id in ids:
        state = _query_state(query_id)
        obs = last_obs.get(query_id)
        metadata_row = metadata_rows.get(query_id, {})
        option = FirstOrderQueryOption(
            query_id=query_id,
            state=state,  # type: ignore[arg-type]
            last_observation_date=obs.isoformat() if obs else None,
            last_location=(metadata_row.get('location') or '').strip() or None,
            easy_match_score=_easy_match_score(state),
            quality=_normalized_quality(metadata_row),
            metadata={key: value for key, value in metadata_row.items() if value},
        )
        if state == 'matched':
            matched.append(option)
        else:
            unmatched.append(option)

    def sort_key(option: FirstOrderQueryOption):
        # Show the newest actionable queries first; keep already matched queries in their own bottom group.
        return (option.last_observation_date is not None, option.last_observation_date or '', option.query_id.lower())

    return FirstOrderQueryOptionsResponse(
        queries=sorted(unmatched, key=sort_key, reverse=True) + sorted(matched, key=sort_key, reverse=True)
    )


def _rank_megastar_by_query_image(query_image_id: str, top_k: int) -> FirstOrderSearchResponse | None:
    query_image_path = resolve_first_order_media_path(query_image_id)
    if query_image_path is None:
        return None
    parts = query_image_id.split(':')
    if len(parts) != 3 or parts[0] != 'query':
        return None
    query_id = parts[1]
    model_key = _active_megastar_model_key()
    if not model_key:
        return None
    try:
        from src.dl.similarity_lookup import get_image_lookup
        lookup = get_image_lookup(model_key)
    except Exception:
        return None
    if not lookup or not lookup.is_loaded():
        return None
    ranked = lookup.rank_gallery_by_query_image(str(query_image_path), top_k=top_k)
    if not ranked:
        return None
    candidates: list[FirstOrderCandidate] = []
    for gallery_id, raw_score, gallery_image_path, local_idx in ranked:
        score = round(float(raw_score), 6)
        preferred_image_id = first_order_image_id_for_path('gallery', gallery_id, gallery_image_path) or f'gallery:{gallery_id}:{local_idx}'
        candidates.append(
            FirstOrderCandidate(
                entity_id=gallery_id,
                score=score,
                k_contrib=1,
                field_breakdown={'megastar': score},
                preferred_image_id=preferred_image_id,
            )
        )
    return FirstOrderSearchResponse(query_id=query_id, query_image_id=query_image_id, preset='megastar', candidates=candidates)


def _rank_megastar(query_id: str, top_k: int, query_image_id: str | None = None) -> FirstOrderSearchResponse:
    if query_image_id:
        image_response = _rank_megastar_by_query_image(query_image_id, top_k)
        if image_response is not None:
            return image_response
    store = _load_megastar_score_store()
    candidates: list[FirstOrderCandidate] = []
    if store is not None and query_id in store.query_ids:
        row_idx = store.query_ids.index(query_id)
        row = np.asarray(store.scores[row_idx], dtype=np.float32).reshape(-1)
        top_n = min(max(int(top_k), 0), len(store.gallery_ids), row.size)
        if top_n > 0:
            ranked = np.argsort(row)[::-1][:top_n]
            for col_idx in ranked.tolist():
                score = round(float(row[col_idx]), 6)
                candidates.append(
                    FirstOrderCandidate(
                        entity_id=store.gallery_ids[col_idx],
                        score=score,
                        k_contrib=1,
                        field_breakdown={'megastar': score},
                    )
                )
    return FirstOrderSearchResponse(query_id=query_id, preset='megastar', candidates=candidates)


@lru_cache(maxsize=1)
def _active_megastar_model_key() -> str | None:
    precompute_root = ap.archive_root() / '_dl_precompute'
    registry_path = precompute_root / '_dl_registry.json'
    if not registry_path.exists():
        return None
    try:
        registry = json.loads(registry_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    model_key = registry.get('active_model')
    return model_key if isinstance(model_key, str) and model_key else None


@lru_cache(maxsize=1)
def _load_megastar_score_store() -> _MegaStarScoreStore | None:
    model_key = _active_megastar_model_key()
    if not model_key:
        return None
    precompute_root = ap.archive_root() / '_dl_precompute'
    similarity_dir = precompute_root / model_key / 'similarity'
    mapping_path = similarity_dir / 'id_mapping.json'
    scores_path = similarity_dir / 'query_gallery_scores.npz'
    if not mapping_path.exists() or not scores_path.exists():
        return None
    try:
        mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
        query_ids = tuple(str(value) for value in mapping.get('query_ids', []))
        gallery_ids = tuple(str(value) for value in mapping.get('gallery_ids', []))
        with np.load(scores_path) as score_file:
            scores = np.asarray(score_file['similarity'], dtype=np.float32)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    if scores.ndim != 2 or scores.shape != (len(query_ids), len(gallery_ids)):
        return None
    return _MegaStarScoreStore(query_ids=query_ids, gallery_ids=gallery_ids, scores=scores)


def run_first_order_search(query_id: str, top_k: int = 10, preset: str = 'all', query_image_id: str | None = None) -> FirstOrderSearchResponse:
    if preset == 'megastar':
        return _rank_megastar(query_id, top_k, query_image_id=query_image_id)

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
