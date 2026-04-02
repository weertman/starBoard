from __future__ import annotations

from fastapi import HTTPException, status

from ..config import get_settings
from ..adapters.archive_paths import entity_exists, latest_metadata_row, list_entity_ids, latest_metadata_map
from ..adapters.image_manifest_adapter import window_for_entity


def get_entity(entity_type: str, entity_id: str, window_size: int | None = None) -> dict:
    if not entity_exists(entity_type, entity_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'{entity_type} ID not found: {entity_id}')
    settings = get_settings()
    size = window_size or settings.initial_image_window
    return {
        'entity_type': entity_type,
        'entity_id': entity_id,
        'metadata_summary': latest_metadata_row(entity_type, entity_id),
        'image_window': window_for_entity(entity_type, entity_id, 0, size),
    }


def get_entity_images(entity_type: str, entity_id: str, offset: int, limit: int | None = None) -> dict:
    if not entity_exists(entity_type, entity_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'{entity_type} ID not found: {entity_id}')
    settings = get_settings()
    size = limit or settings.image_page_size
    return window_for_entity(entity_type, entity_id, offset, size)


def suggest_entity_ids(entity_type: str, query: str, limit: int = 8) -> dict:
    q = query.strip().lower()
    ids = list_entity_ids(entity_type)
    if q:
        ids = [item for item in ids if q in item.lower()]
    ids = ids[:limit]
    return {
        'entity_type': entity_type,
        'query': query,
        'items': ids,
    }


def lookup_options(entity_type: str, location: str = '', limit: int = 200) -> dict:
    location_q = location.strip().lower()
    meta = latest_metadata_map(entity_type)
    ids = list_entity_ids(entity_type)
    if location_q:
        ids = [entity_id for entity_id in ids if location_q in str(meta.get(entity_id, {}).get('location', '')).lower()]
    ids = ids[:limit]
    locations = sorted({str(row.get('location', '')).strip() for row in meta.values() if str(row.get('location', '')).strip()})
    return {
        'entity_type': entity_type,
        'location': location,
        'locations': locations,
        'ids': ids,
    }
