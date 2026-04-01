from __future__ import annotations

from fastapi import HTTPException, status

from ..config import get_settings
from ..adapters.archive_paths import entity_exists, latest_metadata_row
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
