from __future__ import annotations

from ..adapters.gallery_adapter import load_gallery_entity
from ..models.gallery_api import GalleryEntityResponse


class GalleryNotFoundError(Exception):
    pass


def get_gallery_entity(entity_id: str) -> GalleryEntityResponse:
    metadata_summary, encounters, images = load_gallery_entity(entity_id)
    if not metadata_summary and not encounters and not images:
        raise GalleryNotFoundError(entity_id)
    return GalleryEntityResponse(
        entity_id=entity_id,
        metadata_summary=metadata_summary,
        encounters=encounters,
        images=images,
    )
