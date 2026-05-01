from __future__ import annotations

from ..adapters.gallery_adapter import load_id_review_entity
from ..models.gallery_api import GalleryEntityResponse


class GalleryNotFoundError(Exception):
    pass


def get_gallery_entity(entity_id: str) -> GalleryEntityResponse:
    return get_id_review_entity('gallery', entity_id)


def get_id_review_entity(archive_type: str, entity_id: str) -> GalleryEntityResponse:
    metadata_summary, metadata_rows, timeline, encounters, images = load_id_review_entity(archive_type, entity_id)
    if not metadata_summary and not metadata_rows and not timeline and not encounters and not images:
        raise GalleryNotFoundError(entity_id)
    return GalleryEntityResponse(
        archive_type=archive_type,
        entity_id=entity_id,
        metadata_summary=metadata_summary,
        metadata_rows=metadata_rows,
        timeline=timeline,
        encounters=encounters,
        images=images,
    )
