from __future__ import annotations

from typing import Literal
from fastapi import APIRouter, Depends, Query

from ..auth import require_authenticated_email
from ..models.api import ArchiveEntityResponse, ImageWindowResponse
from ..services.lookup_service import get_entity, get_entity_images
from ..services.audit import audit

router = APIRouter()


@router.get('/archive/entities/{entity_id}', response_model=ArchiveEntityResponse)
def archive_entity(
    entity_id: str,
    entity_type: Literal['gallery', 'query'] = Query('gallery'),
    window_size: int | None = Query(None),
    user_email: str = Depends(require_authenticated_email),
):
    result = get_entity(entity_type, entity_id, window_size)
    audit('lookup_entity', user_email, entity_type=entity_type, entity_id=entity_id)
    return result


@router.get('/archive/entities/{entity_id}/images', response_model=ImageWindowResponse)
def archive_entity_images(
    entity_id: str,
    entity_type: Literal['gallery', 'query'] = Query('gallery'),
    offset: int = Query(0, ge=0),
    limit: int | None = Query(None, gt=0),
    user_email: str = Depends(require_authenticated_email),
):
    result = get_entity_images(entity_type, entity_id, offset, limit)
    audit('lookup_entity_images', user_email, entity_type=entity_type, entity_id=entity_id, offset=offset, limit=limit)
    return result
