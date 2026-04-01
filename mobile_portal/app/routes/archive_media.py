from __future__ import annotations

from fastapi import APIRouter, Depends

from ..auth import require_authenticated_email
from ..services.media_service import get_full_image_response, get_preview_image_response
from ..services.audit import audit

router = APIRouter()


@router.get('/archive/media/{image_id}/full')
def archive_media_full(image_id: str, user_email: str = Depends(require_authenticated_email)):
    audit('media_full', user_email, image_id=image_id)
    return get_full_image_response(image_id)


@router.get('/archive/media/{image_id}/preview')
def archive_media_preview(image_id: str, user_email: str = Depends(require_authenticated_email)):
    audit('media_preview', user_email, image_id=image_id)
    return get_preview_image_response(image_id)
