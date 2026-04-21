from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from ..auth import require_authenticated_email
from ..models.gallery_api import GalleryEntityResponse
from ..adapters.gallery_adapter import resolve_gallery_image_path
from ..services.gallery_service import GalleryNotFoundError, get_gallery_entity

router = APIRouter()


@router.get('/gallery/entities/{entity_id}', response_model=GalleryEntityResponse)
def gallery_entity(entity_id: str, _user_email: str = Depends(require_authenticated_email)):
    try:
        return get_gallery_entity(entity_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='gallery_not_found')


@router.get('/gallery/media/{image_id}/full')
def gallery_media_full(image_id: str, _user_email: str = Depends(require_authenticated_email)):
    path = resolve_gallery_image_path(image_id)
    if path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='gallery_image_not_found')
    return FileResponse(path)


@router.get('/gallery/media/{image_id}/preview')
def gallery_media_preview(image_id: str, _user_email: str = Depends(require_authenticated_email)):
    path = resolve_gallery_image_path(image_id)
    if path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='gallery_image_not_found')
    return FileResponse(path)
