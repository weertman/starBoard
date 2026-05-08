from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from ..auth import require_authenticated_email
from ..models.gallery_api import GalleryEntityResponse, IdReviewMetadataUpdateRequest, IdReviewOptionsResponse, IdReviewRenameRequest, SetBestImageResponse
from ..adapters.gallery_adapter import resolve_gallery_image_path, resolve_id_review_image_path
from ..services.gallery_service import (
    GalleryNotFoundError,
    IdReviewEditError,
    get_gallery_entity,
    get_id_review_entity,
    get_id_review_options,
    rename_id_review_entity_and_load,
    set_id_review_first_image,
    update_id_review_metadata_and_load,
)

router = APIRouter()


@router.get('/gallery/entities/{entity_id}', response_model=GalleryEntityResponse)
def gallery_entity(entity_id: str, _user_email: str = Depends(require_authenticated_email)):
    try:
        return get_gallery_entity(entity_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='gallery_not_found')


@router.get('/id-review/options/{archive_type}', response_model=IdReviewOptionsResponse)
def id_review_options(archive_type: str, _user_email: str = Depends(require_authenticated_email)):
    if archive_type not in {'gallery', 'query'}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='archive_type_not_found')
    return get_id_review_options(archive_type)


@router.get('/id-review/entities/{archive_type}/{entity_id}', response_model=GalleryEntityResponse)
def id_review_entity(archive_type: str, entity_id: str, _user_email: str = Depends(require_authenticated_email)):
    if archive_type not in {'gallery', 'query'}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='archive_type_not_found')
    try:
        return get_id_review_entity(archive_type, entity_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='id_not_found')


@router.patch('/id-review/entities/{archive_type}/{entity_id}/id', response_model=GalleryEntityResponse)
def id_review_rename_entity(
    archive_type: str,
    entity_id: str,
    request: IdReviewRenameRequest,
    _user_email: str = Depends(require_authenticated_email),
):
    if archive_type not in {'gallery', 'query'}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='archive_type_not_found')
    try:
        return rename_id_review_entity_and_load(archive_type, entity_id, request.new_entity_id)
    except IdReviewEditError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except GalleryNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='id_not_found')


@router.patch('/id-review/entities/{archive_type}/{entity_id}/metadata', response_model=GalleryEntityResponse)
def id_review_update_metadata(
    archive_type: str,
    entity_id: str,
    request: IdReviewMetadataUpdateRequest,
    _user_email: str = Depends(require_authenticated_email),
):
    if archive_type not in {'gallery', 'query'}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='archive_type_not_found')
    try:
        return update_id_review_metadata_and_load(archive_type, entity_id, request.metadata)
    except IdReviewEditError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except GalleryNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='id_not_found')


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


@router.get('/id-review/media/{image_id}/full')
def id_review_media_full(image_id: str, _user_email: str = Depends(require_authenticated_email)):
    path = resolve_id_review_image_path(image_id)
    if path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='id_review_image_not_found')
    return FileResponse(path)


@router.get('/id-review/media/{image_id}/preview')
def id_review_media_preview(image_id: str, _user_email: str = Depends(require_authenticated_email)):
    path = resolve_id_review_image_path(image_id)
    if path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='id_review_image_not_found')
    return FileResponse(path)


@router.post('/id-review/media/{image_id}/set-first', response_model=SetBestImageResponse)
def id_review_set_first_image(image_id: str, _user_email: str = Depends(require_authenticated_email)):
    try:
        return set_id_review_first_image(image_id)
    except GalleryNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='id_review_image_not_found')
