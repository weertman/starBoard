from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from ..auth import require_authenticated_email
from ..models.search_api import FirstOrderMediaResponse, FirstOrderQueryOptionsResponse, FirstOrderSearchRequest, FirstOrderSearchResponse
from ..services.first_order_media_service import list_first_order_media, resized_preview_response, resolve_first_order_media_path
from ..services.first_order_service import list_first_order_query_options, run_first_order_search

router = APIRouter()


@router.get('/first-order/queries', response_model=FirstOrderQueryOptionsResponse)
def first_order_queries(
    _user_email: str = Depends(require_authenticated_email),
):
    return list_first_order_query_options()


@router.post('/first-order/search', response_model=FirstOrderSearchResponse)
def first_order_search(
    request: FirstOrderSearchRequest,
    _user_email: str = Depends(require_authenticated_email),
):
    return run_first_order_search(request.query_id, top_k=request.top_k, preset=request.preset)


@router.get('/first-order/queries/{query_id}/media', response_model=FirstOrderMediaResponse)
def first_order_query_media(
    query_id: str,
    _user_email: str = Depends(require_authenticated_email),
):
    return list_first_order_media('query', query_id)


@router.get('/first-order/candidates/{gallery_id}/media', response_model=FirstOrderMediaResponse)
def first_order_candidate_media(
    gallery_id: str,
    _user_email: str = Depends(require_authenticated_email),
):
    return list_first_order_media('gallery', gallery_id)


@router.get('/first-order/media/{image_id}/preview')
def first_order_media_preview(
    image_id: str,
    _user_email: str = Depends(require_authenticated_email),
):
    path = resolve_first_order_media_path(image_id)
    if path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='first_order_image_not_found')
    return resized_preview_response(path)


@router.get('/first-order/media/{image_id}/full')
def first_order_media_full(
    image_id: str,
    _user_email: str = Depends(require_authenticated_email),
):
    path = resolve_first_order_media_path(image_id)
    if path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='first_order_image_not_found')
    return FileResponse(path)
