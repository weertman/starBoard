from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse

from src.data.activity_log import request_context, try_record_activity_event

from ..auth import require_authenticated_email
from ..models.search_api import FirstOrderGalleryFiltersResponse, FirstOrderMatchLabelRequest, FirstOrderMatchLabelResponse, FirstOrderMediaResponse, FirstOrderQueryOptionsResponse, FirstOrderSearchRequest, FirstOrderSearchResponse
from ..services.first_order_media_service import list_first_order_media, resized_preview_response, resolve_first_order_media_path
from ..services.first_order_service import list_first_order_gallery_filter_options, list_first_order_query_options, run_first_order_search, save_first_order_match_label

router = APIRouter()


@router.get('/first-order/queries', response_model=FirstOrderQueryOptionsResponse)
def first_order_queries(
    _user_email: str = Depends(require_authenticated_email),
):
    return list_first_order_query_options()


@router.get('/first-order/gallery-filters', response_model=FirstOrderGalleryFiltersResponse)
def first_order_gallery_filters(
    _user_email: str = Depends(require_authenticated_email),
):
    return list_first_order_gallery_filter_options()


@router.post('/first-order/search', response_model=FirstOrderSearchResponse)
def first_order_search(
    request: FirstOrderSearchRequest,
    http_request: Request,
    user_email: str = Depends(require_authenticated_email),
):
    result = run_first_order_search(
        request.query_id,
        top_k=request.top_k,
        preset=request.preset,
        query_image_id=request.query_image_id,
        gallery_filters=request.gallery_filters,
    )
    ctx = request_context(http_request)
    session_id = ctx.pop('session_id') or ''
    try_record_activity_event(
        surface='star_browser',
        user_email=user_email,
        session_id=session_id,
        event_type='query_matcher.search.succeeded',
        workflow='query_matcher',
        entity_type='query',
        entity_id=request.query_id,
        query_id=request.query_id,
        success=True,
        details={
            'preset': request.preset,
            'top_k': request.top_k,
            'query_image_id': request.query_image_id,
            'gallery_filters': request.gallery_filters or {},
            'result_count': len(result.candidates),
        },
        **ctx,
    )
    return result


@router.post('/first-order/match-labels', response_model=FirstOrderMatchLabelResponse)
def first_order_match_labels(
    request: FirstOrderMatchLabelRequest,
    http_request: Request,
    user_email: str = Depends(require_authenticated_email),
):
    try:
        result = save_first_order_match_label(request.query_id, request.gallery_id, request.verdict, request.notes)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    ctx = request_context(http_request)
    session_id = ctx.pop('session_id') or ''
    try_record_activity_event(
        surface='star_browser',
        user_email=user_email,
        session_id=session_id,
        event_type='query_matcher.match_label.saved',
        workflow='query_matcher',
        entity_type='gallery',
        entity_id=request.gallery_id,
        query_id=request.query_id,
        gallery_id=request.gallery_id,
        success=True,
        details={'verdict': request.verdict},
        **ctx,
    )
    return result


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
