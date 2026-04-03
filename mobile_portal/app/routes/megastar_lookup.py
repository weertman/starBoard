from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..auth import require_authenticated_email
from ..config import get_megastar_capability_status
from ..models.megastar_api import MegaStarLookupResponse
from ..services.audit import audit
from ..services.megastar_lookup_service import MegaStarLookupUnavailable, get_megastar_lookup_service

router = APIRouter()


@router.post('/megastar/lookup', response_model=MegaStarLookupResponse)
async def megastar_lookup(
    file: UploadFile = File(...),
    user_email: str = Depends(require_authenticated_email),
):
    started = perf_counter()
    capability = get_megastar_capability_status()
    if not capability.enabled:
        payload = MegaStarLookupResponse(
            query_image_name=file.filename or '',
            status='unavailable',
            candidates=[],
            processing_ms=int((perf_counter() - started) * 1000),
            capability_state=capability.state,
            availability_reason=capability.reason,
        )
        audit(
            'megastar_lookup_unavailable',
            user_email,
            capability_state=capability.state,
            availability_reason=capability.reason or '',
            filename=file.filename or '',
        )
        await file.close()
        return JSONResponse(status_code=503, content=payload.model_dump())

    try:
        payload = await file.read()
        response = get_megastar_lookup_service().lookup_upload(
            filename=file.filename or 'upload.jpg',
            content=payload,
            content_type=file.content_type,
        )
    except MegaStarLookupUnavailable as exc:
        error_payload = MegaStarLookupResponse(
            query_image_name=file.filename or '',
            status='unavailable',
            candidates=[],
            processing_ms=int((perf_counter() - started) * 1000),
            capability_state='unavailable',
            availability_reason=str(exc),
        )
        audit(
            'megastar_lookup_unavailable',
            user_email,
            capability_state='unavailable',
            availability_reason=str(exc),
            filename=file.filename or '',
        )
        return JSONResponse(status_code=503, content=error_payload.model_dump())
    except HTTPException:
        raise
    finally:
        await file.close()

    audit(
        'megastar_lookup',
        user_email,
        filename=file.filename or '',
        content_type=file.content_type or '',
        result_status=response.status,
        candidate_count=len(response.candidates),
        top_entity_id=response.candidates[0].entity_id if response.candidates else '',
        processing_ms=response.processing_ms,
        capability_state=response.capability_state or '',
    )
    return response
