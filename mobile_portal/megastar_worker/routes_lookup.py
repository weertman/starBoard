from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from mobile_portal.app.models.megastar_api import MegaStarLookupResponse
from mobile_portal.app.services.megastar_lookup_service import MegaStarLookupUnavailable
from .config import capability_status
from .services.lookup_service import get_megastar_worker_lookup_service

router = APIRouter()


@router.post('/lookup', response_model=MegaStarLookupResponse)
async def lookup(file: UploadFile = File(...)):
    started = perf_counter()
    capability = capability_status()
    if not capability.enabled:
        payload = MegaStarLookupResponse(
            query_image_name=file.filename or '',
            status='unavailable',
            candidates=[],
            processing_ms=int((perf_counter() - started) * 1000),
            capability_state=capability.state,
            availability_reason=capability.reason,
        )
        await file.close()
        return JSONResponse(status_code=503, content=payload.model_dump())

    try:
        payload = await file.read()
        response = get_megastar_worker_lookup_service().lookup_upload(
            filename=file.filename or 'upload.jpg',
            content=payload,
            content_type=file.content_type,
        )
        return response
    except MegaStarLookupUnavailable as exc:
        error_payload = MegaStarLookupResponse(
            query_image_name=file.filename or '',
            status='unavailable',
            candidates=[],
            processing_ms=int((perf_counter() - started) * 1000),
            capability_state='unavailable',
            availability_reason=str(exc),
        )
        return JSONResponse(status_code=503, content=error_payload.model_dump())
    except HTTPException:
        raise
    finally:
        await file.close()
