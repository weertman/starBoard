from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import JSONResponse

from mobile_portal.app.models.megastar_api import MegaStarLookupResponse
from mobile_portal.app.services.megastar_backend_selector import get_megastar_capability_status, get_megastar_lookup_backend
from mobile_portal.app.services.megastar_lookup_service import MegaStarLookupUnavailable

from ..auth import require_authenticated_email

router = APIRouter()


def _capability_payload():
    capability = get_megastar_capability_status()
    return {
        'enabled': capability.enabled,
        'state': capability.state,
        'backend': capability.backend,
        'reason': capability.reason,
        'model_key': capability.model_key,
    }


@router.get('/megastar/status')
def megastar_status(_user_email: str = Depends(require_authenticated_email)):
    return _capability_payload()


@router.post('/megastar/lookup', response_model=MegaStarLookupResponse)
async def megastar_lookup(
    file: UploadFile = File(...),
    max_candidates: int = Query(default=5, ge=1, le=50),
    _user_email: str = Depends(require_authenticated_email),
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
        await file.close()
        return JSONResponse(status_code=503, content=payload.model_dump())

    try:
        payload = await file.read()
        return get_megastar_lookup_backend().lookup_upload(
            filename=file.filename or 'upload.jpg',
            content=payload,
            content_type=file.content_type,
            max_candidates=max_candidates,
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
        return JSONResponse(status_code=503, content=error_payload.model_dump())
    finally:
        await file.close()
