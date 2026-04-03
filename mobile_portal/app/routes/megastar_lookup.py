from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse

from ..auth import require_authenticated_email
from ..config import get_megastar_capability_status
from ..models.megastar_api import MegaStarLookupResponse
from ..services.audit import audit

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
        return JSONResponse(status_code=503, content=payload.model_dump())

    payload = MegaStarLookupResponse(
        query_image_name=file.filename or '',
        status='unavailable',
        candidates=[],
        processing_ms=int((perf_counter() - started) * 1000),
        capability_state=capability.state,
        availability_reason='not_implemented',
    )
    audit('megastar_lookup_stub', user_email, filename=file.filename or '', model_key=capability.model_key or '')
    return JSONResponse(status_code=501, content=payload.model_dump())
