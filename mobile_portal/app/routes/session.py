from __future__ import annotations
from fastapi import APIRouter, Depends, Request

from src.data.activity_log import request_context, try_record_activity_event

from ..auth import require_authenticated_email
from ..services.megastar_backend_selector import get_megastar_capability_status
from ..models.api import SessionResponse

router = APIRouter()


@router.get('/session', response_model=SessionResponse)
def session(request: Request, user_email: str = Depends(require_authenticated_email)):
    ctx = request_context(request)
    session_id = ctx.pop('session_id') or ''
    try_record_activity_event(
        surface='mobile_portal',
        user_email=user_email,
        session_id=session_id,
        event_type='session.loaded',
        workflow='session',
        success=True,
        **ctx,
    )
    megastar = get_megastar_capability_status()
    return {
        'authenticated_email': user_email,
        'capabilities': {
            'lookup': True,
            'submit_query': True,
            'submit_gallery': True,
            'megastar_lookup': megastar.enabled,
        },
        'megastar_lookup': {
            'enabled': megastar.enabled,
            'state': megastar.state,
            'backend': megastar.backend,
            'reason': megastar.reason,
            'model_key': megastar.model_key,
        },
    }
