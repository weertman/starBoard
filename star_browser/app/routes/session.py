from __future__ import annotations

from fastapi import APIRouter, Depends

from ..auth import require_authenticated_email
from ..models.session_api import SessionResponse

router = APIRouter()


@router.get('/session', response_model=SessionResponse)
def session(user_email: str = Depends(require_authenticated_email)):
    return SessionResponse(
        authenticated_email=user_email,
        capabilities={
            'single_entry': True,
            'batch_upload': True,
            'first_order': True,
            'gallery_review': True,
        },
    )
