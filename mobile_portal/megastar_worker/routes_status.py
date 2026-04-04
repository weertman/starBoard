from __future__ import annotations

from fastapi import APIRouter

from .config import capability_status
from .models import WorkerStatusResponse

router = APIRouter()


@router.get('/status', response_model=WorkerStatusResponse)
def status():
    cap = capability_status()
    return {
        'enabled': bool(cap.enabled),
        'state': cap.state,
        'reason': getattr(cap, 'reason', None),
        'model_key': getattr(cap, 'model_key', None),
    }
