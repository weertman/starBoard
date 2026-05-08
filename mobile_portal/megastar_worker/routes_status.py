from __future__ import annotations

from fastapi import APIRouter

from .config import capability_status
from .models import WorkerStatusResponse


def _queue_status():
    try:
        from src.dl.megastar_queue import get_queue_status
        return get_queue_status()
    except Exception as exc:
        return {'error': str(exc)}

router = APIRouter()


@router.get('/status', response_model=WorkerStatusResponse)
def status():
    cap = capability_status()
    return {
        'enabled': bool(cap.enabled),
        'state': cap.state,
        'reason': getattr(cap, 'reason', None),
        'model_key': getattr(cap, 'model_key', None),
        'megastar_queue': _queue_status(),
    }
