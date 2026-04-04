from __future__ import annotations

from fastapi import APIRouter

from .models import WorkerHealthResponse

router = APIRouter()


@router.get('/health', response_model=WorkerHealthResponse)
def health():
    return {
        'status': 'ok',
        'service': 'starboard-megastar-worker',
        'version': '0.1.0',
    }
