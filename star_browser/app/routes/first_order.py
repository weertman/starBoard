from __future__ import annotations

from fastapi import APIRouter, Depends

from ..auth import require_authenticated_email
from ..models.search_api import FirstOrderSearchRequest, FirstOrderSearchResponse
from ..services.first_order_service import run_first_order_search

router = APIRouter()


@router.post('/first-order/search', response_model=FirstOrderSearchResponse)
def first_order_search(
    request: FirstOrderSearchRequest,
    _user_email: str = Depends(require_authenticated_email),
):
    return run_first_order_search(request.query_id, top_k=request.top_k)
