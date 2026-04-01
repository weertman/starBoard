from __future__ import annotations

from typing import Annotated
from fastapi import APIRouter, Depends, File, Form, UploadFile

from ..auth import require_authenticated_email
from ..models.api import SubmissionResponse
from ..services.submission_service import submit
from ..services.audit import audit

router = APIRouter()


@router.post('/submissions', response_model=SubmissionResponse)
async def submissions(
    payload: Annotated[str, Form(...)],
    files: Annotated[list[UploadFile], File(...)],
    user_email: str = Depends(require_authenticated_email),
):
    result = await submit(payload, files)
    audit('submission', user_email, entity_type=result['entity_type'], entity_id=result['entity_id'], accepted_images=result['accepted_images'])
    return result
