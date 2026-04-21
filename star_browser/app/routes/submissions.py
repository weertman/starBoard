from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ..auth import require_authenticated_email
from ..models.data_entry_api import SubmissionResponse
from ..services.submission_service import submit

router = APIRouter()


@router.post('/submissions', response_model=SubmissionResponse)
async def submissions(
    payload: Annotated[str, Form(...)],
    files: Annotated[list[UploadFile], File(...)],
    _user_email: str = Depends(require_authenticated_email),
):
    return await submit(payload, files)
