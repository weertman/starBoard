from __future__ import annotations

from typing import Annotated
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from src.data.activity_log import request_context, try_record_activity_event

from ..auth import require_authenticated_email
from ..models.api import SubmissionResponse
from ..services.submission_service import submit
from ..services.audit import audit

router = APIRouter()


@router.post('/submissions', response_model=SubmissionResponse)
async def submissions(
    request: Request,
    payload: Annotated[str, Form(...)],
    files: Annotated[list[UploadFile], File(...)],
    user_email: str = Depends(require_authenticated_email),
):
    result = await submit(payload, files)
    audit('submission', user_email, entity_type=result['entity_type'], entity_id=result['entity_id'], accepted_images=result['accepted_images'])
    ctx = request_context(request)
    session_id = ctx.pop('session_id') or ''
    try_record_activity_event(
        surface='mobile_portal',
        user_email=user_email,
        session_id=session_id,
        event_type='mobile.submission.succeeded',
        workflow='observation',
        entity_type=result['entity_type'],
        entity_id=result['entity_id'],
        success=True,
        details={
            'accepted_images': result['accepted_images'],
            'skipped_images': result.get('skipped_images', 0),
            'encounter_folder': result['encounter_folder'],
        },
        **ctx,
    )
    return result
