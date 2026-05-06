from __future__ import annotations

from typing import Any
import json

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.data.activity_log import record_activity_event, request_context

from ..auth import require_authenticated_email

router = APIRouter()
MAX_ACTIVITY_EVENTS_PER_REQUEST = 100
MAX_ACTIVITY_EVENT_TYPE_LENGTH = 120
MAX_ACTIVITY_DETAILS_BYTES = 8192


class ActivityEventIn(BaseModel):
    event_type: str
    client_timestamp_utc: str | None = None
    workflow: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    query_id: str | None = None
    gallery_id: str | None = None
    success: bool | None = None
    duration_ms: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ActivityEventsRequest(BaseModel):
    session_id: str | None = None
    events: list[ActivityEventIn]


@router.post('/activity/events')
def activity_events(
    payload: ActivityEventsRequest,
    request: Request,
    user_email: str = Depends(require_authenticated_email),
):
    ctx = request_context(request)
    header_session_id = ctx.pop('session_id') or ''
    session_id = payload.session_id or header_session_id
    if len(payload.events) > MAX_ACTIVITY_EVENTS_PER_REQUEST:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail='too_many_activity_events')
    for event in payload.events:
        if len(event.event_type) > MAX_ACTIVITY_EVENT_TYPE_LENGTH:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='activity_event_type_too_long')
        if len(json.dumps(event.details, ensure_ascii=False, default=str)) > MAX_ACTIVITY_DETAILS_BYTES:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail='activity_event_details_too_large')
        record_activity_event(
            surface='mobile_portal',
            user_email=user_email,
            session_id=session_id,
            event_type=event.event_type,
            client_timestamp_utc=event.client_timestamp_utc,
            workflow=event.workflow,
            entity_type=event.entity_type,
            entity_id=event.entity_id,
            query_id=event.query_id,
            gallery_id=event.gallery_id,
            success=event.success,
            duration_ms=event.duration_ms,
            details=event.details,
            **ctx,
        )
    return {'accepted_events': len(payload.events)}
