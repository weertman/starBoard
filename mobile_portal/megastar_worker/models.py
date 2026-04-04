from __future__ import annotations

from typing import Literal
from pydantic import BaseModel


class WorkerHealthResponse(BaseModel):
    status: Literal['ok']
    service: str
    version: str


class WorkerStatusResponse(BaseModel):
    enabled: bool
    state: Literal['enabled', 'disabled', 'unavailable']
    reason: str | None = None
    model_key: str | None = None
