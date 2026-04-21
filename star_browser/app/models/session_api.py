from __future__ import annotations

from pydantic import BaseModel


class SessionResponse(BaseModel):
    authenticated_email: str
    capabilities: dict[str, bool]
