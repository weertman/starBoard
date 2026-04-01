from __future__ import annotations

from fastapi import Header, HTTPException, Request, status

from .config import get_settings


AUTH_EMAIL_HEADER = 'cf-access-authenticated-user-email'


def require_authenticated_email(
    request: Request,
    cf_access_authenticated_user_email: str | None = Header(default=None),
) -> str:
    settings = get_settings()
    if cf_access_authenticated_user_email:
        return cf_access_authenticated_user_email

    client_host = request.client.host if request.client else None
    if settings.cf_bypass_localhost and client_host in {'127.0.0.1', '::1', 'testclient', None}:
        return 'localhost-bypass@local'

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Cloudflare Access authentication required',
    )
