from __future__ import annotations

import os
from fastapi import Header, HTTPException, Request, status


def require_authenticated_email(
    request: Request,
    cf_email: str | None = Header(default=None, alias='cf-access-authenticated-user-email'),
) -> str:
    if cf_email:
        return cf_email

    allow_localhost = os.getenv('STAR_BROWSER_CF_BYPASS_LOCALHOST', '0') == '1'
    if allow_localhost and request.client and request.client.host in {'127.0.0.1', '::1'}:
        return 'local@example.invalid'

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Cloudflare Access authentication required',
    )
