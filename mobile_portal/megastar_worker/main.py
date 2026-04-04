from __future__ import annotations

from fastapi import FastAPI
from .routes_health import router as health_router
from .routes_status import router as status_router
from .routes_lookup import router as lookup_router


def create_app() -> FastAPI:
    app = FastAPI(title='starBoard MegaStar Worker', version='0.1.0')
    app.include_router(health_router)
    app.include_router(status_router)
    app.include_router(lookup_router)
    return app


app = create_app()
