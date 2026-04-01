from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routes.health import router as health_router
from .routes.session import router as session_router
from .routes.schema import router as schema_router
from .routes.archive_lookup import router as archive_lookup_router
from .routes.archive_media import router as archive_media_router
from .routes.submissions import router as submissions_router


def create_app() -> FastAPI:
    app = FastAPI(title='starBoard Mobile Portal', version='0.1.0')
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )
    app.include_router(health_router, prefix='/api')
    app.include_router(session_router, prefix='/api')
    app.include_router(schema_router, prefix='/api')
    app.include_router(archive_lookup_router, prefix='/api')
    app.include_router(archive_media_router, prefix='/api')
    app.include_router(submissions_router, prefix='/api')

    built_assets_dir = Path(__file__).resolve().parents[1] / 'frontend' / 'dist' / 'assets'
    fallback_assets_dir = Path(__file__).resolve().parent / 'static'
    built_index = Path(__file__).resolve().parents[1] / 'frontend' / 'dist' / 'index.html'
    fallback_index = Path(__file__).resolve().parent / 'templates' / 'index.html'

    if built_assets_dir.exists():
        app.mount('/assets', StaticFiles(directory=built_assets_dir), name='assets')
    elif fallback_assets_dir.exists():
        app.mount('/assets', StaticFiles(directory=fallback_assets_dir), name='assets')

    @app.get('/')
    def root():
        if built_index.exists():
            return FileResponse(built_index)
        if fallback_index.exists():
            return FileResponse(fallback_index)
        return {'status': 'ok', 'service': 'starboard-mobile-portal'}

    return app


logging.getLogger('starboard.mobile_portal').setLevel(logging.INFO)
app = create_app()
