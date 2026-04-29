from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .routes.batch_upload import router as batch_upload_router
from .routes.first_order import router as first_order_router
from .routes.gallery import router as gallery_router
from .routes.locations import router as locations_router
from .routes.megastar_lookup import router as megastar_lookup_router
from .routes.schema import router as schema_router
from .routes.session import router as session_router
from .routes.submissions import router as submissions_router


def create_app() -> FastAPI:
    app = FastAPI(title='star_browser', version='0.1.0')

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'service': 'star-browser'}

    app.include_router(session_router, prefix='/api')
    app.include_router(gallery_router, prefix='/api')
    app.include_router(batch_upload_router, prefix='/api')
    app.include_router(first_order_router, prefix='/api')
    app.include_router(locations_router, prefix='/api')
    app.include_router(megastar_lookup_router, prefix='/api')
    app.include_router(schema_router, prefix='/api')
    app.include_router(submissions_router, prefix='/api')

    built_assets_dir = Path(__file__).resolve().parents[1] / 'frontend' / 'dist' / 'assets'
    built_index = Path(__file__).resolve().parents[1] / 'frontend' / 'dist' / 'index.html'

    if built_assets_dir.exists():
        app.mount('/assets', StaticFiles(directory=built_assets_dir), name='assets')

    @app.get('/')
    def root():
        if built_index.exists():
            return FileResponse(
                built_index,
                headers={
                    'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
                    'Pragma': 'no-cache',
                    'Expires': '0',
                },
            )
        return {'status': 'ok', 'service': 'star-browser'}

    @app.get('/batch-upload')
    def batch_upload_app_entry():
        if built_index.exists():
            return FileResponse(
                built_index,
                headers={
                    'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
                    'Pragma': 'no-cache',
                    'Expires': '0',
                },
            )
        return Response(status_code=404)

    return app


app = create_app()
