from __future__ import annotations

from fastapi import FastAPI

from .routes.batch_upload import router as batch_upload_router
from .routes.first_order import router as first_order_router
from .routes.gallery import router as gallery_router
from .routes.session import router as session_router


def create_app() -> FastAPI:
    app = FastAPI(title='star_browser', version='0.1.0')

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'service': 'star-browser'}

    app.include_router(session_router, prefix='/api')
    app.include_router(gallery_router, prefix='/api')
    app.include_router(batch_upload_router, prefix='/api')
    app.include_router(first_order_router, prefix='/api')

    return app


app = create_app()
