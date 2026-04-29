from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


SITE_ROOT = Path(__file__).resolve().parents[1] / 'static'
INDEX_HTML = SITE_ROOT / 'index.html'


def create_app() -> FastAPI:
    app = FastAPI(title='starBoard public site', version='0.1.0')

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'service': 'starboard-public-site'}

    app.mount('/static', StaticFiles(directory=SITE_ROOT), name='static')

    @app.get('/')
    def root():
        return FileResponse(INDEX_HTML)

    return app


app = create_app()
