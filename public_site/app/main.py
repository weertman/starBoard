from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException


SITE_ROOT = Path(__file__).resolve().parents[1] / 'static'
INDEX_HTML = SITE_ROOT / 'index.html'

SECURITY_HEADERS = {
    'Content-Security-Policy': "default-src 'self'; style-src 'self'; script-src 'self' https://static.cloudflareinsights.com; img-src 'self' data: https:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
    'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
}


def create_app() -> FastAPI:
    app = FastAPI(title='starBoard public site', version='0.1.0')

    @app.middleware('http')
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    @app.exception_handler(StarletteHTTPException)
    async def not_found_handler(request: Request, exc: StarletteHTTPException):
        if exc.status_code != 404 or request.url.path.startswith('/api/'):
            return JSONResponse({'detail': exc.detail}, status_code=exc.status_code)
        return HTMLResponse((SITE_ROOT / '404.html').read_text(), status_code=404)

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'service': 'starboard-public-site'}

    app.mount('/static', StaticFiles(directory=SITE_ROOT), name='static')

    def static_page(name: str) -> FileResponse:
        return FileResponse(SITE_ROOT / name)

    @app.api_route('/', methods=['GET', 'HEAD'])
    def root():
        return static_page('index.html')

    @app.api_route('/contact', methods=['GET', 'HEAD'])
    def contact():
        return static_page('contact.html')

    @app.api_route('/lab', methods=['GET', 'HEAD'])
    def lab():
        return static_page('lab.html')

    return app


app = create_app()
