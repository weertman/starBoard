# Mobile Portal Deployment

Target hostname:
- `mobile.fhl-star-board.com`

Runtime model:
- FastAPI service on `127.0.0.1:8091`
- Cloudflare Access in front of the hostname
- cloudflared ingress routes hostname to local FastAPI service
- service runs independently of desktop `main.py`

Required env:
- `STARBOARD_ARCHIVE_DIR`
- `STARBOARD_MOBILE_HOST`
- `STARBOARD_MOBILE_PORT`
- `STARBOARD_MOBILE_INITIAL_IMAGE_WINDOW`
- `STARBOARD_MOBILE_IMAGE_PAGE_SIZE`
- `STARBOARD_MOBILE_MAX_UPLOAD_MB`
- optional local dev only: `STARBOARD_MOBILE_CF_BYPASS_LOCALHOST=1`

Dev run:
```bash
cd /home/weertman/Documents/starBoard
mobile_portal/scripts/run_dev
```

Frontend build:
```bash
cd /home/weertman/Documents/starBoard/mobile_portal/frontend
npm install
npm run build
```
