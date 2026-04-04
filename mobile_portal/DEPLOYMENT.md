# Mobile Portal Deployment

Target hostname:
- `mobile.fhl-star-board.com`

Runtime model:
- Public portal FastAPI service on `127.0.0.1:8091`
- Internal MegaStar worker on `127.0.0.1:8093`
- Cloudflare Access in front of the public hostname only
- cloudflared ingress routes hostname to the local portal service
- worker remains localhost-only and is called internally by the portal
- services run independently of desktop `main.py`

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
