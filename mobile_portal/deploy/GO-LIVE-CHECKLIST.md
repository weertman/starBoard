# Mobile Portal + MegaStar Worker Go-Live Checklist

Target public app
- https://mobile.fhl-star-board.com

Authentication model
- Public access is controlled by Cloudflare Access
- The portal trusts the Cloudflare-injected authenticated email header:
  - `cf-access-authenticated-user-email`
- The MegaStar worker is not public
- The worker stays bound to localhost only and is reachable only by the portal process

Final runtime topology
- Cloudflare Access
  -> `mobile.fhl-star-board.com`
  -> cloudflared tunnel ingress
  -> portal FastAPI on `127.0.0.1:8091`
  -> internal MegaStar worker on `127.0.0.1:8093`

Important principle
- Only the portal is public
- The worker must remain private/internal

---

## 1. Preconditions

Confirm these files/paths exist on the machine:
- repo: `/home/weertman/Documents/starBoard`
- archive: `/home/weertman/Documents/starBoard/archive`
- main checkpoint:
  - `/home/weertman/Documents/starBoard/star_identification/checkpoints/default/best.pth`
- refreshed retrieval artifacts under:
  - `/home/weertman/Documents/starBoard/archive/_dl_precompute/default_megastarid_v1`
- registry:
  - `/home/weertman/Documents/starBoard/archive/_dl_precompute/_dl_registry.json`

Confirm registry freshness:
- `active_model == default_megastarid_v1`
- `pending_ids.gallery == []`
- `pending_ids.queries == []`

Verification command:
```bash
cd /home/weertman/Documents/starBoard
python3 - <<'PY'
import json
from pathlib import Path
p = Path('archive/_dl_precompute/_dl_registry.json')
obj = json.loads(p.read_text())
print('active_model=', obj.get('active_model'))
print('pending_gallery=', len(obj.get('pending_ids', {}).get('gallery', [])))
print('pending_queries=', len(obj.get('pending_ids', {}).get('queries', [])))
print('checkpoint_path=', obj['models'][obj['active_model']]['checkpoint_path'])
PY
```

Expected:
- active_model = default_megastarid_v1
- pending_gallery = 0
- pending_queries = 0

---

## 2. Build frontend

```bash
cd /home/weertman/Documents/starBoard/mobile_portal/frontend
npm install
npm run build
```

Expected:
- successful Vite build into `mobile_portal/frontend/dist`

---

## 3. Start the internal MegaStar worker

Recommended worker bind:
- `127.0.0.1:8093`

Manual start command:
```bash
cd /home/weertman/Documents/starBoard
STARBOARD_ARCHIVE_DIR=/home/weertman/Documents/starBoard/archive \
STARBOARD_MEGASTAR_WORKER_ENABLED=1 \
STARBOARD_MEGASTAR_MODEL_KEY=default_megastarid_v1 \
STARBOARD_MEGASTAR_REQUIRE_FRESH_ASSETS=1 \
STARBOARD_MEGASTAR_WORKER_HOST=127.0.0.1 \
STARBOARD_MEGASTAR_WORKER_PORT=8093 \
bash ./scripts/python -m uvicorn mobile_portal.megastar_worker.main:app --host 127.0.0.1 --port 8093
```

Health checks:
```bash
curl http://127.0.0.1:8093/health
curl http://127.0.0.1:8093/status
```

Expected:
- `/health` => `{"status":"ok", ...}`
- `/status` => enabled true, state enabled, model_key default_megastarid_v1

---

## 4. Start the public-facing portal in worker mode

Manual start command:
```bash
cd /home/weertman/Documents/starBoard
STARBOARD_ARCHIVE_DIR=/home/weertman/Documents/starBoard/archive \
STARBOARD_MOBILE_HOST=127.0.0.1 \
STARBOARD_MOBILE_PORT=8091 \
STARBOARD_MOBILE_MEGASTAR_ENABLED=1 \
STARBOARD_MOBILE_MEGASTAR_BACKEND=worker \
STARBOARD_MOBILE_MEGASTAR_WORKER_URL=http://127.0.0.1:8093 \
STARBOARD_MOBILE_MEGASTAR_REQUIRE_FRESH_ASSETS=1 \
bash ./scripts/python -m uvicorn mobile_portal.app.main:app --host 127.0.0.1 --port 8091
```

Important
- Do not set `STARBOARD_MOBILE_CF_BYPASS_LOCALHOST=1` in production
- That bypass is local testing only

Portal checks:
```bash
curl http://127.0.0.1:8091/api/health
curl -H 'cf-access-authenticated-user-email: field@example.org' http://127.0.0.1:8091/api/session
```

Expected session result:
- `megastar_lookup.enabled == true`
- `megastar_lookup.backend == worker`
- `megastar_lookup.state == enabled`

---

## 5. Cloudflare tunnel ingress

The tunnel should expose only the portal, not the worker.

Public hostname route:
- hostname: `mobile.fhl-star-board.com`
- service: `http://127.0.0.1:8091`

Do NOT add:
- any public route to `127.0.0.1:8093`

If dashboard-managed tunnel ingress is used:
- add hostname `mobile.fhl-star-board.com`
- point to `http://127.0.0.1:8091`

If file-managed ingress is used, rule is:
```yaml
ingress:
  - hostname: mobile.fhl-star-board.com
    service: http://127.0.0.1:8091
  - service: http_status:404
```

---

## 6. Cloudflare Access policy

Use Cloudflare Access in front of `mobile.fhl-star-board.com`.

Desired auth behavior:
- Cloudflare authenticates users
- Cloudflare injects `cf-access-authenticated-user-email`
- portal reads that header
- worker has no external auth surface because it is not public

Policy requirement:
- allow the intended email identities or groups

Verification:
```bash
curl -I https://mobile.fhl-star-board.com
```

Expected:
- unauthenticated: redirect/challenge to Access login
- authenticated browser session: 200

---

## 7. Functional verification

After login, verify in browser:
1. home screen loads
2. New Observation opens
3. MegaStar section is visible
4. MegaStar no longer says unavailable
5. add/select a local image
6. tap MegaStar Lookup
7. ranked candidate list appears
8. `Compare here` works
9. `Open in archive browser` works
10. metadata/submit flow still works and remains separate

API smoke test through portal:
```bash
python3 - <<'PY'
from PIL import Image
img = Image.new('RGB', (160, 120), (200, 120, 80))
img.save('/tmp/megastar_smoke.jpg', format='JPEG')
print('/tmp/megastar_smoke.jpg')
PY
curl -H 'cf-access-authenticated-user-email: field@example.org' \
  -F file=@/tmp/megastar_smoke.jpg \
  http://127.0.0.1:8091/api/megastar/lookup
```

Expected:
- `status` one of `ok`, `weak`, or `empty`
- not `unavailable`

---

## 8. Suggested systemd split

Recommended units:
- `starboard-mobile-portal.service`
- `starboard-megastar-worker.service`
- existing `cloudflared-starboard.service`

Example unit files included in this repo:
- `mobile_portal/deploy/starboard-mobile-portal.service.example`
- `mobile_portal/deploy/starboard-megastar-worker.service.example`

Operational goal:
- restart worker without restarting portal
- stop worker and leave portal up
- public hostname remains the portal only

Minimal example layout
- portal unit runs uvicorn on `127.0.0.1:8091`
- worker unit runs uvicorn on `127.0.0.1:8093`
- cloudflared routes hostname to `127.0.0.1:8091`

---

## 9. Rollback

If MegaStar misbehaves but portal should stay live:
- stop worker service
or
- set `STARBOARD_MOBILE_MEGASTAR_ENABLED=0` on the portal

Expected result:
- portal remains available
- MegaStar section becomes unavailable/disabled
- archive browsing and submission still work

If a bad artifact refresh caused trouble:
- restore backup registry/artifact snapshot
- restart worker and portal

Current backup artifacts created during refresh:
- `archive/_dl_precompute/_dl_registry.json.bak.20260404-122532`
- `archive/_dl_precompute/default_megastarid_v1.bak.20260404-122645.tar.gz`

---

## 10. Go-live criteria

Go live only when all are true:
- portal frontend build succeeds
- worker `/health` succeeds
- worker `/status` reports enabled
- portal `/api/session` reports `megastar_lookup.enabled == true`
- Cloudflare Access protects `mobile.fhl-star-board.com`
- worker remains localhost-only
- end-to-end MegaStar lookup works through portal
- no archive write regressions in normal metadata/submit flow
