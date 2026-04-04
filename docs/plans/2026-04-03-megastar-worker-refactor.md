# MegaStar Worker Refactor Implementation Plan

> For Hermes: Use subagent-driven-development skill to implement this plan task-by-task.

Goal: Refactor MegaStar lookup so it can work in both cases: (1) direct in-process execution inside the mobile portal for development/fallback, and (2) a separate backend worker/service for cleaner production operation, while preserving the current mobile portal workflow and fail-closed safety guarantees.

Architecture: The mobile portal FastAPI app remains the main user-facing web API for auth, archive browsing, metadata, and submission. MegaStar execution should support two backends behind one stable portal contract: a local in-process backend and a dedicated worker FastAPI service with its own health/capability endpoints, model lifecycle, artifact loading, and retrieval execution. The portal computes MegaStar capability from local feature flags plus backend availability, and routes requests through a backend selector that can use either direct local execution or worker proxy mode.

Tech Stack: FastAPI, Python adapters/services in mobile_portal/app, a new MegaStar worker app under mobile_portal/megastar_worker or similar, existing src/dl and star_identification/megastarid code, current portal frontend React/TypeScript, optional systemd service config for the worker.

---

Hard constraints
- Preserve current archive browsing and submission flows
- Both local mode and worker mode must use the same chosen model checkpoint / model key source of truth
- Keep MegaStar advisory-only in v1 of worker refactor
- No archive writes from MegaStar worker
- Worker must be independently enableable/disableable
- Portal must continue working if worker is down, disabled, stale, or missing assets
- Keep fail-closed artifact policy unless explicitly changed later
- Do not break current session/capability semantics; extend them if needed

Design summary
Current state:
- mobile_portal FastAPI app directly hosts MegaStar capability logic, route, preprocessing, model adapter, artifact loading, and retrieval service in-process
- frontend hides or shows MegaStar based on session megastar_lookup state
- this is good for prototyping but mixes inference lifecycle with normal portal request serving
Target state:
- MegaStar has two supported execution backends behind one stable portal API:
  - both backends must resolve the same active model key/checkpoint contract for a given deployment state
  - local mode: existing in-process execution inside the mobile portal process
  - worker mode: separate MegaStar worker service
- the selected backend owns:
  - artifact availability checks
  - model loading / caching
  - query preprocessing
  - retrieval/index search
  - request-level inference execution
- main portal always owns:
  - auth
  - user session/capability exposure
  - UI API surface
  - backend selection and request dispatch
- portal capability becomes a composition of:
  - local feature flag enabled
  - selected backend available/healthy
  - selected backend reports assets fresh/enabled

Recommended directory split
- mobile_portal/app/...
  - portal-only API and proxy layer
- mobile_portal/megastar_worker/...
  - dedicated worker app, config, routes, services, adapters

Worker responsibilities
- startup config and health reporting
- artifact source-of-truth checks
- model loading and warm state
- query preprocessing
- retrieval execution
- status response with exact availability reason
- request-level inference execution

Portal responsibilities after refactor
- call worker /status or /health+capability endpoint
- expose megastar_lookup status to frontend via /api/session
- proxy POST /api/megastar/lookup to worker
- preserve auth boundary and UI contract

Worker API proposal
- GET /health
  - liveness only
- GET /status
  - feature enabled/disabled
  - state
  - reason
  - model_key
  - artifact freshness / metadata summary
- POST /lookup
  - one uploaded image
  - returns same MegaStarLookupResponse contract already used by portal UI

Portal API after refactor
- keep GET /api/session unchanged in shape, but its megastar_lookup info is derived from the selected backend status + local config
- keep POST /api/megastar/lookup unchanged for frontend, but internally dispatch through a backend selector
- backend selector modes for v1 refactor:
  - local: call existing in-process service path
  - worker: proxy to worker

Important design choice
Keep the frontend contract stable.
That means:
- App.tsx and ObservationWorkspace should not need major semantic changes
- mobile_portal/app/routes/megastar_lookup.py becomes a thin proxy layer instead of inference host

Failure/availability states to support
- feature_flag_disabled
- worker_unreachable
- worker_unhealthy
- stale_artifacts
- model_not_configured
- required_assets_missing
- not_implemented (only during transition phase, not final)

Deployment/runtime recommendation
Final runtime should look like:
- main mobile portal FastAPI service on 127.0.0.1:8091
- MegaStar worker FastAPI service on localhost separate port, e.g. 8092 or unix socket
- independent systemd unit for MegaStar worker
- worker can be stopped/started without interrupting normal portal browsing/submission

Configuration additions
Portal config
- STARBOARD_MOBILE_MEGASTAR_ENABLED
- STARBOARD_MOBILE_MEGASTAR_WORKER_URL
- STARBOARD_MOBILE_MEGASTAR_PROXY_TIMEOUT_S
- optional startup mode / strictness flags

Worker config
- STARBOARD_MEGASTAR_WORKER_ENABLED
- STARBOARD_MEGASTAR_MODEL_KEY
- STARBOARD_MEGASTAR_ARTIFACT_ROOT
- STARBOARD_MEGASTAR_REGISTRY_PATH
- STARBOARD_MEGASTAR_REQUIRE_FRESH_ASSETS
- STARBOARD_MEGASTAR_MAX_UPLOAD_MB
- STARBOARD_MEGASTAR_DEVICE or equivalent if needed later

Validation strategy
- Portal still works with worker absent
- Worker status correctly drives session capability
- Portal MegaStar proxy returns controlled unavailable when worker unavailable
- Portal MegaStar proxy returns successful candidate results when worker healthy
- No archive writes still guaranteed
- Frontend still behaves the same except availability explanation becomes more accurate

Execution phases

### Phase 0: Freeze dual-mode boundary
Objective: Decide exactly what remains shared, what runs in local mode, and what runs in worker mode.

Files:
- Modify: docs/plans/2026-04-03-megastar-worker-refactor.md
- Reference: mobile_portal/app/routes/megastar_lookup.py
- Reference: mobile_portal/app/config.py
- Reference: mobile_portal/app/services/megastar_lookup_service.py

Tasks:
1. Freeze worker API shape (/health, /status, /lookup).
2. Freeze the shared model-key/checkpoint source-of-truth rule for both local and worker backends.
3. Freeze portal API stability requirement (/api/session and /api/megastar/lookup stay stable).
4. Freeze backend selector contract: local mode and worker mode must both satisfy the same portal request/response semantics.
5. Freeze what code remains shared versus duplicated.
6. Commit docs.

### Phase 1: Scaffold worker service
Objective: Create a standalone MegaStar worker app with health and status routes.

Files:
- Create: mobile_portal/megastar_worker/main.py
- Create: mobile_portal/megastar_worker/config.py
- Create: mobile_portal/megastar_worker/routes/health.py
- Create: mobile_portal/megastar_worker/routes/status.py
- Create: mobile_portal/megastar_worker/routes/lookup.py
- Create: mobile_portal/megastar_worker/models/api.py
- Test: mobile_portal/tests/test_megastar_worker.py

Tasks:
1. Build standalone worker FastAPI app.
2. Add health route.
3. Add status route exposing enabled/state/reason/model_key.
4. Reuse or move existing capability logic into worker-side service.
5. Add tests.

### Phase 2: Move inference host logic into worker
Objective: Worker becomes the place where inference actually runs.

Files:
- Create or move: mobile_portal/megastar_worker/services/lookup_service.py
- Create or move: mobile_portal/megastar_worker/adapters/*
- Possibly keep shared utility code in a neutral shared module if duplication is avoidable
- Test: mobile_portal/tests/test_megastar_worker.py

Tasks:
1. Move or factor artifact loader into worker-side code.
2. Move or factor query preprocessing into worker-side code.
3. Move or factor model adapter into worker-side code.
4. Move or factor result resolver into worker-side code.
5. Implement worker /lookup using the current real retrieval service logic.
6. Keep no-write behavior and tests.

### Phase 3: Add portal-side backend selector and worker client
Objective: Main portal can choose between local execution and worker execution without changing the frontend contract.

Files:
- Create: mobile_portal/app/adapters/megastar_worker_client.py
- Modify: mobile_portal/app/config.py
- Modify: mobile_portal/app/routes/session.py
- Modify: mobile_portal/app/routes/megastar_lookup.py
- Test: mobile_portal/tests/test_megastar_lookup.py

Tasks:
1. Add worker client for /status and /lookup.
2. Add a backend selector abstraction with at least two modes:
   - local: call current in-process MegaStar service path
   - worker: proxy to worker
3. Replace direct capability computation in /api/session with backend-selector-aware capability computation.
4. Replace direct /api/megastar/lookup implementation with backend-selector dispatch.
5. Preserve current response contract.
6. Add tests for local mode, worker mode, worker unavailable, stale, and enabled paths.

### Phase 4: Preserve frontend behavior
Objective: Ensure frontend contract remains stable.

Files:
- Modify minimally if needed: mobile_portal/frontend/src/api/client.ts
- Modify minimally if needed: mobile_portal/frontend/src/App.tsx
- Modify minimally if needed: mobile_portal/frontend/src/screens/ObservationWorkspace.tsx

Tasks:
1. Keep current frontend response handling stable.
2. Ensure unavailable reasons still appear.
3. Avoid changing user-visible workflow except for improved diagnostics.
4. Add tests if frontend tests exist; otherwise verify by build/manual flow.

### Phase 5: Deployment and operator control
Objective: Make worker operationally togglable.

Files:
- Create: mobile_portal/deploy/megastar-worker.service.example
- Modify: mobile_portal/DEPLOYMENT.md
- Optional: docs/plans/2026-04-03-megastar-worker-refactor.md

Tasks:
1. Document worker port and environment variables.
2. Add example systemd service file.
3. Document how to stop/start/disable worker independently.
4. Document rollback path: disable feature flag or stop worker.

### Phase 6: Cleanup and de-duplication
Objective: Reduce split-brain logic after proxy path is stable.

Files:
- Modify/remove old in-process portal MegaStar code under mobile_portal/app/services and adapters if no longer needed
- Keep shared neutral helpers only where duplication is genuinely harmful

Tasks:
1. Remove dead in-process inference path from portal app once worker path is stable.
2. Keep compatibility tests green.
3. Verify portal still works if worker is absent.

Key technical decisions to keep
- Fail closed on stale artifacts in worker status
- Keep frontend contract stable
- Keep advisory-only workflow semantics
- No archive writes from worker
- Use worker status for diagnostics rather than hiding failure state entirely

Recommended first implementation chunk
Do only Phases 0-1 first:
- freeze dual-mode boundary
- scaffold worker app with /health and /status
- no portal cutover yet

That gives us a clean service boundary while preserving the current local in-process path as a working fallback.

Verification commands
- Portal backend tests:
  - bash ./scripts/test -q mobile_portal/tests/test_auth.py mobile_portal/tests/test_megastar_lookup.py mobile_portal/tests/test_megastar_worker.py
- Frontend build:
  - npm run build
  - workdir: /home/weertman/Documents/starBoard/mobile_portal/frontend

Go/no-go before full cutover
- Worker returns correct status reasons
- Portal session reflects worker state correctly
- Portal still boots with worker off
- No regression in archive lookup / submission flows
