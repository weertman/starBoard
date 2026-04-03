# MegaStar Lookup for Mobile Portal Execution Plan

> For Hermes: implement in order. Do not begin feature coding until Tasks 1-6 are complete and the provenance decision is written down. Keep this additive-only. Commit after each task group.

Goal
Build a portal-only MegaStar Lookup in New Observation that runs on exactly one selected local image, performs transient image-only retrieval against read-only archive embeddings, returns ranked gallery IDs, and never writes to archive state.

Architecture
The implementation should add a new mobile_portal-only retrieval path that reuses existing read-only DL / artifact machinery wherever safe, while preserving all current lookup and submission behavior. The first work is not UI; it is freezing model/artifact provenance and capability behavior so the eventual route is scientifically and operationally correct.

Tech stack
- Frontend: React/TypeScript in mobile_portal/frontend/src
- Backend: FastAPI in mobile_portal/app
- Retrieval/model code: src/dl and star_identification/megastarid
- Tests: mobile_portal/tests

Non-negotiable constraints
- One selected local image only
- Gallery IDs only in v1
- No archive writes
- Manual lookup unchanged
- Metadata/submission flow unchanged
- Feature hidden when capability unavailable
- Portal still boots when MegaStar assets are absent

---

## Phase 0: Provenance freeze and contract capture

### Task 1: Record the chosen model/artifact source of truth
Objective: Freeze whether portal lookup uses the existing DL registry/artifact path or a portal-specific override.

Files:
- Modify: docs/plans/2026-04-03-megastar-lookup-mobile-portal.md
- Reference: src/dl/registry.py
- Reference: archive/_dl_precompute/_dl_registry.json

Steps:
1. Inspect the currently active/default DL model and artifact metadata.
2. Write a short “Chosen source of truth” subsection into the revised plan.
3. Explicitly state:
   - model key
   - artifact root
   - whether portal uses active model or explicit override
4. Commit.

Verification:
- The plan explicitly names one source of truth.
- No ambiguous “maybe reuse/maybe new artifact” language remains.

### Task 2: Record preprocessing provenance
Objective: Freeze the retrieval preprocessing contract that query images must match.

Files:
- Modify: docs/plans/2026-04-03-megastar-lookup-mobile-portal.md
- Reference: src/dl/reid_adapter.py
- Reference: src/dl/precompute.py
- Reference: star_identification/megastarid/transforms.py
- Reference: wildlife_reid_inference/preprocessing.py if used by current pipeline

Steps:
1. Trace how current archive embeddings were actually generated.
2. Decide whether the artifact provenance includes:
   - raw image only
   - YOLO crop/segmentation
   - cached preprocessed images
   - TTA
3. Add a “Frozen preprocessing provenance” subsection to the plan.
4. Explicitly state what the query path must match.
5. Commit.

Verification:
- The plan names one canonical query preprocessing contract.
- The plan no longer says “exact MegaStar preprocessing” without qualification.

### Task 3: Freeze stale-artifact policy
Objective: Decide fail-closed vs warn-open for stale retrieval artifacts.

Files:
- Modify: docs/plans/2026-04-03-megastar-lookup-mobile-portal.md
- Reference: archive/_dl_precompute/_dl_registry.json

Steps:
1. Check whether pending/stale state currently exists.
2. Choose v1 policy.
3. Write policy text into the plan.
4. Recommended default: fail closed and disable capability.
5. Commit.

Verification:
- The plan contains an explicit stale-artifact behavior.
- It is testable later.

### Task 4: Freeze result semantics
Objective: Decide exactly what MegaStar results do in the current mobile workflow.

Files:
- Modify: docs/plans/2026-04-03-megastar-lookup-mobile-portal.md
- Reference: mobile_portal/frontend/src/App.tsx
- Reference: mobile_portal/frontend/src/screens/ObservationWorkspace.tsx

Steps:
1. Add explicit result actions to the plan:
   - Compare best image
   - Open ID in archive browser (optional secondary action)
   - Clear results
2. Add invalidation rules:
   - selected local image changes -> stale/clear results
   - selected local image removed -> clear results
3. Explicitly state that metadata fields are not auto-filled from results in v1.
4. Commit.

Verification:
- The plan describes exact frontend result behavior.
- No hidden metadata mutation remains.

### Task 5: Freeze API and capability contract
Objective: Define the backend/public contract before coding.

Files:
- Modify: docs/plans/2026-04-03-megastar-lookup-mobile-portal.md

Steps:
1. Add the exact route name:
   - POST /api/megastar/lookup
2. Add session capability key:
   - megastar_lookup
3. Add required response fields for candidates.
4. Add explicit disabled/unavailable behavior.
5. Commit.

Verification:
- The plan contains exact route and capability names.
- The response shape is concrete enough to implement.

### Task 6: Freeze v1 ranking scope
Objective: Remove ambiguous scope around queries and ranking style.

Files:
- Modify: docs/plans/2026-04-03-megastar-lookup-mobile-portal.md

Steps:
1. State that v1 ranks gallery IDs only.
2. State that retrieval is against per-image gallery embeddings.
3. State that centroid-only retrieval is out of scope for v1.
4. Commit.

Verification:
- Gallery-only is explicit.
- Per-image retrieval is explicit.

---

## Phase 1: Backend scaffolding and capability gating

### Task 7: Add MegaStar capability to session response
Objective: Create a clean rollback seam before adding UI.

Files:
- Modify: mobile_portal/app/models/api.py
- Modify: mobile_portal/app/routes/session.py
- Modify: mobile_portal/app/config.py
- Test: mobile_portal/tests/test_auth.py or new test file

Steps:
1. Add config flag(s) for megastar enablement and required asset paths.
2. Extend session capability map with `megastar_lookup`.
3. Capability should be false when:
   - feature flag disabled
   - required assets missing
   - stale policy fails
4. Add tests.
5. Commit.

Verification:
- /api/session includes megastar_lookup.
- Existing capabilities remain unchanged.

### Task 8: Add API models for lookup request/response
Objective: Freeze typed backend/frontend contracts.

Files:
- Create: mobile_portal/app/models/megastar_api.py
- Modify: mobile_portal/frontend/src/api/client.ts
- Test: mobile_portal/tests/test_health.py or new route tests later

Steps:
1. Add backend response models for candidate rows and lookup response.
2. Add matching TypeScript types.
3. Keep request multipart-based for one file upload.
4. Commit.

Verification:
- Models exist on both backend and frontend.
- Candidate payload includes image-descriptor-compatible best match fields.

### Task 9: Add disabled stub route
Objective: Establish route wiring safely before model logic.

Files:
- Create: mobile_portal/app/routes/megastar_lookup.py
- Modify: mobile_portal/app/main.py
- Test: mobile_portal/tests/test_megastar_lookup.py

Steps:
1. Add POST /api/megastar/lookup.
2. If capability disabled/unavailable, return a controlled error.
3. Register router in main.py.
4. Add tests for disabled behavior and auth requirement.
5. Commit.

Verification:
- Route exists.
- Disabled behavior is predictable.
- Portal still boots when assets are absent.

---

## Phase 2: Artifact loading and preprocessing contract

### Task 10: Implement artifact availability checker
Objective: Centralize asset/path validation and stale checks.

Files:
- Create: mobile_portal/app/adapters/megastar_artifact_loader.py
- Modify: mobile_portal/app/config.py
- Test: mobile_portal/tests/test_megastar_lookup.py

Steps:
1. Add helper to locate chosen artifact root.
2. Validate expected files exist.
3. Validate artifact metadata needed by v1.
4. Apply stale policy.
5. Return a structured availability result.
6. Commit.

Verification:
- Loader reports available/unavailable cleanly.
- Capability logic can depend on it.

### Task 11: Implement query preprocessing adapter
Objective: Create one explicit preprocessing path for portal queries.

Files:
- Create: mobile_portal/app/adapters/megastar_query_preprocess.py
- Reference: src/dl/reid_adapter.py
- Reference: src/dl/precompute.py
- Reference: star_identification/megastarid/transforms.py
- Test: mobile_portal/tests/test_megastar_lookup.py

Steps:
1. Add safe image decode.
2. Add EXIF transpose.
3. Add RGB conversion.
4. Add the frozen transform chain matching chosen artifact provenance.
5. Reject or clearly fail on preprocessing errors.
6. Do not silently use zero-tensor fallback.
7. Commit.

Verification:
- Adapter produces tensor/image input matching chosen contract.
- Failure path is explicit.

### Task 12: Implement model adapter with cached loading
Objective: Load and reuse the MegaStar model safely.

Files:
- Create: mobile_portal/app/adapters/megastar_model_adapter.py
- Reference: src/dl/reid_adapter.py
- Reference: star_identification/megastarid/inference.py
- Test: mobile_portal/tests/test_megastar_lookup.py

Steps:
1. Implement one-time lazy model load.
2. Put model in eval mode.
3. Add single-image embedding extraction.
4. Ensure normalized embedding output.
5. Add clear unavailable/error path when checkpoint/model config missing.
6. Commit.

Verification:
- Adapter can produce one embedding from one preprocessed image.
- Lazy loading does not break app startup.

### Task 13: Implement portable result-image resolver
Objective: Make best-match archive images plug into current compare UI.

Files:
- Create: mobile_portal/app/adapters/megastar_result_resolver.py
- Reference: mobile_portal/app/adapters/image_manifest_adapter.py
- Test: mobile_portal/tests/test_megastar_lookup.py

Steps:
1. Convert artifact match references into portal-compatible image descriptors.
2. Avoid reliance on host-specific absolute paths.
3. Ensure preview/fullres URLs are resolvable by current archive media routes.
4. Commit.

Verification:
- Returned candidate image can be shown by existing frontend image components.

---

## Phase 3: Retrieval service with real ranking logic

### Task 14: Implement single-image search against per-image gallery embeddings
Objective: Search the read-only gallery image index with one query embedding.

Files:
- Create: mobile_portal/app/services/megastar_lookup_service.py
- Modify: mobile_portal/app/adapters/megastar_artifact_loader.py if needed
- Test: mobile_portal/tests/test_megastar_lookup.py

Steps:
1. Load read-only gallery per-image embeddings.
2. Compute query-to-gallery similarity.
3. Get top image matches.
4. Keep this image-level only.
5. Commit.

Verification:
- Service returns top image matches for one query embedding.

### Task 15: Add ID aggregation policy
Objective: Convert image-level hits into ranked gallery IDs.

Files:
- Modify: mobile_portal/app/services/megastar_lookup_service.py
- Test: mobile_portal/tests/test_megastar_lookup.py

Steps:
1. Implement the chosen v1 aggregation rule.
2. Group image hits by gallery ID.
3. Compute ranked ID candidates.
4. Attach best supporting image for each ID.
5. Commit.

Verification:
- Service returns ranked IDs, not just ranked images.
- Best supporting image is included.

### Task 16: Add no-candidate / weak-candidate behavior
Objective: Prevent forced misleading output.

Files:
- Modify: mobile_portal/app/services/megastar_lookup_service.py
- Modify: mobile_portal/app/models/megastar_api.py
- Test: mobile_portal/tests/test_megastar_lookup.py

Steps:
1. Define empty result behavior.
2. Optionally define weak-result threshold/margin behavior if chosen in plan.
3. Return structured empty/weak state instead of generic failure.
4. Commit.

Verification:
- Service distinguishes success with candidates from success with no strong candidates.

### Task 17: Add transient request image handling and cleanup
Objective: Enforce no-write, ephemeral processing.

Files:
- Modify: mobile_portal/app/routes/megastar_lookup.py
- Modify: mobile_portal/app/services/megastar_lookup_service.py
- Test: mobile_portal/tests/test_megastar_lookup.py

Steps:
1. Accept one uploaded file.
2. Validate type and size.
3. Process in-memory if possible.
4. If temp files are required, ensure cleanup on success/failure.
5. Add test proving no archive writes occur.
6. Commit.

Verification:
- Lookup path leaves no archive artifacts.
- Cleanup is robust.

### Task 18: Wire the real route
Objective: Expose the actual service through FastAPI.

Files:
- Modify: mobile_portal/app/routes/megastar_lookup.py
- Test: mobile_portal/tests/test_megastar_lookup.py

Steps:
1. Call the real service.
2. Return typed response.
3. Add audit logging that records request metadata only.
4. Commit.

Verification:
- Route returns ranked candidate IDs for one uploaded image.

---

## Phase 4: Frontend integration

### Task 19: Add MegaStar capability to frontend session handling
Objective: Hide the feature cleanly when unavailable.

Files:
- Modify: mobile_portal/frontend/src/state/session.ts
- Modify: mobile_portal/frontend/src/api/client.ts
- Modify: mobile_portal/frontend/src/App.tsx

Steps:
1. Ensure session capability typing includes megastar_lookup.
2. Pass capability into ObservationWorkspace or derive from session.
3. Commit.

Verification:
- UI can conditionally hide/show MegaStar lookup.

### Task 20: Add MegaStar UI state to App or ObservationWorkspace parent
Objective: Store transient lookup results and invalidation state cleanly.

Files:
- Modify: mobile_portal/frontend/src/App.tsx
- Modify: mobile_portal/frontend/src/screens/ObservationWorkspace.tsx

Steps:
1. Add state for:
   - running
   - error
   - result list
   - source selected image binding
2. Add invalidation on selected local image change.
3. Add clear-results behavior.
4. Commit.

Verification:
- Changing selected image clears or marks results stale.

### Task 21: Add MegaStar Lookup button
Objective: Let user trigger lookup from selected local image only.

Files:
- Modify: mobile_portal/frontend/src/screens/ObservationWorkspace.tsx

Steps:
1. Add button near local image controls.
2. Disable when no selected local image or capability unavailable.
3. Show loading state when request runs.
4. Commit.

Verification:
- Button appears only in correct conditions.

### Task 22: Add frontend client call
Objective: Connect the selected local image to the new backend route.

Files:
- Modify: mobile_portal/frontend/src/api/client.ts
- Modify: mobile_portal/frontend/src/App.tsx or ObservationWorkspace container

Steps:
1. Add `megastarLookup(file: File)` helper.
2. Call it with the currently selected local image only.
3. Store results in transient state only.
4. Commit.

Verification:
- Exactly one selected file is sent.

### Task 23: Add results panel
Objective: Show ranked candidates inside ObservationWorkspace.

Files:
- Modify: mobile_portal/frontend/src/screens/ObservationWorkspace.tsx
- Optional create: mobile_portal/frontend/src/components/MegaStarResults.tsx

Steps:
1. Show ranked candidate list.
2. Include rank, ID, score, best supporting image preview.
3. Show empty/weak/error states distinctly.
4. Add Clear and Retry affordances.
5. Commit.

Verification:
- Results render and remain local/transient.

### Task 24: Add Compare candidate action
Objective: Reuse existing archive compare state.

Files:
- Modify: mobile_portal/frontend/src/App.tsx
- Modify: mobile_portal/frontend/src/screens/ObservationWorkspace.tsx

Steps:
1. Define candidate action callback.
2. Hydrate archiveImages with best-match image descriptor.
3. Set selectedArchiveIndex.
4. Keep user in ObservationWorkspace.
5. Commit.

Verification:
- User can compare the selected local image against the returned archive best match immediately.

### Task 25: Optional Open in archive browser action
Objective: Allow manual deeper inspection without changing default flow.

Files:
- Modify: mobile_portal/frontend/src/App.tsx
- Modify: mobile_portal/frontend/src/screens/ObservationWorkspace.tsx
- Possibly modify: mobile_portal/frontend/src/screens/LookupWorkspace.tsx

Steps:
1. Add action to jump into existing lookup/open flow for a returned gallery ID.
2. Preserve observation state.
3. Commit.

Verification:
- User can inspect more images for a candidate without breaking compare flow.

---

## Phase 5: Validation and rollout

### Task 26: Add non-regression tests for current workflows
Objective: Prove MegaStar does not break existing portal behavior.

Files:
- Create/modify: mobile_portal/tests/test_megastar_lookup.py
- Possibly add frontend/manual verification notes in mobile_portal/README.md

Steps:
1. Add tests that manual archive lookup still works.
2. Add tests that submission flow still works.
3. Add tests that disabled capability hides/rejects MegaStar path without affecting other routes.
4. Commit.

Verification:
- Existing workflows remain intact.

### Task 27: Add no-write guarantee tests
Objective: Make isolation enforceable.

Files:
- Modify: mobile_portal/tests/test_megastar_lookup.py

Steps:
1. Assert no archive folders are created.
2. Assert no metadata CSV rows are appended.
3. Assert no durable result records are written.
4. Commit.

Verification:
- Feature is truly transient.

### Task 28: Add startup resilience test
Objective: Ensure portal boots without MegaStar assets.

Files:
- Modify: mobile_portal/tests/test_health.py or test_auth.py
- Possibly add new test file

Steps:
1. Simulate missing assets / disabled flag.
2. Verify app still starts and existing routes function.
3. Verify megastar_lookup capability is false.
4. Commit.

Verification:
- Missing assets do not take down portal.

### Task 29: Record manual evaluation checklist
Objective: Make rollout measurable.

Files:
- Modify: docs/plans/2026-04-03-megastar-lookup-mobile-portal.md
- Optional create: docs/plans/2026-04-03-megastar-lookup-manual-eval.md

Steps:
1. Record offline retrieval metrics to check.
2. Record manual portal UX checks.
3. Record latency targets.
4. Record rollback steps.
5. Commit.

Verification:
- There is an explicit go/no-go checklist.

---

## Recommended commit sequence
1. docs: freeze megastar provenance and contracts
2. feat: add megastar capability gating and route scaffold
3. feat: add megastar preprocessing and model adapters
4. feat: add megastar retrieval service
5. feat: integrate megastar lookup into observation workspace
6. test: add megastar no-write and non-regression coverage

## Exact commands to use during implementation
- Frontend build:
  - npm run build
  - workdir: /home/weertman/Documents/starBoard/mobile_portal/frontend
- Backend tests:
  - bash ./scripts/test -q mobile_portal/tests/test_health.py mobile_portal/tests/test_auth.py mobile_portal/tests/test_schema.py mobile_portal/tests/test_archive_lookup.py mobile_portal/tests/test_archive_media.py mobile_portal/tests/test_submission_policy.py mobile_portal/tests/test_submissions.py mobile_portal/tests/test_audit.py mobile_portal/tests/test_megastar_lookup.py
  - workdir: /home/weertman/Documents/starBoard

## Recommended first implementation chunk
Only execute Tasks 1-9 first.
That gets us:
- frozen provenance
- frozen API/capability contract
- safe rollback seam
- route scaffold
without prematurely committing to invalid retrieval behavior.
