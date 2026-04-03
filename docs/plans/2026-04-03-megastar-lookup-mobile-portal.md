# MegaStar Lookup for Mobile Portal Implementation Plan

> For Hermes: this plan has been revised after council critique. Do not start implementation until Phase 0 provenance freeze is complete.

Goal
Add a portal-only MegaStar Lookup feature to the New Observation workflow so a user can run visual retrieval on exactly one currently selected local image and receive a ranked list of likely archive gallery IDs, without persisting the query image or results into the starBoard archive.

Architecture summary
This feature will be implemented as an additive mobile_portal inference path with a narrow read-only boundary into the existing starBoard DL / MegaStar retrieval assets. The frontend will trigger lookup from the currently selected local image in ObservationWorkspace. The backend will transiently process that one image, apply a query preprocessing contract matched to the provenance of the active archive embedding artifacts, extract a query embedding, search a read-only per-image archive embedding index, aggregate image-level matches into ID-level ranking, return candidates with best-supporting archive image descriptors, and remove any temporary query artifact.

Tech stack / code areas
- Frontend: /home/weertman/Documents/starBoard/mobile_portal/frontend/src
- Backend portal app: /home/weertman/Documents/starBoard/mobile_portal/app
- Existing DL / retrieval stack: /home/weertman/Documents/starBoard/src/dl
- MegaStar package: /home/weertman/Documents/starBoard/star_identification/megastarid
- Plan file: /home/weertman/Documents/starBoard/docs/plans/2026-04-03-megastar-lookup-mobile-portal.md

Hard constraints
- Operates on one currently selected local image only
- Uses image-level visual comparison only; no metadata-assisted ranking
- Returns ranked gallery IDs, derived from image-to-image similarity
- No archive writes: no Query/Gallery creation, no metadata CSV append, no encounter-folder creation
- Keep route/service isolated from core submission codepaths
- Explicitly account for all pipeline stages, including preprocessing provenance
- Prefer existing read-only DL artifact ecosystem if compatible; do not create duplicate truth casually
- Feature must be additive only and instantly disableable

Frozen compatibility contract
Must not change or break:
- Existing manual archive lookup workflow in mobile portal
- Existing observation metadata + submission workflow
- Existing archive storage format
- Existing desktop starBoard workflows
- Existing mobile portal startup behavior when MegaStar assets are absent

Additive-only contract
- Manual “Look up star” remains unchanged and first-class
- MegaStar Lookup is a second, separate action in ObservationWorkspace
- MegaStar results are advisory only in v1
- MegaStar results must not silently change metadata target type, target mode, or target ID
- Portal must continue to work if MegaStar is disabled, unconfigured, stale, or unavailable

Current diagnosis
The mobile portal already supports local image selection, metadata gating, archive lookup, and compare flow, but it has no transient visual retrieval path from a selected local observation image into the archive. The repo already contains an existing DL retrieval ecosystem under src/dl and precomputed artifacts under archive/_dl_precompute. The main implementation risk is not route wiring; it is matching portal query preprocessing and result semantics to the provenance of the existing archive artifacts.

Success criteria
- ObservationWorkspace shows MegaStar Lookup only when capability is enabled and a selected local image exists
- Pressing MegaStar Lookup sends exactly the currently selected local image
- Backend runs explicit query preprocessing matched to archive artifact provenance
- Backend computes a query embedding and compares it to a read-only per-image gallery embedding index
- Backend returns top ranked gallery IDs with best supporting archive image descriptors
- No query image or result is persisted into archive state
- User can compare a candidate directly in ObservationWorkspace using existing archive comparison state
- Manual lookup and actual archive submission continue to work unchanged

Required state model
Local image selection
- Feature is enabled when at least one local image exists and a selectedPreview exists
- Request always uses only the currently selected local image
- Results are bound to the selected local image at request start
- If selected local image changes, existing MegaStar results become stale and must be cleared or marked stale
- If selected local image is removed, MegaStar results must be cleared

Archive comparison interaction
- MegaStar candidate actions must hydrate the same archiveImages / selectedArchiveIndex state already used for compare flow
- “Compare candidate” should keep user in ObservationWorkspace and set the chosen archive image for comparison
- “Open candidate in archive browser” may optionally hand off to existing LookupWorkspace, but v1 should prioritize inline comparison

Metadata interaction
- MegaStar lookup must not automatically populate metadata target fields in v1
- Any later “Use candidate ID in metadata” shortcut must be a separate explicit user action and separate phase

Capability / rollback contract
- Add session capability: megastar_lookup
- Capability false when env flag disabled, model missing, artifact missing, artifact stale per policy, or startup validation fails
- Frontend renders no MegaStar UI when capability false
- Existing portal startup must succeed even if MegaStar assets are missing

Pipeline contract
1. Frontend selected-image capture
2. Transient request image handling
3. Safe decode + EXIF normalization
4. RGB normalization
5. Query preprocessing contract matched to archive artifact provenance
6. Query embedding extraction with cached eval-mode model
7. Read-only archive per-image embedding search
8. Image-level similarity scoring
9. ID-level aggregation from image matches
10. Candidate response aligned to existing portal image descriptor usage
11. Temp cleanup / no-write guarantee

Provenance freeze requirement
Before implementation, freeze these facts:
- Which exact active/default model artifact source will be used: existing DLRegistry active model or an explicit portal override
- Whether current archive artifacts were built from raw images, YOLO-cached crops, or another pipeline
- Which exact transform path produced those artifacts
- TTA policy used during artifact generation
- Whether artifacts are portable across hosts or still rely on absolute host-specific paths
- Whether stale artifacts are acceptable for v1

The feature must not proceed until query preprocessing and archive artifact provenance are frozen.

Preprocessing requirements
Do not treat “MegaStar preprocessing” as already settled. The repo currently has multiple transform paths.

Required preprocessing spec for portal query path
- Decode image safely
- Apply EXIF orientation correction
- Convert to RGB
- Match the exact retrieval preprocessing contract used by the chosen archive artifact set
- If the chosen archive artifact set depends on YOLO segmentation/cropping or cached preprocessed images, the query path must explicitly match that provenance or the archive artifact set must be rebuilt
- Use the same image sizing / interpolation / normalization contract as the chosen archive artifact builder
- TTA policy must match the chosen retrieval contract or be explicitly disabled for both
- Version preprocessing assumptions in artifact metadata

Do not silently use placeholder or zero-tensor fallbacks for failed preprocessing.

Read-only retrieval artifact contract
Preferred source of truth
- Reuse the existing src/dl artifact ecosystem if it can satisfy the portal contract cleanly
- Avoid inventing a parallel artifact format unless necessary

Required artifact properties
- Per-image gallery embeddings, not centroid-only retrieval
- Stable, portable references, not absolute host-specific cache paths
- Fields should include at minimum:
  - entity_type (gallery for v1)
  - entity_id
  - encounter/date if available
  - stable archive image reference or mobile-portal-compatible image_id
  - preview/fullres resolvable reference if possible
  - embedding vector
  - model key / checkpoint hash
  - preprocessing mode / transform version
  - YOLO enabled/disabled and version if applicable
  - TTA policy
  - reranking policy if any
  - artifact schema version
  - build timestamp / archive snapshot marker
  - stale/pending state visibility

Stale artifact policy
Must be explicit before rollout. Recommended v1 policy:
- If active artifact set is stale or pending_ids is non-empty for the selected model, fail closed and disable MegaStar capability
Alternative warning-open mode is only acceptable if explicitly approved.

Ranking contract
Retrieval target for v1
- Gallery IDs only
- Query IDs are out of scope for v1

Similarity stage
- Search query embedding against per-image gallery embeddings
- Use a clearly defined similarity metric, likely cosine on normalized embeddings
- Do not use centroid-only retrieval for v1

Aggregation stage
This must be empirically chosen, not left ambiguous.
At minimum evaluate:
- max image score per ID
- mean of top-N image scores per ID
- normalized logsumexp or similar top-N aggregation

V1 recommendation pending ablation
- Use a simple interpretable aggregator only after offline validation
- Scores must be labeled as retrieval similarity / retrieval score, not probability

Reranking
- Disabled by default in v1 unless a dedicated single-image transient-query ablation shows clear value

Response contract
POST /api/megastar/lookup should return structured candidates that are compatible with existing portal comparison flow.

Candidate payload should include at minimum:
- rank
- entity_type
- entity_id
- retrieval_score
- best_match_image as an ImageDescriptor-compatible object or enough info to generate one
- encounter/date if available
- optional support metadata such as best-supporting image label

Service-level metadata should include:
- query image name
- processing time
- model / artifact version metadata in logs, and optionally response/debug fields
- stale flag if warning-open mode is ever allowed

Frontend UX spec
ObservationWorkspace additions
- MegaStar Lookup button near the selected local image controls
- Loading state: “Analyzing selected image…”
- Result panel below local image workspace
- Clear results action
- Retry path

Required result actions
- Compare best image: hydrate existing archive comparison state and stay in ObservationWorkspace
- Open ID in archive browser: optional secondary action
- Show support details: best image and encounter/date; optional top support images later

Required UX states
- idle: select one local image to run lookup
- running
- success with candidates
- empty / no strong candidates
- degraded / unavailable / timeout / model warming
- stale results after local image selection changes

No-regression rules
- Existing manual archive lookup button remains unchanged
- Existing metadata ready / submit flow remains unchanged
- MegaStar failures must not block actual archive submission

Proposed backend modules
- mobile_portal/app/routes/megastar_lookup.py
- mobile_portal/app/services/megastar_lookup_service.py
- mobile_portal/app/adapters/megastar_query_preprocess.py
- mobile_portal/app/adapters/megastar_model_adapter.py
- mobile_portal/app/adapters/megastar_artifact_loader.py
- mobile_portal/app/adapters/megastar_result_resolver.py
- mobile_portal/app/models/megastar_api.py

Explicit architecture guardrails
- MegaStar lookup service must not import or call submission_service, ingest_adapter, or csv-writing adapters
- Route/service may only read from archive-derived artifacts and existing archive media resolution paths
- Transient request image handling should be in-memory by default; use temp files only if required by shared adapters

Phases

Phase 0: provenance freeze and compatibility audit
- Inspect current src/dl + MegaStar + artifact provenance
- Freeze active model source, artifact source, preprocessing contract, TTA policy, stale policy, and portability strategy
- Decide whether v1 reuses existing artifacts or requires rebuilding them
- Freeze gallery-only scope

Phase 1: artifact and API contract freeze
- Freeze portable per-image artifact contract
- Freeze API response schema aligned to existing ImageDescriptor usage
- Freeze capability / rollback behavior via /api/session
- Freeze result interaction semantics in ObservationWorkspace

Phase 2: offline retrieval validity work
- Build or validate the chosen gallery per-image artifact set
- Run ablations on preprocessing, TTA, and aggregation choices
- Pick one ranking contract for v1 based on measured retrieval quality
- Record quality metrics and latency budget before frontend exposure

Phase 3: backend lookup service
- Implement transient request image validation and handling
- Implement query preprocessing path matched to frozen artifact provenance
- Implement cached model loading and single-image embedding extraction
- Implement artifact load/search and ID aggregation
- Implement no-write guarantee and cleanup
- Implement clear failure modes for unavailable assets, stale index, decode failure, preprocessing failure, no candidates, and timeout

Phase 4: portal route and capability gating
- Add POST /api/megastar/lookup
- Add megastar_lookup capability to session response
- Ensure app still starts if assets are missing; capability false instead of startup failure

Phase 5: frontend integration
- Add MegaStar Lookup button in ObservationWorkspace
- Bind results to selected local image and invalidate on selection change/removal
- Add result panel, compare action, clear action, retry action
- Preserve current manual lookup and submit flow unchanged

Phase 6: validation and rollout
- Non-regression tests for manual lookup and actual submission
- No-write tests for MegaStar path
- Startup-with-assets-missing test
- Feature-flag disable / rollback validation
- Manual relevance and latency validation

Validation strategy
Functional
- Route validates file type/size
- No archive folders or metadata rows created on success/failure
- Candidate compare action populates existing archive comparison state
- Existing lookup and submit flows remain unchanged

Quality / retrieval
- ID-level Rank-1 / Rank-5 / Rank-10
- mAP or MRR as appropriate
- Query success rate
- Preprocessing/crop success rate
- Coverage of archive images in artifact set
- Failure bucket counts
- Stratified metrics by identity image count / encounter diversity if possible
- Latency split by decode / preprocess / embed / search / aggregate

Recommended first implementation chunk
Do not start with frontend.
Start with Phase 0 and Phase 1 only:
- freeze provenance
- freeze stale policy
- freeze gallery-only scope
- freeze capability behavior
- freeze artifact and response contracts

Open questions to resolve before coding
- Which exact active model / checkpoint should v1 use?
- Are current archive artifacts portable and trustworthy enough for portal reuse?
- Must query preprocessing include YOLO segmentation/cropping to match artifact provenance?
- Which aggregation rule wins the offline ablation for single-image query -> ID ranking?
- Is fail-closed on stale artifacts acceptable for v1?
