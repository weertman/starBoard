# starBoard activity events v1

This document is the v1 contract for browser/mobile activity logging in starBoard.

The goal is to record who used each interface, which browser/app session they used, and what major workflow actions happened, without adding a new user database or making telemetry failures block archive workflows.

## Storage

Activity events are append-only JSONL files under the configured starBoard archive root:

```text
<archive_root>/logs/activity/activity_YYYY-MM-DD.jsonl
```

On Willem's local checkout, the normal archive root is:

```text
/home/weertman/Documents/starBoard/archive
```

The archive root is resolved by `src.data.archive_paths.archive_root()` and can be overridden with:

```text
STARBOARD_ARCHIVE_DIR
```

Implementation:

```text
src/data/activity_log.py
```

The logger uses a process-local append lock and writes one JSON object per line.

## Authentication and identity

User identity comes from the existing Cloudflare Access header used by the app auth dependencies:

```text
cf-access-authenticated-user-email
```

There is no separate starBoard user table or login system in v1.

Local development may use localhost bypass identities from the existing app auth code, for example:

```text
local@example.invalid
localhost-bypass@local
```

The event field is:

```json
"user_email": "field@example.org"
```

The auth source field is currently:

```json
"auth_source": "cloudflare_access"
```

## Session identity

Frontend code creates a stable per-tab/app-session ID in `sessionStorage` and sends it on API calls with:

```text
X-Starboard-Session-Id
```

Frontend session storage keys:

```text
star_browser:  starboard.activity.session_id
mobile_portal: starboard.mobile.activity.session_id
```

For `POST /api/activity/events`, the request body may also include `session_id`. If both body and header are present, the body `session_id` wins. This precedence is regression-tested because the smoke test caught an earlier duplicate-kwarg bug in this path.

## Surfaces

Current valid surface values:

| surface | Meaning |
|---|---|
| `star_browser` | Browser/desktop web interface. |
| `mobile_portal` | Mobile field portal. |

## Ingestion endpoint

Both web apps expose the same authenticated endpoint:

```text
POST /api/activity/events
```

Implemented in:

```text
star_browser/app/routes/activity.py
mobile_portal/app/routes/activity.py
```

Request shape:

```json
{
  "session_id": "browser-session-uuid-or-other-stable-id",
  "events": [
    {
      "event_type": "ui.tab.open",
      "client_timestamp_utc": "2026-05-06T22:00:00.000Z",
      "workflow": "single-entry",
      "entity_type": "query",
      "entity_id": "q123",
      "query_id": "q123",
      "gallery_id": null,
      "success": true,
      "duration_ms": 125,
      "details": {"tab": "single-entry"}
    }
  ]
}
```

Response shape:

```json
{"accepted_events": 1}
```

Limits:

| Limit | Value | Failure |
|---|---:|---|
| Events per request | 100 | 413 `too_many_activity_events` |
| Event type length | 120 chars | 400 `activity_event_type_too_long` |
| Serialized details size per event | 8192 bytes | 413 `activity_event_details_too_large` |

The endpoint requires Cloudflare Access authentication. Anonymous requests return 401 through the existing auth dependency.

## Event record schema

Every JSONL row uses `schema_version = 1`.

Example row, redacted:

```json
{
  "schema_version": 1,
  "event_id": "8eab0123-1111-4444-9999-abcdefabcdef",
  "timestamp_utc": "2026-05-06T22:00:00.000000Z",
  "client_timestamp_utc": "2026-05-06T22:00:00.000Z",
  "surface": "star_browser",
  "session_id": "browser-session-uuid",
  "user_email": "field@example.org",
  "auth_source": "cloudflare_access",
  "request_path": "/api/activity/events",
  "request_method": "POST",
  "event_type": "query_matcher.search.completed",
  "entity_type": "query",
  "entity_id": "q123",
  "query_id": "q123",
  "gallery_id": null,
  "workflow": "query_matcher",
  "success": true,
  "duration_ms": 240,
  "details": {"preset": "megastar", "result_count": 10},
  "user_agent": "Mozilla/5.0 ...",
  "client_ip": "203.0.113.10"
}
```

Field meanings:

| Field | Type | Required in row | Meaning |
|---|---|---:|---|
| `schema_version` | integer | yes | Activity schema version. Current value: `1`. |
| `event_id` | string UUID | yes | Server-generated unique event ID unless explicitly supplied internally. |
| `timestamp_utc` | string | yes | Server receive/write time in UTC. |
| `client_timestamp_utc` | string or null | yes | Browser timestamp when the frontend event was created. Server-side events usually leave this null. |
| `surface` | string | yes | `star_browser` or `mobile_portal`. |
| `session_id` | string | yes | Frontend/browser session ID, or empty string if none was supplied. |
| `user_email` | string | yes | Authenticated user email from Cloudflare Access/local bypass. |
| `auth_source` | string | yes | Currently `cloudflare_access`. |
| `request_path` | string or null | yes | FastAPI request path that caused the event. |
| `request_method` | string or null | yes | HTTP method for the request. |
| `event_type` | string | yes | Dot-delimited event name. |
| `entity_type` | string or null | yes | Entity class when relevant, usually `query` or `gallery`. |
| `entity_id` | string or null | yes | Primary entity ID when relevant. |
| `query_id` | string or null | yes | Query ID when the event involves a query/gallery pair. |
| `gallery_id` | string or null | yes | Gallery ID when the event involves a query/gallery pair. |
| `workflow` | string or null | yes | Higher-level workflow grouping. |
| `success` | boolean or null | yes | True/false for completed actions; null for neutral/open/start events. |
| `duration_ms` | integer or null | yes | Client- or server-measured duration for completed actions. |
| `details` | object | yes | Small event-specific metadata object. Never store raw notes, raw errors, or filenames here. |
| `user_agent` | string or null | yes | Request user-agent. |
| `client_ip` | string or null | yes | `cf-connecting-ip`, then `x-forwarded-for`, then request client host fallback. |

## Workflow names

Current workflow values:

| workflow | Surface(s) | Meaning |
|---|---|---|
| `session` | both | Session load/auth context. |
| `single-entry` | star_browser | Browser top-level tab name for Single Entry open events. |
| `single_entry` | star_browser | Single Entry form interactions and submissions. |
| `batch` | star_browser | Browser top-level tab name for Batch Upload open events. |
| `batch_upload` | star_browser | Batch Upload source/preflight/preview/submit interactions. |
| `gallery` | star_browser | Browser top-level tab name for ID Review open events. |
| `id_review` | star_browser | ID Review entity load, encounter filter, rename, metadata save. |
| `first-order` | star_browser | Browser top-level tab name for Query Matcher open events. |
| `query_matcher` | star_browser | Query Matcher search, proposal navigation, match decisions. |
| `home` | mobile_portal | Mobile home screen open. |
| `observation` | mobile_portal | Mobile new-observation and submission workflow. |
| `lookup` | mobile_portal | Mobile archive lookup and MegaStar lookup workflow. |

Note: top-level browser tab events currently use the existing tab state strings (`single-entry`, `batch`, `gallery`, `first-order`), while deeper workflow events use more semantic workflow strings (`single_entry`, `batch_upload`, `id_review`, `query_matcher`). Preserve this distinction unless intentionally doing a schema v2 cleanup.

## Event catalog

### Shared/backend events

| Event type | Surface | Workflow | Emitted by | Notes |
|---|---|---|---|---|
| `session.loaded` | both | `session` | Backend `/api/session` routes | Server-side event; best-effort so session responses are not broken by logging failure. |

### star_browser top-level UI events

| Event type | Workflow | Details |
|---|---|---|
| `ui.tab.open` | tab string: `single-entry`, `batch`, `gallery`, `first-order` | `{tab}` |

### Single Entry events

| Event type | Workflow | Entity fields | Details |
|---|---|---|---|
| `single_entry.location.changed` | `single_entry` | none | `{mode: "new_location"|"saved_location", has_value}` |
| `single_entry.submit.started` | `single_entry` | `entity_type`, `entity_id` from current target | `{target_mode, file_count, has_location}` |
| `single_entry.submit.completed` | `single_entry` | success: response entity; failure: attempted target | success: `{accepted_images, skipped_images, target_mode}`; failure: `{target_mode, file_count}` |
| `single_entry.submit.succeeded` | `single_entry` | response entity | Backend success event after archive submission succeeds. Details include accepted/skipped image counts and encounter folder. |

### Batch Upload events

| Event type | Workflow | Details |
|---|---|---|
| `batch_upload.location.changed` | `batch_upload` | `{mode: "new_location"|"saved_location", has_value}` |
| `batch_upload.source.selected` | `batch_upload` | zip: `{source_type: "zip", has_file, file_count}`; folder: `{source_type: "folder", file_count}` |
| `batch_upload.zip_preflight.completed` | `batch_upload` | success-ish: `{requested_mode, resolved_mode, importable_images}`; failure: `{requested_mode}` |
| `batch_upload.source_prepared` | `batch_upload` | `{source_type, file_count}` on success; summarized source info on failure. |
| `batch_upload.preview.completed` | `batch_upload` | success: `{target_archive, discovery_mode, detected_rows, detected_ids, total_images}`; failure: `{target_archive, discovery_mode}` |
| `batch_upload.submit.completed` | `batch_upload` | success: `{plan_id, selected_rows, accepted_images, executed_rows, status}`; failure: `{plan_id, selected_rows}` |

### ID Review events

| Event type | Workflow | Entity fields | Details |
|---|---|---|---|
| `id_review.entity.loaded` | `id_review` | `entity_type`, `entity_id` | success: `{image_count, metadata_rows}` |
| `id_review.encounter_filter.changed` | `id_review` | current entity | `{filter: "all"|"encounter"}`; does not store the raw encounter ID. |
| `id_review.rename.completed` | `id_review` | new/current entity | success includes `{previous_entity_id}`. |
| `id_review.metadata_save.completed` | `id_review` | current entity | `{field_count}` |

### Query Matcher events

| Event type | Workflow | Entity fields | Details |
|---|---|---|---|
| `query_matcher.search.started` | `query_matcher` | query entity/query_id | `{preset, top_k, gallery_filter_count, query_image_id?}` |
| `query_matcher.search.completed` | `query_matcher` | query entity/query_id | success: `{preset, result_count, gallery_filter_count, query_image_id?}`; failure: `{preset, top_k, gallery_filter_count}` |
| `query_matcher.search.succeeded` | `query_matcher` | query entity/query_id | Backend success event after search returns. Details include preset/top_k/query image/filter/result count. |
| `query_matcher.proposal.changed` | `query_matcher` | active gallery candidate + query_id/gallery_id | `{index, total}` |
| `query_matcher.match_decision.completed` | `query_matcher` | gallery entity + query_id/gallery_id | `{verdict, has_notes}`; does not store note text. |
| `query_matcher.match_label.saved` | `query_matcher` | gallery entity + query_id/gallery_id | Backend success event after label save. Details include verdict. |

### mobile_portal screen/workflow events

| Event type | Workflow | Entity fields | Details |
|---|---|---|---|
| `mobile.screen.open` | `home`, `observation`, or `lookup` | none | `{mode}` |
| `mobile.megastar_lookup.started` | `lookup` | none | `{max_candidates}` |
| `mobile.megastar_lookup.completed` | `lookup` | none | success: `{max_candidates, result_count}`; failure: `{max_candidates}` |
| `mobile.megastar_candidate.compare` | `lookup` | candidate entity | `{rank, score}` |
| `mobile.megastar_candidate.open_lookup` | `lookup` | candidate entity | `{rank, score}`; does not also log `compare` for the same click. |
| `mobile.archive_lookup.completed` | `lookup` | looked-up entity | success: `{encounter_selected, image_count, metadata_field_count}`; failure has no raw error string. |
| `mobile.archive_lookup.more_images_loaded` | `lookup` | current entity | `{loaded_count, next_offset}` |
| `mobile.submission.started` | `observation` | attempted target | `{target_mode, file_count}` |
| `mobile.submission.completed` | `observation` | success: response entity; failure: attempted target | success: `{accepted_images, skipped_images, target_mode}`; failure: `{target_mode, file_count}` |
| `mobile.submission.succeeded` | `observation` | response entity | Backend success event after archive submission succeeds. Details include accepted/skipped image counts and encounter folder. |

## Privacy and sensitivity rules

The activity log is operational telemetry, not a raw interaction transcript.

Do not put these in `details`:

- raw filenames
- raw notes text
- raw error strings or tracebacks
- secrets, tokens, cookies, or headers
- identity documents or payment/legal/tax content
- unbounded form values
- full metadata rows

Preferred patterns:

- counts instead of lists: `file_count`, `image_count`, `result_count`
- booleans instead of text: `has_notes`, `has_location`, `encounter_selected`
- controlled values instead of free text: `target_mode`, `preset`, `source_type`, `verdict`
- durations in `duration_ms`

Entity IDs (`entity_id`, `query_id`, `gallery_id`) are allowed because they are needed to connect activity to archive workflows, but avoid adding unrelated raw identifiers inside `details`.

## Failure behavior

Primary archive workflows should not fail just because activity logging fails.

Backend route hooks for primary workflows should use:

```python
try_record_activity_event(...)
```

This is used for server-side session/submission/search/label-save events.

The explicit frontend ingestion endpoint, `POST /api/activity/events`, is allowed to return validation errors for malformed or oversized telemetry requests. Frontend callers must ignore failures, and `trackActivity()` is best-effort.

## Testing

Targeted activity tests:

```bash
cd /home/weertman/Documents/starBoard
./scripts/test -q star_browser/tests/test_activity.py mobile_portal/tests/test_activity.py
```

Current activity test coverage includes:

- authentication required for `/api/activity/events`
- authenticated events write JSONL rows with user/session/surface fields
- `/api/session` writes server-side `session.loaded`
- body `session_id` wins over `X-Starboard-Session-Id` when both are supplied

## Smoke-test report

The initial end-to-end local smoke test report is stored at:

```text
/home/weertman/Desktop/Deep Investigation/starBoard-activity-smoke-20260506-154634/report.md
```

That smoke verified the JSONL path, schema fields, local rendered UI pages, and caught/fixed the body/header session-id bug now covered by regression tests.
