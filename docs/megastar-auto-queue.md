# MegaStar automatic precompute queue

starBoard can now track MegaStar embedding/precompute work as durable queue state instead of relying only on manual DL-tab updates.

## What gets queued

Normal browser, mobile, and desktop archive uploads share `src.data.ingest.place_images()`. After files are successfully written into the archive, `place_images()` now:

1. marks the affected `Gallery` or `Queries` ID in `DLRegistry.pending_ids`, and
2. creates/coalesces a durable SQLite queue job for the same model/target/ID.

The queue payload is ID-centric. Final archive paths from `FileOp.dest` are stored as optional context; temporary upload paths are never authoritative.

Direct archive mutation paths that bypass `place_images()` also enqueue affected IDs:

- `src.data.archive_merge.execute_merge`
- `src.data.merge_yes.merge_yeses_for_gallery`
- `src.data.merge_yes.revert_merge_batch_for_gallery`
- `src.data.batch_undo.undo_batch`
- `src.data.batch_undo.redo_batch`

## Queue and lock files

Under the active archive root:

- queue DB: `_dl_precompute/megastar_queue.sqlite`
- artifact lock: `_dl_precompute/.megastar_artifact.lock`

The SQLite queue coalesces queued work by `(model_key, target, id_str)`. If the same ID is already queued, repeated uploads update the queued job context instead of creating duplicates. If that ID is currently running, a new queued follow-up job may exist at the same time so changes that arrive during an active precompute are not lost.

## Worker commands

From the repo root:

```bash
cd /home/weertman/Documents/starBoard
./scripts/python -m src.dl.megastar_queue_worker --status
./scripts/python -m src.dl.megastar_queue_worker --once --batch-size 1
./scripts/python -m src.dl.megastar_queue_worker --poll-seconds 10 --batch-size 1
```

`--batch-size 1` is the default and should remain the production default unless explicitly changed.

## Important safety semantics

The worker does not silently run a full precompute. If the active model is missing or not marked precomputed, it reports `full_precompute_required` and leaves jobs queued.

The worker uses `PrecomputeWorker(..., only_pending=True, batch_size=1)` for real work. This re-embeds only pending IDs, but it still rewrites global MegaStar artifact files and similarity matrices. For that reason, there must be only one active MegaStar artifact writer at a time. The automatic queue worker uses the artifact lock for this.

Initial automatic worker behavior does not run verification precompute by default. Use `--include-verification` only after deciding that the extra pairwise verification work is acceptable.

## Status

The mobile MegaStar worker `/status` response now includes a `megastar_queue` object with queued/running/completed/failed counts and the queue DB path.

## Recommended rollout

1. Run tests.
2. Check status with `--status`.
3. Run one canary pass with `--once --batch-size 1`.
4. Confirm registry pending IDs decrease and MegaStar lookup artifacts still load.
5. Only then launch the long-running worker with `--poll-seconds 10 --batch-size 1`.
