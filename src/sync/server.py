"""
starBoard Sync Server — FastAPI application.

Accepts pushes from field machines and serves filtered pulls.
Runs behind a Cloudflare Tunnel for auth and TLS termination.

Start via:  python -m src.sync
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

# Ensure project root is importable
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.sync.config import get_lab_id
from src.sync.server_index import SyncIndex
from src.sync.merge_logic import (
    ingest_encounter_files,
    merge_metadata_rows,
    merge_decisions,
)

log = logging.getLogger("starBoard.sync.server")

# ─── Configuration ──────────────────────────────────────────────────────────

ARCHIVE_ROOT = Path(os.getenv(
    "STARBOARD_ARCHIVE_ROOT",
    str(_project_root / "archive"),
)).resolve()

PACKAGE_TTL = int(os.getenv("STARBOARD_SYNC_PACKAGE_TTL", "3600"))
MAX_UPLOAD_SIZE = int(os.getenv("STARBOARD_SYNC_MAX_UPLOAD_SIZE", str(50 * 1024**3)))

# ─── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="starBoard Sync Server",
    version="0.1.0",
    description="Central archive sync for starBoard photo-ID platform",
)

# Shared state
_index: Optional[SyncIndex] = None
_packages: Dict[str, Dict[str, Any]] = {}  # package_id -> manifest + expiry


def get_index() -> SyncIndex:
    global _index
    if _index is None:
        _index = SyncIndex(ARCHIVE_ROOT)
        log.info("Sync index initialized at %s", ARCHIVE_ROOT)
    return _index


def _get_user_email(request: Request) -> str:
    """Extract authenticated user email from Cloudflare Access header."""
    return request.headers.get("cf-access-authenticated-user-email", "")


def _cleanup_expired_packages():
    """Remove expired package manifests."""
    now = time.time()
    expired = [pid for pid, pkg in _packages.items() if pkg["expires_at"] < now]
    for pid in expired:
        _packages.pop(pid, None)


# ─── Startup / Shutdown ────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    idx = get_index()
    # Rebuild index on startup to ensure it's current
    stats = idx.rebuild_index(compute_hashes=False)
    log.info("Startup index rebuild: %s", stats)


@app.on_event("shutdown")
async def shutdown():
    global _index
    if _index:
        _index.close()
        _index = None


# ─── Health ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health(request: Request):
    idx = get_index()
    catalog = idx.query_catalog()
    sync_log = idx.get_sync_log(limit=1)
    last_sync = sync_log[0]["timestamp"] if sync_log else None

    return {
        "status": "ok",
        "service": "starboard-sync",
        "version": "0.1.0",
        "archive_root": str(ARCHIVE_ROOT),
        "lab_id": get_lab_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_email": _get_user_email(request),
        "totals": catalog["totals"],
        "last_sync": last_sync,
    }


# ─── Catalog ────────────────────────────────────────────────────────────────

@app.get("/api/catalog")
async def catalog(
    location: Optional[str] = None,
    lab: Optional[str] = None,
    date_after: Optional[str] = None,
    date_before: Optional[str] = None,
):
    """Browse the archive catalog with optional filters."""
    idx = get_index()
    locations = [location] if location else None
    labs = [lab] if lab else None
    return idx.query_catalog(
        locations=locations, labs=labs,
        date_after=date_after, date_before=date_before,
    )


# ─── Push: Encounters ──────────────────────────────────────────────────────

@app.post("/api/push/encounters")
async def push_encounters(
    request: Request,
    entity_type: str = Form(...),
    entity_id: str = Form(...),
    encounter_folder: str = Form(...),
    source_lab: str = Form(""),
    files: List[UploadFile] = File(...),
):
    """
    Upload image files for an encounter.
    Deduplicates by SHA-256 hash against existing archive images.
    """
    user = _get_user_email(request)
    lab = source_lab or get_lab_id()

    if entity_type not in ("gallery", "query"):
        return JSONResponse(
            {"error": "entity_type must be 'gallery' or 'query'"},
            status_code=400,
        )

    # Read uploaded files into memory
    file_pairs = []
    for f in files:
        data = await f.read()
        file_pairs.append((f.filename, data))

    # Collect existing hashes from the index for dedup
    idx = get_index()
    existing_hashes = set()
    rows = idx.conn.execute(
        "SELECT sha256 FROM images WHERE sha256 != ''"
    ).fetchall()
    existing_hashes = {r["sha256"] for r in rows}

    report = ingest_encounter_files(
        archive_root=ARCHIVE_ROOT,
        entity_type=entity_type,
        entity_id=entity_id,
        encounter_folder=encounter_folder,
        files=file_pairs,
        source_lab=lab,
        existing_hashes=existing_hashes,
    )

    # Re-index this entity
    idx.update_index_for_entity(entity_type, entity_id, compute_hashes=True)

    # Log
    idx._log_action(
        "push",
        lab_id=lab,
        user_email=user,
        details={
            "type": "encounters",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "encounter_folder": encounter_folder,
            "accepted": report.accepted_images,
            "duplicates": report.skipped_duplicates,
        },
    )

    return {
        "accepted_images": report.accepted_images,
        "skipped_duplicates": report.skipped_duplicates,
        "new_encounter_created": report.new_encounter_created,
        "errors": report.errors,
    }


# ─── Push: Metadata ────────────────────────────────────────────────────────

@app.post("/api/push/metadata")
async def push_metadata(request: Request):
    """
    Push metadata rows. Server compares timestamps and applies
    newer rows (last-writer-wins by last_modified_utc).
    """
    user = _get_user_email(request)
    body = await request.json()
    rows = body.get("rows", [])
    target = body.get("target", "")  # 'gallery' or 'queries'

    if target not in ("gallery", "queries"):
        return JSONResponse(
            {"error": "target must be 'gallery' or 'queries'"},
            status_code=400,
        )

    if not rows:
        return {"updated_count": 0, "skipped_count": 0, "conflicts": []}

    report = merge_metadata_rows(
        archive_root=ARCHIVE_ROOT,
        target=target,
        client_rows=rows,
    )

    # Rebuild index since metadata changed
    idx = get_index()
    idx.rebuild_index(compute_hashes=False)

    idx._log_action(
        "push",
        lab_id=body.get("lab_id", ""),
        user_email=user,
        details={
            "type": "metadata",
            "target": target,
            "updated": report.updated_count,
            "skipped": report.skipped_count,
        },
    )

    return {
        "updated_count": report.updated_count,
        "skipped_count": report.skipped_count,
        "conflicts": report.conflicts,
    }


# ─── Push: Decisions ────────────────────────────────────────────────────────

@app.post("/api/push/decisions")
async def push_decisions(request: Request):
    """
    Push match decisions. Appends to master log, dedup by
    (query_id, gallery_id, timestamp).
    """
    user = _get_user_email(request)
    body = await request.json()
    decisions = body.get("decisions", [])

    if not decisions:
        return {"appended_count": 0, "duplicate_count": 0}

    report = merge_decisions(
        archive_root=ARCHIVE_ROOT,
        client_decisions=decisions,
    )

    idx = get_index()
    idx._log_action(
        "push",
        lab_id=body.get("lab_id", ""),
        user_email=user,
        details={
            "type": "decisions",
            "appended": report.appended_count,
            "duplicates": report.duplicate_count,
        },
    )

    return {
        "appended_count": report.appended_count,
        "duplicate_count": report.duplicate_count,
    }


# ─── Pull: Package (create manifest) ───────────────────────────────────────

@app.post("/api/pull/package")
async def pull_package(request: Request):
    """
    Create a download package manifest. Returns a package_id
    that can be used with /api/pull/stream/{package_id}.
    """
    body = await request.json()
    _cleanup_expired_packages()

    idx = get_index()
    manifest = idx.build_package_manifest(
        gallery_ids=body.get("gallery_ids"),
        query_ids=body.get("query_ids"),
        locations=body.get("locations"),
        labs=body.get("labs"),
        date_after=body.get("date_after"),
        date_before=body.get("date_before"),
    )

    package_id = uuid.uuid4().hex[:12]
    expires_at = time.time() + PACKAGE_TTL

    _packages[package_id] = {
        "manifest": manifest,
        "expires_at": expires_at,
        "include_metadata": body.get("include_metadata", True),
        "include_decisions": body.get("include_decisions", True),
        "created_by": _get_user_email(request),
    }

    return {
        "package_id": package_id,
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "entities": {
            "gallery": len(manifest["entities"]["gallery"]),
            "queries": len(manifest["entities"]["queries"]),
        },
        "expires_at": datetime.fromtimestamp(
            expires_at, tz=timezone.utc
        ).isoformat(),
    }


# ─── Pull: Stream ──────────────────────────────────────────────────────────

@app.get("/api/pull/stream/{package_id}")
async def pull_stream(package_id: str):
    """
    Stream a tar.gz of the files in a package manifest.
    Includes images, and optionally metadata CSV subsets and decisions.
    """
    _cleanup_expired_packages()

    pkg = _packages.get(package_id)
    if not pkg:
        return JSONResponse(
            {"error": "Package not found or expired"},
            status_code=404,
        )

    manifest = pkg["manifest"]

    def generate_tar():
        """Generator that yields tar.gz chunks."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            # Add image files
            for file_info in manifest["files"]:
                abs_path = ARCHIVE_ROOT / file_info["path"]
                if abs_path.exists():
                    tar.add(str(abs_path), arcname=file_info["path"])

            # Add DL precompute cache — sliced to only requested entities
            _add_sliced_embeddings_to_tar(tar, manifest["entities"])

            # Add metadata CSVs if requested
            if pkg.get("include_metadata", True):
                _add_metadata_to_tar(tar, manifest["entities"])

            # Add decisions if requested
            if pkg.get("include_decisions", True):
                _add_decisions_to_tar(tar, manifest["entities"])

        buf.seek(0)
        yield buf.read()

    return StreamingResponse(
        generate_tar(),
        media_type="application/gzip",
        headers={
            "Content-Disposition": f"attachment; filename=starboard_package_{package_id}.tar.gz",
        },
    )


def _add_sliced_embeddings_to_tar(tar: tarfile.TarFile, entities: Dict[str, List[str]]):
    """Add DL embeddings sliced to only the requested entities."""
    try:
        import numpy as np

        dl_root = ARCHIVE_ROOT / "_dl_precompute"
        if not dl_root.exists():
            return

        gallery_ids = set(entities.get("gallery", []))
        query_ids = set(entities.get("queries", []))

        # Always include the registry
        registry = dl_root / "_dl_registry.json"
        if registry.exists():
            tar.add(str(registry), arcname="_dl_precompute/_dl_registry.json")

        # Process each model directory
        for model_dir in dl_root.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            prefix = f"_dl_precompute/{model_name}"

            # Copy evaluation results as-is (small JSON)
            for json_file in model_dir.glob("*.json"):
                tar.add(str(json_file), arcname=f"{prefix}/{json_file.name}")

            # Slice embeddings
            emb_dir = model_dir / "embeddings"
            if emb_dir.exists():
                for npz_file in emb_dir.glob("*.npz"):
                    data = dict(np.load(npz_file))
                    # Filter keys to only requested entities
                    if "gallery" in npz_file.name:
                        filtered = {k: v for k, v in data.items() if k in gallery_ids}
                    elif "query" in npz_file.name:
                        filtered = {k: v for k, v in data.items() if k in query_ids}
                    else:
                        filtered = data  # unknown type, include all

                    if filtered:
                        buf = io.BytesIO()
                        np.savez(buf, **filtered)
                        buf.seek(0)
                        info = tarfile.TarInfo(name=f"{prefix}/embeddings/{npz_file.name}")
                        info.size = len(buf.getvalue())
                        tar.addfile(info, buf)

                # Slice image path JSONs
                for json_file in emb_dir.glob("*.json"):
                    with json_file.open() as f:
                        path_data = json.load(f)
                    if isinstance(path_data, dict):
                        if "gallery" in json_file.name:
                            filtered = {k: v for k, v in path_data.items() if k in gallery_ids}
                        elif "query" in json_file.name:
                            filtered = {k: v for k, v in path_data.items() if k in query_ids}
                        else:
                            filtered = path_data
                    else:
                        filtered = path_data

                    content = json.dumps(filtered).encode("utf-8")
                    info = tarfile.TarInfo(name=f"{prefix}/embeddings/{json_file.name}")
                    info.size = len(content)
                    tar.addfile(info, io.BytesIO(content))

            # Slice similarity scores
            sim_dir = model_dir / "similarity"
            if sim_dir.exists():
                # Load id_mapping to know index positions
                id_map_file = sim_dir / "id_mapping.json"
                if id_map_file.exists():
                    with id_map_file.open() as f:
                        id_mapping = json.load(f)

                    orig_gallery = id_mapping.get("gallery_ids", [])
                    orig_query = id_mapping.get("query_ids", [])

                    # Find indices of requested entities
                    g_indices = [i for i, gid in enumerate(orig_gallery) if gid in gallery_ids]
                    q_indices = [i for i, qid in enumerate(orig_query) if qid in query_ids]

                    new_gallery = [orig_gallery[i] for i in g_indices]
                    new_query = [orig_query[i] for i in q_indices]

                    # Write filtered id_mapping
                    new_mapping = {"gallery_ids": new_gallery, "query_ids": new_query}
                    content = json.dumps(new_mapping).encode("utf-8")
                    info = tarfile.TarInfo(name=f"{prefix}/similarity/id_mapping.json")
                    info.size = len(content)
                    tar.addfile(info, io.BytesIO(content))

                    # Slice score matrices
                    for npz_file in sim_dir.glob("*.npz"):
                        data = dict(np.load(npz_file))
                        for key, arr in data.items():
                            if arr.ndim == 2:
                                # query_gallery_scores: (n_queries, n_gallery)
                                if arr.shape == (len(orig_query), len(orig_gallery)):
                                    data[key] = arr[np.ix_(q_indices, g_indices)] if q_indices and g_indices else np.array([]).reshape(0, 0)
                                # image_similarity_matrix: (n_query_imgs, n_gallery_imgs)
                                # Can't easily slice by entity without image-level index
                                # Include as-is for now (it'll be recomputed on use)

                        buf = io.BytesIO()
                        np.savez(buf, **data)
                        buf.seek(0)
                        info = tarfile.TarInfo(name=f"{prefix}/similarity/{npz_file.name}")
                        info.size = len(buf.getvalue())
                        tar.addfile(info, buf)

                    # Copy other JSON files in similarity/
                    for json_file in sim_dir.glob("*.json"):
                        if json_file.name == "id_mapping.json":
                            continue  # already handled
                        # Filter image index JSONs if possible
                        with json_file.open() as f:
                            jdata = json.load(f)
                        content = json.dumps(jdata).encode("utf-8")
                        info = tarfile.TarInfo(name=f"{prefix}/similarity/{json_file.name}")
                        info.size = len(content)
                        tar.addfile(info, io.BytesIO(content))

            # Handle verification subdirectories
            verif_dir = model_dir / "verification"
            if verif_dir.exists():
                for f in verif_dir.iterdir():
                    if f.is_file():
                        tar.add(str(f), arcname=f"{prefix}/verification/{f.name}")

    except Exception as e:
        log.warning("Could not add sliced embeddings to tar: %s", e)


def _add_metadata_to_tar(tar: tarfile.TarFile, entities: Dict[str, List[str]]):
    """Add filtered metadata CSVs to the tar archive."""
    try:
        from src.data.archive_paths import metadata_csv_paths_for_read, id_column_name
        from src.data.csv_io import read_rows_multi, normalize_id_value

        for target, entity_ids in [("gallery", entities["gallery"]), ("queries", entities["queries"])]:
            if not entity_ids:
                continue

            paths = metadata_csv_paths_for_read(target)
            all_rows = read_rows_multi(paths)
            id_col = id_column_name(target)
            id_set = {normalize_id_value(eid) for eid in entity_ids}

            # Filter rows for selected entities
            filtered = [r for r in all_rows
                        if normalize_id_value(r.get(id_col, "")) in id_set]

            if not filtered:
                continue

            # Write to CSV in memory
            header = list(filtered[0].keys())
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=header)
            writer.writeheader()
            writer.writerows(filtered)

            csv_bytes = buf.getvalue().encode("utf-8-sig")
            csv_name = f"_sync_metadata/{target}_metadata.csv"

            info = tarfile.TarInfo(name=csv_name)
            info.size = len(csv_bytes)
            tar.addfile(info, io.BytesIO(csv_bytes))

    except Exception as e:
        log.warning("Could not add metadata to tar: %s", e)


def _add_decisions_to_tar(tar: tarfile.TarFile, entities: Dict[str, List[str]]):
    """Add relevant match decisions to the tar archive."""
    try:
        import csv as csv_mod

        master_csv = ARCHIVE_ROOT / "reports" / "past_matches_master.csv"
        if not master_csv.exists():
            return

        # Read all decisions
        with master_csv.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv_mod.DictReader(f)
            all_decisions = list(reader)

        if not all_decisions:
            return

        # Filter for decisions involving our entities
        gallery_set = set(entities.get("gallery", []))
        query_set = set(entities.get("queries", []))

        filtered = [
            d for d in all_decisions
            if d.get("gallery_id", "").strip() in gallery_set
            or d.get("query_id", "").strip() in query_set
        ]

        if not filtered:
            return

        header = list(filtered[0].keys())
        buf = io.StringIO()
        writer = csv_mod.DictWriter(buf, fieldnames=header)
        writer.writeheader()
        writer.writerows(filtered)

        csv_bytes = buf.getvalue().encode("utf-8-sig")
        info = tarfile.TarInfo(name="_sync_metadata/decisions.csv")
        info.size = len(csv_bytes)
        tar.addfile(info, io.BytesIO(csv_bytes))

    except Exception as e:
        log.warning("Could not add decisions to tar: %s", e)


# ─── Pull: Metadata only ───────────────────────────────────────────────────

@app.get("/api/pull/metadata")
async def pull_metadata(target: Optional[str] = None):
    """
    Return the full canonical metadata CSV(s).
    For machines that want complete metadata without images.
    """
    import csv as csv_mod
    from src.data.archive_paths import metadata_csv_paths_for_read, id_column_name
    from src.data.csv_io import read_rows_multi

    result = {}
    targets = [target] if target else ["gallery", "queries"]

    for t in targets:
        paths = metadata_csv_paths_for_read(t)
        rows = read_rows_multi(paths)
        result[t] = {
            "row_count": len(rows),
            "rows": rows,
        }

    return result


# ─── Rebuild index (admin) ──────────────────────────────────────────────────

@app.post("/api/admin/rebuild-index")
async def rebuild_index(request: Request):
    """Force a full index rebuild."""
    idx = get_index()
    stats = idx.rebuild_index(compute_hashes=False)
    return {"status": "rebuilt", "stats": stats}


# ─── Sync log ──────────────────────────────────────────────────────────────

@app.get("/api/sync-log")
async def sync_log_endpoint(limit: int = 50):
    """Return recent sync log entries."""
    idx = get_index()
    return {"entries": idx.get_sync_log(limit=limit)}
