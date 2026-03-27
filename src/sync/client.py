"""
starBoard Sync Client — CLI tool for field machines.

Pushes local archive data to the central server and pulls
filtered subsets back. Designed to run on field machines
that don't keep a full copy of the archive.

Usage:
    python -m src.sync.client config --server URL --lab LAB_ID
    python -m src.sync.client push
    python -m src.sync.client pull --gallery anchovy pepperoni
    python -m src.sync.client pull --location "Eagle point"
    python -m src.sync.client pull --all
    python -m src.sync.client catalog
    python -m src.sync.client status
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

log = logging.getLogger("starBoard.sync.client")

# Ensure project root importable
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ─── Cloudflare Access Auth ─────────────────────────────────────────────────

_cf_token: Optional[str] = None


def _get_cf_token(server_url: str) -> Optional[str]:
    """Get a cached Cloudflare Access JWT token, or None."""
    global _cf_token
    if _cf_token:
        return _cf_token

    # Check if we have a saved token
    cfg = load_config()
    token = cfg.get("cf_access_token", "")
    if token:
        _cf_token = token
        return token

    return None


def _authenticate_cloudflare(server_url: str) -> str:
    """
    Authenticate with Cloudflare Access via browser.

    Uses `cloudflared access login` if available, which opens a browser
    for email verification and returns a JWT token.
    If cloudflared is not installed, opens the browser directly and
    instructs the user.
    """
    global _cf_token
    import subprocess, shutil

    cloudflared = shutil.which("cloudflared")
    if cloudflared:
        print("Opening browser for Cloudflare Access login...")
        print("Please verify your email in the browser window.")
        try:
            result = subprocess.run(
                [cloudflared, "access", "login", server_url],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                # Token is the last non-empty line of stdout
                lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
                token = lines[-1] if lines else ""
                if token and token.startswith("ey"):
                    _cf_token = token
                    # Save to config
                    cfg = load_config()
                    cfg["cf_access_token"] = token
                    save_config(cfg)
                    print("Authenticated successfully.")
                    return token
            print(f"Authentication failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("Authentication timed out (120s). Please try again.")
        except Exception as e:
            print(f"Authentication error: {e}")
    else:
        print("ERROR: cloudflared is not installed.")
        print("Install it to enable Cloudflare Access authentication:")
        print("  https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
        print()
        print("Or install on Debian/Ubuntu:")
        print("  curl -fsSL https://pkg.cloudflare.com/cloudflare-public-v2.gpg | sudo tee /usr/share/keyrings/cloudflare-public-v2.gpg >/dev/null")
        print("  echo 'deb [signed-by=/usr/share/keyrings/cloudflare-public-v2.gpg] https://pkg.cloudflare.com/cloudflared any main' | sudo tee /etc/apt/sources.list.d/cloudflared.list")
        print("  sudo apt-get update && sudo apt-get install cloudflared")

    sys.exit(1)


# ─── Config ─────────────────────────────────────────────────────────────────

def _config_path() -> Path:
    from src.data.archive_paths import archive_root
    return archive_root() / "starboard_sync_config.json"


def load_config() -> Dict[str, Any]:
    p = _config_path()
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: Dict[str, Any]) -> None:
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def get_server_url(cfg: Dict[str, Any]) -> str:
    url = cfg.get("server_url", "")
    if not url:
        print("ERROR: No server configured. Run:")
        print("  python -m src.sync.client config --server URL --lab LAB_ID")
        sys.exit(1)
    return url.rstrip("/")


def get_lab_id(cfg: Dict[str, Any]) -> str:
    lab = cfg.get("lab_id", "")
    if not lab:
        from src.sync.config import get_lab_id as _get
        lab = _get()
    return lab


# ─── HTTP helpers ───────────────────────────────────────────────────────────

_USER_AGENT = "starBoard-Sync/0.1"


def _make_request(url: str, data: bytes = None, headers: Dict[str, str] = None,
                   timeout: int = 30) -> Any:
    """Make an HTTP request with Cloudflare Access auth, retrying on 403."""
    hdrs = dict(headers or {})
    hdrs.setdefault("User-Agent", _USER_AGENT)

    # Add CF auth token if available
    cfg = load_config()
    token = _get_cf_token(cfg.get("server_url", ""))
    if token:
        hdrs["cf-access-token"] = token
        hdrs["Cookie"] = f"CF_Authorization={token}"

    req = Request(url, data=data, headers=hdrs)
    try:
        resp = urlopen(req, timeout=timeout)
        # Check if we got the Cloudflare Access login page instead of JSON
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type and "application/json" not in content_type:
            # Got login page — need to authenticate
            raise HTTPError(url, 403, "Cloudflare Access login required", resp.headers, resp)
        return resp
    except HTTPError as e:
        if e.code == 403:
            # Auth required — trigger browser login
            server = cfg.get("server_url", "")
            if server:
                print("\nCloudflare Access authentication required.")
                _authenticate_cloudflare(server)
                # Retry with new token
                token = _get_cf_token(server)
                if token:
                    hdrs["cf-access-token"] = token
                    hdrs["Cookie"] = f"CF_Authorization={token}"
                    req = Request(url, data=data, headers=hdrs)
                    return urlopen(req, timeout=timeout)
        raise


def _api_get(base_url: str, path: str, params: Optional[Dict] = None) -> Any:
    url = f"{base_url}{path}"
    if params:
        url += "?" + urlencode({k: v for k, v in params.items() if v is not None})
    try:
        r = _make_request(url, timeout=30)
        return json.loads(r.read())
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"ERROR: {e.code} from {path}: {body}")
        sys.exit(1)
    except URLError as e:
        print(f"ERROR: Cannot reach server at {base_url}: {e.reason}")
        sys.exit(1)


def _api_post_json(base_url: str, path: str, data: Any) -> Any:
    body = json.dumps(data).encode()
    try:
        r = _make_request(
            f"{base_url}{path}", data=body,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        return json.loads(r.read())
    except HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        print(f"ERROR: {e.code} from {path}: {body_text}")
        sys.exit(1)


def _api_post_multipart(
    base_url: str, path: str,
    fields: Dict[str, str],
    files: List[tuple],
) -> Any:
    """POST multipart/form-data with fields and file uploads."""
    import uuid
    boundary = uuid.uuid4().hex
    body = b""

    for key, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()

    for filename, file_data in files:
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'.encode()
        body += b"Content-Type: application/octet-stream\r\n\r\n"
        body += file_data + b"\r\n"

    body += f"--{boundary}--\r\n".encode()

    try:
        r = _make_request(
            f"{base_url}{path}", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=300,
        )
        return json.loads(r.read())
    except HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        print(f"ERROR: {e.code} from {path}: {body_text}")
        sys.exit(1)


def _api_get_bytes(base_url: str, path: str) -> bytes:
    try:
        r = _make_request(f"{base_url}{path}", timeout=600)
        return r.read()
    except HTTPError as e:
        print(f"ERROR: {e.code} from {path}")
        sys.exit(1)


# ─── Commands ───────────────────────────────────────────────────────────────

def cmd_config(args):
    """Configure the sync client."""
    cfg = load_config()

    if args.server:
        cfg["server_url"] = args.server.rstrip("/")
    if args.lab:
        cfg["lab_id"] = args.lab
        # Also update the sync config module
        from src.sync.config import set_lab_id
        set_lab_id(args.lab)

    save_config(cfg)
    print("Sync config saved:")
    print(f"  Server:  {cfg.get('server_url', '(not set)')}")
    print(f"  Lab ID:  {cfg.get('lab_id', '(not set)')}")
    print(f"  Config:  {_config_path()}")

    # Test connection
    if cfg.get("server_url"):
        print("\nTesting connection...")
        health = _api_get(cfg["server_url"], "/api/health")
        print(f"  Server status: {health['status']}")
        print(f"  Server lab:    {health['lab_id']}")
        print(f"  Archive:       {health['totals']['gallery_ids']} gallery, "
              f"{health['totals']['query_ids']} queries, "
              f"{health['totals']['images']} images")


def cmd_status(args):
    """Show sync status."""
    cfg = load_config()
    server = get_server_url(cfg)
    lab = get_lab_id(cfg)

    print(f"Sync Status")
    print(f"  Server:     {server}")
    print(f"  Lab ID:     {lab}")
    print(f"  Last push:  {cfg.get('last_push_utc', 'never')}")
    print(f"  Last pull:  {cfg.get('last_pull_utc', 'never')}")

    # Check server
    print("\nServer:")
    health = _api_get(server, "/api/health")
    print(f"  Status:     {health['status']}")
    print(f"  Gallery:    {health['totals']['gallery_ids']}")
    print(f"  Queries:    {health['totals']['query_ids']}")
    print(f"  Images:     {health['totals']['images']}")
    print(f"  Last sync:  {health.get('last_sync', 'unknown')}")


def cmd_catalog(args):
    """Browse the central catalog."""
    cfg = load_config()
    server = get_server_url(cfg)

    params = {}
    if args.location:
        params["location"] = args.location
    if args.lab:
        params["lab"] = args.lab

    cat = _api_get(server, "/api/catalog", params)

    print(f"Central Archive Catalog")
    print(f"{'=' * 60}")
    print(f"  Gallery IDs:  {cat['totals']['gallery_ids']}")
    print(f"  Query IDs:    {cat['totals']['query_ids']}")
    print(f"  Total images: {cat['totals']['images']}")
    print(f"  Encounters:   {cat['totals']['encounters']}")
    print(f"  Locations:    {', '.join(cat['all_locations']) or '(none)'}")
    print(f"  Labs:         {', '.join(cat['all_labs']) or '(none)'}")
    print(f"  Date range:   {cat['date_range']['earliest']} to {cat['date_range']['latest']}")

    if cat["gallery"]:
        print(f"\nGallery ({len(cat['gallery'])}):")
        print(f"  {'ID':<25} {'Encounters':>10} {'Images':>8} {'Lab':<15} {'Dates'}")
        print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*15} {'-'*21}")
        for g in cat["gallery"]:
            dates = ""
            if g["date_range_start"]:
                dates = f"{g['date_range_start']} — {g['date_range_end']}"
            print(f"  {g['id']:<25} {g['encounter_count']:>10} {g['image_count']:>8} "
                  f"{g['source_lab'] or '':>15} {dates}")

    if cat["queries"] and not args.no_queries:
        print(f"\nQueries ({len(cat['queries'])}):")
        print(f"  {'ID':<35} {'Encounters':>10} {'Images':>8} {'Date':<12} {'Location'}")
        print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*12} {'-'*15}")
        for q in cat["queries"][:50]:  # limit display
            print(f"  {q['id']:<35} {q['encounter_count']:>10} {q['image_count']:>8} "
                  f"{q['date'] or '':<12} {q['location'] or ''}")
        if len(cat["queries"]) > 50:
            print(f"  ... and {len(cat['queries']) - 50} more")


def cmd_push(args):
    """Push local archive data to the central server."""
    cfg = load_config()
    server = get_server_url(cfg)
    lab = get_lab_id(cfg)

    from src.data.archive_paths import (
        archive_root, gallery_root, queries_root,
        metadata_csv_paths_for_read, id_column_name,
        GALLERY_HEADER, QUERIES_HEADER,
    )
    from src.data.csv_io import read_rows_multi, last_row_per_id
    from src.data.id_registry import list_ids

    archive = archive_root()
    last_push = cfg.get("last_push_utc", "")

    print(f"Pushing to {server}")
    print(f"Lab: {lab}")
    print(f"Last push: {last_push or 'never'}")
    print()

    total_images = 0
    total_metadata = 0
    total_decisions = 0

    # ── Push encounters ──────────────────────────────────────────────
    for target, target_name in [("gallery", "Gallery"), ("queries", "Queries")]:
        ids = list_ids(target)
        print(f"Scanning {target_name} ({len(ids)} IDs)...")

        for entity_id in ids:
            if target == "gallery":
                entity_dir = gallery_root() / entity_id
                entity_type = "gallery"
            else:
                entity_dir = queries_root() / entity_id
                entity_type = "query"

            if not entity_dir.exists():
                continue

            for enc_dir in sorted(entity_dir.iterdir()):
                if not enc_dir.is_dir() or enc_dir.name.startswith(("_", ".")):
                    continue

                # Collect image file paths
                img_paths = [
                    img for img in enc_dir.iterdir()
                    if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS
                ]

                if not img_paths:
                    continue

                # Upload in batches to stay under Cloudflare's ~100MB request limit
                MAX_BATCH_BYTES = 50 * 1024 * 1024  # 50MB per request
                batch = []
                batch_bytes = 0
                enc_accepted = 0
                enc_skipped = 0

                for img_path in sorted(img_paths):
                    data = img_path.read_bytes()
                    if batch and batch_bytes + len(data) > MAX_BATCH_BYTES:
                        # Send current batch
                        r = _api_post_multipart(server, "/api/push/encounters", {
                            "entity_type": entity_type,
                            "entity_id": entity_id,
                            "encounter_folder": enc_dir.name,
                            "source_lab": lab,
                        }, batch)
                        enc_accepted += r["accepted_images"]
                        enc_skipped += r["skipped_duplicates"]
                        batch = []
                        batch_bytes = 0
                    batch.append((img_path.name, data))
                    batch_bytes += len(data)

                # Send remaining batch
                if batch:
                    r = _api_post_multipart(server, "/api/push/encounters", {
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "encounter_folder": enc_dir.name,
                        "source_lab": lab,
                    }, batch)
                    enc_accepted += r["accepted_images"]
                    enc_skipped += r["skipped_duplicates"]

                total_images += enc_accepted
                print(f"  {entity_type}/{entity_id}/{enc_dir.name}: "
                      f"{enc_accepted} new, {enc_skipped} dupes "
                      f"({len(img_paths)} images)")

    # ── Push metadata ────────────────────────────────────────────────
    print("\nPushing metadata...")
    for target in ["gallery", "queries"]:
        paths = metadata_csv_paths_for_read(target)
        rows = read_rows_multi(paths)
        id_col = id_column_name(target)
        latest = last_row_per_id(rows, id_col)

        # Filter to rows modified by this lab, or modified after last push
        push_rows = []
        for entity_id, row in latest.items():
            modified_by = row.get("modified_by_lab", "")
            modified_utc = row.get("last_modified_utc", "")

            # Push if: this lab modified it, OR it's newer than last push
            should_push = False
            if modified_by == lab:
                should_push = True
            elif last_push and modified_utc and modified_utc > last_push:
                should_push = True
            elif not last_push:
                should_push = True  # First push: send everything

            if should_push:
                push_rows.append(row)

        if push_rows:
            result = _api_post_json(server, "/api/push/metadata", {
                "target": target,
                "lab_id": lab,
                "rows": push_rows,
            })
            print(f"  {target}: {result['updated_count']} updated, "
                  f"{result['skipped_count']} skipped")
            total_metadata += result["updated_count"]

    # ── Push decisions ───────────────────────────────────────────────
    print("\nPushing decisions...")
    master_csv = archive / "reports" / "past_matches_master.csv"
    if master_csv.exists():
        with master_csv.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            all_decisions = list(reader)

        # Map to the sync format
        decisions = []
        for d in all_decisions:
            decisions.append({
                "query_id": d.get("query_id", ""),
                "gallery_id": d.get("gallery_id", ""),
                "decision": d.get("verdict", d.get("decision", "")),
                "timestamp": d.get("updated_utc", d.get("timestamp", "")),
                "lab_id": lab,
                "user": d.get("user", ""),
                "notes": d.get("notes", ""),
            })

        if decisions:
            result = _api_post_json(server, "/api/push/decisions", {
                "lab_id": lab,
                "decisions": decisions,
            })
            print(f"  {result['appended_count']} appended, "
                  f"{result['duplicate_count']} duplicates")
            total_decisions = result["appended_count"]

    # ── Update last push time ────────────────────────────────────────
    cfg["last_push_utc"] = datetime.now(timezone.utc).isoformat()
    save_config(cfg)

    print(f"\nPush complete:")
    print(f"  Images:    {total_images} new")
    print(f"  Metadata:  {total_metadata} updated")
    print(f"  Decisions: {total_decisions} new")


def cmd_pull(args):
    """Pull filtered data from the central server."""
    cfg = load_config()
    server = get_server_url(cfg)

    from src.data.archive_paths import archive_root

    archive = archive_root()

    # Build filter
    pull_filter = {}
    if args.gallery:
        pull_filter["gallery_ids"] = args.gallery
    if args.query:
        pull_filter["query_ids"] = args.query
    if args.location:
        pull_filter["locations"] = [args.location]
    if args.lab:
        pull_filter["labs"] = [args.lab]
    if args.date_after:
        pull_filter["date_after"] = args.date_after
    if args.date_before:
        pull_filter["date_before"] = args.date_before
    if args.all:
        pass  # No filters = everything

    pull_filter["include_metadata"] = True
    pull_filter["include_decisions"] = True

    if not pull_filter.get("gallery_ids") and not pull_filter.get("query_ids") \
       and not pull_filter.get("locations") and not pull_filter.get("labs") \
       and not args.all:
        print("ERROR: Specify what to pull. Options:")
        print("  --gallery ID [ID ...]")
        print("  --query ID [ID ...]")
        print("  --location LOCATION")
        print("  --lab LAB")
        print("  --all")
        sys.exit(1)

    # Create package
    print(f"Creating pull package from {server}...")
    pkg = _api_post_json(server, "/api/pull/package", pull_filter)

    file_count = pkg["file_count"]
    total_bytes = pkg["total_bytes"]
    gallery_count = pkg["entities"]["gallery"]
    query_count = pkg["entities"]["queries"]

    print(f"  Files:     {file_count}")
    print(f"  Size:      {total_bytes / (1024**2):.1f} MB")
    print(f"  Gallery:   {gallery_count} IDs")
    print(f"  Queries:   {query_count} IDs")
    print(f"  Expires:   {pkg['expires_at']}")

    if file_count == 0:
        print("\nNothing to pull.")
        return

    # Confirm
    if not args.yes:
        answer = input(f"\nDownload {total_bytes / (1024**2):.1f} MB? [y/N] ").strip().lower()
        if answer != "y":
            print("Cancelled.")
            return

    # Download
    print(f"\nDownloading package {pkg['package_id']}...")
    t0 = time.time()
    tar_bytes = _api_get_bytes(server, f"/api/pull/stream/{pkg['package_id']}")
    elapsed = time.time() - t0
    print(f"  Downloaded {len(tar_bytes) / (1024**2):.1f} MB in {elapsed:.1f}s")

    # Extract
    print(f"Extracting to {archive}...")
    extracted = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            # Skip sync metadata files (handle separately)
            if member.name.startswith("_sync_metadata/"):
                continue

            # Merge embedding files instead of overwriting
            if member.name.startswith("_dl_precompute/") and member.isfile():
                with tar.extractfile(member) as src:
                    _merge_pulled_embedding_file(archive, member.name, src.read())
                continue

            dest = archive / member.name
            if member.isfile():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src:
                    dest.write_bytes(src.read())
                extracted += 1

        # Handle metadata CSV
        for member in tar.getmembers():
            if "_sync_metadata/" in member.name and member.name.endswith("_metadata.csv"):
                print(f"  Merging metadata: {member.name}")
                with tar.extractfile(member) as src:
                    content = src.read().decode("utf-8-sig")
                    _merge_pulled_metadata(archive, member.name, content)

        # Handle decisions
        for member in tar.getmembers():
            if member.name == "_sync_metadata/decisions.csv":
                print(f"  Merging decisions...")
                with tar.extractfile(member) as src:
                    content = src.read().decode("utf-8-sig")
                    _merge_pulled_decisions(archive, content)

    # Update last pull time
    cfg["last_pull_utc"] = datetime.now(timezone.utc).isoformat()
    save_config(cfg)

    print(f"\nPull complete:")
    print(f"  Extracted {extracted} image files")


def _merge_pulled_embedding_file(archive: Path, arcname: str, data: bytes):
    """Merge a pulled embedding file into the local cache.

    NPZ files: merge by key (entity ID) — new entries added, existing updated.
    JSON files: merge by key if dict, otherwise overwrite.
    Other files: overwrite.
    """
    dest = archive / arcname
    dest.parent.mkdir(parents=True, exist_ok=True)

    if arcname.endswith(".npz"):
        try:
            import numpy as np

            # Load new data
            new_data = dict(np.load(io.BytesIO(data)))

            # Load existing if present
            if dest.exists():
                existing = dict(np.load(str(dest)))
                existing.update(new_data)  # new entries overwrite same-key
                merged = existing
            else:
                merged = new_data

            np.savez(str(dest), **merged)
        except Exception as e:
            # Fallback: just write the new file
            dest.write_bytes(data)
            log.warning("Could not merge NPZ %s, overwrote: %s", arcname, e)

    elif arcname.endswith(".json"):
        try:
            new_data = json.loads(data.decode("utf-8"))

            if dest.exists() and isinstance(new_data, dict):
                with dest.open("r") as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    existing.update(new_data)
                    merged = existing
                else:
                    merged = new_data
            else:
                merged = new_data

            with dest.open("w") as f:
                json.dump(merged, f)
        except Exception:
            dest.write_bytes(data)

    else:
        dest.write_bytes(data)


def _merge_pulled_metadata(archive: Path, csv_name: str, content: str):
    """Merge pulled metadata CSV into local archive."""
    from src.data.csv_io import read_rows, last_row_per_id, append_row, ensure_header
    from src.data.archive_paths import metadata_csv_for, id_column_name

    # Determine target from filename
    if "gallery" in csv_name:
        target = "gallery"
    else:
        target = "queries"

    csv_path, header = metadata_csv_for(target)
    id_col = id_column_name(target)

    # Read pulled rows
    reader = csv.DictReader(io.StringIO(content))
    pulled_rows = list(reader)

    # Read local state
    local_rows = read_rows(csv_path)
    local_latest = last_row_per_id(local_rows, id_col)

    updated = 0
    for row in pulled_rows:
        entity_id = row.get(id_col, "").strip()
        if not entity_id:
            continue

        pulled_ts = row.get("last_modified_utc", "")
        local_row = local_latest.get(entity_id)

        if local_row:
            local_ts = local_row.get("last_modified_utc", "")
            if local_ts and pulled_ts and pulled_ts <= local_ts:
                continue  # Local is newer or equal

        # Append pulled row (preserve its timestamps, don't re-stamp)
        ensure_header(csv_path, header)
        ordered = [row.get(col, "") for col in header]
        with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(ordered)
        updated += 1

    print(f"    {target}: {updated} rows merged")


def _merge_pulled_decisions(archive: Path, content: str):
    """Merge pulled decisions into local master CSV."""
    reader = csv.DictReader(io.StringIO(content))
    pulled = list(reader)

    if not pulled:
        return

    master_csv = archive / "reports" / "past_matches_master.csv"

    # Ensure reports directory exists
    master_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load existing keys
    existing_keys = set()
    existing_header = []
    if master_csv.exists():
        with master_csv.open("r", newline="", encoding="utf-8-sig") as f:
            r = csv.DictReader(f)
            if r.fieldnames:
                existing_header = list(r.fieldnames)
            for row in r:
                ts = (row.get("updated_utc", "") or row.get("timestamp", "")).strip()
                key = (row.get("query_id", "").strip(),
                       row.get("gallery_id", "").strip(), ts)
                existing_keys.add(key)

    appended = 0
    with master_csv.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for d in pulled:
            ts = (d.get("updated_utc", "") or d.get("timestamp", "")).strip()
            key = (d.get("query_id", "").strip(),
                   d.get("gallery_id", "").strip(), ts)
            if key in existing_keys:
                continue
            if existing_header:
                writer.writerow([d.get(col, "") for col in existing_header])
            appended += 1
            existing_keys.add(key)

    print(f"    {appended} decisions merged")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="starboard-sync",
        description="starBoard Sync Client — push and pull archive data",
    )
    sub = parser.add_subparsers(dest="command")

    # config
    p_cfg = sub.add_parser("config", help="Configure sync client")
    p_cfg.add_argument("--server", help="Central server URL")
    p_cfg.add_argument("--lab", help="Lab/machine identifier")

    # status
    sub.add_parser("status", help="Show sync status")

    # catalog
    p_cat = sub.add_parser("catalog", help="Browse central catalog")
    p_cat.add_argument("--location", help="Filter by location")
    p_cat.add_argument("--lab", help="Filter by lab")
    p_cat.add_argument("--no-queries", action="store_true",
                        help="Hide queries in output")

    # push
    sub.add_parser("push", help="Push local data to central server")

    # pull
    p_pull = sub.add_parser("pull", help="Pull data from central server")
    p_pull.add_argument("--gallery", nargs="+", help="Gallery IDs to pull")
    p_pull.add_argument("--query", nargs="+", help="Query IDs to pull")
    p_pull.add_argument("--location", help="Pull by location")
    p_pull.add_argument("--lab", help="Pull by source lab")
    p_pull.add_argument("--date-after", help="Pull encounters after date (YYYY-MM-DD)")
    p_pull.add_argument("--date-before", help="Pull encounters before date (YYYY-MM-DD)")
    p_pull.add_argument("--all", action="store_true", help="Pull everything")
    p_pull.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompt")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.command == "config":
        cmd_config(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "catalog":
        cmd_catalog(args)
    elif args.command == "push":
        cmd_push(args)
    elif args.command == "pull":
        cmd_pull(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
