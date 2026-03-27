# starBoard Sync Guide

How to sync archive data between field machines and the central server.

## Overview

starBoard supports multi-machine sync:

- **One central machine** runs the sync server and holds the full archive
- **Field machines** push their local data to the central server and pull subsets back
- Sync works over HTTPS through a Cloudflare Tunnel — no VPN or special network setup needed on field machines

## Architecture

```
Field Machine A                     Central Machine
┌──────────────────────┐            ┌──────────────────────────┐
│ starBoard app         │            │ starBoard app             │
│ Local archive (partial)│   PUSH ►  │ Full archive (all data)   │
│                        │  ◄ PULL   │                            │
│ Sync tab or CLI client │           │ Sync server (port 8090)   │
└──────────────────────┘            │ Cloudflare Tunnel          │
                                    └──────────────────────────┘
Field Machine B                              ▲
┌──────────────────────┐                     │
│ starBoard app         │      PUSH / PULL ──┘
│ Local archive (partial)│
│ Sync tab or CLI client │
└──────────────────────┘
```

## For Field Machine Users

### Requirements

On top of the normal starBoard dependencies:

```bash
pip install fastapi uvicorn python-multipart
```

You also need `cloudflared` installed for authentication. The central server uses
Cloudflare Access email verification — `cloudflared` handles the browser login flow.

**Debian/Ubuntu:**
```bash
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-public-v2.gpg | sudo tee /usr/share/keyrings/cloudflare-public-v2.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-public-v2.gpg] https://pkg.cloudflare.com/cloudflared any main' | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update && sudo apt-get install cloudflared
```

**macOS:**
```bash
brew install cloudflared
```

**Other platforms:** https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

> Note: field machines do NOT need to configure a tunnel. `cloudflared` is only
> used for the `cloudflared access login` command which handles browser-based
> email verification.

### Authentication

The first time you push, pull, or test connection, the app will:

1. Detect that authentication is required (403 from server)
2. Automatically run `cloudflared access login` which opens your browser
3. You enter your email address on the Cloudflare login page
4. You receive a one-time code via email and enter it
5. The JWT token is saved locally and reused for 1 week

Your email must be authorized by the central server operator. If you get
"access denied", contact the server operator to add your email to the
Cloudflare Access policy.

### Setup (One Time)

#### Option A: Using the Sync Tab (GUI)

1. Launch starBoard: `python main.py`
2. Click the **Sync** tab (rightmost tab)
3. In the **Connection** section:
   - **Server URL**: enter the central server URL (e.g. `https://upload.fhl-star-board.com`)
   - **Lab ID**: enter your lab/machine identifier (e.g. `hodin_lab`, `fhl_dock`)
4. Click **Save Config**
5. Click **Test Connection** — you should see "Connected ✓" with archive stats

#### Option B: Using the CLI

```bash
python -m src.sync.client config \
  --server https://upload.fhl-star-board.com \
  --lab hodin_lab
```

### Pushing Data

Push sends your local images, metadata, and match decisions to the central server.
Only new/modified data is sent — duplicates are automatically skipped.

**GUI:** Click **Push All Local Data** in the Sync tab.

**CLI:**
```bash
python -m src.sync.client push
```

What gets pushed:
- All encounter folders (images) for every gallery and query ID
- Metadata rows from gallery_metadata.csv and queries_metadata.csv
- Match decisions from reports/past_matches_master.csv

Deduplication:
- Images are deduplicated by SHA-256 hash — re-pushing the same images is safe
- Metadata uses timestamp-based merge: newer rows win, older rows are skipped
- Decisions are deduplicated by (query_id, gallery_id, timestamp)

### Pulling Data

Pull downloads images and metadata from the central server with optional filters.

**GUI:**
1. The catalog auto-refreshes on startup (or click **Refresh Catalog from Server**)
2. Use the searchable multi-select lists to check gallery IDs, query IDs, or locations
3. Optionally enable date range filters with the calendar pickers
4. Click **Pull Selected Data**
5. Or click **Pull Everything** to download the full archive

**CLI:**
```bash
# Pull specific gallery IDs
python -m src.sync.client pull --gallery anchovy pepperoni feta

# Pull by location
python -m src.sync.client pull --location "Eagle point"

# Pull by date range
python -m src.sync.client pull --date-after 2026-01-01

# Pull everything
python -m src.sync.client pull --all -y
```

### Checking Status

**GUI:** The Sync tab shows last push/pull timestamps and server stats.

**CLI:**
```bash
python -m src.sync.client status
```

### Browsing the Catalog

**GUI:** The Sync tab shows the full catalog with searchable filter lists.

**CLI:**
```bash
python -m src.sync.client catalog
python -m src.sync.client catalog --no-queries  # hide queries for cleaner output
```

---

## For Central Server Operators

### Requirements

The central machine needs:
- starBoard installed with `pip install fastapi uvicorn python-multipart`
- `cloudflared` installed and configured as a systemd service
- The sync server running as a systemd service

### Server Setup

The sync server is a FastAPI application that runs on port 8090.

#### Start manually (for testing):

```bash
cd /path/to/starBoard
PYTHONPATH=. python -m src.sync --port 8090
```

#### Run as a systemd service:

Create `/etc/systemd/system/starboard-sync.service`:

```ini
[Unit]
Description=starBoard Sync Server (FastAPI on port 8090)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=<your-username>
Group=<your-group>
WorkingDirectory=/path/to/starBoard
Environment=PYTHONPATH=/path/to/starBoard
ExecStart=/path/to/python -m src.sync --host 127.0.0.1 --port 8090
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable starboard-sync
sudo systemctl start starboard-sync
```

### Cloudflare Tunnel Setup

The tunnel exposes the local server to the internet securely.

1. Install `cloudflared` on the central machine
2. Create a tunnel in the Cloudflare Zero Trust dashboard
3. Add a hostname route pointing to `http://localhost:8090`
4. Install and enable the tunnel as a systemd service

Field machines connect to the public hostname (e.g. `https://upload.fhl-star-board.com`) — they don't need any tunnel software.

### Rebuilding the Index

The server rebuilds its SQLite index on every startup. To force a manual rebuild:

```bash
PYTHONPATH=. python -m src.sync --rebuild-index
```

Or via the API:

```
POST /api/admin/rebuild-index
```

### Server API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Server status and archive stats |
| `/api/catalog` | GET | Browsable archive index with optional filters |
| `/api/push/encounters` | POST | Upload image files for an encounter |
| `/api/push/metadata` | POST | Push metadata rows (timestamp-based merge) |
| `/api/push/decisions` | POST | Push match decisions (dedup append) |
| `/api/pull/package` | POST | Create a download package manifest |
| `/api/pull/stream/{id}` | GET | Download a package as tar.gz |
| `/api/pull/metadata` | GET | Download full metadata CSVs |
| `/api/admin/rebuild-index` | POST | Force index rebuild |
| `/api/sync-log` | GET | View sync audit log |

### Lab ID Configuration

Each machine has a lab ID used to tag data. Resolution order:

1. `STARBOARD_LAB_ID` environment variable
2. `archive/starboard_sync_config.json` → `lab_id` field
3. Machine hostname (fallback)

### CSV Schema Changes

The sync system adds 3 columns to the end of gallery_metadata.csv and queries_metadata.csv:

| Column | Description | Behavior |
|--------|-------------|----------|
| `last_modified_utc` | ISO 8601 timestamp | Auto-set on every save |
| `modified_by_lab` | Lab ID of the machine that made the change | Auto-set on every save |
| `source_lab` | Lab ID that originally created the record | Set once, never overwritten |

These columns are added automatically when the app starts (via the existing CSV header auto-upgrade mechanism). Existing data is not modified — new columns are filled with empty strings for existing rows.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No server configured" | Enter the server URL in the Sync tab and click Save Config |
| Connection failed | Check that the central server is running and the URL is correct |
| Push shows all duplicates | Normal on re-push — means all data was already on the server |
| Pull downloads 0 files | Check your filters — try "Pull Everything" to verify connectivity |
| Catalog lists are empty | Click "Refresh Catalog from Server" or check connection |
| Server won't start | Check `journalctl -u starboard-sync` for errors |
| Tunnel not routing | Check `systemctl status cloudflared-starboard` |
| 403 on every request | Token expired — delete `cf_access_token` from `archive/starboard_sync_config.json` and retry to trigger re-auth |
| "cloudflared is not installed" | Install cloudflared — required for email verification on field machines |
| Browser doesn't open for auth | Run `cloudflared access login https://upload.fhl-star-board.com` manually in a terminal |
