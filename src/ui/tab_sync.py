# src/ui/tab_sync.py
"""
Sync tab for starBoard — push/pull archive data to/from a central server.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal, QTimer, QDate, QSortFilterProxyModel, QStringListModel
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QLineEdit, QComboBox, QCheckBox, QProgressBar, QTextEdit,
    QMessageBox, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem,
    QHeaderView, QFormLayout, QFrame, QSplitter, QDateEdit, QListWidget,
    QListWidgetItem, QCompleter, QAbstractItemView, QStyledItemDelegate,
)

from src.ui.collapsible import CollapsibleSection
from src.utils.interaction_logger import get_interaction_logger

log = logging.getLogger("starBoard.ui.tab_sync")


class SearchableMultiSelect(QWidget):
    """A searchable combo-like widget with multi-select via checkboxes.

    Shows a search field at top; below it a scrollable list of checkboxes.
    Call set_items() to populate, selected_items() to read.
    """
    selectionChanged = Signal()

    def __init__(self, placeholder: str = "Search...", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        # Top row: search + select all / deselect all
        top_row = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText(placeholder)
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._filter)
        top_row.addWidget(self._search)

        self._btn_all = QPushButton("All")
        self._btn_all.setFixedWidth(40)
        self._btn_all.setToolTip("Select all")
        self._btn_all.clicked.connect(self._select_all)
        top_row.addWidget(self._btn_all)

        self._btn_none = QPushButton("None")
        self._btn_none.setFixedWidth(45)
        self._btn_none.setToolTip("Deselect all")
        self._btn_none.clicked.connect(self._deselect_all)
        top_row.addWidget(self._btn_none)

        lay.addLayout(top_row)

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.NoSelection)
        self._list.setMinimumHeight(90)
        self._list.setMaximumHeight(150)
        lay.addWidget(self._list)

        self._all_items: List[str] = []

    def set_items(self, items: List[str]):
        """Replace all items (unchecked by default)."""
        self._all_items = sorted(set(items))
        self._rebuild()

    def _rebuild(self):
        checked = self.selected_items()
        self._list.clear()
        for name in self._all_items:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in checked else Qt.Unchecked)
            self._list.addItem(item)
        self._list.itemChanged.connect(lambda _: self.selectionChanged.emit())

    def _filter(self, text: str):
        text_lower = text.lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(text_lower not in item.text().lower())

    def selected_items(self) -> List[str]:
        result = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.Checked:
                result.append(item.text())
        return result

    def set_selected(self, names: List[str]):
        name_set = set(names)
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setCheckState(Qt.Checked if item.text() in name_set else Qt.Unchecked)

    def add_and_select(self, name: str):
        """Add an item if not present and check it."""
        if name not in self._all_items:
            self._all_items.append(name)
            self._all_items.sort()
            self._rebuild()
        self.set_selected(self.selected_items() + [name])

    def _select_all(self):
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.Checked)
        self.selectionChanged.emit()

    def _deselect_all(self):
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.Unchecked)
        self.selectionChanged.emit()


class TabSync(QWidget):
    """Sync tab — push/pull archive data to/from central server."""

    # Emitted from worker threads to update UI safely
    _log_signal = Signal(str)
    _status_signal = Signal(dict)
    _catalog_signal = Signal(dict)
    _progress_signal = Signal(int, int)  # current, total
    _progress_detail_signal = Signal(str)  # detail text (speed, ETA, etc.)
    _done_signal = Signal(str)  # "push" or "pull"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ilog = get_interaction_logger()
        self._build_ui()
        self._connect_signals()
        self._load_config()

        # Auto-refresh catalog on startup (after a short delay so UI is ready)
        QTimer.singleShot(2000, self._auto_refresh_catalog)

    # ── UI Construction ─────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        scroll.setWidget(scroll_widget)
        outer.addWidget(scroll)

        # ── Connection Config ──
        cfg_section = CollapsibleSection("Connection", start_collapsed=False)
        cfg_content = QWidget()
        cfg_lay = QFormLayout(cfg_content)

        self._server_input = QLineEdit()
        self._server_input.setPlaceholderText("https://upload.fhl-star-board.com")
        cfg_lay.addRow("Server URL:", self._server_input)

        self._lab_input = QLineEdit()
        self._lab_input.setPlaceholderText("e.g. hodin_lab, fhl_dock")
        cfg_lay.addRow("Lab ID:", self._lab_input)

        btn_row = QHBoxLayout()
        self._btn_save_config = QPushButton("Save Config")
        self._btn_test_connection = QPushButton("Test Connection")
        btn_row.addWidget(self._btn_save_config)
        btn_row.addWidget(self._btn_test_connection)
        btn_row.addStretch()
        cfg_lay.addRow(btn_row)

        self._lbl_connection_status = QLabel("")
        cfg_lay.addRow("Status:", self._lbl_connection_status)

        cfg_section.setContent(cfg_content)
        layout.addWidget(cfg_section)

        # ── Status ──
        status_section = CollapsibleSection("Sync Status", start_collapsed=False)
        status_content = QWidget()
        status_lay = QFormLayout(status_content)

        self._lbl_last_push = QLabel("never")
        self._lbl_last_pull = QLabel("never")
        self._lbl_server_gallery = QLabel("-")
        self._lbl_server_queries = QLabel("-")
        self._lbl_server_images = QLabel("-")

        status_lay.addRow("Last push:", self._lbl_last_push)
        status_lay.addRow("Last pull:", self._lbl_last_pull)
        status_lay.addRow("Server gallery IDs:", self._lbl_server_gallery)
        status_lay.addRow("Server query IDs:", self._lbl_server_queries)
        status_lay.addRow("Server images:", self._lbl_server_images)

        status_section.setContent(status_content)
        layout.addWidget(status_section)

        # ── Push ──
        push_section = CollapsibleSection("Push to Central", start_collapsed=False)
        push_content = QWidget()
        push_lay = QVBoxLayout(push_content)

        push_desc = QLabel(
            "Push your local archive data (images, metadata, match decisions) "
            "to the central server. Only new/modified data is sent — duplicates "
            "are automatically skipped."
        )
        push_desc.setWordWrap(True)
        push_lay.addWidget(push_desc)

        push_btn_row = QHBoxLayout()
        self._btn_push = QPushButton("Push All Local Data")
        self._btn_push.setMinimumHeight(36)
        self._btn_push.setStyleSheet("QPushButton { font-weight: bold; }")
        push_btn_row.addWidget(self._btn_push)
        push_btn_row.addStretch()
        push_lay.addLayout(push_btn_row)

        push_section.setContent(push_content)
        layout.addWidget(push_section)

        # ── Pull ──
        pull_section = CollapsibleSection("Pull from Central", start_collapsed=False)
        pull_content = QWidget()
        pull_lay = QVBoxLayout(pull_content)

        pull_desc = QLabel(
            "Pull images and metadata from the central server. "
            "Use the catalog below to see what's available, then "
            "select what to download."
        )
        pull_desc.setWordWrap(True)
        pull_lay.addWidget(pull_desc)

        # Filters — three columns: Gallery | Query | Location
        filter_row = QHBoxLayout()

        # Gallery multi-select
        gallery_col = QVBoxLayout()
        gallery_col.addWidget(QLabel("Gallery IDs:"))
        self._pull_gallery_select = SearchableMultiSelect("Search gallery...")
        gallery_col.addWidget(self._pull_gallery_select)
        filter_row.addLayout(gallery_col)

        # Query multi-select
        query_col = QVBoxLayout()
        query_col.addWidget(QLabel("Query IDs:"))
        self._pull_query_select = SearchableMultiSelect("Search queries...")
        query_col.addWidget(self._pull_query_select)
        filter_row.addLayout(query_col)

        # Location multi-select
        location_col = QVBoxLayout()
        location_col.addWidget(QLabel("Locations:"))
        self._pull_location_select = SearchableMultiSelect("Search locations...")
        location_col.addWidget(self._pull_location_select)
        filter_row.addLayout(location_col)

        pull_lay.addLayout(filter_row)

        # Date range — Before (left) | After (right)
        date_lay = QHBoxLayout()

        self._chk_date_after = QCheckBox("After:")
        self._pull_date_after = QDateEdit()
        self._pull_date_after.setCalendarPopup(True)
        self._pull_date_after.setDisplayFormat("yyyy-MM-dd")
        self._pull_date_after.setDate(QDate.currentDate().addMonths(-6))
        self._pull_date_after.setEnabled(False)
        self._chk_date_after.toggled.connect(self._pull_date_after.setEnabled)
        date_lay.addWidget(self._chk_date_after)
        date_lay.addWidget(self._pull_date_after)

        date_lay.addSpacing(20)

        self._chk_date_before = QCheckBox("Before:")
        self._pull_date_before = QDateEdit()
        self._pull_date_before.setCalendarPopup(True)
        self._pull_date_before.setDisplayFormat("yyyy-MM-dd")
        self._pull_date_before.setDate(QDate.currentDate())
        self._pull_date_before.setEnabled(False)
        self._chk_date_before.toggled.connect(self._pull_date_before.setEnabled)
        date_lay.addWidget(self._chk_date_before)
        date_lay.addWidget(self._pull_date_before)

        date_lay.addStretch()
        pull_lay.addLayout(date_lay)

        pull_btn_row = QHBoxLayout()
        self._btn_pull = QPushButton("Pull Selected Data")
        self._btn_pull.setMinimumHeight(36)
        self._btn_pull.setStyleSheet("QPushButton { font-weight: bold; }")
        self._btn_pull_all = QPushButton("Pull Everything")
        pull_btn_row.addWidget(self._btn_pull)
        pull_btn_row.addWidget(self._btn_pull_all)
        pull_btn_row.addStretch()
        pull_lay.addLayout(pull_btn_row)

        pull_section.setContent(pull_content)
        layout.addWidget(pull_section)

        # ── Refresh Catalog (always visible, not in a collapsible) ──
        cat_btn_row = QHBoxLayout()
        self._btn_refresh_catalog = QPushButton("Refresh Catalog from Server")
        self._btn_refresh_catalog.setMinimumHeight(32)
        cat_btn_row.addWidget(self._btn_refresh_catalog)
        cat_btn_row.addStretch()
        self._lbl_catalog_summary = QLabel("")
        cat_btn_row.addWidget(self._lbl_catalog_summary)
        layout.addLayout(cat_btn_row)

        # ── Catalog table (collapsible) ──
        catalog_section = CollapsibleSection("Central Archive Catalog", start_collapsed=True)
        catalog_content = QWidget()
        catalog_lay = QVBoxLayout(catalog_content)

        self._catalog_table = QTableWidget()
        self._catalog_table.setColumnCount(5)
        self._catalog_table.setHorizontalHeaderLabels(
            ["Type", "ID", "Encounters", "Images", "Date Range"]
        )
        self._catalog_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self._catalog_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._catalog_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._catalog_table.setMinimumHeight(250)
        self._catalog_table.doubleClicked.connect(self._on_catalog_double_click)
        catalog_lay.addWidget(self._catalog_table)

        catalog_section.setContent(catalog_content)
        layout.addWidget(catalog_section)

        # ── Progress / Log ──
        progress_row = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setMinimumHeight(22)
        progress_row.addWidget(self._progress, 1)

        self._lbl_progress_detail = QLabel("")
        self._lbl_progress_detail.setVisible(False)
        self._lbl_progress_detail.setMinimumWidth(280)
        progress_row.addWidget(self._lbl_progress_detail)
        layout.addLayout(progress_row)

        self._log_output = QTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setMaximumHeight(150)
        self._log_output.setPlaceholderText("Sync activity log...")
        layout.addWidget(self._log_output)

        layout.addStretch()

    # ── Signal wiring ───────────────────────────────────────────────────

    def _connect_signals(self):
        self._btn_save_config.clicked.connect(self._save_config)
        self._btn_test_connection.clicked.connect(self._test_connection)
        self._btn_push.clicked.connect(self._do_push)
        self._btn_pull.clicked.connect(self._do_pull)
        self._btn_pull_all.clicked.connect(self._do_pull_all)
        self._btn_refresh_catalog.clicked.connect(self._refresh_catalog)

        # Thread-safe UI updates
        self._log_signal.connect(self._append_log)
        self._status_signal.connect(self._update_status)
        self._catalog_signal.connect(self._populate_catalog)
        self._progress_signal.connect(self._update_progress)
        self._progress_detail_signal.connect(self._update_progress_detail)
        self._done_signal.connect(self._on_done)

    # ── Config ──────────────────────────────────────────────────────────

    def _load_config(self):
        try:
            from src.sync.config import get_lab_id
            from src.data.archive_paths import archive_root
            cfg_path = archive_root() / "starboard_sync_config.json"
            if cfg_path.exists():
                with cfg_path.open("r") as f:
                    cfg = json.load(f)
                self._server_input.setText(cfg.get("server_url", ""))
                self._lab_input.setText(cfg.get("lab_id", ""))
                self._lbl_last_push.setText(cfg.get("last_push_utc", "never") or "never")
                self._lbl_last_pull.setText(cfg.get("last_pull_utc", "never") or "never")
            else:
                self._lab_input.setText(get_lab_id())
        except Exception as e:
            log.warning("Could not load sync config: %s", e)

    def _save_config(self):
        self._ilog.log("button_click", "btn_sync_save_config", value="clicked")
        try:
            from src.data.archive_paths import archive_root
            from src.sync.config import set_lab_id

            cfg_path = archive_root() / "starboard_sync_config.json"
            cfg = {}
            if cfg_path.exists():
                with cfg_path.open("r") as f:
                    cfg = json.load(f)

            cfg["server_url"] = self._server_input.text().strip().rstrip("/")
            cfg["lab_id"] = self._lab_input.text().strip()

            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with cfg_path.open("w") as f:
                json.dump(cfg, f, indent=2)

            if cfg["lab_id"]:
                set_lab_id(cfg["lab_id"])

            self._append_log("Config saved.")
            self._lbl_connection_status.setText("Config saved")
        except Exception as e:
            self._append_log(f"Error saving config: {e}")

    def _get_server(self) -> str:
        url = self._server_input.text().strip().rstrip("/")
        if not url:
            QMessageBox.warning(self, "No Server",
                                "Please enter a server URL and save config first.")
        return url

    # ── Test Connection ─────────────────────────────────────────────────

    def _cf_auth_headers(self) -> Dict[str, str]:
        """Return headers with Cloudflare Access token if available."""
        headers = {"User-Agent": "starBoard-Sync/0.1"}
        try:
            cfg_path = self._config_path()
            if cfg_path and cfg_path.exists():
                with cfg_path.open("r") as f:
                    cfg = json.load(f)
                token = cfg.get("cf_access_token", "")
                if token:
                    headers["cf-access-token"] = token
                    headers["Cookie"] = f"CF_Authorization={token}"
        except Exception:
            pass
        return headers

    def _config_path(self):
        try:
            from src.data.archive_paths import archive_root
            return archive_root() / "starboard_sync_config.json"
        except Exception:
            return None

    def _cf_urlopen(self, url: str, data: bytes = None, timeout: int = 30):
        """urlopen with CF Access auth headers. Auto-triggers browser auth on 403."""
        from urllib.request import Request, urlopen  # noqa: F811
        from urllib.error import HTTPError  # noqa: F811
        headers = self._cf_auth_headers()
        if data and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        req = Request(url, data=data, headers=headers)
        try:
            resp = urlopen(req, timeout=timeout)
            # Detect Cloudflare Access login page (HTML instead of JSON)
            ct = resp.headers.get("Content-Type", "")
            if "text/html" in ct and "application/json" not in ct:
                raise HTTPError(url, 403, "Cloudflare Access login required", resp.headers, resp)
            return resp
        except HTTPError as e:
            if e.code == 403:
                self._log_signal.emit("Authentication required — opening browser...")
                token = self._run_cf_auth()
                if token:
                    headers["cf-access-token"] = token
                    headers["Cookie"] = f"CF_Authorization={token}"
                    req = Request(url, data=data, headers=headers)
                    return urlopen(req, timeout=timeout)
            raise

    def _run_cf_auth(self) -> Optional[str]:
        """Run cloudflared access login and return the JWT token."""
        import subprocess, shutil

        cloudflared = shutil.which("cloudflared")
        if not cloudflared:
            self._log_signal.emit(
                "ERROR: cloudflared is not installed. "
                "Install it to enable authentication: "
                "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
            )
            return None

        server = self._server_input.text().strip().rstrip("/")
        if not server:
            return None

        try:
            self._log_signal.emit("Waiting for email verification in browser...")
            result = subprocess.run(
                [cloudflared, "access", "login", server],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
                token = lines[-1] if lines else ""
                if token and token.startswith("ey"):
                    # Save token to config
                    cfg_path = self._config_path()
                    if cfg_path:
                        cfg = {}
                        if cfg_path.exists():
                            with cfg_path.open("r") as f:
                                cfg = json.load(f)
                        cfg["cf_access_token"] = token
                        with cfg_path.open("w") as f:
                            json.dump(cfg, f, indent=2)
                    self._log_signal.emit("Authenticated successfully.")
                    return token
            self._log_signal.emit(f"Authentication failed: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            self._log_signal.emit("Authentication timed out (120s).")
        except Exception as e:
            self._log_signal.emit(f"Authentication error: {e}")
        return None

    def _test_connection(self):
        self._ilog.log("button_click", "btn_sync_test_connection", value="clicked")
        server = self._get_server()
        if not server:
            return
        self._lbl_connection_status.setText("Testing...")

        def worker():
            try:
                r = self._cf_urlopen(f"{server}/api/health", timeout=10)
                data = json.loads(r.read())
                self._status_signal.emit(data)
                self._log_signal.emit(
                    f"Connected: {data['totals']['gallery_ids']} gallery, "
                    f"{data['totals']['query_ids']} queries, "
                    f"{data['totals']['images']} images"
                )
                self._lbl_connection_status.setText("Connected ✓")
            except Exception as e:
                self._log_signal.emit(f"Connection failed: {e}")
                self._lbl_connection_status.setText(f"Failed: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _update_status(self, data: dict):
        totals = data.get("totals", {})
        self._lbl_server_gallery.setText(str(totals.get("gallery_ids", "-")))
        self._lbl_server_queries.setText(str(totals.get("query_ids", "-")))
        self._lbl_server_images.setText(str(totals.get("images", "-")))

    # ── Push ────────────────────────────────────────────────────────────

    def _do_push(self):
        self._ilog.log("button_click", "btn_sync_push", value="clicked")
        server = self._get_server()
        if not server:
            return

        self._btn_push.setEnabled(False)
        self._show_progress()
        self._progress.setRange(0, 0)  # indeterminate until we know total
        self._append_log("Starting push...")

        def worker():
            try:
                import sys, os, csv, io
                from urllib.request import Request, urlopen
                from urllib.error import HTTPError
                import uuid

                cf_headers = self._cf_auth_headers()

                def _authed_open(req_or_url, timeout=300):
                    """urlopen with auto-retry on 403 via browser auth."""
                    try:
                        return urlopen(req_or_url, timeout=timeout)
                    except HTTPError as e:
                        if e.code == 403:
                            self._log_signal.emit("Authentication required — opening browser...")
                            token = self._run_cf_auth()
                            if token:
                                cf_headers["cf-access-token"] = token
                                cf_headers["Cookie"] = f"CF_Authorization={token}"
                                if isinstance(req_or_url, Request):
                                    req_or_url.add_header("cf-access-token", token)
                                    req_or_url.add_header("Cookie", f"CF_Authorization={token}")
                                return urlopen(req_or_url, timeout=timeout)
                        raise

                sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
                from src.data.archive_paths import (
                    archive_root, gallery_root, queries_root,
                    metadata_csv_paths_for_read, id_column_name,
                )
                from src.data.csv_io import read_rows_multi, last_row_per_id
                from src.data.id_registry import list_ids
                from src.sync.config import get_lab_id

                archive = archive_root()
                lab = self._lab_input.text().strip() or get_lab_id()
                IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

                total_new = 0
                total_dupes = 0
                total_meta = 0
                total_decisions = 0

                # Count total encounters first for progress bar
                import time as _time
                all_encounters = []  # (target, entity_type, entity_id, enc_dir)
                for target, target_name in [("gallery", "Gallery"), ("queries", "Queries")]:
                    ids = list_ids(target)
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
                            if enc_dir.is_dir() and not enc_dir.name.startswith(("_", ".")):
                                all_encounters.append((target, entity_type, entity_id, enc_dir))

                total_enc = len(all_encounters)
                self._log_signal.emit(f"Found {total_enc} encounters to push")
                self._progress_signal.emit(0, total_enc + 2)  # +2 for metadata + decisions
                push_start = _time.time()

                # Push encounters
                for enc_idx, (target, entity_type, entity_id, enc_dir) in enumerate(all_encounters):
                    elapsed = _time.time() - push_start
                    if enc_idx > 0:
                        rate = enc_idx / elapsed
                        remaining = (total_enc - enc_idx) / rate
                        mins, secs = divmod(int(remaining), 60)
                        self._progress_detail_signal.emit(
                            f"Encounter {enc_idx+1}/{total_enc}  |  "
                            f"ETA: {mins}m {secs}s  |  "
                            f"{total_new} new, {total_dupes} dupes"
                        )
                    self._progress_signal.emit(enc_idx, total_enc + 2)

                    # Re-derive entity_dir for the old scanning code below
                    # (already have entity_type, entity_id, enc_dir from the list)
                    pass  # variables already set from all_encounters

                    # Scan for images in this encounter
                    images = []
                    for img in enc_dir.iterdir():
                        if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                            images.append((img.name, img.read_bytes()))

                    if not images:
                        continue

                    # Multipart upload
                    boundary = uuid.uuid4().hex
                    body = b""
                    for key, value in {
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "encounter_folder": enc_dir.name,
                        "source_lab": lab,
                    }.items():
                        body += f"--{boundary}\r\n".encode()
                        body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
                        body += f"{value}\r\n".encode()

                    for fname, fdata in images:
                        body += f"--{boundary}\r\n".encode()
                        body += f'Content-Disposition: form-data; name="files"; filename="{fname}"\r\n'.encode()
                        body += b"Content-Type: application/octet-stream\r\n\r\n"
                        body += fdata + b"\r\n"

                    body += f"--{boundary}--\r\n".encode()

                    enc_headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
                    enc_headers.update(cf_headers)
                    req = Request(
                        f"{server}/api/push/encounters",
                        data=body,
                        headers=enc_headers,
                    )
                    r = _authed_open(req, timeout=300)
                    result = json.loads(r.read())
                    accepted = result["accepted_images"]
                    skipped = result["skipped_duplicates"]
                    total_new += accepted
                    total_dupes += skipped

                    if accepted > 0:
                        self._log_signal.emit(
                            f"  {entity_type}/{entity_id}/{enc_dir.name}: "
                            f"{accepted} new, {skipped} dupes"
                        )

                # Push metadata
                self._progress_signal.emit(total_enc, total_enc + 2)
                self._progress_detail_signal.emit("Pushing metadata...")
                self._log_signal.emit("Pushing metadata...")
                for target in ["gallery", "queries"]:
                    paths = metadata_csv_paths_for_read(target)
                    rows = read_rows_multi(paths)
                    id_col = id_column_name(target)
                    latest = last_row_per_id(rows, id_col)
                    push_rows = list(latest.values())

                    if push_rows:
                        body = json.dumps({
                            "target": target, "lab_id": lab, "rows": push_rows,
                        }).encode()
                        meta_headers = {"Content-Type": "application/json"}
                        meta_headers.update(cf_headers)
                        req = Request(
                            f"{server}/api/push/metadata",
                            data=body,
                            headers=meta_headers,
                        )
                        r = _authed_open(req, timeout=60)
                        result = json.loads(r.read())
                        total_meta += result["updated_count"]
                        self._log_signal.emit(
                            f"  {target}: {result['updated_count']} updated, "
                            f"{result['skipped_count']} skipped"
                        )

                # Push decisions
                self._progress_signal.emit(total_enc + 1, total_enc + 2)
                self._progress_detail_signal.emit("Pushing decisions...")
                self._log_signal.emit("Pushing decisions...")
                master_csv = archive / "reports" / "past_matches_master.csv"
                if master_csv.exists():
                    with master_csv.open("r", newline="", encoding="utf-8-sig") as f:
                        reader = csv.DictReader(f)
                        all_dec = list(reader)
                    decisions = [{
                        "query_id": d.get("query_id", ""),
                        "gallery_id": d.get("gallery_id", ""),
                        "decision": d.get("verdict", d.get("decision", "")),
                        "timestamp": d.get("updated_utc", d.get("timestamp", "")),
                        "lab_id": lab,
                        "user": d.get("user", ""),
                        "notes": d.get("notes", ""),
                    } for d in all_dec]

                    if decisions:
                        body = json.dumps({
                            "lab_id": lab, "decisions": decisions,
                        }).encode()
                        dec_headers = {"Content-Type": "application/json"}
                        dec_headers.update(cf_headers)
                        req = Request(
                            f"{server}/api/push/decisions",
                            data=body,
                            headers=dec_headers,
                        )
                        r = _authed_open(req, timeout=60)
                        result = json.loads(r.read())
                        total_decisions = result["appended_count"]
                        self._log_signal.emit(
                            f"  {result['appended_count']} new, "
                            f"{result['duplicate_count']} dupes"
                        )

                # Update config
                cfg_path = archive / "starboard_sync_config.json"
                cfg = {}
                if cfg_path.exists():
                    with cfg_path.open("r") as f:
                        cfg = json.load(f)
                cfg["last_push_utc"] = datetime.now(timezone.utc).isoformat()
                with cfg_path.open("w") as f:
                    json.dump(cfg, f, indent=2)

                self._log_signal.emit(
                    f"\nPush complete: {total_new} new images, "
                    f"{total_meta} metadata updates, "
                    f"{total_decisions} new decisions"
                )
                self._done_signal.emit("push")

            except Exception as e:
                log.error("[sync] Push error: %s", e, exc_info=True)
                self._log_signal.emit(f"Push error: {e}")
                self._done_signal.emit("push")

        threading.Thread(target=worker, daemon=True).start()

    # ── Pull ────────────────────────────────────────────────────────────

    def _do_pull(self):
        self._ilog.log("button_click", "btn_sync_pull", value="clicked")
        server = self._get_server()
        if not server:
            return

        # Build filter from widgets
        pull_filter = {"include_metadata": True, "include_decisions": True}

        gallery_ids = self._pull_gallery_select.selected_items()
        if gallery_ids:
            pull_filter["gallery_ids"] = gallery_ids

        query_ids = self._pull_query_select.selected_items()
        if query_ids:
            pull_filter["query_ids"] = query_ids

        locations = self._pull_location_select.selected_items()
        if locations:
            pull_filter["locations"] = locations

        if self._chk_date_after.isChecked():
            pull_filter["date_after"] = self._pull_date_after.date().toString("yyyy-MM-dd")

        if self._chk_date_before.isChecked():
            pull_filter["date_before"] = self._pull_date_before.date().toString("yyyy-MM-dd")

        if len(pull_filter) <= 2:
            QMessageBox.warning(self, "No Filters",
                                "Please specify at least one filter, or use 'Pull Everything'.")
            return

        self._run_pull(server, pull_filter)

    def _do_pull_all(self):
        self._ilog.log("button_click", "btn_sync_pull_all", value="clicked")
        server = self._get_server()
        if not server:
            return

        reply = QMessageBox.question(
            self, "Pull Everything",
            "This will download the entire central archive. "
            "This may be very large. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._run_pull(server, {"include_metadata": True, "include_decisions": True})

    def _run_pull(self, server: str, pull_filter: dict):
        self._btn_pull.setEnabled(False)
        self._btn_pull_all.setEnabled(False)
        self._show_progress()
        self._progress.setRange(0, 0)
        self._append_log("Creating pull package...")

        def worker():
            try:
                import io, tarfile, csv as csv_mod
                from urllib.request import Request, urlopen
                from urllib.error import HTTPError
                from src.data.archive_paths import archive_root

                cf_headers = self._cf_auth_headers()

                def _authed_open(req_or_url, timeout=300):
                    try:
                        return urlopen(req_or_url, timeout=timeout)
                    except HTTPError as e:
                        if e.code == 403:
                            self._log_signal.emit("Authentication required — opening browser...")
                            token = self._run_cf_auth()
                            if token:
                                cf_headers["cf-access-token"] = token
                                cf_headers["Cookie"] = f"CF_Authorization={token}"
                                if isinstance(req_or_url, Request):
                                    req_or_url.add_header("cf-access-token", token)
                                    req_or_url.add_header("Cookie", f"CF_Authorization={token}")
                                return urlopen(req_or_url, timeout=timeout)
                        raise

                archive = archive_root()
                IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

                # Scan local images to exclude from pull
                self._log_signal.emit("Scanning local images to skip duplicates...")
                local_paths = set()
                for target_dir in ["gallery", "queries", "querries"]:
                    target_path = archive / target_dir
                    if not target_path.exists():
                        continue
                    for img in target_path.rglob("*"):
                        if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                            local_paths.add(str(img.relative_to(archive)))
                if local_paths:
                    pull_filter["exclude_paths"] = list(local_paths)
                    self._log_signal.emit(f"  {len(local_paths)} local images will be skipped")

                # Create package
                body = json.dumps(pull_filter).encode()
                pkg_headers = {"Content-Type": "application/json"}
                pkg_headers.update(cf_headers)
                req = Request(
                    f"{server}/api/pull/package",
                    data=body,
                    headers=pkg_headers,
                )
                r = _authed_open(req, timeout=30)
                pkg = json.loads(r.read())

                self._log_signal.emit(
                    f"Package: {pkg['file_count']} files, "
                    f"{pkg['total_bytes'] / (1024**2):.1f} MB, "
                    f"{pkg['entities']['gallery']} gallery, "
                    f"{pkg['entities']['queries']} queries"
                )

                if pkg["file_count"] == 0:
                    self._log_signal.emit("Nothing to pull.")
                    self._done_signal.emit("pull")
                    return

                # Download with progress
                import time as _time

                # Waiting messages while server builds the tar.gz
                _waiting_msgs = [
                    "Wrangling sea stars into a tarball...",
                    "Compressing tentacles... er, arms...",
                    "Packing images like sardines...",
                    "Server is doing star math...",
                    "Herding sunflower stars into a package...",
                    "Convincing 24 arms to hold still...",
                    "Rolling up the archive, one arm at a time...",
                    "Counting spines... this takes a moment...",
                    "Zipping up the deep blue archive...",
                    "Asking Pycnopodia to pose for compression...",
                ]

                total_bytes = pkg["total_bytes"]
                self._log_signal.emit(f"Downloading {total_bytes / (1024**2):.1f} MB...")
                self._progress_signal.emit(0, 0)  # indeterminate while waiting

                dl_req = Request(f"{server}/api/pull/stream/{pkg['package_id']}", headers=cf_headers)

                # Show rotating messages while waiting for server response
                import random
                random.shuffle(_waiting_msgs)
                _waiting = True
                _msg_idx = [0]

                def _rotate_messages():
                    while _waiting:
                        self._progress_detail_signal.emit(_waiting_msgs[_msg_idx[0] % len(_waiting_msgs)])
                        _msg_idx[0] += 1
                        _time.sleep(3)

                msg_thread = threading.Thread(target=_rotate_messages, daemon=True)
                msg_thread.start()

                r = _authed_open(dl_req, timeout=600)
                _waiting = False  # stop rotating messages

                # Use Content-Length for accurate progress (compressed size)
                content_length = r.headers.get("Content-Length")
                dl_total = int(content_length) if content_length else total_bytes

                chunks = []
                downloaded = 0
                dl_start = _time.time()
                self._progress_signal.emit(0, max(dl_total, 1))  # switch to determinate
                while True:
                    chunk = r.read(256 * 1024)  # 256KB chunks
                    if not chunk:
                        break
                    chunks.append(chunk)
                    downloaded += len(chunk)
                    elapsed = _time.time() - dl_start
                    speed = downloaded / elapsed if elapsed > 0 else 0
                    remaining_bytes = max(dl_total - downloaded, 0)
                    eta_secs = remaining_bytes / speed if speed > 0 else 0
                    mins, secs = divmod(int(eta_secs), 60)
                    pct = int(100 * downloaded / dl_total) if dl_total > 0 else 0

                    self._progress_signal.emit(downloaded, max(dl_total, 1))
                    self._progress_detail_signal.emit(
                        f"{pct}%  |  "
                        f"{downloaded / (1024**2):.0f} / {dl_total / (1024**2):.0f} MB  |  "
                        f"{speed / (1024**2):.1f} MB/s  |  "
                        f"ETA: {mins}m {secs}s"
                    )

                tar_bytes = b"".join(chunks)
                elapsed = _time.time() - dl_start
                self._log_signal.emit(
                    f"Downloaded {len(tar_bytes) / (1024**2):.1f} MB in {elapsed:.0f}s "
                    f"({len(tar_bytes) / elapsed / (1024**2):.1f} MB/s)"
                )

                # Extract
                self._log_signal.emit("Extracting...")
                extracted = 0
                with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
                    for member in tar.getmembers():
                        if member.name.startswith("_sync_metadata/"):
                            continue
                        # Merge embedding files instead of overwriting
                        if member.name.startswith("_dl_precompute/") and member.isfile():
                            with tar.extractfile(member) as src:
                                from src.sync.client import _merge_pulled_embedding_file
                                _merge_pulled_embedding_file(archive, member.name, src.read())
                            continue
                        if member.isfile():
                            dest = archive / member.name
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            with tar.extractfile(member) as src:
                                dest.write_bytes(src.read())
                            extracted += 1

                # Update config
                cfg_path = archive / "starboard_sync_config.json"
                cfg = {}
                if cfg_path.exists():
                    with cfg_path.open("r") as f:
                        cfg = json.load(f)
                cfg["last_pull_utc"] = datetime.now(timezone.utc).isoformat()
                with cfg_path.open("w") as f:
                    json.dump(cfg, f, indent=2)

                self._log_signal.emit(
                    f"\nPull complete: {extracted} image files extracted"
                )
                self._done_signal.emit("pull")

            except Exception as e:
                log.error("[sync] Pull error: %s", e, exc_info=True)
                self._log_signal.emit(f"Pull error: {e}")
                self._done_signal.emit("pull")

        threading.Thread(target=worker, daemon=True).start()

    # ── Catalog ─────────────────────────────────────────────────────────

    def _auto_refresh_catalog(self):
        """Auto-refresh on startup — silent if no server configured."""
        server = self._server_input.text().strip().rstrip("/")
        if not server:
            return
        self._refresh_catalog_impl(server)

    def _refresh_catalog(self):
        self._ilog.log("button_click", "btn_sync_refresh_catalog", value="clicked")
        server = self._get_server()
        if not server:
            return
        self._refresh_catalog_impl(server)

    def _refresh_catalog_impl(self, server: str):
        self._btn_refresh_catalog.setEnabled(False)
        self._append_log("Fetching catalog...")

        def worker():
            try:
                r = self._cf_urlopen(f"{server}/api/catalog", timeout=30)
                data = json.loads(r.read())
                self._catalog_signal.emit(data)
                self._log_signal.emit(
                    f"Catalog: {data['totals']['gallery_ids']} gallery, "
                    f"{data['totals']['query_ids']} queries"
                )
            except Exception as e:
                self._log_signal.emit(f"Catalog error: {e}")
            finally:
                # Re-enable button from main thread
                QTimer.singleShot(0, lambda: self._btn_refresh_catalog.setEnabled(True))

        threading.Thread(target=worker, daemon=True).start()

    def _populate_catalog(self, data: dict):
        table = self._catalog_table

        gallery = data.get("gallery", [])
        queries = data.get("queries", [])
        total = len(gallery) + len(queries)

        table.setRowCount(total)
        row_idx = 0

        for g in gallery:
            dates = ""
            if g.get("date_range_start"):
                dates = f"{g['date_range_start']} — {g['date_range_end']}"
            table.setItem(row_idx, 0, QTableWidgetItem("Gallery"))
            table.setItem(row_idx, 1, QTableWidgetItem(g["id"]))
            table.setItem(row_idx, 2, QTableWidgetItem(str(g["encounter_count"])))
            table.setItem(row_idx, 3, QTableWidgetItem(str(g["image_count"])))
            table.setItem(row_idx, 4, QTableWidgetItem(dates))
            row_idx += 1

        for q in queries:
            table.setItem(row_idx, 0, QTableWidgetItem("Query"))
            table.setItem(row_idx, 1, QTableWidgetItem(q["id"]))
            table.setItem(row_idx, 2, QTableWidgetItem(str(q["encounter_count"])))
            table.setItem(row_idx, 3, QTableWidgetItem(str(q["image_count"])))
            table.setItem(row_idx, 4, QTableWidgetItem(q.get("date", "")))
            row_idx += 1

        self._lbl_catalog_summary.setText(
            f"{len(gallery)} gallery, {len(queries)} queries, "
            f"{data['totals']['images']} images"
        )

        # Populate the pull filter multi-select widgets
        self._pull_gallery_select.set_items([g["id"] for g in gallery])
        self._pull_query_select.set_items([q["id"] for q in queries])
        self._pull_location_select.set_items(data.get("all_locations", []))

    def _on_catalog_double_click(self, index):
        """Double-click a catalog row to populate the pull filter."""
        self._ilog.log("button_click", "catalog_table_dblclick",
                       value=str(index.row()))
        row = index.row()
        type_item = self._catalog_table.item(row, 0)
        id_item = self._catalog_table.item(row, 1)
        if not type_item or not id_item:
            return
        entity_type = type_item.text()
        entity_id = id_item.text()

        if entity_type == "Gallery":
            self._pull_gallery_select.add_and_select(entity_id)
        else:
            self._pull_query_select.add_and_select(entity_id)

        self._append_log(f"Added {entity_type} '{entity_id}' to pull filter")

    # ── UI helpers ──────────────────────────────────────────────────────

    def _append_log(self, msg: str, level: str = "info"):
        """Log to UI text area, Python logger, and interaction logger."""
        ts = datetime.now().strftime('%H:%M:%S')
        self._log_output.append(f"[{ts}] {msg}")

        # Python logger (goes to starboard.log)
        if level == "error":
            log.error("[sync] %s", msg)
        elif level == "warning":
            log.warning("[sync] %s", msg)
        else:
            log.info("[sync] %s", msg)

    def _update_progress(self, current: int, total: int):
        self._progress.setRange(0, total)
        self._progress.setValue(current)

    def _update_progress_detail(self, text: str):
        self._lbl_progress_detail.setText(text)

    def _show_progress(self):
        self._progress.setVisible(True)
        self._lbl_progress_detail.setVisible(True)
        self._lbl_progress_detail.setText("")

    def _hide_progress(self):
        self._progress.setVisible(False)
        self._lbl_progress_detail.setVisible(False)
        self._lbl_progress_detail.setText("")

    def _on_done(self, action: str):
        self._hide_progress()
        self._btn_push.setEnabled(True)
        self._btn_pull.setEnabled(True)
        self._btn_pull_all.setEnabled(True)
        self._ilog.log("button_click", f"sync_{action}_completed",
                       value="done", context={"action": action})
        self._load_config()  # refresh timestamps
