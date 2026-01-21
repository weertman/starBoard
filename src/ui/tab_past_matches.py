# src/ui/tab_past_matches.py
from __future__ import annotations
import platform, subprocess, os, csv
from pathlib import Path
from typing import List, Tuple

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QGridLayout,
    QMessageBox, QFileDialog, QComboBox
)
from PySide6.QtCore import Qt

from src.data.past_matches import (
    build_past_matches_dataset,
    export_past_matches_master_csv,
    export_past_matches_summaries_csv,
    PastMatchesDataset,
)
from src.data.archive_paths import archive_root
from src.ui.vis_past_matches import (
    TotalsDialog, TimelineDialog, ByQueryDialog, ByGalleryDialog, QueryInflowDialog,
    OutingStatsDialog, QueryGridDialog
)
from src.ui.vis_interaction_logs import (
    SessionProductivityDialog, FeatureUsageDialog, WorkEstimationDialog
)
from src.ui.matrix_matches_dialog import MatrixMatchesDialog
from src.data.matches_matrix import MatchMatrixData, load_match_matrix

# NEW: merge helpers
from src.data.merge_yes import (
    list_mergeable_queries_for_gallery,
    list_galleries_with_merge_candidates,
    list_galleries_with_history,
    list_batches_for_gallery,
    merge_yeses_for_gallery,
    revert_merge_batch_for_gallery,
    revert_last_merge_for_gallery,           # kept for convenience
    read_merge_history_for_gallery,
    SILENT_MARKER_FILENAME,
)
from src.utils.interaction_logger import get_interaction_logger

def _open_folder(path: Path) -> None:
    try:
        if platform.system() == "Windows":
            os.startfile(str(path))
        elif platform.system() == "Darwin":
            subprocess.call(["open", str(path)])
        else:
            subprocess.call(["xdg-open", str(path)])
    except Exception:
        pass

class TabPastMatches(QWidget):
    """
    Analytics & History tab with:
      - Refresh, Export master/summaries, open reports
      - Visualization dialogs: Totals, Timeline, By Query, By Gallery
      - Matrix dialog (Queries × Gallery)
      - Export tidy CSV of decided pairs
      - NEW: Merge YES’s to Gallery (with filtered gallery list + reversible history)
      - NEW: Revert old merges (pick gallery & batch)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ds: PastMatchesDataset | None = None
        self._dialogs: List[QWidget] = []
        self._ilog = get_interaction_logger()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ---- Row 1: Utility bar ----
        util = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh dataset")
        self.btn_export_master = QPushButton("Export master CSV")
        self.btn_export_summaries = QPushButton("Export summaries CSVs")
        self.btn_open_reports = QPushButton("Open reports folder")
        util.addWidget(self.btn_refresh); util.addSpacing(6)
        util.addWidget(self.btn_export_master); util.addWidget(self.btn_export_summaries)
        util.addStretch(1); util.addWidget(self.btn_open_reports)
        outer.addLayout(util)

        # ---- Row 2: Visualization group ----
        gb_vis = QGroupBox("Visualizations")
        lay_vis = QGridLayout(gb_vis)
        self.btn_totals = QPushButton("Totals…")
        self.btn_timeline = QPushButton("Timeline…")
        self.btn_by_query = QPushButton("By Query…")
        self.btn_by_gallery = QPushButton("By Gallery…")
        self.btn_matrix = QPushButton("Matches Matrix…")
        self.btn_query_inflow = QPushButton("Query Inflow…")
        self.btn_query_inflow.setToolTip("Stacked bar chart of matched vs unmatched queries by first observation date")
        lay_vis.addWidget(self.btn_totals, 0, 0)
        lay_vis.addWidget(self.btn_timeline, 0, 1)
        lay_vis.addWidget(self.btn_by_query, 1, 0)
        lay_vis.addWidget(self.btn_by_gallery, 1, 1)
        lay_vis.addWidget(self.btn_matrix, 0, 2)
        lay_vis.addWidget(self.btn_query_inflow, 1, 2)
        
        # Outing Stats (observations per outing)
        self.btn_outing_stats = QPushButton("Outing Stats…")
        self.btn_outing_stats.setToolTip("Stacked bar chart of observations per outing by identification status")
        lay_vis.addWidget(self.btn_outing_stats, 2, 0)
        
        # Query Grid (best images grid)
        self.btn_query_grid = QPushButton("Query Grid…")
        self.btn_query_grid.setToolTip("Grid of best images for queries organized by outing")
        lay_vis.addWidget(self.btn_query_grid, 2, 1)
        
        # Interaction Analytics (new)
        lay_vis.addWidget(QLabel(""), 0, 3)  # Spacer
        self.btn_session_productivity = QPushButton("Session Activity…")
        self.btn_feature_usage = QPushButton("Feature Usage…")
        self.btn_work_estimation = QPushButton("Work Estimation…")
        self.btn_session_productivity.setToolTip("View activity timeline across sessions")
        self.btn_feature_usage.setToolTip("View hierarchical feature usage patterns")
        self.btn_work_estimation.setToolTip("Estimate remaining work to match all queries")
        lay_vis.addWidget(self.btn_session_productivity, 0, 4)
        lay_vis.addWidget(self.btn_feature_usage, 1, 4)
        lay_vis.addWidget(self.btn_work_estimation, 0, 5)
        
        outer.addWidget(gb_vis)

        # ---- Row 3: NEW — Merge YES's to Gallery ----
        gb_merge = QGroupBox("Merge YES’s to Gallery")
        lay_merge = QHBoxLayout(gb_merge)
        lay_merge.addWidget(QLabel("Gallery:"))
        self.cmb_gallery = QComboBox()
        self.cmb_gallery.setMinimumWidth(260)
        lay_merge.addWidget(self.cmb_gallery)

        self.lbl_count = QLabel("—")
        lay_merge.addWidget(self.lbl_count)
        lay_merge.addStretch(1)

        self.btn_merge = QPushButton("Merge YES's")
        lay_merge.addWidget(self.btn_merge)

        self.btn_merge_all = QPushButton("Merge All")
        self.btn_merge_all.setToolTip("Merge YES queries for all galleries at once")
        lay_merge.addWidget(self.btn_merge_all)

        outer.addWidget(gb_merge)

        # ---- Row 4: NEW — Revert old merges ----
        gb_revert = QGroupBox("Revert old merges")
        lay_rev = QHBoxLayout(gb_revert)

        lay_rev.addWidget(QLabel("Gallery:"))
        self.cmb_gallery_rev = QComboBox()
        self.cmb_gallery_rev.setMinimumWidth(260)
        lay_rev.addWidget(self.cmb_gallery_rev)

        lay_rev.addSpacing(12)
        lay_rev.addWidget(QLabel("Batch:"))
        self.cmb_batch = QComboBox()
        self.cmb_batch.setMinimumWidth(360)
        lay_rev.addWidget(self.cmb_batch)

        lay_rev.addStretch(1)
        self.btn_revert_sel = QPushButton("Revert selected batch")
        self.btn_open_hist = QPushButton("Open history CSV")
        lay_rev.addWidget(self.btn_revert_sel)
        lay_rev.addWidget(self.btn_open_hist)

        outer.addWidget(gb_revert)

        # ---- Row 5: Tidy CSV for decisions ----
        gb_tidy = QGroupBox("Export decided pairs (tidy CSV)")
        lay_tidy = QHBoxLayout(gb_tidy)
        self.btn_export_tidy = QPushButton("Export CSV…")
        lay_tidy.addWidget(QLabel(
            "One row per (Query, Gallery) with verdict, update time, notes, and query last observed date."
        ))
        lay_tidy.addStretch(1)
        lay_tidy.addWidget(self.btn_export_tidy)
        outer.addWidget(gb_tidy)

        # ---- Signals ----
        self.btn_refresh.clicked.connect(self._reload)
        self.btn_export_master.clicked.connect(self._export_master)
        self.btn_export_summaries.clicked.connect(self._export_summaries)
        self.btn_open_reports.clicked.connect(self._open_reports)

        self.btn_totals.clicked.connect(lambda: self._open_dialog(TotalsDialog))
        self.btn_timeline.clicked.connect(lambda: self._open_dialog(TimelineDialog))
        self.btn_by_query.clicked.connect(lambda: self._open_dialog(ByQueryDialog))
        self.btn_by_gallery.clicked.connect(lambda: self._open_dialog(ByGalleryDialog))
        self.btn_matrix.clicked.connect(self._open_matrix_dialog)
        self.btn_query_inflow.clicked.connect(lambda: self._open_dialog(QueryInflowDialog))
        self.btn_outing_stats.clicked.connect(self._open_outing_stats_dialog)
        self.btn_query_grid.clicked.connect(self._open_query_grid_dialog)
        
        # Interaction analytics dialogs
        self.btn_session_productivity.clicked.connect(self._open_session_productivity_dialog)
        self.btn_feature_usage.clicked.connect(self._open_feature_usage_dialog)
        self.btn_work_estimation.clicked.connect(self._open_work_estimation_dialog)

        self.btn_export_tidy.clicked.connect(self._export_decisions_csv)

        # Merge signals
        self.cmb_gallery.currentIndexChanged.connect(self._update_merge_preview)
        self.btn_merge.clicked.connect(self._on_merge)
        self.btn_merge_all.clicked.connect(self._on_merge_all)

        # Revert signals
        self.cmb_gallery_rev.currentIndexChanged.connect(self._on_revert_gallery_changed)
        self.btn_revert_sel.clicked.connect(self._on_revert_selected_batch)
        self.btn_open_hist.clicked.connect(self._on_open_history_rev)

        # Initial load
        self._reload()
        self._refresh_merge_gallery_list()
        self._refresh_revert_gallery_list()

    # ----------------- existing actions -----------------
    def _reload(self):
        self._ilog.log("button_click", "btn_refresh_dataset", value="clicked")
        self._ds = build_past_matches_dataset()

        # Keep the merge/revert lists in sync with current state
        self._refresh_merge_gallery_list()
        self._refresh_revert_gallery_list()

    def _export_master(self):
        self._ilog.log("button_click", "btn_export_master", value="clicked")
        ds = self._ds or build_past_matches_dataset()
        path = export_past_matches_master_csv(ds)
        QMessageBox.information(self, "starBoard", f"Master CSV exported:\n{path}")

    def _export_summaries(self):
        self._ilog.log("button_click", "btn_export_summaries", value="clicked")
        ds = self._ds or build_past_matches_dataset()
        p_q, p_g, p_t = export_past_matches_summaries_csv(ds)
        QMessageBox.information(
            self, "starBoard",
            "Summary CSVs exported:\n"
            f"By Query: {p_q}\nBy Gallery: {p_g}\nTimeline: {p_t}"
        )

    def _open_reports(self):
        self._ilog.log("button_click", "btn_open_reports", value="clicked")
        p = archive_root() / "reports"
        _open_folder(p)

    def _open_dialog(self, cls):
        self._ilog.log("dialog_open", f"dialog_{cls.__name__}", value="opened")
        dlg = cls(self._ds or build_past_matches_dataset(), self)
        dlg.finished.connect(lambda _: self._on_dialog_closed(dlg))
        self._dialogs.append(dlg)
        dlg.show()

    def _open_matrix_dialog(self):
        self._ilog.log("dialog_open", "dialog_matrix", value="opened")
        dlg = MatrixMatchesDialog(self)
        dlg.finished.connect(lambda _: self._on_dialog_closed(dlg))
        self._dialogs.append(dlg)
        dlg.show()

    def _open_session_productivity_dialog(self):
        self._ilog.log("dialog_open", "dialog_session_productivity", value="opened")
        dlg = SessionProductivityDialog(self)
        dlg.finished.connect(lambda _: self._on_dialog_closed(dlg))
        self._dialogs.append(dlg)
        dlg.show()

    def _open_feature_usage_dialog(self):
        self._ilog.log("dialog_open", "dialog_feature_usage", value="opened")
        dlg = FeatureUsageDialog(self)
        dlg.finished.connect(lambda _: self._on_dialog_closed(dlg))
        self._dialogs.append(dlg)
        dlg.show()

    def _open_work_estimation_dialog(self):
        self._ilog.log("dialog_open", "dialog_work_estimation", value="opened")
        dlg = WorkEstimationDialog(self)
        dlg.finished.connect(lambda _: self._on_dialog_closed(dlg))
        self._dialogs.append(dlg)
        dlg.show()

    def _open_outing_stats_dialog(self):
        self._ilog.log("dialog_open", "dialog_outing_stats", value="opened")
        dlg = OutingStatsDialog(self)
        dlg.finished.connect(lambda _: self._on_dialog_closed(dlg))
        self._dialogs.append(dlg)
        dlg.show()

    def _open_query_grid_dialog(self):
        self._ilog.log("dialog_open", "dialog_query_grid", value="opened")
        dlg = QueryGridDialog(self)
        dlg.finished.connect(lambda _: self._on_dialog_closed(dlg))
        self._dialogs.append(dlg)
        dlg.show()

    def _on_dialog_closed(self, dlg: QWidget):
        try:
            self._dialogs.remove(dlg)
        except ValueError:
            pass

    # ---- Export decided pairs tidy CSV (kept) ----
    def _export_decisions_csv(self):
        self._ilog.log("button_click", "btn_export_decisions", value="clicked")
        data = load_match_matrix()
        logs = archive_root() / "reports"; logs.mkdir(parents=True, exist_ok=True)
        default = logs / "past_matches_decisions.csv"
        path_str, _ = QFileDialog.getSaveFileName(self, "Export decisions (tidy CSV)", str(default), "CSV files (*.csv)")
        if not path_str:
            return
        try:
            self._write_decisions_csv(path_str, data)
            QMessageBox.information(self, "starBoard", f"Exported to:\n{path_str}")
        except Exception as e:
            QMessageBox.warning(self, "starBoard", f"Export failed:\n{e}")

    @staticmethod
    def _write_decisions_csv(path_str: str, data: MatchMatrixData) -> None:
        from datetime import date
        def _fmt_date(d: date | None) -> str: return d.isoformat() if d else ""
        with open(path_str, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["query_id","gallery_id","verdict","updated_utc","notes","query_last_observed"])
            for (qid, gid), verdict in sorted(data.verdict_by_pair.items()):
                w.writerow([qid, gid, verdict,
                            data.updated_by_pair.get((qid, gid), ""),
                            data.notes_by_pair.get((qid, gid), ""),
                            _fmt_date(data.last_obs_by_query.get(qid))])

    # ----------------- NEW: Merge panel logic -----------------
    def _refresh_merge_gallery_list(self):
        prev = self.cmb_gallery.currentText()
        gids = list_galleries_with_merge_candidates(require_encounters=True)
        self.cmb_gallery.blockSignals(True)
        self.cmb_gallery.clear()
        self.cmb_gallery.addItems(gids)
        self.cmb_gallery.blockSignals(False)
        if prev and (idx := self.cmb_gallery.findText(prev)) >= 0:
            self.cmb_gallery.setCurrentIndex(idx)
        self._update_merge_preview()
        # Enable Merge All only if there are galleries with merge candidates
        self.btn_merge_all.setEnabled(len(gids) > 0)

    def _current_gallery(self) -> str:
        return self.cmb_gallery.currentText() or ""

    def _update_merge_preview(self, _index: int = -1) -> None:
        gid = self._current_gallery()
        if gid:
            self._ilog.log("combo_change", "cmb_gallery_merge", value=gid)
        if not gid:
            self.lbl_count.setText("—")
            self.btn_merge.setEnabled(False)
            return
        qids = list_mergeable_queries_for_gallery(gid, require_encounters=True)
        self.lbl_count.setText(f"YES queries for {gid}: <b>{len(qids)}</b>")
        self.btn_merge.setEnabled(len(qids) > 0)

    def _on_merge(self, checked: bool = False) -> None:
        gid = self._current_gallery()
        if not gid:
            return
        self._ilog.log("button_click", "btn_merge", value=gid)
        qids = list_mergeable_queries_for_gallery(gid, require_encounters=True)
        if not qids:
            QMessageBox.information(self, "starBoard", f"No merge-able YES queries for gallery '{gid}'.")
            return

        # Dry-run to show counts
        dry = merge_yeses_for_gallery(gid, dry_run=True)
        n_q = dry.num_queries
        n_dirs = dry.num_encounter_dirs
        hist_path = (archive_root() / "gallery" / gid / "_merge_history.csv")
        marker = SILENT_MARKER_FILENAME

        reply = QMessageBox.question(
            self, "Merge YES’s",
            "This will:\n"
            f" • Copy <b>{n_dirs}</b> encounter folder(s) from <b>{n_q}</b> YES query/queries into <b>{gid}</b>\n"
            f" • Mark each query as <b>silent</b> by writing <code>{marker}</code> in its folder\n"
            f" • Append to history CSV:\n    <code>{hist_path}</code>\n\n"
            "Proceed?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        rep = merge_yeses_for_gallery(gid, dry_run=False)
        if rep.errors:
            QMessageBox.warning(
                self,
                "starBoard",
                f"Merged with errors.\nCreated: {rep.num_encounter_dirs} encounter folder(s)\n"
                f"Errors:\n - " + "\n - ".join(rep.errors)
            )
        else:
            QMessageBox.information(
                self,
                "starBoard",
                f"Merge complete.\nCreated: {rep.num_encounter_dirs} encounter folder(s)\n"
                f"Batch: {rep.batch_id}"
            )
        # Refresh both panels (lists change after merge)
        self._refresh_merge_gallery_list()
        self._refresh_revert_gallery_list()

    def _on_merge_all(self, checked: bool = False) -> None:
        """Merge YES queries for all galleries at once."""
        self._ilog.log("button_click", "btn_merge_all", value="clicked")

        gids = list_galleries_with_merge_candidates(require_encounters=True)
        if not gids:
            QMessageBox.information(self, "starBoard", "No galleries have merge-able YES queries.")
            return

        # Dry-run for all galleries to get totals
        total_queries = 0
        total_dirs = 0
        for gid in gids:
            dry = merge_yeses_for_gallery(gid, dry_run=True)
            total_queries += dry.num_queries
            total_dirs += dry.num_encounter_dirs

        marker = SILENT_MARKER_FILENAME
        reply = QMessageBox.question(
            self, "Merge All YES's",
            f"This will merge YES queries across <b>{len(gids)}</b> gallery/galleries:\n\n"
            f" • Copy <b>{total_dirs}</b> encounter folder(s) from <b>{total_queries}</b> YES query/queries\n"
            f" • Mark each query as <b>silent</b> by writing <code>{marker}</code> in its folder\n"
            f" • Append to each gallery's history CSV\n\n"
            "Proceed?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Execute merges
        merged_galleries = 0
        merged_queries = 0
        merged_dirs = 0
        all_errors: list[str] = []

        for gid in gids:
            rep = merge_yeses_for_gallery(gid, dry_run=False)
            merged_galleries += 1
            merged_queries += rep.num_queries
            merged_dirs += rep.num_encounter_dirs
            if rep.errors:
                all_errors.extend([f"{gid}: {e}" for e in rep.errors])

        if all_errors:
            QMessageBox.warning(
                self,
                "starBoard",
                f"Merged with errors.\n\n"
                f"Galleries: {merged_galleries}\n"
                f"Queries: {merged_queries}\n"
                f"Encounter folders: {merged_dirs}\n\n"
                f"Errors ({len(all_errors)}):\n - " + "\n - ".join(all_errors[:10])
                + ("\n..." if len(all_errors) > 10 else "")
            )
        else:
            QMessageBox.information(
                self,
                "starBoard",
                f"Merge complete.\n\n"
                f"Galleries: {merged_galleries}\n"
                f"Queries: {merged_queries}\n"
                f"Encounter folders: {merged_dirs}"
            )

        # Refresh both panels
        self._refresh_merge_gallery_list()
        self._refresh_revert_gallery_list()

    # ----------------- NEW: Revert panel logic -----------------
    def _refresh_revert_gallery_list(self):
        prev = self.cmb_gallery_rev.currentText()
        gids = list_galleries_with_history()
        self.cmb_gallery_rev.blockSignals(True)
        self.cmb_gallery_rev.clear()
        self.cmb_gallery_rev.addItems(gids)
        self.cmb_gallery_rev.blockSignals(False)
        if prev and (idx := self.cmb_gallery_rev.findText(prev)) >= 0:
            self.cmb_gallery_rev.setCurrentIndex(idx)
        self._refresh_batch_list()

    def _on_revert_gallery_changed(self, _index: int = -1) -> None:
        gid = self.cmb_gallery_rev.currentText()
        self._ilog.log("combo_change", "cmb_gallery_revert", value=gid)
        self._refresh_batch_list()

    def _refresh_batch_list(self):
        gid = self.cmb_gallery_rev.currentText()
        self.cmb_batch.clear()
        if not gid:
            self.btn_revert_sel.setEnabled(False)
            return
        batches = list_batches_for_gallery(gid)
        # Human-friendly labels
        for b in batches:
            ts = b.get("timestamp_utc", "") or ""
            num_dirs = b.get("num_dirs", "0")
            num_q = b.get("num_queries", "0")
            bid = b.get("batch_id", "")
            label = f"{ts} — {num_dirs} folder(s) from {num_q} query/queries — {bid}"
            self.cmb_batch.addItem(label, bid)
        self.btn_revert_sel.setEnabled(self.cmb_batch.count() > 0)

    def _on_revert_selected_batch(self, checked: bool = False) -> None:
        gid = self.cmb_gallery_rev.currentText()
        if not gid or self.cmb_batch.count() == 0:
            return

        bid = self.cmb_batch.currentData()
        self._ilog.log("button_click", "btn_revert_batch", value=bid,
                      context={"gallery_id": gid})
        rep = revert_merge_batch_for_gallery(gid, bid)
        if not rep.batch_id:
            QMessageBox.information(self, "starBoard", "Nothing to revert for this gallery.")
            return

        if rep.errors:
            QMessageBox.warning(
                self,
                "starBoard",
                f"Reverted batch {rep.batch_id} with errors.\n"
                f"Removed encounter dirs: {rep.num_encounter_dirs}\n"
                "Errors:\n - " + "\n - ".join(rep.errors)
            )
        else:
            QMessageBox.information(
                self,
                "starBoard",
                f"Reverted batch {rep.batch_id}.\n"
                f"Removed encounter dirs: {rep.num_encounter_dirs}\n"
                f"Unsilenced queries: {rep.num_queries}"
            )

        # Refresh both panels (lists change after revert)
        self._refresh_merge_gallery_list()
        self._refresh_revert_gallery_list()

    def _on_open_history_rev(self):
        gid = self.cmb_gallery_rev.currentText()
        self._ilog.log("button_click", "btn_open_history", value=gid)
        if not gid:
            return
        p = archive_root() / "gallery" / gid / "_merge_history.csv"
        if not p.exists():
            QMessageBox.information(self, "starBoard",
                                    "No history file yet.\n"
                                    f"Expected path:\n{p}")
            return
        _open_folder(p.parent)
