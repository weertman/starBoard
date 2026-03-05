from __future__ import annotations

from datetime import datetime, date
import csv
import os
from typing import Dict, List, Tuple, Optional, Any

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QLineEdit,
    QDateEdit,
    QSpinBox,
)
from PySide6.QtCore import Qt, QDate

from src.data.archive_paths import archive_root
from src.data.morphometric_analytics import (
    MORPHOMETRIC_METRIC_LABELS,
    MORPHOMETRIC_METRIC_FIELDS,
    MorphometricAnalyticsDataset,
    build_morphometric_analytics_dataset,
    filter_morphometric_rows,
    available_locations,
    available_identity_labels,
)
from src.ui.mpl_embed import MplWidget
from src.utils.interaction_logger import get_interaction_logger

DEFAULT_EXPORT_FIELDS: List[str] = [
    "identity_type",
    "identity_id",
    "identity_label",
    "measurement_day",
    "location",
    "morph_area_mm2",
    "morph_major_axis_mm",
    "morph_minor_axis_mm",
    "morph_mean_arm_length_mm",
    "morph_max_arm_length_mm",
    "morph_tip_to_tip_mm",
    "morph_num_arms",
    "mfolder",
    "measurement_sequence",
]


def _try_import_dataframe():
    try:
        import pandas as pd
        return pd
    except Exception:
        return None


def _rows_to_dataframe(rows: List[Dict[str, Any]]):
    pd = _try_import_dataframe()
    if pd is None:
        return None
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).copy()
    if "measurement_day" in df.columns:
        df["measurement_day"] = pd.to_datetime(df["measurement_day"], errors="coerce")
    return df


def _visible_metrics(df, preferred: List[str]) -> List[str]:
    out: List[str] = []
    if df is None or df.empty:
        return out
    for metric in preferred:
        if metric in df.columns:
            series = df[metric]
            if hasattr(series, "notna") and bool(series.notna().any()):
                out.append(metric)
    return out


def _parse_iso_day(value: str) -> Optional[date]:
    s = (value or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _date_to_iso(qd: QDate) -> str:
    return f"{qd.year():04d}-{qd.month():02d}-{qd.day():02d}"


class MorphometricDialogBase(QDialog):
    """Reusable shell for morphometric analytics dialogs with shared filters."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(1100, 780)

        self._dataset: MorphometricAnalyticsDataset = build_morphometric_analytics_dataset()
        self._ilog = get_interaction_logger()
        self._reports_dir = archive_root() / "reports" / "figures"
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._last_export_rows: List[Dict[str, Any]] = []
        self._updating_filters = False
        self._all_identity_labels: List[str] = []

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(8, 8, 8, 8)
        self._root.setSpacing(6)

        top = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_png = QPushButton("Export PNG")
        self.btn_csv = QPushButton("Export CSV")
        self.lbl_hint = QLabel("")
        self.lbl_hint.setStyleSheet("color:#666")
        top.addWidget(self.btn_refresh)
        top.addSpacing(6)
        top.addWidget(self.btn_png)
        top.addWidget(self.btn_csv)
        top.addStretch(1)
        top.addWidget(self.lbl_hint)
        self._root.addLayout(top)

        self.btn_refresh.clicked.connect(self._on_refresh)
        self.btn_png.clicked.connect(self._on_export_png)
        self.btn_csv.clicked.connect(self._on_export_csv)

        self._build_body()
        self._refresh_filter_options(preserve_selection=False)
        self.render()

    def _build_body(self):
        self._build_shared_filters()
        self._chart = MplWidget(self)
        self._root.addWidget(self._chart, 1)

    def _build_shared_filters(self):
        group = QGroupBox("Filters")
        layout = QVBoxLayout(group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Scope:"))
        self.cmb_scope = QComboBox()
        self.cmb_scope.addItems(["All", "Gallery", "Query"])
        row1.addWidget(self.cmb_scope)

        row1.addWidget(QLabel("Location:"))
        self.cmb_location = QComboBox()
        self.cmb_location.addItem("All")
        row1.addWidget(self.cmb_location)

        self.chk_date_range = QCheckBox("Date range")
        row1.addWidget(self.chk_date_range)

        self.date_start = QDateEdit()
        self.date_start.setCalendarPopup(True)
        self.date_start.setDisplayFormat("yyyy-MM-dd")
        self.date_start.setDate(QDate.currentDate())
        self.date_start.setEnabled(False)
        row1.addWidget(self.date_start)

        row1.addWidget(QLabel("to"))

        self.date_end = QDateEdit()
        self.date_end.setCalendarPopup(True)
        self.date_end.setDisplayFormat("yyyy-MM-dd")
        self.date_end.setDate(QDate.currentDate())
        self.date_end.setEnabled(False)
        row1.addWidget(self.date_end)

        row1.addStretch(1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Identities:"))
        self.edit_identity_search = QLineEdit()
        self.edit_identity_search.setPlaceholderText("Search identities...")
        row2.addWidget(self.edit_identity_search, 1)
        self.btn_identity_all = QPushButton("Select all")
        self.btn_identity_none = QPushButton("Clear")
        row2.addWidget(self.btn_identity_all)
        row2.addWidget(self.btn_identity_none)
        layout.addLayout(row2)

        self.lst_identities = QListWidget()
        self.lst_identities.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lst_identities.setMinimumHeight(90)
        self.lst_identities.setMaximumHeight(140)
        layout.addWidget(self.lst_identities)

        self._root.addWidget(group)

        self.cmb_scope.currentIndexChanged.connect(self._on_scope_changed)
        self.cmb_location.currentIndexChanged.connect(self._on_location_changed)
        self.chk_date_range.stateChanged.connect(self._on_date_range_toggled)
        self.date_start.dateChanged.connect(self._on_filter_changed)
        self.date_end.dateChanged.connect(self._on_filter_changed)
        self.edit_identity_search.textChanged.connect(self._on_identity_search_changed)
        self.btn_identity_all.clicked.connect(self._on_select_all_identities)
        self.btn_identity_none.clicked.connect(self._on_clear_identities)
        self.lst_identities.itemSelectionChanged.connect(self._on_filter_changed)

    def _set_message(self, message: str):
        self._chart.clear()
        self._chart.ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        self._chart.draw()

    def _on_scope_changed(self):
        if self._updating_filters:
            return
        self._refresh_filter_options(preserve_selection=False)
        self.render()

    def _on_location_changed(self):
        if self._updating_filters:
            return
        self._refresh_identity_options(preserve_selection=False)
        self.render()

    def _on_date_range_toggled(self):
        enabled = self.chk_date_range.isChecked()
        self.date_start.setEnabled(enabled)
        self.date_end.setEnabled(enabled)
        self._on_filter_changed()

    def _on_identity_search_changed(self):
        self._apply_identity_search()

    def _on_select_all_identities(self):
        self._updating_filters = True
        try:
            for i in range(self.lst_identities.count()):
                self.lst_identities.item(i).setSelected(True)
        finally:
            self._updating_filters = False
        self.render()

    def _on_clear_identities(self):
        self._updating_filters = True
        try:
            self.lst_identities.clearSelection()
        finally:
            self._updating_filters = False
        self.render()

    def _on_filter_changed(self):
        if self._updating_filters:
            return
        self.render()

    def _selected_identity_labels(self) -> List[str]:
        return [item.text() for item in self.lst_identities.selectedItems()]

    def _refresh_filter_options(self, *, preserve_selection: bool):
        prev_location = self.cmb_location.currentText().strip() if preserve_selection else "All"
        prev_identities = set(self._selected_identity_labels()) if preserve_selection else set()

        self._updating_filters = True
        try:
            scope = self.cmb_scope.currentText().strip().lower() if hasattr(self, "cmb_scope") else "all"

            locations = ["All"] + available_locations(self._dataset.rows, scope=scope)
            self.cmb_location.blockSignals(True)
            self.cmb_location.clear()
            self.cmb_location.addItems(locations)
            self.cmb_location.blockSignals(False)
            if prev_location in locations:
                self.cmb_location.setCurrentText(prev_location)
            else:
                self.cmb_location.setCurrentIndex(0)

            self._refresh_identity_options(
                preserve_selection=preserve_selection,
                selected_labels=prev_identities,
            )
            self._set_date_bounds()
        finally:
            self._updating_filters = False

    def _refresh_identity_options(self, *, preserve_selection: bool, selected_labels: Optional[set[str]] = None):
        scope = self.cmb_scope.currentText().strip().lower()
        loc = self.cmb_location.currentText().strip()
        location_filter = None if loc.lower() == "all" else loc
        labels = available_identity_labels(self._dataset.rows, scope=scope, location=location_filter)
        self._all_identity_labels = labels

        if selected_labels is None:
            selected_labels = set(self._selected_identity_labels()) if preserve_selection else set()

        self.lst_identities.blockSignals(True)
        self.lst_identities.clear()
        for label in labels:
            item = QListWidgetItem(label)
            self.lst_identities.addItem(item)
            if preserve_selection and label in selected_labels:
                item.setSelected(True)
            elif not preserve_selection:
                item.setSelected(True)
        self.lst_identities.blockSignals(False)
        self._apply_identity_search()

    def _apply_identity_search(self):
        query = self.edit_identity_search.text().strip().lower()
        for i in range(self.lst_identities.count()):
            item = self.lst_identities.item(i)
            visible = (not query) or (query in item.text().lower())
            item.setHidden(not visible)

    def _set_date_bounds(self):
        days = []
        scope = self.cmb_scope.currentText().strip().lower() if hasattr(self, "cmb_scope") else "all"
        loc = self.cmb_location.currentText().strip() if hasattr(self, "cmb_location") else "All"
        locations = None if loc.lower() == "all" else [loc]
        scoped_rows = filter_morphometric_rows(self._dataset.rows, scope=scope, locations=locations)

        for row in scoped_rows:
            d = _parse_iso_day(str(row.get("measurement_day") or ""))
            if d is not None:
                days.append(d)

        if not days:
            today = QDate.currentDate()
            self.date_start.setDate(today)
            self.date_end.setDate(today)
            return

        min_d = min(days)
        max_d = max(days)
        min_q = QDate(min_d.year, min_d.month, min_d.day)
        max_q = QDate(max_d.year, max_d.month, max_d.day)

        self.date_start.setMinimumDate(min_q)
        self.date_start.setMaximumDate(max_q)
        self.date_end.setMinimumDate(min_q)
        self.date_end.setMaximumDate(max_q)

        if not self.chk_date_range.isChecked():
            self.date_start.setDate(min_q)
            self.date_end.setDate(max_q)
            return

        if self.date_start.date() < min_q or self.date_start.date() > max_q:
            self.date_start.setDate(min_q)
        if self.date_end.date() < min_q or self.date_end.date() > max_q:
            self.date_end.setDate(max_q)

    def _current_filtered_rows(self) -> List[Dict[str, Any]]:
        scope = self.cmb_scope.currentText().strip().lower()
        location = self.cmb_location.currentText().strip()
        locations = None if location.lower() == "all" else [location]

        selected_labels = self._selected_identity_labels()
        if len(selected_labels) == 0 or len(selected_labels) == self.lst_identities.count():
            selected_labels = None

        start_day = None
        end_day = None
        if self.chk_date_range.isChecked():
            start_day = _date_to_iso(self.date_start.date())
            end_day = _date_to_iso(self.date_end.date())

        return filter_morphometric_rows(
            self._dataset.rows,
            scope=scope,
            identity_labels=selected_labels,
            locations=locations,
            start_day=start_day,
            end_day=end_day,
        )

    def render(self):
        self._chart.clear()
        self._chart.ax.text(0.5, 0.5, "No content", ha="center", va="center")
        self._chart.draw()

    def export_rows(self) -> Tuple[List[str], List[List[str]]]:
        if not self._last_export_rows:
            return [], []
        rows = [[str(r.get(k, "")) for k in DEFAULT_EXPORT_FIELDS] for r in self._last_export_rows]
        return DEFAULT_EXPORT_FIELDS, rows

    def _on_refresh(self):
        self._ilog.log("button_click", f"btn_refresh_{self.windowTitle().lower().replace(' ', '_')}", value="clicked")
        self._dataset = build_morphometric_analytics_dataset()
        self._refresh_filter_options(preserve_selection=False)
        self.render()

    def _on_export_png(self):
        self._ilog.log("button_click", f"btn_export_png_{self.windowTitle().lower().replace(' ', '_')}", value="clicked")
        if not getattr(self._chart, "fig", None):
            return
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        fname = f"{self.windowTitle().lower().replace(' ', '_')}_{ts}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", str(self._reports_dir / fname), "PNG (*.png)"
        )
        if not path:
            return
        try:
            self._chart.fig.savefig(path, dpi=150)
            self.lbl_hint.setText(f"Saved {os.path.basename(path)}")
        except Exception as e:
            self.lbl_hint.setText(f"Save failed: {e}")

    def _on_export_csv(self):
        self._ilog.log("button_click", f"btn_export_csv_{self.windowTitle().lower().replace(' ', '_')}", value="clicked")
        header, rows = self.export_rows()
        if not header:
            return
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        fname = f"{self.windowTitle().lower().replace(' ', '_')}_{ts}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", str(self._reports_dir / fname), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
            self.lbl_hint.setText(f"Saved {os.path.basename(path)}")
        except Exception as e:
            self.lbl_hint.setText(f"Save failed: {e}")


class MorphometricTrendsDialog(MorphometricDialogBase):
    """Notebook-inspired longitudinal size trends by measurement day."""

    _TREND_METRICS = [
        "morph_area_mm2",
        "morph_major_axis_mm",
        "morph_minor_axis_mm",
        "morph_mean_arm_length_mm",
        "morph_tip_to_tip_mm",
    ]

    def __init__(self, parent=None):
        super().__init__("Morphometric Size Trends", parent)

    def _build_body(self):
        self._build_shared_filters()

        row = QHBoxLayout()
        self.chk_lines = QCheckBox("Connect same ID")
        self.chk_lines.setChecked(True)
        row.addWidget(self.chk_lines)

        self.chk_band = QCheckBox("Show daily mean +/- std")
        self.chk_band.setChecked(True)
        row.addWidget(self.chk_band)
        row.addStretch(1)
        self._root.addLayout(row)

        self.chk_lines.stateChanged.connect(self._on_filter_changed)
        self.chk_band.stateChanged.connect(self._on_filter_changed)

        self._chart = MplWidget(self)
        self._root.addWidget(self._chart, 1)

    def render(self):
        rows = self._current_filtered_rows()
        self._last_export_rows = rows

        df = _rows_to_dataframe(rows)
        if df is None:
            self._set_message("Pandas is required for morphometric analytics views.")
            return
        if df.empty:
            self._set_message("No morphometric measurements found for current filters.")
            return

        metrics = _visible_metrics(df, self._TREND_METRICS)
        if not metrics:
            self._set_message("No numeric morphometric fields available to plot.")
            return

        try:
            import matplotlib.dates as mdates
        except Exception:
            self._set_message("Matplotlib dependencies for date plotting are unavailable.")
            return

        fig = self._chart.fig
        fig.clear()
        axes = fig.subplots(1, len(metrics), squeeze=False)[0]

        show_lines = self.chk_lines.isChecked()
        show_band = self.chk_band.isChecked()

        for ax, metric in zip(axes, metrics):
            tmp = df[["measurement_day", "identity_label", metric]].dropna(subset=["measurement_day", metric]).copy()
            if tmp.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(MORPHOMETRIC_METRIC_LABELS.get(metric, metric))
                continue

            tmp = tmp.sort_values("measurement_day")
            if show_lines:
                for _, grp in tmp.groupby("identity_label"):
                    if len(grp) > 1:
                        ax.plot(grp["measurement_day"], grp[metric], color="#444", alpha=0.35, linewidth=1.0)

            ax.scatter(tmp["measurement_day"], tmp[metric], color="#111", s=14, alpha=0.8, edgecolors="none")

            if show_band:
                grouped = tmp.groupby("measurement_day")[metric].agg(["mean", "std"]).reset_index().sort_values("measurement_day")
                if not grouped.empty:
                    x = grouped["measurement_day"]
                    y_mean = grouped["mean"]
                    y_std = grouped["std"].fillna(0.0)
                    ax.plot(x, y_mean, color="#c62828", linewidth=1.2)
                    ax.fill_between(x, y_mean - y_std, y_mean + y_std, color="#c62828", alpha=0.15)

            locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.set_title(MORPHOMETRIC_METRIC_LABELS.get(metric, metric), fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle("Morphometric size trends by measurement day", fontsize=12)
        fig.tight_layout()
        self._chart.draw()


class MorphometricRelationshipsDialog(MorphometricDialogBase):
    """Pairwise metric relationships inspired by notebook pairplots."""

    _REL_METRICS = [
        "morph_area_mm2",
        "morph_major_axis_mm",
        "morph_minor_axis_mm",
        "morph_mean_arm_length_mm",
        "morph_tip_to_tip_mm",
    ]

    def __init__(self, parent=None):
        super().__init__("Morphometric Metric Relationships", parent)

    def _build_body(self):
        self._build_shared_filters()

        row = QHBoxLayout()
        row.addWidget(QLabel("Color by:"))
        self.cmb_color = QComboBox()
        self.cmb_color.addItems(["None", "Identity Type", "Location", "ID"])
        row.addWidget(self.cmb_color)
        row.addStretch(1)
        self._root.addLayout(row)

        self.cmb_color.currentIndexChanged.connect(self._on_filter_changed)

        self._chart = MplWidget(self)
        self._root.addWidget(self._chart, 1)

    def _color_column(self) -> Optional[str]:
        text = self.cmb_color.currentText().strip().lower()
        if text == "identity type":
            return "identity_type"
        if text == "location":
            return "location"
        if text == "id":
            return "identity_id"
        return None

    def render(self):
        rows = self._current_filtered_rows()
        self._last_export_rows = rows

        df = _rows_to_dataframe(rows)
        if df is None:
            self._set_message("Pandas is required for morphometric analytics views.")
            return
        if df.empty:
            self._set_message("No morphometric measurements found for current filters.")
            return

        metrics = _visible_metrics(df, self._REL_METRICS)
        if len(metrics) < 2:
            self._set_message("Need at least two populated morphometric metrics for relationship plots.")
            return

        color_col = self._color_column()

        try:
            from matplotlib import cm
        except Exception:
            self._set_message("Matplotlib plotting backend is unavailable.")
            return

        fig = self._chart.fig
        fig.clear()
        n = len(metrics)
        axes = fig.subplots(n, n, squeeze=False)

        categories: List[str] = []
        color_map: Dict[str, Any] = {}
        if color_col:
            cats = [str(v) for v in df[color_col].fillna("Unknown").astype(str).tolist()]
            categories = sorted(set(cats))
            cmap = cm.get_cmap("tab10")
            for idx, cat in enumerate(categories):
                color_map[cat] = cmap(idx % 10)

        for i, y_metric in enumerate(metrics):
            for j, x_metric in enumerate(metrics):
                ax = axes[i][j]
                if i == j:
                    vals = df[y_metric].dropna()
                    if not vals.empty:
                        ax.hist(vals, bins=16, color="#607d8b", alpha=0.8, edgecolor="white")
                else:
                    cols = [x_metric, y_metric]
                    if color_col:
                        cols.append(color_col)
                    tmp = df[cols].dropna(subset=[x_metric, y_metric]).copy()
                    if not tmp.empty:
                        if color_col:
                            series = tmp[color_col].fillna("Unknown").astype(str)
                            for cat in categories:
                                sub = tmp[series == cat]
                                if sub.empty:
                                    continue
                                ax.scatter(
                                    sub[x_metric],
                                    sub[y_metric],
                                    s=12,
                                    alpha=0.55,
                                    color=color_map[cat],
                                    edgecolors="none",
                                )
                        else:
                            ax.scatter(tmp[x_metric], tmp[y_metric], s=12, alpha=0.55, color="#1f77b4", edgecolors="none")

                if i == n - 1:
                    ax.set_xlabel(MORPHOMETRIC_METRIC_LABELS.get(x_metric, x_metric), fontsize=8)
                else:
                    ax.set_xlabel("")
                    ax.set_xticklabels([])

                if j == 0:
                    ax.set_ylabel(MORPHOMETRIC_METRIC_LABELS.get(y_metric, y_metric), fontsize=8)
                else:
                    ax.set_ylabel("")
                    ax.set_yticklabels([])

                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.12)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

        if color_col and categories:
            handles = []
            labels = []
            for cat in categories[:8]:
                handles.append(axes[0][0].scatter([], [], color=color_map[cat], s=25))
                labels.append(cat)
            if handles:
                fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8, title=color_col)

        fig.suptitle("Morphometric metric relationships", fontsize=12)
        fig.tight_layout()
        self._chart.draw()


class MorphometricPcaDialog(MorphometricDialogBase):
    """PCA and biplot-style morphology space view."""

    _PCA_METRICS = [
        "morph_area_mm2",
        "morph_major_axis_mm",
        "morph_minor_axis_mm",
        "morph_mean_arm_length_mm",
        "morph_max_arm_length_mm",
        "morph_tip_to_tip_mm",
    ]

    def __init__(self, parent=None):
        self._last_pca_rows: List[Dict[str, Any]] = []
        super().__init__("Morphometric PCA", parent)

    def _build_body(self):
        self._build_shared_filters()

        row = QHBoxLayout()
        row.addWidget(QLabel("Color by:"))
        self.cmb_color = QComboBox()
        self.cmb_color.addItems(["Measurement Day", "Identity Type", "Location", "ID", "None"])
        row.addWidget(self.cmb_color)

        self.chk_connect = QCheckBox("Connect same ID")
        self.chk_connect.setChecked(True)
        row.addWidget(self.chk_connect)

        self.chk_loadings = QCheckBox("Show loadings")
        self.chk_loadings.setChecked(True)
        row.addWidget(self.chk_loadings)

        row.addStretch(1)
        self._root.addLayout(row)

        self.cmb_color.currentIndexChanged.connect(self._on_filter_changed)
        self.chk_connect.stateChanged.connect(self._on_filter_changed)
        self.chk_loadings.stateChanged.connect(self._on_filter_changed)

        self._chart = MplWidget(self)
        self._root.addWidget(self._chart, 1)

    def _color_column(self) -> Optional[str]:
        text = self.cmb_color.currentText().strip().lower()
        if text == "measurement day":
            return "measurement_day"
        if text == "identity type":
            return "identity_type"
        if text == "location":
            return "location"
        if text == "id":
            return "identity_id"
        return None

    def render(self):
        self._last_pca_rows = []
        rows = self._current_filtered_rows()
        self._last_export_rows = rows

        df = _rows_to_dataframe(rows)
        if df is None:
            self._set_message("Pandas is required for morphometric analytics views.")
            return
        if df.empty:
            self._set_message("No morphometric measurements found for current filters.")
            return

        metrics = _visible_metrics(df, self._PCA_METRICS)
        if len(metrics) < 2:
            self._set_message("Need at least two populated metrics to compute PCA.")
            return

        needed = ["identity_label", "identity_id", "identity_type", "location", "measurement_day"] + metrics
        tmp = df[needed].dropna(subset=metrics).copy()
        if len(tmp) < 3:
            self._set_message("Need at least three complete rows to compute PCA.")
            return

        try:
            import numpy as np
            from matplotlib import cm
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except Exception:
            self._set_message("scikit-learn and matplotlib are required for PCA visualization.")
            return

        X = tmp[metrics].astype(float).values
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_scaled)
        tmp["PC1"] = pcs[:, 0]
        tmp["PC2"] = pcs[:, 1]
        self._last_pca_rows = tmp.to_dict(orient="records")

        fig = self._chart.fig
        fig.clear()
        ax = fig.add_subplot(111)

        color_col = self._color_column()
        if color_col == "measurement_day":
            dates = tmp["measurement_day"]
            if dates.isna().all():
                ax.scatter(tmp["PC1"], tmp["PC2"], s=28, alpha=0.8, color="#2c7fb8")
            else:
                ordinals = dates.map(lambda d: d.toordinal() if d is not None and str(d) != "NaT" else np.nan)
                ordinals = ordinals.fillna(ordinals.min())
                sc = ax.scatter(tmp["PC1"], tmp["PC2"], c=ordinals, cmap="viridis", s=30, alpha=0.85)
                cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Measurement day")
        elif color_col in ("identity_type", "location", "identity_id"):
            categories = sorted(set(str(v) for v in tmp[color_col].fillna("Unknown")))
            cmap = cm.get_cmap("tab10")
            handles = []
            for idx, cat in enumerate(categories):
                sub = tmp[tmp[color_col].fillna("Unknown").astype(str) == cat]
                h = ax.scatter(sub["PC1"], sub["PC2"], s=30, alpha=0.85, color=cmap(idx % 10), label=cat)
                handles.append(h)
            if handles:
                ax.legend(loc="best", fontsize=8, frameon=False, title=color_col)
        else:
            ax.scatter(tmp["PC1"], tmp["PC2"], s=30, alpha=0.85, color="#2c7fb8")

        if self.chk_connect.isChecked():
            for _, grp in tmp.sort_values("measurement_day").groupby("identity_label"):
                if len(grp) > 1:
                    ax.plot(grp["PC1"], grp["PC2"], color="#444", linewidth=1.0, alpha=0.45)

        if self.chk_loadings.isChecked():
            loadings = pca.components_.T
            span_x = max(float(tmp["PC1"].max() - tmp["PC1"].min()), 1e-6)
            span_y = max(float(tmp["PC2"].max() - tmp["PC2"].min()), 1e-6)
            arrow_scale = 0.3 * min(span_x, span_y)
            for i, metric in enumerate(metrics):
                dx = float(loadings[i, 0]) * arrow_scale
                dy = float(loadings[i, 1]) * arrow_scale
                ax.arrow(0, 0, dx, dy, color="#8e24aa", alpha=0.75, width=0.003, head_width=0.04)
                ax.text(dx * 1.08, dy * 1.08, MORPHOMETRIC_METRIC_LABELS.get(metric, metric), fontsize=8, color="#5e35b1")

        pc1_var = pca.explained_variance_ratio_[0] * 100.0
        pc2_var = pca.explained_variance_ratio_[1] * 100.0
        ax.set_xlabel(f"PC1 ({pc1_var:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pc2_var:.1f}% var)")
        ax.set_title("Morphometric PCA")
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        self._chart.draw()

    def export_rows(self) -> Tuple[List[str], List[List[str]]]:
        if not self._last_pca_rows:
            return [], []
        ordered = [
            "identity_type",
            "identity_id",
            "identity_label",
            "measurement_day",
            "location",
            "PC1",
            "PC2",
            "morph_area_mm2",
            "morph_major_axis_mm",
            "morph_minor_axis_mm",
            "morph_mean_arm_length_mm",
            "morph_max_arm_length_mm",
            "morph_tip_to_tip_mm",
        ]
        rows = [[str(r.get(k, "")) for k in ordered] for r in self._last_pca_rows]
        return ordered, rows


class MorphometricIdentityLongitudinalDialog(MorphometricDialogBase):
    """
    Identity-first longitudinal analysis view for repeated measurements over time.
    """

    def __init__(self, parent=None):
        self._last_series_rows: List[Dict[str, Any]] = []
        super().__init__("Morphometric Identity Longitudinal", parent)

    def _build_body(self):
        self._build_shared_filters()

        row = QHBoxLayout()
        row.addWidget(QLabel("Metric:"))
        self.cmb_metric = QComboBox()
        for m in MORPHOMETRIC_METRIC_FIELDS:
            self.cmb_metric.addItem(MORPHOMETRIC_METRIC_LABELS.get(m, m), userData=m)
        row.addWidget(self.cmb_metric)

        row.addWidget(QLabel("Min measurements/identity:"))
        self.spin_min = QSpinBox()
        self.spin_min.setRange(1, 30)
        self.spin_min.setValue(2)
        row.addWidget(self.spin_min)

        self.chk_delta = QCheckBox("Delta from first measurement")
        self.chk_delta.setChecked(False)
        row.addWidget(self.chk_delta)

        self.chk_mean = QCheckBox("Show daily mean")
        self.chk_mean.setChecked(True)
        row.addWidget(self.chk_mean)

        row.addStretch(1)
        self._root.addLayout(row)

        self.cmb_metric.currentIndexChanged.connect(self._on_filter_changed)
        self.spin_min.valueChanged.connect(self._on_filter_changed)
        self.chk_delta.stateChanged.connect(self._on_filter_changed)
        self.chk_mean.stateChanged.connect(self._on_filter_changed)

        self._chart = MplWidget(self)
        self._root.addWidget(self._chart, 1)

    def render(self):
        self._last_series_rows = []
        rows = self._current_filtered_rows()
        self._last_export_rows = rows

        df = _rows_to_dataframe(rows)
        if df is None:
            self._set_message("Pandas is required for morphometric analytics views.")
            return
        if df.empty:
            self._set_message("No morphometric measurements found for current filters.")
            return

        metric = self.cmb_metric.currentData()
        if not metric:
            self._set_message("No metric selected.")
            return

        cols = ["measurement_day", "identity_label", "identity_type", "identity_id", "location", metric]
        tmp = df[cols].dropna(subset=["measurement_day", metric]).copy()
        if tmp.empty:
            self._set_message("No rows available after metric/date filtering.")
            return

        min_n = int(self.spin_min.value())
        normalize_delta = self.chk_delta.isChecked()
        show_mean = self.chk_mean.isChecked()

        grouped = []
        for identity_label, grp in tmp.groupby("identity_label"):
            g = grp.sort_values("measurement_day").copy()
            if len(g) < min_n:
                continue
            if normalize_delta:
                baseline = float(g[metric].iloc[0])
                g["plot_value"] = g[metric].astype(float) - baseline
            else:
                g["plot_value"] = g[metric].astype(float)
            grouped.append((identity_label, g))

        if not grouped:
            self._set_message("No identities have enough repeated measurements under current filters.")
            return

        try:
            from matplotlib import cm
            import matplotlib.dates as mdates
        except Exception:
            self._set_message("Matplotlib date plotting is unavailable.")
            return

        fig = self._chart.fig
        fig.clear()
        ax = fig.add_subplot(111)

        cmap = cm.get_cmap("tab20")
        handles = []
        labels = []

        for idx, (identity_label, grp) in enumerate(grouped):
            color = cmap(idx % 20)
            h = ax.plot(
                grp["measurement_day"],
                grp["plot_value"],
                marker="o",
                linewidth=1.4,
                markersize=4,
                alpha=0.9,
                color=color,
            )[0]
            if idx < 20:
                handles.append(h)
                labels.append(identity_label)

            for _, row in grp.iterrows():
                self._last_series_rows.append(
                    {
                        "identity_label": identity_label,
                        "identity_type": str(row.get("identity_type") or ""),
                        "identity_id": str(row.get("identity_id") or ""),
                        "location": str(row.get("location") or ""),
                        "measurement_day": str(row.get("measurement_day") or ""),
                        "metric": metric,
                        "metric_label": MORPHOMETRIC_METRIC_LABELS.get(metric, metric),
                        "raw_value": row.get(metric, ""),
                        "plot_value": row.get("plot_value", ""),
                    }
                )

        if show_mean:
            mean_by_day = tmp.groupby("measurement_day")[metric].mean().reset_index().sort_values("measurement_day")
            if not mean_by_day.empty:
                if normalize_delta:
                    baseline = float(mean_by_day[metric].iloc[0])
                    mean_vals = mean_by_day[metric].astype(float) - baseline
                else:
                    mean_vals = mean_by_day[metric].astype(float)
                ax.plot(
                    mean_by_day["measurement_day"],
                    mean_vals,
                    color="#111",
                    linewidth=2.2,
                    linestyle="--",
                    label="daily mean",
                )

        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.tick_params(axis="x", rotation=30)

        y_label = MORPHOMETRIC_METRIC_LABELS.get(metric, metric)
        if normalize_delta:
            y_label = f"Delta {y_label}"

        ax.set_ylabel(y_label)
        ax.set_xlabel("Measurement day")
        ax.set_title("Identity longitudinal morphometric trajectories")
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if handles:
            ax.legend(handles, labels, loc="best", fontsize=7, frameon=False, title="Identities (sample)")

        fig.tight_layout()
        self._chart.draw()

    def export_rows(self) -> Tuple[List[str], List[List[str]]]:
        if not self._last_series_rows:
            return [], []
        ordered = [
            "identity_type",
            "identity_id",
            "identity_label",
            "location",
            "measurement_day",
            "metric",
            "metric_label",
            "raw_value",
            "plot_value",
        ]
        rows = [[str(r.get(k, "")) for k in ordered] for r in self._last_series_rows]
        return ordered, rows

