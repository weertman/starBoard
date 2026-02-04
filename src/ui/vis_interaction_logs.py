# src/ui/vis_interaction_logs.py
"""
Visualization dialogs for user interaction logs.

Provides insights into how users interact with starBoard through:
1. SessionProductivityDialog - Activity timeline across sessions
2. FeatureUsageDialog - Hierarchical feature usage patterns (sunburst/treemap)
"""
from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QTableWidget, QTableWidgetItem, QWidget, QSpinBox, QFileDialog,
    QGroupBox, QCheckBox
)
from PySide6.QtCore import Qt

from src.ui.mpl_embed import MplWidget
from src.data.archive_paths import archive_root, logs_root
from src.utils.interaction_logger import get_interaction_logger

# Color palette for visualizations
COLORS = {
    "primary": "#2E86AB",      # Blue
    "secondary": "#A23B72",    # Magenta
    "tertiary": "#F18F01",     # Orange
    "quaternary": "#C73E1D",   # Red
    "quinary": "#3B1F2B",      # Dark purple
    "success": "#1b9e77",      # Green
    "warning": "#d95f02",      # Orange
    "danger": "#d73027",       # Red
}

# Tab colors for consistency
TAB_COLORS = {
    "First Order": "#2E86AB",
    "Second Order": "#A23B72",
    "Data Entry": "#F18F01",
    "DL": "#C73E1D",
    "Past": "#1b9e77",
    "Other": "#888888",
}

# Event category colors
EVENT_CATEGORY_COLORS = {
    "button_click": "#2E86AB",
    "combo_change": "#A23B72",
    "checkbox_toggle": "#F18F01",
    "dialog_open": "#C73E1D",
    "dialog_close": "#3B1F2B",
    "tab_switch": "#1b9e77",
    "decision_save": "#FFD700",
    "other": "#888888",
}


def load_all_interaction_logs() -> List[Dict[str, Any]]:
    """Load all interaction log CSV files and return as list of event dicts."""
    logs_dir = logs_root()
    if not logs_dir.exists():
        return []
    
    all_events = []
    csv_files = sorted(logs_dir.glob("interactions_*.csv"))
    
    for csv_path in csv_files:
        session_id = csv_path.stem.replace("interactions_", "")
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["session_id"] = session_id
                    # Parse timestamp
                    try:
                        row["datetime"] = datetime.fromisoformat(row["timestamp"])
                    except (ValueError, KeyError):
                        row["datetime"] = None
                    all_events.append(row)
        except Exception:
            continue
    
    return all_events


def get_session_stats(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute per-session statistics."""
    sessions: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "event_count": 0,
        "decision_count": 0,
        "tabs_used": set(),
        "event_types": Counter(),
        "widgets_used": Counter(),
        "start_time": None,
        "end_time": None,
        "duration_minutes": 0,
    })
    
    for e in events:
        sid = e.get("session_id", "unknown")
        s = sessions[sid]
        s["event_count"] += 1
        
        event_type = e.get("event_type", "")
        if event_type == "decision_save":
            s["decision_count"] += 1
        
        tab = e.get("tab", "")
        if tab:
            s["tabs_used"].add(tab)
        
        s["event_types"][event_type] += 1
        s["widgets_used"][e.get("widget", "")] += 1
        
        dt = e.get("datetime")
        if dt:
            if s["start_time"] is None or dt < s["start_time"]:
                s["start_time"] = dt
            if s["end_time"] is None or dt > s["end_time"]:
                s["end_time"] = dt
    
    # Compute durations
    for sid, s in sessions.items():
        if s["start_time"] and s["end_time"]:
            delta = s["end_time"] - s["start_time"]
            s["duration_minutes"] = delta.total_seconds() / 60.0
        s["tabs_used"] = list(s["tabs_used"])  # Convert set to list for serialization
    
    return dict(sessions)


class InteractionLogDialogBase(QDialog):
    """Base class for interaction log visualization dialogs."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(1000, 700)
        
        self._ilog = get_interaction_logger()
        self._reports_dir = archive_root() / "reports" / "figures"
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        
        self._events: List[Dict[str, Any]] = []
        self._session_stats: Dict[str, Dict[str, Any]] = {}
        
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(8, 8, 8, 8)
        self._root.setSpacing(6)
        
        # Top toolbar
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
        self._load_data()
        self.render()
    
    def _build_body(self):
        """Override in subclasses to build the body of the dialog."""
        self._chart = MplWidget(self)
        self._root.addWidget(self._chart, 1)
    
    def _load_data(self):
        """Load interaction log data."""
        self._events = load_all_interaction_logs()
        self._session_stats = get_session_stats(self._events)
        self.lbl_hint.setText(f"{len(self._events)} events from {len(self._session_stats)} sessions")
    
    def render(self):
        """Override in subclasses to render the visualization."""
        self._chart.clear()
        self._chart.ax.text(0.5, 0.5, "No content", ha="center", va="center")
        self._chart.draw()
    
    def export_rows(self) -> Tuple[List[str], List[List[str]]]:
        """Override to provide CSV export data."""
        return [], []
    
    def _on_refresh(self):
        self._ilog.log("button_click", f"btn_refresh_{self.__class__.__name__}", value="clicked")
        self._load_data()
        self.render()
    
    def _on_export_png(self):
        self._ilog.log("button_click", f"btn_export_png_{self.__class__.__name__}", value="clicked")
        if not getattr(self._chart, "fig", None):
            return
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        fname = f"{self.__class__.__name__.lower()}_{ts}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", str(self._reports_dir / fname), "PNG (*.png)"
        )
        if not path:
            return
        try:
            self._chart.fig.savefig(path, dpi=150, bbox_inches="tight")
            self.lbl_hint.setText(f"Saved {os.path.basename(path)}")
        except Exception as e:
            self.lbl_hint.setText(f"Save failed: {e}")
    
    def _on_export_csv(self):
        self._ilog.log("button_click", f"btn_export_csv_{self.__class__.__name__}", value="clicked")
        header, rows = self.export_rows()
        if not header:
            return
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        fname = f"{self.__class__.__name__.lower()}_{ts}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", str(self._reports_dir / fname), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)
            self.lbl_hint.setText(f"Saved {os.path.basename(path)}")
        except Exception as e:
            self.lbl_hint.setText(f"Save failed: {e}")


class SessionProductivityDialog(InteractionLogDialogBase):
    """
    Session Productivity Timeline Visualization.
    
    Shows activity patterns over time:
    - Bar chart of events per session (or per day)
    - Overlaid line for decisions made
    - Color-coded by primary activity tab
    """
    
    def __init__(self, parent=None):
        self._granularity = "Session"
        super().__init__("Session Productivity Timeline", parent)
    
    def _build_body(self):
        # Controls row
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Granularity:"))
        self.cmb_granularity = QComboBox()
        self.cmb_granularity.addItems(["Session", "Day", "Week"])
        self.cmb_granularity.currentIndexChanged.connect(self._on_granularity_changed)
        controls.addWidget(self.cmb_granularity)
        
        controls.addSpacing(20)
        self.chk_show_decisions = QCheckBox("Show Decisions Line")
        self.chk_show_decisions.setChecked(True)
        self.chk_show_decisions.toggled.connect(self.render)
        controls.addWidget(self.chk_show_decisions)
        
        controls.addStretch(1)
        self._root.addLayout(controls)
        
        # Chart
        self._chart = MplWidget(self)
        self._root.addWidget(self._chart, 1)
        
        # Stats summary
        self.lbl_stats = QLabel("")
        self.lbl_stats.setWordWrap(True)
        self.lbl_stats.setStyleSheet("color: #555; font-size: 11px;")
        self._root.addWidget(self.lbl_stats)
    
    def _on_granularity_changed(self):
        self._granularity = self.cmb_granularity.currentText()
        self._ilog.log("combo_change", "cmb_granularity_productivity", value=self._granularity)
        self.render()
    
    def render(self):
        if not self._events:
            self._chart.clear()
            self._chart.ax.text(0.5, 0.5, "No interaction logs found.\nStart using the app to generate data.",
                               ha="center", va="center", fontsize=12)
            self._chart.draw()
            return
        
        self._chart.clear()
        ax = self._chart.ax
        
        # Aggregate data based on granularity
        if self._granularity == "Session":
            data = self._aggregate_by_session()
        elif self._granularity == "Day":
            data = self._aggregate_by_day()
        else:  # Week
            data = self._aggregate_by_week()
        
        if not data:
            ax.text(0.5, 0.5, "No data to display", ha="center", va="center")
            self._chart.draw()
            return
        
        labels = [d["label"] for d in data]
        events = [d["events"] for d in data]
        decisions = [d["decisions"] for d in data]
        primary_tabs = [d.get("primary_tab", "Other") for d in data]
        
        x = range(len(labels))
        
        # Bar colors based on primary tab
        bar_colors = [TAB_COLORS.get(tab, TAB_COLORS["Other"]) for tab in primary_tabs]
        
        # Create bars
        bars = ax.bar(x, events, color=bar_colors, alpha=0.7, label="Events", edgecolor="#333", linewidth=0.5)
        
        # Add decision line if enabled
        if self.chk_show_decisions.isChecked() and sum(decisions) > 0:
            ax2 = ax.twinx()
            ax2.plot(x, decisions, color=COLORS["success"], marker="o", linewidth=2, 
                     markersize=5, label="Decisions", zorder=5)
            ax2.set_ylabel("Decisions", color=COLORS["success"])
            ax2.tick_params(axis="y", labelcolor=COLORS["success"])
            ax2.set_ylim(bottom=0)
        
        # Labels
        ax.set_ylabel("Events")
        ax.set_xlabel(self._granularity)
        ax.set_title(f"Activity by {self._granularity}")
        
        # X-axis labels
        if len(labels) <= 20:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        else:
            # Show every Nth label
            step = max(1, len(labels) // 15)
            ax.set_xticks([i for i in x if i % step == 0])
            ax.set_xticklabels([labels[i] for i in x if i % step == 0], rotation=45, ha="right", fontsize=8)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)
        
        # Add value labels on bars (for small datasets)
        if len(labels) <= 15:
            for bar, val, dec in zip(bars, events, decisions):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height, f"{val}",
                       ha="center", va="bottom", fontsize=8)
        
        self._chart.fig.tight_layout()
        self._chart.draw()
        
        # Update stats summary
        total_events = sum(events)
        total_decisions = sum(decisions)
        avg_events = total_events / len(data) if data else 0
        self.lbl_stats.setText(
            f"Total: {total_events} events, {total_decisions} decisions | "
            f"Average per {self._granularity.lower()}: {avg_events:.1f} events | "
            f"Periods: {len(data)}"
        )
    
    def _aggregate_by_session(self) -> List[Dict[str, Any]]:
        """Aggregate data by session."""
        data = []
        for sid, stats in sorted(self._session_stats.items(), 
                                  key=lambda x: x[1].get("start_time") or datetime.min):
            primary_tab = max(stats["event_types"].items(), key=lambda x: x[1])[0] if stats["event_types"] else "Other"
            # Map event type to tab (rough heuristic)
            tabs_count = Counter(e.get("tab", "") for e in self._events if e.get("session_id") == sid)
            if tabs_count:
                primary_tab = max(tabs_count.items(), key=lambda x: x[1])[0] or "Other"
            
            # Create short label
            if stats["start_time"]:
                label = stats["start_time"].strftime("%m/%d %H:%M")
            else:
                label = sid[:10]
            
            data.append({
                "label": label,
                "events": stats["event_count"],
                "decisions": stats["decision_count"],
                "primary_tab": primary_tab,
                "session_id": sid,
            })
        return data
    
    def _aggregate_by_day(self) -> List[Dict[str, Any]]:
        """Aggregate data by day."""
        daily: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "events": 0, "decisions": 0, "tabs": Counter()
        })
        
        for e in self._events:
            dt = e.get("datetime")
            if not dt:
                continue
            day = dt.strftime("%Y-%m-%d")
            daily[day]["events"] += 1
            if e.get("event_type") == "decision_save":
                daily[day]["decisions"] += 1
            tab = e.get("tab", "")
            if tab:
                daily[day]["tabs"][tab] += 1
        
        data = []
        for day in sorted(daily.keys()):
            d = daily[day]
            primary_tab = max(d["tabs"].items(), key=lambda x: x[1])[0] if d["tabs"] else "Other"
            data.append({
                "label": day[5:],  # MM-DD
                "events": d["events"],
                "decisions": d["decisions"],
                "primary_tab": primary_tab,
            })
        return data
    
    def _aggregate_by_week(self) -> List[Dict[str, Any]]:
        """Aggregate data by week."""
        weekly: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "events": 0, "decisions": 0, "tabs": Counter()
        })
        
        for e in self._events:
            dt = e.get("datetime")
            if not dt:
                continue
            # Get ISO week
            week = dt.strftime("%Y-W%W")
            weekly[week]["events"] += 1
            if e.get("event_type") == "decision_save":
                weekly[week]["decisions"] += 1
            tab = e.get("tab", "")
            if tab:
                weekly[week]["tabs"][tab] += 1
        
        data = []
        for week in sorted(weekly.keys()):
            w = weekly[week]
            primary_tab = max(w["tabs"].items(), key=lambda x: x[1])[0] if w["tabs"] else "Other"
            data.append({
                "label": week,
                "events": w["events"],
                "decisions": w["decisions"],
                "primary_tab": primary_tab,
            })
        return data
    
    def export_rows(self) -> Tuple[List[str], List[List[str]]]:
        """Export session data to CSV."""
        header = ["session_id", "start_time", "end_time", "duration_min", 
                  "event_count", "decision_count", "tabs_used"]
        rows = []
        for sid, stats in sorted(self._session_stats.items(),
                                  key=lambda x: x[1].get("start_time") or datetime.min):
            rows.append([
                sid,
                stats["start_time"].isoformat() if stats["start_time"] else "",
                stats["end_time"].isoformat() if stats["end_time"] else "",
                f"{stats['duration_minutes']:.1f}",
                stats["event_count"],
                stats["decision_count"],
                ";".join(stats["tabs_used"]),
            ])
        return header, rows


class FeatureUsageDialog(InteractionLogDialogBase):
    """
    Feature Usage & Workflow Patterns Visualization.
    
    Shows hierarchical breakdown of feature usage:
    - Treemap or sunburst showing Tab → Event Type → Widget
    - Helps identify most-used features and workflows
    """
    
    def __init__(self, parent=None):
        self._view_mode = "Treemap"
        super().__init__("Feature Usage Patterns", parent)
    
    def _build_body(self):
        # Controls row
        controls = QHBoxLayout()
        controls.addWidget(QLabel("View:"))
        self.cmb_view = QComboBox()
        self.cmb_view.addItems(["Treemap", "Horizontal Bars", "Sunburst"])
        self.cmb_view.currentIndexChanged.connect(self._on_view_changed)
        controls.addWidget(self.cmb_view)
        
        controls.addSpacing(20)
        controls.addWidget(QLabel("Group by:"))
        self.cmb_group = QComboBox()
        self.cmb_group.addItems(["Tab → Event Type", "Event Type → Tab", "Widget Only"])
        self.cmb_group.currentIndexChanged.connect(self.render)
        controls.addWidget(self.cmb_group)
        
        controls.addStretch(1)
        self._root.addLayout(controls)
        
        # Chart
        self._chart = MplWidget(self)
        self._root.addWidget(self._chart, 1)
        
        # Stats table
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Feature", "Count", "% of Total", "Trend"])
        self._table.setMaximumHeight(180)
        self._root.addWidget(self._table)
    
    def _on_view_changed(self):
        self._view_mode = self.cmb_view.currentText()
        self._ilog.log("combo_change", "cmb_view_feature_usage", value=self._view_mode)
        self.render()
    
    def render(self):
        if not self._events:
            self._chart.clear()
            self._chart.ax.text(0.5, 0.5, "No interaction logs found.\nStart using the app to generate data.",
                               ha="center", va="center", fontsize=12)
            self._chart.draw()
            return
        
        self._chart.clear()
        
        group_mode = self.cmb_group.currentText()
        
        if self._view_mode == "Treemap":
            self._render_treemap(group_mode)
        elif self._view_mode == "Horizontal Bars":
            self._render_bars(group_mode)
        else:  # Sunburst
            self._render_sunburst(group_mode)
        
        self._update_table()
    
    def _get_hierarchical_data(self, group_mode: str) -> Dict[str, Dict[str, int]]:
        """Get hierarchical data based on grouping mode."""
        data: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for e in self._events:
            tab = e.get("tab", "Other") or "Other"
            event_type = e.get("event_type", "other") or "other"
            widget = e.get("widget", "unknown") or "unknown"
            
            if group_mode == "Tab → Event Type":
                data[tab][event_type] += 1
            elif group_mode == "Event Type → Tab":
                data[event_type][tab] += 1
            else:  # Widget Only
                data["All"][widget] += 1
        
        return dict(data)
    
    def _render_treemap(self, group_mode: str):
        """Render a treemap visualization using squarified layout."""
        ax = self._chart.ax
        data = self._get_hierarchical_data(group_mode)
        
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            self._chart.draw()
            return
        
        # Flatten and sort by size
        flat_data = []
        for group, items in data.items():
            group_color = TAB_COLORS.get(group, COLORS["primary"])
            for item, count in items.items():
                flat_data.append({
                    "label": f"{group}: {item}",
                    "short_label": item[:15],
                    "group": group,
                    "count": count,
                    "color": group_color,
                })
        
        if not flat_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            self._chart.draw()
            return
        
        # Sort by count and take top 16 for a 4x4-ish grid
        flat_data = sorted(flat_data, key=lambda x: -x["count"])[:16]
        total = sum(d["count"] for d in flat_data)
        
        # Simple squarify algorithm
        def squarify(data, x, y, width, height):
            """Recursively divide rectangle into squares."""
            if not data:
                return []
            if len(data) == 1:
                return [(data[0], x, y, width, height)]
            
            # Split data into two groups
            total_val = sum(d["count"] for d in data)
            running = 0
            split_idx = 0
            for i, d in enumerate(data):
                running += d["count"]
                if running >= total_val / 2:
                    split_idx = i + 1
                    break
            
            if split_idx == 0:
                split_idx = 1
            if split_idx >= len(data):
                split_idx = len(data) - 1
            
            left_data = data[:split_idx]
            right_data = data[split_idx:]
            
            left_ratio = sum(d["count"] for d in left_data) / total_val if total_val > 0 else 0.5
            
            rects = []
            if width > height:
                # Split horizontally
                rects.extend(squarify(left_data, x, y, width * left_ratio, height))
                rects.extend(squarify(right_data, x + width * left_ratio, y, width * (1 - left_ratio), height))
            else:
                # Split vertically
                rects.extend(squarify(left_data, x, y, width, height * left_ratio))
                rects.extend(squarify(right_data, x, y + height * left_ratio, width, height * (1 - left_ratio)))
            
            return rects
        
        rects = squarify(flat_data, 0, 0, 1, 1)
        
        from matplotlib.patches import Rectangle
        
        for item, rx, ry, rw, rh in rects:
            # Draw rectangle
            rect = Rectangle((rx, ry), rw, rh, 
                            facecolor=item["color"], edgecolor="white", 
                            linewidth=2, alpha=0.85)
            ax.add_patch(rect)
            
            # Add label if box is big enough
            area = rw * rh
            if area > 0.02:  # Only label boxes > 2% of total area
                # Determine font size based on area
                fontsize = max(6, min(10, int(area * 80)))
                
                # Choose label based on space
                if area > 0.08:
                    label = item["label"]
                else:
                    label = item["short_label"]
                
                # Wrap long labels
                if len(label) > 12 and rh > rw:
                    # Vertical orientation - can use more lines
                    words = label.split(": ")
                    label = "\n".join(words)
                
                ax.text(rx + rw/2, ry + rh/2, label,
                       ha="center", va="center", fontsize=fontsize, 
                       color="white", fontweight="bold",
                       wrap=True)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Feature Usage ({group_mode}) - Top 16")
        
        self._chart.draw()
    
    def _render_bars(self, group_mode: str):
        """Render horizontal stacked bars."""
        ax = self._chart.ax
        data = self._get_hierarchical_data(group_mode)
        
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            self._chart.draw()
            return
        
        # Get top groups
        group_totals = {g: sum(items.values()) for g, items in data.items()}
        top_groups = sorted(group_totals.keys(), key=lambda g: -group_totals[g])[:10]
        
        # Get all item types
        all_items = set()
        for g in top_groups:
            all_items.update(data[g].keys())
        all_items = sorted(all_items, key=lambda i: -sum(data[g].get(i, 0) for g in top_groups))[:8]
        
        # Create stacked bars
        y_pos = range(len(top_groups))
        left = [0] * len(top_groups)
        
        color_list = list(EVENT_CATEGORY_COLORS.values())
        
        for i, item in enumerate(all_items):
            widths = [data[g].get(item, 0) for g in top_groups]
            color = color_list[i % len(color_list)]
            ax.barh(y_pos, widths, left=left, label=item, color=color, alpha=0.8, edgecolor="white")
            left = [l + w for l, w in zip(left, widths)]
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_groups)
        ax.set_xlabel("Event Count")
        ax.set_title(f"Feature Usage by {group_mode.split(' →')[0]}")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        self._chart.fig.tight_layout()
        self._chart.draw()
    
    def _render_sunburst(self, group_mode: str):
        """Render a sunburst/pie chart visualization with legend."""
        ax = self._chart.ax
        data = self._get_hierarchical_data(group_mode)
        
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            self._chart.draw()
            return
        
        # Inner ring data (groups) - sorted by size
        group_totals = {g: sum(items.values()) for g, items in data.items()}
        total_events = sum(group_totals.values())
        sorted_groups = sorted(group_totals.keys(), key=lambda g: -group_totals[g])
        inner_sizes = [group_totals[g] for g in sorted_groups]
        inner_colors = [TAB_COLORS.get(g, COLORS["primary"]) for g in sorted_groups]
        
        # Outer ring data (items within each group)
        outer_sizes = []
        outer_colors = []
        outer_labels = []  # Store labels for each wedge
        
        import matplotlib.colors as mcolors
        
        for i, group in enumerate(sorted_groups):
            items = data[group]
            base_color = inner_colors[i]
            sorted_items = sorted(items.items(), key=lambda x: -x[1])[:5]  # Top 5 per group
            
            # Create lighter shades for sub-items
            try:
                rgb = mcolors.to_rgb(base_color)
                for j, (item, count) in enumerate(sorted_items):
                    outer_sizes.append(count)
                    outer_labels.append(item)
                    # Lighten the color for each successive item
                    factor = 1.0 - (j * 0.12)
                    lighter = tuple(min(1.0, c * factor + (1 - factor) * 0.3) for c in rgb)
                    outer_colors.append(lighter)
            except:
                for j, (item, count) in enumerate(sorted_items):
                    outer_sizes.append(count)
                    outer_labels.append(item)
                    outer_colors.append(base_color)
        
        # Draw outer ring (items) with labels on larger wedges
        outer_total = sum(outer_sizes)
        if outer_sizes:
            wedges_outer, _ = ax.pie(
                outer_sizes, radius=1.0, colors=outer_colors,
                wedgeprops=dict(width=0.3, edgecolor="white", linewidth=0.5),
                startangle=90
            )
            
            # Add labels to outer wedges that are large enough (>5%)
            import matplotlib.patheffects as pe
            for wedge, label, size in zip(wedges_outer, outer_labels, outer_sizes):
                pct = 100 * size / outer_total if outer_total > 0 else 0
                if pct > 5:  # Only label wedges > 5%
                    # Calculate angle for text placement
                    angle = (wedge.theta2 + wedge.theta1) / 2
                    # Place text in middle of outer ring (radius 0.85)
                    import math
                    x = 0.85 * math.cos(math.radians(angle))
                    y = 0.85 * math.sin(math.radians(angle))
                    
                    # Shorten label if needed
                    short_label = label[:10] if len(label) > 10 else label
                    
                    # Rotate text to follow the wedge
                    rotation = angle - 90 if angle < 180 else angle + 90
                    
                    ax.text(x, y, short_label, ha="center", va="center",
                           fontsize=7, fontweight="bold", color="white",
                           rotation=rotation,
                           path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        
        # Draw inner ring (groups) - with selective labeling
        # Ring goes from radius 0.4 to 0.7, so middle is at 0.55
        # pctdistance is relative to radius, so 0.55/0.7 ≈ 0.78
        def make_label(pct):
            if pct > 8:
                return f"{pct:.0f}%"
            return ""
        
        wedges_inner, texts_inner, autotexts = ax.pie(
            inner_sizes, radius=0.7, colors=inner_colors,
            wedgeprops=dict(width=0.3, edgecolor="white", linewidth=1.5),
            autopct=make_label,
            pctdistance=0.78, startangle=90, 
            textprops={"fontsize": 10, "fontweight": "bold", "color": "white"}
        )
        
        # Ensure autotext labels are visible with shadow/outline effect
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight("bold")
            # Add black outline for visibility
            autotext.set_path_effects([
                __import__("matplotlib.patheffects", fromlist=["withStroke"]).withStroke(linewidth=2, foreground="black")
            ])
        
        # Create legend instead of inline labels
        legend_labels = []
        for g, size in zip(sorted_groups, inner_sizes):
            pct = 100 * size / total_events if total_events > 0 else 0
            legend_labels.append(f"{g} ({pct:.0f}%)")
        
        ax.legend(wedges_inner, legend_labels, 
                  title="Tabs", loc="center left", 
                  bbox_to_anchor=(1.0, 0.5), fontsize=9)
        
        ax.set_title(f"Feature Usage by {group_mode.split(' →')[0]}")
        
        # Adjust layout to make room for legend
        self._chart.fig.tight_layout()
        self._chart.fig.subplots_adjust(right=0.7)
        
        self._chart.draw()
    
    def _update_table(self):
        """Update the summary table."""
        # Get widget counts
        widget_counts = Counter(e.get("widget", "unknown") for e in self._events)
        total = sum(widget_counts.values())
        
        top_widgets = widget_counts.most_common(10)
        self._table.setRowCount(len(top_widgets))
        
        for i, (widget, count) in enumerate(top_widgets):
            self._table.setItem(i, 0, QTableWidgetItem(widget))
            self._table.setItem(i, 1, QTableWidgetItem(str(count)))
            self._table.setItem(i, 2, QTableWidgetItem(f"{100*count/total:.1f}%"))
            self._table.setItem(i, 3, QTableWidgetItem("—"))  # Trend placeholder
    
    def export_rows(self) -> Tuple[List[str], List[List[str]]]:
        """Export feature usage data to CSV."""
        header = ["tab", "event_type", "widget", "count"]
        rows = []
        
        # Aggregate by tab, event_type, widget
        agg: Dict[Tuple[str, str, str], int] = defaultdict(int)
        for e in self._events:
            key = (
                e.get("tab", "Other") or "Other",
                e.get("event_type", "other") or "other",
                e.get("widget", "unknown") or "unknown",
            )
            agg[key] += 1
        
        for (tab, event_type, widget), count in sorted(agg.items(), key=lambda x: -x[1]):
            rows.append([tab, event_type, widget, count])
        
        return header, rows


class WorkEstimationDialog(InteractionLogDialogBase):
    """
    Work Estimation Visualization.
    
    Estimates remaining work based on:
    - Number of unmatched queries (active queries without a YES verdict)
    - Historical decision rate (decisions per hour from logged sessions)
    - Projected time to completion
    
    Shows:
    - Progress pie chart (matched vs remaining)
    - Cumulative decision trend over time
    - Estimated sessions/hours/days to completion
    """
    
    def __init__(self, parent=None):
        self._matrix = None
        self._active_query_ids = []
        super().__init__("Work Estimation", parent)
    
    def _build_body(self):
        # Summary stats at top
        self._stats_group = QGroupBox("Current Progress")
        stats_layout = QHBoxLayout(self._stats_group)
        
        self.lbl_queries_matched = QLabel("—")
        self.lbl_queries_remaining = QLabel("—")
        self.lbl_decisions_total = QLabel("—")
        self.lbl_rate = QLabel("—")
        
        for label_name, label_widget in [
            ("Queries Matched:", self.lbl_queries_matched),
            ("Queries Remaining:", self.lbl_queries_remaining),
            ("Total Decisions:", self.lbl_decisions_total),
            ("Decisions/Hour:", self.lbl_rate),
        ]:
            v = QVBoxLayout()
            v.addWidget(QLabel(label_name))
            label_widget.setStyleSheet("font-size: 18px; font-weight: bold; color: #2E86AB;")
            v.addWidget(label_widget)
            stats_layout.addLayout(v)
            stats_layout.addSpacing(20)
        
        stats_layout.addStretch(1)
        self._root.addWidget(self._stats_group)
        
        # Chart area
        self._chart = MplWidget(self)
        self._root.addWidget(self._chart, 1)
        
        # Estimation summary
        self._estimate_group = QGroupBox("Estimated Time to Completion")
        est_layout = QHBoxLayout(self._estimate_group)
        
        self.lbl_est_sessions = QLabel("—")
        self.lbl_est_hours = QLabel("—")
        self.lbl_est_days = QLabel("—")
        
        for label_name, label_widget in [
            ("Sessions Needed:", self.lbl_est_sessions),
            ("Hours Needed:", self.lbl_est_hours),
            ("Days (at 2h/day):", self.lbl_est_days),
        ]:
            v = QVBoxLayout()
            v.addWidget(QLabel(label_name))
            label_widget.setStyleSheet("font-size: 18px; font-weight: bold; color: #1b9e77;")
            v.addWidget(label_widget)
            est_layout.addLayout(v)
            est_layout.addSpacing(20)
        
        est_layout.addStretch(1)
        self._root.addWidget(self._estimate_group)
    
    def _load_data(self):
        """Load both interaction logs and match matrix data."""
        super()._load_data()
        
        # Load match matrix data
        try:
            from src.data.matches_matrix import load_match_matrix
            self._matrix = load_match_matrix()
        except Exception as e:
            self._matrix = None
            print(f"Failed to load match matrix: {e}")
        
        # Load query IDs (both active and all)
        try:
            from src.data.id_registry import list_ids
            self._active_query_ids = list_ids("Queries", exclude_silent=True)
            self._all_query_ids = list_ids("Queries", exclude_silent=False)
        except Exception as e:
            self._active_query_ids = []
            self._all_query_ids = []
            print(f"Failed to load queries: {e}")
    
    def render(self):
        if not self._matrix:
            self._chart.clear()
            self._chart.ax.text(0.5, 0.5, "Failed to load match data.\nCheck console for errors.",
                               ha="center", va="center", fontsize=12)
            self._chart.draw()
            return
        
        # Calculate key metrics
        # Silent queries = matched and merged (done)
        # Active queries = still need matching (remaining work)
        all_query_set = set(self._all_query_ids)
        active_query_set = set(self._active_query_ids)
        silent_query_ids = all_query_set - active_query_set
        
        total_queries = len(self._all_query_ids)  # All queries ever
        queries_matched = len(silent_query_ids)   # Matched & merged (silent)
        queries_remaining = len(self._active_query_ids)  # Still need work (active)
        
        total_galleries = len(self._matrix.gallery_ids)
        total_decisions = len(self._matrix.verdict_by_pair)
        
        # Calculate decision rate from logged sessions
        # Filter out idle sessions (app left open) using event density
        total_logged_time_min = 0
        total_logged_decisions = 0
        active_sessions = 0
        
        for sid, stats in self._session_stats.items():
            duration = stats.get("duration_minutes", 0)
            event_count = stats.get("event_count", 0)
            decisions = stats.get("decision_count", 0)
            
            # Skip test sessions and sessions with no meaningful duration
            if duration < 1 or "test" in sid.lower():
                continue
            
            # Calculate event density (events per minute)
            # If < 0.5 events/min over a long session, it was likely idle
            event_density = event_count / duration if duration > 0 else 0
            
            # Cap session duration at 60 min to avoid idle-time inflation
            # Also skip very low-density sessions (< 1 event per 2 min over 30+ min)
            if duration > 30 and event_density < 0.5:
                # Likely an idle session - cap at reasonable active time estimate
                effective_duration = min(duration, event_count * 2)  # ~2 min per event
            else:
                effective_duration = min(duration, 60)  # Cap at 1 hour max
            
            total_logged_time_min += effective_duration
            total_logged_decisions += decisions
            active_sessions += 1
        
        # Calculate decisions per hour based on ACTIVE time
        if total_logged_time_min > 5 and total_logged_decisions > 0:
            decisions_per_hour = (total_logged_decisions / total_logged_time_min) * 60
        else:
            # Fallback: use historical data (164 decisions for 100 matched queries)
            # Assume ~30 min active work per matched query
            if queries_matched > 0:
                decisions_per_hour = (total_decisions / queries_matched) * 2  # ~2 decisions/hour estimate
            else:
                decisions_per_hour = 10.0  # Default reasonable estimate
        
        # Average decisions needed per query to find a match
        if queries_matched > 0:
            avg_decisions_per_match = total_decisions / queries_matched
        else:
            avg_decisions_per_match = 3.0  # Default estimate
        
        # Estimate remaining work
        remaining_decisions = queries_remaining * avg_decisions_per_match
        
        if decisions_per_hour > 0:
            hours_remaining = remaining_decisions / decisions_per_hour
            avg_session_hours = 0.75  # Assume 45 min effective per session
            sessions_remaining = hours_remaining / avg_session_hours
            work_hours_per_day = 2.0
            days_remaining = hours_remaining / work_hours_per_day
        else:
            hours_remaining = float('inf')
            sessions_remaining = float('inf')
            days_remaining = float('inf')
        
        # Update summary labels
        if total_queries > 0:
            pct = 100 * queries_matched / total_queries
            self.lbl_queries_matched.setText(f"{queries_matched} / {total_queries} ({pct:.0f}%)")
        else:
            self.lbl_queries_matched.setText("0 / 0")
        
        self.lbl_queries_remaining.setText(str(queries_remaining))
        self.lbl_decisions_total.setText(str(total_decisions))
        
        if decisions_per_hour < 100:
            self.lbl_rate.setText(f"{decisions_per_hour:.1f}")
        else:
            self.lbl_rate.setText("—")
        
        # Update estimation labels
        if hours_remaining < float('inf') and queries_remaining > 0:
            self.lbl_est_sessions.setText(f"~{max(1, int(sessions_remaining))}")
            self.lbl_est_hours.setText(f"~{hours_remaining:.1f}")
            self.lbl_est_days.setText(f"~{max(1, int(days_remaining))}")
        elif queries_remaining == 0:
            self.lbl_est_sessions.setText("0")
            self.lbl_est_hours.setText("0")
            self.lbl_est_days.setText("Done! 🎉")
        else:
            self.lbl_est_sessions.setText("—")
            self.lbl_est_hours.setText("—")
            self.lbl_est_days.setText("—")
        
        # Render charts
        self._chart.clear()
        fig = self._chart.fig
        
        # Create two subplots
        ax1 = fig.add_subplot(121)  # Progress pie
        ax2 = fig.add_subplot(122)  # Trend over time
        
        # Progress pie chart
        if total_queries > 0:
            sizes = [queries_matched, queries_remaining]
            labels = [f"Matched\n({queries_matched})", f"Remaining\n({queries_remaining})"]
            colors = [COLORS["success"], COLORS["danger"]]
            explode = (0.02, 0)
            
            wedges, texts, autotexts = ax1.pie(
                sizes, explode=explode, labels=labels, colors=colors,
                autopct=lambda p: f"{p:.0f}%" if p > 3 else "",
                startangle=90, textprops={"fontsize": 10}
            )
            ax1.set_title("Query Matching Progress", fontsize=12, fontweight="bold")
        else:
            ax1.text(0.5, 0.5, "No queries found", ha="center", va="center")
            ax1.set_title("Query Matching Progress", fontsize=12, fontweight="bold")
        
        # Trend chart - cumulative decisions over time from logs
        decision_events = [e for e in self._events if e.get("event_type") == "decision_save"]
        
        if decision_events:
            # Sort by datetime
            decision_events_sorted = sorted(
                [e for e in decision_events if e.get("datetime")],
                key=lambda e: e.get("datetime")
            )
            
            if decision_events_sorted:
                times = []
                cumulative = []
                count = 0
                for e in decision_events_sorted:
                    dt = e.get("datetime")
                    if dt:
                        count += 1
                        times.append(dt)
                        cumulative.append(count)
                
                if times:
                    ax2.plot(times, cumulative, marker="o", color=COLORS["primary"], 
                            linewidth=2, markersize=5)
                    ax2.fill_between(times, cumulative, alpha=0.3, color=COLORS["primary"])
                    ax2.set_xlabel("Date")
                    ax2.set_ylabel("Cumulative Logged Decisions")
                    ax2.set_title("Decision Rate Trend", fontsize=12, fontweight="bold")
                    ax2.tick_params(axis="x", rotation=45)
                    
                    # Add trend line if we have enough data points
                    if len(times) >= 3:
                        try:
                            import numpy as np
                            x_numeric = [(t - times[0]).total_seconds() for t in times]
                            z = np.polyfit(x_numeric, cumulative, 1)
                            p = np.poly1d(z)
                            ax2.plot(times, [p(x) for x in x_numeric], "--", 
                                    color=COLORS["tertiary"], alpha=0.7, linewidth=2,
                                    label="Trend")
                            ax2.legend(fontsize=8)
                        except Exception:
                            pass  # Skip trend line if numpy not available or fit fails
                else:
                    ax2.text(0.5, 0.5, "No timestamped\ndecision events",
                            ha="center", va="center", fontsize=10, transform=ax2.transAxes)
                    ax2.set_title("Decision Rate Trend", fontsize=12, fontweight="bold")
            else:
                ax2.text(0.5, 0.5, "No timestamped\ndecision events",
                        ha="center", va="center", fontsize=10, transform=ax2.transAxes)
                ax2.set_title("Decision Rate Trend", fontsize=12, fontweight="bold")
        else:
            ax2.text(0.5, 0.5, "No decision events logged yet.\n\nStart making decisions\nto see trends.",
                    ha="center", va="center", fontsize=10, transform=ax2.transAxes)
            ax2.set_title("Decision Rate Trend", fontsize=12, fontweight="bold")
        
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        
        fig.tight_layout()
        self._chart.draw()
        
        # Update hint
        self.lbl_hint.setText(
            f"{total_queries} active queries, {total_galleries} galleries, "
            f"{total_decisions} total decisions | "
            f"Logged: {len(self._events)} events, {total_logged_decisions} decisions"
        )
    
    def export_rows(self) -> Tuple[List[str], List[List[str]]]:
        """Export work estimation data to CSV."""
        if not self._matrix:
            return [], []
        
        header = ["metric", "value"]
        
        # Silent queries = matched and merged
        all_query_set = set(self._all_query_ids)
        active_query_set = set(self._active_query_ids)
        silent_query_ids = all_query_set - active_query_set
        
        total_queries = len(self._all_query_ids)
        queries_matched = len(silent_query_ids)
        queries_remaining = len(self._active_query_ids)
        
        rows = [
            ["total_queries", total_queries],
            ["queries_matched_merged", queries_matched],
            ["queries_remaining", queries_remaining],
            ["total_decisions", len(self._matrix.verdict_by_pair)],
            ["total_galleries", len(self._matrix.gallery_ids)],
            ["logged_sessions", len(self._session_stats)],
            ["logged_events", len(self._events)],
        ]
        
        return header, rows

