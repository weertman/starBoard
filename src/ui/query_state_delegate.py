# src/ui/query_state_delegate.py
"""
Custom delegate for QComboBox that color-codes query items based on their
identification workflow state and displays image quality indicator symbols.

State-based background colors:
  - Default (no color): Query has not been attempted (no pins, no second-order labels)
  - Orange: Query has pins but no second-order labels (pinned but not attempted)
  - Yellow/Amber: Query has been attempted but has no positive match
  - Green: Query has at least one confirmed positive match (verdict = "yes")

Quality indicator symbols (appended after ID text):
  - ● (closed circle): Madreporite visibility
  - ○ (open circle): Anus visibility  
  - ★ (star): Postural visibility
  - Color gradient: Red (poor) → Yellow (medium) → Green (excellent)
  - Only shown if the annotation exists for that ID
"""
from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QModelIndex, QRectF
from PySide6.QtGui import QColor, QPalette, QPainter, QFont, QPen, QBrush
from PySide6.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem, QStyle

from src.data.compare_labels import load_latest_map_for_query
from src.data import archive_paths as ap
from src.data.csv_io import read_rows_multi, last_row_per_id


class QueryState(IntEnum):
    """Workflow state for a query."""
    NOT_ATTEMPTED = 0   # No pins, no second-order labels
    PINNED = 1          # Has pins but no second-order labels
    ATTEMPTED = 2       # Has labels but no "yes" verdict
    MATCHED = 3         # Has at least one "yes" verdict


# Custom item data roles
QUERY_STATE_ROLE = Qt.UserRole + 100
# Quality indicator roles (store normalized 0.0-1.0 values, or -1 for not present)
QUALITY_MADREPORITE_ROLE = Qt.UserRole + 101
QUALITY_ANUS_ROLE = Qt.UserRole + 102
QUALITY_POSTURE_ROLE = Qt.UserRole + 103

# Quality field names in metadata
QUALITY_FIELDS = ["madreporite_visibility", "anus_visibility", "postural_visibility"]
# Max values for normalization
QUALITY_MAX_VALUES = {
    "madreporite_visibility": 3,  # 0-3 (Not visible to Excellently visible)
    "anus_visibility": 3,          # 0-3 (Not visible to Excellently visible)
    "postural_visibility": 4,      # 0-4 (Very poor to Excellent)
}

# Symbols for each quality indicator
SYMBOL_MADREPORITE = "●"  # Closed circle
SYMBOL_ANUS = "○"          # Open circle
SYMBOL_POSTURE = "★"       # Star

# Color definitions for workflow state backgrounds
COLOR_NOT_ATTEMPTED = None  # Default/no special background
COLOR_PINNED = QColor(255, 200, 130)          # Light orange
COLOR_PINNED_DARK = QColor(200, 130, 50)      # Darker orange for dark mode
COLOR_ATTEMPTED = QColor(255, 235, 156)       # Light amber/yellow
COLOR_ATTEMPTED_DARK = QColor(180, 150, 50)   # Darker amber for dark mode
COLOR_MATCHED = QColor(180, 230, 180)         # Light green
COLOR_MATCHED_DARK = QColor(80, 140, 80)      # Darker green for dark mode


def _quality_to_color(normalized_value: float) -> QColor:
    """
    Convert a normalized quality value (0.0-1.0) to a color.
    Uses red → yellow → green gradient.
    
    0.0 = Red (poor quality)
    0.5 = Yellow (medium quality)
    1.0 = Green (excellent quality)
    """
    v = max(0.0, min(1.0, normalized_value))
    
    if v < 0.5:
        # Red to Yellow (0.0 -> 0.5)
        t = v / 0.5
        r = 220
        g = int(60 + 180 * t)  # 60 -> 240
        b = 60
    else:
        # Yellow to Green (0.5 -> 1.0)
        t = (v - 0.5) / 0.5
        r = int(220 - 160 * t)  # 220 -> 60
        g = int(240 - 40 * t)   # 240 -> 200
        b = 60
    
    return QColor(r, g, b)


def _has_pins(query_id: str) -> bool:
    """Check if a query has any first-order pins."""
    try:
        pins_path = ap.queries_root(prefer_new=True) / query_id / "_pins_first_order.json"
        if pins_path.exists():
            data = json.loads(pins_path.read_text(encoding="utf-8"))
            pins = data.get("pinned", [])
            return len(pins) > 0
    except Exception:
        pass
    return False


def get_query_state(query_id: str) -> QueryState:
    """
    Determine the workflow state of a query by checking its pins and second-order labels.
    
    Returns:
        QueryState.NOT_ATTEMPTED: No pins and no labels exist for this query
        QueryState.PINNED: Has pins but no second-order labels
        QueryState.ATTEMPTED: Has labels but no "yes" verdict
        QueryState.MATCHED: At least one "yes" verdict exists
    """
    if not query_id:
        return QueryState.NOT_ATTEMPTED
    
    try:
        latest = load_latest_map_for_query(query_id)
        
        if not latest:
            # No second-order labels - check for pins
            if _has_pins(query_id):
                return QueryState.PINNED
            return QueryState.NOT_ATTEMPTED
        
        # Check if any verdict is "yes"
        for row in latest.values():
            verdict = (row.get("verdict", "") or "").strip().lower()
            if verdict == "yes":
                return QueryState.MATCHED
        
        # Has labels but no positive match
        return QueryState.ATTEMPTED
        
    except Exception:
        return QueryState.NOT_ATTEMPTED


def get_query_states_batch(query_ids: list[str]) -> Dict[str, QueryState]:
    """
    Get states for multiple queries efficiently.
    
    Returns:
        Dict mapping query_id to QueryState
    """
    return {qid: get_query_state(qid) for qid in query_ids}


def _load_metadata_for_target(target: str) -> Dict[str, Dict[str, str]]:
    """
    Load metadata for all IDs of a target (Gallery or Queries).
    
    Returns:
        Dict mapping ID to metadata row dict
    """
    try:
        id_col = "gallery_id" if target.lower() == "gallery" else "query_id"
        csv_paths = ap.metadata_csv_paths_for_read(target)
        rows = read_rows_multi(csv_paths)
        return last_row_per_id(rows, id_col)
    except Exception:
        return {}


def get_quality_values(metadata_row: Dict[str, str]) -> Tuple[float, float, float]:
    """
    Extract normalized quality values from a metadata row.
    
    Returns:
        Tuple of (madreporite, anus, posture) values.
        Each value is normalized to 0.0-1.0, or -1.0 if not present.
    """
    results = []
    for field in QUALITY_FIELDS:
        raw = (metadata_row.get(field, "") or "").strip()
        if not raw:
            results.append(-1.0)  # Not present
        else:
            try:
                val = float(raw)
                max_val = QUALITY_MAX_VALUES.get(field, 4)
                normalized = val / max_val if max_val > 0 else 0.0
                results.append(max(0.0, min(1.0, normalized)))
            except (ValueError, TypeError):
                results.append(-1.0)
    
    return tuple(results)


def get_quality_for_ids(target: str, ids: List[str]) -> Dict[str, Tuple[float, float, float]]:
    """
    Get quality values for multiple IDs.
    
    Args:
        target: "Gallery" or "Queries"
        ids: List of IDs to look up
        
    Returns:
        Dict mapping ID to (madreporite, anus, posture) normalized values
    """
    metadata = _load_metadata_for_target(target)
    result = {}
    for id_str in ids:
        row = metadata.get(id_str, {})
        result[id_str] = get_quality_values(row)
    return result


class QueryStateDelegate(QStyledItemDelegate):
    """
    Custom delegate that draws query/gallery items with:
    1. Background colors based on identification workflow state
    2. Quality indicator symbols (●○★) with color coding
    """
    
    def __init__(self, parent=None, show_quality_symbols: bool = True):
        super().__init__(parent)
        self._show_quality_symbols = show_quality_symbols
    
    def _get_color_for_state(self, state: QueryState, is_dark: bool) -> Optional[QColor]:
        """Get the background color for a given state."""
        if state == QueryState.PINNED:
            return COLOR_PINNED_DARK if is_dark else COLOR_PINNED
        elif state == QueryState.ATTEMPTED:
            return COLOR_ATTEMPTED_DARK if is_dark else COLOR_ATTEMPTED
        elif state == QueryState.MATCHED:
            return COLOR_MATCHED_DARK if is_dark else COLOR_MATCHED
        return None
    
    def _draw_quality_symbols(self, painter: QPainter, rect, index: QModelIndex):
        """Draw the quality indicator symbols at the right side of the item."""
        # Get quality values from item data
        madreporite = index.data(QUALITY_MADREPORITE_ROLE)
        anus = index.data(QUALITY_ANUS_ROLE)
        posture = index.data(QUALITY_POSTURE_ROLE)
        
        # Check if any symbols should be drawn
        symbols_to_draw = []
        if madreporite is not None and madreporite >= 0:
            symbols_to_draw.append((SYMBOL_MADREPORITE, _quality_to_color(madreporite)))
        if anus is not None and anus >= 0:
            symbols_to_draw.append((SYMBOL_ANUS, _quality_to_color(anus)))
        if posture is not None and posture >= 0:
            symbols_to_draw.append((SYMBOL_POSTURE, _quality_to_color(posture)))
        
        if not symbols_to_draw:
            return
        
        painter.save()
        
        # Set up font for symbols
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        
        # Calculate positions (right-aligned with some padding)
        symbol_width = 14
        padding_right = 6
        spacing = 2
        padding_vertical = 2
        
        total_width = len(symbols_to_draw) * symbol_width + (len(symbols_to_draw) - 1) * spacing
        x_start = rect.right() - padding_right - total_width
        
        # Draw white background behind symbols
        bg_rect = QRectF(
            x_start - 3,
            rect.top() + padding_vertical,
            total_width + 6,
            rect.height() - 2 * padding_vertical
        )
        painter.fillRect(bg_rect, QColor(255, 255, 255))
        
        for i, (symbol, color) in enumerate(symbols_to_draw):
            x = x_start + i * (symbol_width + spacing)
            
            # Draw the symbol
            painter.setPen(QPen(color))
            symbol_rect = QRectF(x, rect.top(), symbol_width, rect.height())
            painter.drawText(symbol_rect, Qt.AlignCenter, symbol)
        
        painter.restore()
    
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        """Paint the item with state-based background color and quality symbols."""
        # Get the query state from item data
        state = index.data(QUERY_STATE_ROLE)
        
        if state is not None:
            # Determine if we're in a dark theme (simple heuristic)
            is_dark = option.palette.color(QPalette.Window).lightness() < 128
            
            # Get the appropriate background color
            bg_color = self._get_color_for_state(state, is_dark)
            
            if bg_color is not None:
                # Fill background
                painter.save()
                painter.fillRect(option.rect, bg_color)
                painter.restore()
        
        # Call parent to draw the text and other elements
        super().paint(painter, option, index)
        
        # Draw quality symbols on top
        if self._show_quality_symbols:
            self._draw_quality_symbols(painter, option.rect, index)
    
    def initStyleOption(self, option: QStyleOptionViewItem, index: QModelIndex):
        """Initialize style options, including background color for state."""
        super().initStyleOption(option, index)
        
        state = index.data(QUERY_STATE_ROLE)
        if state is not None:
            is_dark = option.palette.color(QPalette.Window).lightness() < 128
            bg_color = self._get_color_for_state(state, is_dark)
            
            if bg_color is not None:
                option.backgroundBrush = bg_color


def apply_query_states_to_combobox(combo, query_ids: list[str], states: Optional[Dict[str, QueryState]] = None):
    """
    Apply query state data to all items in a QComboBox.
    
    Args:
        combo: The QComboBox to update
        query_ids: List of query IDs in the same order as combo items
        states: Optional pre-computed states dict. If None, states will be computed.
    """
    if states is None:
        states = get_query_states_batch(query_ids)
    
    model = combo.model()
    for i, qid in enumerate(query_ids):
        if i < model.rowCount():
            idx = model.index(i, 0)
            state = states.get(qid, QueryState.NOT_ATTEMPTED)
            model.setData(idx, state, QUERY_STATE_ROLE)


def apply_quality_to_combobox(combo, ids: List[str], target: str, 
                               quality_data: Optional[Dict[str, Tuple[float, float, float]]] = None):
    """
    Apply quality indicator data to all items in a QComboBox.
    
    Args:
        combo: The QComboBox to update
        ids: List of IDs in the same order as combo items
        target: "Gallery" or "Queries"
        quality_data: Optional pre-computed quality dict. If None, will be loaded.
    """
    if quality_data is None:
        quality_data = get_quality_for_ids(target, ids)
    
    model = combo.model()
    for i, id_str in enumerate(ids):
        if i < model.rowCount():
            idx = model.index(i, 0)
            madreporite, anus, posture = quality_data.get(id_str, (-1.0, -1.0, -1.0))
            model.setData(idx, madreporite, QUALITY_MADREPORITE_ROLE)
            model.setData(idx, anus, QUALITY_ANUS_ROLE)
            model.setData(idx, posture, QUALITY_POSTURE_ROLE)
