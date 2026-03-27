# src/ui/tab_morphometric.py
"""
Morphometric Tab for starBoard.

Integrates webcam-based morphometric measurement functionality from the
starMorphometricTool, allowing live capture and metadata entry.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import date as _date
from pathlib import Path
from typing import Optional, Dict, Any, List

from PySide6.QtCore import Qt, QTimer, QDate, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QScrollArea,
    QLabel, QPushButton, QLineEdit, QComboBox, QDateEdit,
    QFormLayout, QSlider, QTextEdit, QMessageBox, QGroupBox,
    QSizePolicy, QCompleter, QFrame, QDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)

import numpy as np
import cv2

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import QTabWidget

from .collapsible import CollapsibleSection
from .metadata_form_v2 import MetadataFormV2
from src.ui.help_button import HelpButton, HELP_TEXTS
from src.data import archive_paths as ap
from src.data.csv_io import append_row, read_rows_multi, last_row_per_id, normalize_id_value
from src.data.id_registry import list_ids, id_exists
from src.data.metadata_history import (
    record_morphometric_import, get_current_metadata_for_gallery,
)
from src.data.ingest import ensure_encounter_name, place_images
from src.utils.interaction_logger import get_interaction_logger
from src.morphometric.data_bridge import list_mfolders, load_morphometrics_from_mfolder

logger = logging.getLogger("starBoard.ui.tab_morphometric")


def qdate_to_ymd(q) -> tuple:
    """Convert QDate or QDateEdit to (year, month, day)."""
    if hasattr(q, 'date'):
        q = q.date()
    return q.year(), q.month(), q.day()


# =============================================================================
# Checkerboard Preset Configuration
# =============================================================================

def _get_checkerboard_config_path() -> Path:
    """Get path to checkerboard presets JSON file."""
    # Store in starMorphometricTool directory
    return Path(__file__).parent.parent.parent / "starMorphometricTool" / "checkerboard_presets.json"


def _load_checkerboard_presets() -> Dict[str, List]:
    """
    Load checkerboard presets from file.
    
    Returns:
        Dictionary with 'rows', 'cols', 'square_sizes_mm' lists.
        Returns defaults if file doesn't exist.
    """
    defaults = {
        "rows": [8, 6, 7, 9],
        "cols": [10, 8, 11, 12],
        "square_sizes_mm": [25.0, 20.0, 30.0, 15.0]
    }
    
    config_path = _get_checkerboard_config_path()
    if not config_path.exists():
        return defaults
    
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        # Merge with defaults to ensure all keys exist
        for key in defaults:
            if key not in data or not data[key]:
                data[key] = defaults[key]
        return data
    except Exception as e:
        logger.warning("Failed to load checkerboard presets: %s", e)
        return defaults


def _save_checkerboard_presets(presets: Dict[str, List]) -> bool:
    """
    Save checkerboard presets to file.
    
    Args:
        presets: Dictionary with 'rows', 'cols', 'square_sizes_mm' lists.
    
    Returns:
        True if saved successfully.
    """
    config_path = _get_checkerboard_config_path()
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(presets, f, indent=2)
        logger.debug("Saved checkerboard presets to %s", config_path)
        return True
    except Exception as e:
        logger.warning("Failed to save checkerboard presets: %s", e)
        return False


def _add_preset_value(presets: Dict[str, List], key: str, value) -> bool:
    """
    Add a new value to presets if not already present.
    
    Args:
        presets: Presets dictionary to modify.
        key: Key ('rows', 'cols', or 'square_sizes_mm').
        value: Value to add.
    
    Returns:
        True if value was added (was new).
    """
    if key not in presets:
        presets[key] = []
    
    # Normalize value for comparison
    if key == "square_sizes_mm":
        value = float(value)
        # Check if already exists (with float tolerance)
        for existing in presets[key]:
            if abs(float(existing) - value) < 0.001:
                return False
    else:
        value = int(value)
        if value in presets[key]:
            return False
    
    # Add to front of list (most recently used first)
    presets[key].insert(0, value)
    
    # Keep list reasonable size (max 10 entries)
    presets[key] = presets[key][:10]
    
    return True


# =============================================================================
# User Initials Configuration
# =============================================================================

def _get_initials_config_path() -> Path:
    """Get path to user initials JSON file."""
    return Path(__file__).parent.parent.parent / "starMorphometricTool" / "user_initials.json"


def _load_saved_initials() -> List[str]:
    """
    Load saved user initials from file.
    
    Returns:
        List of initials strings (uppercase, 3 letters).
    """
    config_path = _get_initials_config_path()
    if not config_path.exists():
        return []
    
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        return data.get("initials", [])
    except Exception as e:
        logger.warning("Failed to load user initials: %s", e)
        return []


def _save_initials(initials_list: List[str]) -> bool:
    """
    Save user initials to file.
    
    Args:
        initials_list: List of initials strings.
    
    Returns:
        True if saved successfully.
    """
    config_path = _get_initials_config_path()
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump({"initials": initials_list}, f, indent=2)
        logger.debug("Saved user initials to %s", config_path)
        return True
    except Exception as e:
        logger.warning("Failed to save user initials: %s", e)
        return False


def _add_initials(initials: str) -> bool:
    """
    Add new initials to saved list if not already present.
    
    Args:
        initials: 3-letter initials string (will be uppercased).
    
    Returns:
        True if initials were added (was new).
    """
    initials = initials.strip().upper()
    if not initials or len(initials) != 3:
        return False
    
    saved = _load_saved_initials()
    if initials in saved:
        return False
    
    # Add to front (most recently used first)
    saved.insert(0, initials)
    
    # Keep reasonable size
    saved = saved[:20]
    
    _save_initials(saved)
    return True


class TabMorphometric(QWidget):
    """
    Morphometric capture tab integrating starMorphometricTool functionality.
    
    Provides:
    - Webcam capture with checkerboard calibration
    - YOLO-based star detection and segmentation
    - Interactive morphometric analysis
    - Metadata entry with auto-population from measurements
    - Dual-save to morphometric storage and starBoard archive
    """
    
    # Signal emitted when data is saved (for cross-tab refresh)
    dataSaved = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TabMorphometric")
        
        self._ilog = get_interaction_logger()
        
        # State
        self._camera_adapter = None
        self._detection_adapter = None
        self._analysis_adapter = None
        self._stream_active = False
        self._yolo_active = False
        self._camera_config_prompted = False
        self._saved_camera_config: Optional[Dict[str, Any]] = None
        self._last_frame = None
        self._captured_frame = None  # Frame captured during detection (frozen)
        self._corrected_detection = None
        self._current_mfolder = None
        self._edit_mfolder: Optional[Path] = None
        self._measurement_rows: List[Dict[str, Any]] = []
        self._filtered_measurement_rows: List[Dict[str, Any]] = []
        self._loading_measurement = False
        
        # Visualization state
        self._corrected_object_rgb: Optional[np.ndarray] = None
        self._center: Optional[tuple] = None
        self._arm_data: Optional[List] = None
        self._ellipse_data: Optional[tuple] = None
        self._morphometrics_data: Optional[Dict[str, Any]] = None
        self._polar_canvas_available = False
        
        # Timer for camera feed
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)
        
        # Build UI
        self._build_ui()
        
        # Defer heavy initialization
        QTimer.singleShot(100, self._deferred_init)
    
    def _deferred_init(self):
        """Initialize adapters after UI is ready."""
        try:
            from src.morphometric import is_available, get_camera_adapter, get_detection_adapter, get_analysis_adapter
            
            if not is_available():
                self._show_unavailable_message()
                return
            
            self._camera_adapter = get_camera_adapter()
            self._detection_adapter = get_detection_adapter()
            self._analysis_adapter = get_analysis_adapter()
            self._saved_camera_config = self._load_saved_camera_config()
            self._update_camera_status("inactive")
            
            # Load YOLO model
            if self._detection_adapter.load_yolo_model():
                logger.info("YOLO model loaded")
            else:
                logger.warning("Failed to load YOLO model")
            
            # Populate ID list
            self._refresh_id_list()
            
            # Populate initials from existing metadata
            self._populate_initials()
            
        except ImportError as e:
            logger.error("Morphometric module not available: %s", e)
            self._show_unavailable_message()
        except Exception as e:
            logger.exception("Error initializing morphometric tab: %s", e)

    def _load_saved_camera_config(self) -> Optional[Dict[str, Any]]:
        """Load saved camera settings without opening hardware."""
        try:
            from src.morphometric import _ensure_morphometric_path

            _ensure_morphometric_path()
            from camera.config import load_camera_config

            return load_camera_config()
        except Exception as e:
            logger.debug("Unable to load saved camera config: %s", e)
            return None

    def _get_camera_dialog_config(self) -> tuple[Optional[Dict[str, Any]], bool]:
        """Return config defaults for the camera dialog and whether to preselect."""
        if self._camera_adapter is not None and self._camera_adapter.config:
            return dict(self._camera_adapter.config), self._camera_adapter.is_available

        if self._saved_camera_config is None:
            self._saved_camera_config = self._load_saved_camera_config()
        if self._saved_camera_config:
            return dict(self._saved_camera_config), False

        return None, False
    
    def _show_unavailable_message(self):
        """Show message that morphometric features are unavailable."""
        self.lbl_camera.setText(
            "Morphometric features unavailable.\n\n"
            "Required dependencies:\n"
            "- opencv-python\n"
            "- ultralytics\n"
            "- scipy\n\n"
            "Install with: pip install opencv-python ultralytics scipy"
        )
    
    # =========================================================================
    # UI Building
    # =========================================================================
    
    def _build_ui(self):
        """Build the tab UI."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Main horizontal splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel (controls)
        left_widget = self._build_left_panel()
        left_widget.setMaximumWidth(450)
        left_widget.setMinimumWidth(300)
        
        # Right panel (displays)
        right_widget = self._build_right_panel()
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([350, 900])
        
        main_layout.addWidget(splitter)
    
    def _build_left_panel(self) -> QWidget:
        """Build the left control panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        
        # Camera Controls Section
        sec_camera = CollapsibleSection("Camera Controls", start_collapsed=False)
        sec_camera.setContent(self._build_camera_controls())
        _hc = QHBoxLayout(); _hc.addWidget(sec_camera); _hc.addWidget(HelpButton(HELP_TEXTS.get('morph_camera', '')), 0, Qt.AlignTop); layout.addLayout(_hc)
        
        # Archive Selection Section
        sec_archive = CollapsibleSection("Archive Selection", start_collapsed=False)
        sec_archive.setContent(self._build_archive_selection())
        _ha = QHBoxLayout(); _ha.addWidget(sec_archive); _ha.addWidget(HelpButton(HELP_TEXTS.get('morph_archive', '')), 0, Qt.AlignTop); layout.addLayout(_ha)

        # Saved Measurements Section
        sec_saved = CollapsibleSection("Saved Measurements", start_collapsed=True)
        sec_saved.setContent(self._build_saved_measurements())
        _hsv = QHBoxLayout(); _hsv.addWidget(sec_saved); _hsv.addWidget(HelpButton(HELP_TEXTS.get('morph_saved', '')), 0, Qt.AlignTop); layout.addLayout(_hsv)
        
        # Metadata Form Section
        sec_metadata = CollapsibleSection("Metadata", start_collapsed=False)
        self.meta_form = MetadataFormV2()
        self.meta_form.set_target("Gallery")
        sec_metadata.setContent(self.meta_form)
        _hmt = QHBoxLayout(); _hmt.addWidget(sec_metadata, 1); _hmt.addWidget(HelpButton(HELP_TEXTS.get('morph_metadata', '')), 0, Qt.AlignTop); layout.addLayout(_hmt)
        
        # Analysis Controls Section
        sec_analysis = CollapsibleSection("Analysis Controls", start_collapsed=True)
        sec_analysis.setContent(self._build_analysis_controls())
        _han = QHBoxLayout(); _han.addWidget(sec_analysis); _han.addWidget(HelpButton(HELP_TEXTS.get('morph_analysis', '')), 0, Qt.AlignTop); layout.addLayout(_han)
        
        # Action Buttons Section
        sec_actions = CollapsibleSection("Actions", start_collapsed=False)
        sec_actions.setContent(self._build_action_buttons())
        _hac = QHBoxLayout(); _hac.addWidget(sec_actions); _hac.addWidget(HelpButton(HELP_TEXTS.get('morph_actions', '')), 0, Qt.AlignTop); layout.addLayout(_hac)
        
        layout.addStretch()
        scroll.setWidget(container)
        return scroll
    
    def _build_camera_controls(self) -> QWidget:
        """Build camera control widgets."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        # Camera status
        status_row = QHBoxLayout()
        self.lbl_camera_status = QLabel("● Initializing...")
        self.lbl_camera_status.setStyleSheet("color: orange; font-size: 10px;")
        status_row.addWidget(self.lbl_camera_status)
        status_row.addStretch()
        self.btn_config_camera = QPushButton("Configure Camera")
        self.btn_config_camera.setToolTip("Choose camera device, backend, codec, and resolution")
        self.btn_config_camera.clicked.connect(self._on_configure_camera)
        self.btn_config_camera.setEnabled(False)
        status_row.addWidget(self.btn_config_camera)
        layout.addLayout(status_row)
        
        # Checkerboard config - editable combo boxes with presets
        form = QFormLayout()
        form.setSpacing(4)
        
        presets = _load_checkerboard_presets()
        
        self.cmb_rows = QComboBox()
        self.cmb_rows.setEditable(True)
        self.cmb_rows.setFixedWidth(70)
        for val in presets.get("rows", [8]):
            self.cmb_rows.addItem(str(val))
        if self.cmb_rows.count() == 0:
            self.cmb_rows.addItem("8")
        form.addRow("Rows:", self.cmb_rows)
        
        self.cmb_cols = QComboBox()
        self.cmb_cols.setEditable(True)
        self.cmb_cols.setFixedWidth(70)
        for val in presets.get("cols", [10]):
            self.cmb_cols.addItem(str(val))
        if self.cmb_cols.count() == 0:
            self.cmb_cols.addItem("10")
        form.addRow("Columns:", self.cmb_cols)
        
        self.cmb_square_size = QComboBox()
        self.cmb_square_size.setEditable(True)
        self.cmb_square_size.setFixedWidth(70)
        for val in presets.get("square_sizes_mm", [25.0]):
            self.cmb_square_size.addItem(str(val))
        if self.cmb_square_size.count() == 0:
            self.cmb_square_size.addItem("25")
        form.addRow("Square (mm):", self.cmb_square_size)
        
        layout.addLayout(form)
        
        # Camera buttons
        btn_row1 = QHBoxLayout()
        self.btn_start_stream = QPushButton("Start Stream")
        self.btn_start_stream.clicked.connect(self._on_start_stream)
        self.btn_stop_stream = QPushButton("Stop Stream")
        self.btn_stop_stream.clicked.connect(self._on_stop_stream)
        self.btn_stop_stream.setEnabled(False)
        btn_row1.addWidget(self.btn_start_stream)
        btn_row1.addWidget(self.btn_stop_stream)
        layout.addLayout(btn_row1)
        
        btn_row2 = QHBoxLayout()
        self.btn_detect_board = QPushButton("Detect Board")
        self.btn_detect_board.clicked.connect(self._on_detect_checkerboard)
        self.btn_clear_board = QPushButton("Clear Board")
        self.btn_clear_board.clicked.connect(self._on_clear_checkerboard)
        self.btn_clear_board.setEnabled(False)
        btn_row2.addWidget(self.btn_detect_board)
        btn_row2.addWidget(self.btn_clear_board)
        layout.addLayout(btn_row2)
        
        btn_row3 = QHBoxLayout()
        self.btn_start_yolo = QPushButton("Start Detection")
        self.btn_start_yolo.clicked.connect(self._on_start_yolo)
        self.btn_stop_yolo = QPushButton("Stop Detection")
        self.btn_stop_yolo.clicked.connect(self._on_stop_yolo)
        self.btn_stop_yolo.setEnabled(False)
        btn_row3.addWidget(self.btn_start_yolo)
        btn_row3.addWidget(self.btn_stop_yolo)
        layout.addLayout(btn_row3)
        
        return widget
    
    def _build_archive_selection(self) -> QWidget:
        """Build archive selection widgets."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        form = QFormLayout()
        form.setSpacing(4)
        
        # Target (Gallery/Queries)
        self.cmb_target = QComboBox()
        self.cmb_target.addItems(["Gallery", "Queries"])
        self.cmb_target.currentIndexChanged.connect(self._on_target_changed)
        form.addRow("Archive:", self.cmb_target)
        
        # ID selection (searchable)
        self.cmb_id = QComboBox()
        self.cmb_id.setEditable(True)
        self.cmb_id.setInsertPolicy(QComboBox.NoInsert)
        # Create completer with proper search settings
        id_completer = QCompleter()
        id_completer.setFilterMode(Qt.MatchContains)
        id_completer.setCompletionMode(QCompleter.PopupCompletion)
        id_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.cmb_id.setCompleter(id_completer)
        self.cmb_id.currentIndexChanged.connect(self._on_id_changed)
        form.addRow("ID:", self.cmb_id)
        
        # New ID input
        self.edit_new_id = QLineEdit()
        self.edit_new_id.setPlaceholderText("New ID")
        self.edit_new_id.setVisible(False)
        form.addRow("", self.edit_new_id)
        
        # Encounter date
        self.date_encounter = QDateEdit()
        self.date_encounter.setCalendarPopup(True)
        self.date_encounter.setDate(QDate.currentDate())
        form.addRow("Date:", self.date_encounter)
        
        # User initials (searchable combobox)
        self.cmb_initials = QComboBox()
        self.cmb_initials.setEditable(True)
        self.cmb_initials.setInsertPolicy(QComboBox.NoInsert)
        init_completer = QCompleter()
        init_completer.setFilterMode(Qt.MatchContains)
        init_completer.setCompletionMode(QCompleter.PopupCompletion)
        init_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.cmb_initials.setCompleter(init_completer)
        self.cmb_initials.setFixedWidth(80)
        form.addRow("Initials:", self.cmb_initials)
        
        layout.addLayout(form)
        
        # Action buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        
        self.btn_new_query = QPushButton("New Query")
        self.btn_new_query.setToolTip("Create a new query entry")
        self.btn_new_query.clicked.connect(self._on_new_query)
        btn_row.addWidget(self.btn_new_query)
        
        self.btn_reset_metadata = QPushButton("Reset Metadata")
        self.btn_reset_metadata.setToolTip("Clear all metadata fields to defaults")
        self.btn_reset_metadata.clicked.connect(self._on_reset_metadata)
        btn_row.addWidget(self.btn_reset_metadata)
        
        layout.addLayout(btn_row)
        
        return widget

    def _build_saved_measurements(self) -> QWidget:
        """Build saved-measurement browser widgets."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        search_row = QHBoxLayout()
        search_row.setSpacing(4)
        self.edit_measurement_search = QLineEdit()
        self.edit_measurement_search.setPlaceholderText("Search ID, location, folder...")
        self.edit_measurement_search.textChanged.connect(self._apply_saved_measurement_filter)
        search_row.addWidget(self.edit_measurement_search, 1)

        self.cmb_measurement_filter = QComboBox()
        self.cmb_measurement_filter.addItems(["All", "Gallery", "Query"])
        self.cmb_measurement_filter.currentIndexChanged.connect(self._apply_saved_measurement_filter)
        search_row.addWidget(self.cmb_measurement_filter)
        layout.addLayout(search_row)

        self.tbl_measurements = QTableWidget(0, 4)
        self.tbl_measurements.setHorizontalHeaderLabels(["Date", "ID", "Location", "Folder"])
        self.tbl_measurements.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_measurements.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_measurements.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_measurements.setAlternatingRowColors(True)
        self.tbl_measurements.verticalHeader().setVisible(False)
        header = self.tbl_measurements.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.tbl_measurements.doubleClicked.connect(self._open_selected_measurement)
        layout.addWidget(self.tbl_measurements)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_refresh_measurements = QPushButton("Refresh")
        self.btn_refresh_measurements.clicked.connect(self._refresh_saved_measurements)
        btn_row.addWidget(self.btn_refresh_measurements)

        self.btn_open_measurement = QPushButton("Open Selected")
        self.btn_open_measurement.clicked.connect(self._open_selected_measurement)
        btn_row.addWidget(self.btn_open_measurement)
        layout.addLayout(btn_row)

        self.lbl_measurement_status = QLabel("Not editing a historical measurement.")
        self.lbl_measurement_status.setWordWrap(True)
        self.lbl_measurement_status.setStyleSheet("color: #999; font-size: 10px;")
        layout.addWidget(self.lbl_measurement_status)

        self._refresh_saved_measurements()
        return widget

    def _refresh_saved_measurements(self):
        """Refresh the saved-measurements cache from disk."""
        self._measurement_rows = []
        try:
            from src.morphometric import get_measurements_root
            for mfolder in list_mfolders(get_measurements_root()):
                row = self._build_measurement_row(mfolder)
                if row:
                    self._measurement_rows.append(row)
        except Exception as e:
            logger.warning("Failed to refresh saved measurements: %s", e)
        self._apply_saved_measurement_filter()

    def _build_measurement_row(self, mfolder: Path) -> Optional[Dict[str, Any]]:
        """Build one row for the saved-measurements table."""
        try:
            identity_type = mfolder.parents[2].name.lower()
            identity_id = mfolder.parents[1].name
            date_label = mfolder.parents[0].name
        except IndexError:
            return None

        morph_data = load_morphometrics_from_mfolder(mfolder) or {}
        location = str(morph_data.get("location", "")).strip()
        if not location:
            detection_path = mfolder / "corrected_detection.json"
            if detection_path.exists():
                try:
                    with detection_path.open("r", encoding="utf-8") as f:
                        detection_data = json.load(f)
                    location = str(detection_data.get("location", "")).strip()
                except Exception:
                    location = ""

        return {
            "mfolder": mfolder,
            "identity_type": identity_type,
            "identity_id": identity_id,
            "date_label": date_label,
            "location": location,
            "folder_name": mfolder.name,
            "identity_label": f"{identity_type.title()}:{identity_id}",
        }

    def _apply_saved_measurement_filter(self):
        """Apply target/search filter to saved measurements."""
        if not hasattr(self, "tbl_measurements"):
            return

        query = self.edit_measurement_search.text().strip().lower()
        scope_text = self.cmb_measurement_filter.currentText().strip().lower()
        scope = None if scope_text == "all" else scope_text

        filtered: List[Dict[str, Any]] = []
        for row in self._measurement_rows:
            if scope and row["identity_type"] != scope:
                continue
            haystack = " ".join([
                row.get("identity_id", ""),
                row.get("identity_label", ""),
                row.get("location", ""),
                row.get("folder_name", ""),
                row.get("date_label", ""),
            ]).lower()
            if query and query not in haystack:
                continue
            filtered.append(row)

        self._filtered_measurement_rows = filtered
        self.tbl_measurements.setRowCount(len(filtered))

        for r, row in enumerate(filtered):
            values = [
                row["date_label"],
                row["identity_label"],
                row["location"],
                row["folder_name"],
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if c == 0:
                    item.setData(Qt.UserRole, str(row["mfolder"]))
                self.tbl_measurements.setItem(r, c, item)

        self.btn_open_measurement.setEnabled(len(filtered) > 0)
        if filtered:
            self.tbl_measurements.selectRow(0)

    def _selected_saved_measurement(self) -> Optional[Dict[str, Any]]:
        """Return selected saved-measurement row metadata."""
        if not hasattr(self, "tbl_measurements"):
            return None
        selection_model = self.tbl_measurements.selectionModel()
        if selection_model is None:
            return None
        selected = selection_model.selectedRows()
        if not selected:
            return None
        row_index = selected[0].row()
        if row_index < 0 or row_index >= len(self._filtered_measurement_rows):
            return None
        return self._filtered_measurement_rows[row_index]

    def _open_selected_measurement(self, *_args):
        """Open selected historical measurement into active analysis state."""
        selected = self._selected_saved_measurement()
        if not selected:
            QMessageBox.information(self, "Open Measurement", "Select a saved measurement first.")
            return
        self._load_measurement_folder(selected["mfolder"])

    def _set_edit_context(self, mfolder: Optional[Path]):
        """Set or clear edit context for historical measurement mode."""
        self._edit_mfolder = mfolder
        if mfolder is None:
            self.lbl_measurement_status.setText("Not editing a historical measurement.")
            self.lbl_measurement_status.setStyleSheet("color: #999; font-size: 10px;")
            self.btn_save.setText("Save to starBoard")
            return

        self.lbl_measurement_status.setText(f"Editing historical measurement:\n{mfolder}")
        self.lbl_measurement_status.setStyleSheet("color: #2e7d32; font-size: 10px;")
        self.btn_save.setText("Resave to starBoard")

    def _clear_edit_context(self):
        """Exit historical edit context safely."""
        if self._edit_mfolder is None:
            return
        self._set_edit_context(None)

    def _set_corrected_object_rgb(self, corrected_object: Optional[np.ndarray]):
        """Normalize corrected-object image to RGB visualization buffer."""
        if corrected_object is None:
            self._corrected_object_rgb = None
            return
        if len(corrected_object.shape) == 2:
            self._corrected_object_rgb = cv2.cvtColor(corrected_object, cv2.COLOR_GRAY2RGB)
        elif corrected_object.shape[2] == 4:
            self._corrected_object_rgb = cv2.cvtColor(corrected_object, cv2.COLOR_BGRA2RGB)
        else:
            self._corrected_object_rgb = cv2.cvtColor(corrected_object, cv2.COLOR_BGR2RGB)

    def _load_measurement_folder(self, mfolder: Path):
        """Load saved measurement files into current tab analysis pipeline."""
        mfolder = Path(mfolder)
        if not mfolder.exists():
            QMessageBox.warning(self, "Open Measurement", f"Folder not found:\n{mfolder}")
            return

        required_paths = [
            mfolder / "corrected_detection.json",
            mfolder / "corrected_mask.png",
            mfolder / "corrected_object.png",
            mfolder / "morphometrics.json",
        ]
        missing = [p.name for p in required_paths if not p.exists()]
        if missing:
            QMessageBox.warning(
                self,
                "Open Measurement",
                "Selected folder is missing required files:\n" + "\n".join(missing),
            )
            return

        try:
            with (mfolder / "corrected_detection.json").open("r", encoding="utf-8") as f:
                detection_data = json.load(f)
            morph_data = load_morphometrics_from_mfolder(mfolder) or {}

            corrected_mask = cv2.imread(str(mfolder / "corrected_mask.png"), cv2.IMREAD_GRAYSCALE)
            corrected_object = cv2.imread(str(mfolder / "corrected_object.png"), cv2.IMREAD_UNCHANGED)
            raw_frame = cv2.imread(str(mfolder / "raw_frame.png"), cv2.IMREAD_COLOR)

            if corrected_mask is None or corrected_object is None:
                QMessageBox.warning(self, "Open Measurement", "Failed to read corrected images from folder.")
                return

            corrected_detection = dict(detection_data)
            corrected_detection["corrected_mask"] = (corrected_mask > 0).astype(np.uint8)
            corrected_detection["corrected_object"] = corrected_object
            corrected_detection["mm_per_pixel"] = float(corrected_detection.get("mm_per_pixel", 1.0) or 1.0)

            self._corrected_detection = corrected_detection
            self._captured_frame = raw_frame
            self._current_mfolder = mfolder
            self._set_corrected_object_rgb(corrected_object)
            self._display_result_fallback()

            identity_type = str(
                morph_data.get("identity_type")
                or detection_data.get("identity_type")
                or mfolder.parents[2].name
            ).lower()
            identity_id = str(
                morph_data.get("identity_id")
                or detection_data.get("identity_id")
                or mfolder.parents[1].name
            ).strip()

            self._loading_measurement = True
            try:
                self.cmb_target.setCurrentText("Gallery" if identity_type == "gallery" else "Queries")
                idx = self.cmb_id.findText(identity_id)
                if idx >= 0:
                    self.cmb_id.setCurrentIndex(idx)
                else:
                    self.cmb_id.setCurrentIndex(0)
                    self.edit_new_id.setText(identity_id)
                    self.meta_form.set_id_value(identity_id)

                loaded_location = str(
                    morph_data.get("location") or detection_data.get("location") or ""
                ).strip()
                if loaded_location:
                    self.meta_form.apply_values({"location": loaded_location})

                initials = str(morph_data.get("user_initials", "")).strip().upper()
                if initials:
                    self.cmb_initials.setCurrentText(initials)

                date_text = mfolder.parents[0].name
                date_val = QDate.fromString(date_text, "MM_dd_yyyy")
                if date_val.isValid():
                    self.date_encounter.setDate(date_val)
            finally:
                self._loading_measurement = False

            self._on_analyze()
            if self._arm_data is None:
                return

            saved_rotation = int(morph_data.get("arm_rotation", 0) or 0)
            self.slider_rotation.setValue(max(0, min(saved_rotation, self.slider_rotation.maximum())))

            self._set_edit_context(mfolder)
            self.btn_save.setEnabled(True)

            QMessageBox.information(
                self,
                "Measurement Loaded",
                "Historical measurement loaded.\n"
                "You can now edit arm peaks/rotation and resave in place.",
            )
        except Exception as e:
            logger.exception("Failed to load historical measurement %s: %s", mfolder, e)
            QMessageBox.critical(self, "Open Measurement", f"Failed to load measurement:\n{e}")
    
    def _build_analysis_controls(self) -> QWidget:
        """Build analysis control widgets."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        # Smoothing slider
        self.lbl_smoothing = QLabel("Smoothing: 5")
        self.slider_smoothing = QSlider(Qt.Horizontal)
        self.slider_smoothing.setRange(1, 15)
        self.slider_smoothing.setValue(5)
        self.slider_smoothing.valueChanged.connect(self._on_smoothing_changed)
        layout.addWidget(self.lbl_smoothing)
        layout.addWidget(self.slider_smoothing)
        
        # Prominence slider
        self.lbl_prominence = QLabel("Prominence: 0.05")
        self.slider_prominence = QSlider(Qt.Horizontal)
        self.slider_prominence.setRange(1, 100)
        self.slider_prominence.setValue(5)
        self.slider_prominence.valueChanged.connect(self._on_prominence_changed)
        layout.addWidget(self.lbl_prominence)
        layout.addWidget(self.slider_prominence)
        
        # Distance slider
        self.lbl_distance = QLabel("Distance: 5")
        self.slider_distance = QSlider(Qt.Horizontal)
        self.slider_distance.setRange(0, 15)
        self.slider_distance.setValue(5)
        self.slider_distance.valueChanged.connect(self._on_distance_changed)
        layout.addWidget(self.lbl_distance)
        layout.addWidget(self.slider_distance)
        
        # Arm rotation slider
        self.lbl_rotation = QLabel("Arm Rotation: 0")
        self.slider_rotation = QSlider(Qt.Horizontal)
        self.slider_rotation.setRange(0, 24)
        self.slider_rotation.setValue(0)
        self.slider_rotation.valueChanged.connect(self._on_rotation_changed)
        layout.addWidget(self.lbl_rotation)
        layout.addWidget(self.slider_rotation)
        
        return widget
    
    def _build_action_buttons(self) -> QWidget:
        """Build action button widgets."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        
        # Run All button (Detect → Morph sequence)
        self.btn_run_all = QPushButton("Run All")
        self.btn_run_all.setToolTip("Run Detect → Morph in sequence")
        self.btn_run_all.clicked.connect(self._on_run_all)
        self.btn_run_all.setEnabled(False)
        self.btn_run_all.setStyleSheet("QPushButton { background-color: #7b1fa2; color: white; }")
        layout.addWidget(self.btn_run_all)
        
        # Save button
        self.btn_save = QPushButton("Save to starBoard")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("QPushButton { background-color: #2e7d32; color: white; font-weight: bold; }")
        layout.addWidget(self.btn_save)
        
        return widget
    
    def _build_right_panel(self) -> QWidget:
        """Build the right display panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Vertical splitter for top (camera) and bottom (results)
        v_splitter = QSplitter(Qt.Vertical)
        
        # Camera feed
        self.lbl_camera = QLabel("Camera Feed")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setMinimumSize(640, 480)
        self.lbl_camera.setStyleSheet("QLabel { background-color: #1a1a1a; color: #666; }")
        v_splitter.addWidget(self.lbl_camera)
        
        # Bottom splitter for results
        h_splitter = QSplitter(Qt.Horizontal)
        
        # Detection result with matplotlib figure for annotations
        self._result_fig, self._result_ax = plt.subplots(figsize=(4, 4), facecolor='#1a1a1a')
        self._result_ax.set_facecolor('#1a1a1a')
        self._result_ax.axis('off')
        self._result_ax.text(0.5, 0.5, "Detection Result", ha='center', va='center', 
                            color='#666', fontsize=12, transform=self._result_ax.transAxes)
        self._result_canvas = FigureCanvas(self._result_fig)
        self._result_canvas.setMinimumSize(250, 250)
        self._result_fig.tight_layout(pad=0)
        h_splitter.addWidget(self._result_canvas)
        
        # Polar plot placeholder (will be replaced with actual PolarCanvas if available)
        self.polar_widget = QLabel("Polar Plot")
        self.polar_widget.setAlignment(Qt.AlignCenter)
        self.polar_widget.setMinimumSize(250, 250)
        self.polar_widget.setStyleSheet("QLabel { background-color: #1a1a1a; color: #666; }")
        h_splitter.addWidget(self.polar_widget)
        
        # Try to create actual PolarCanvas
        self._init_polar_canvas()
        
        v_splitter.addWidget(h_splitter)
        v_splitter.setSizes([500, 350])
        
        layout.addWidget(v_splitter)
        return widget
    
    def _init_polar_canvas(self):
        """Initialize the polar canvas if available."""
        try:
            from src.morphometric import _ensure_morphometric_path
            _ensure_morphometric_path()
            from ui.components.polar_canvas import PolarCanvas
            
            # Create the actual polar canvas
            self.polar_canvas = PolarCanvas()
            self.polar_canvas.peaksChanged.connect(self._on_peaks_changed)
            self.polar_canvas.setMinimumSize(250, 250)
            
            # Replace placeholder in the splitter
            # QSplitter uses replaceWidget
            parent = self.polar_widget.parent()
            if isinstance(parent, QSplitter):
                idx = parent.indexOf(self.polar_widget)
                if idx >= 0:
                    parent.replaceWidget(idx, self.polar_canvas)
                    self.polar_widget.deleteLater()
                    self.polar_widget = self.polar_canvas  # Update reference
            
            self._polar_canvas_available = True
            logger.info("PolarCanvas initialized and inserted into UI")
            
        except ImportError as e:
            self._polar_canvas_available = False
            logger.debug("PolarCanvas not available: %s", e)
        except Exception as e:
            self._polar_canvas_available = False
            logger.warning("Failed to initialize PolarCanvas: %s", e)
    
    # =========================================================================
    # Camera Control Handlers
    # =========================================================================

    def _sync_detection_camera_info(self) -> None:
        """Pass camera config/device info to detection adapter for calibration identity."""
        if self._camera_adapter is None or self._detection_adapter is None:
            return
        try:
            if hasattr(self._detection_adapter, "set_camera_info"):
                config = self._camera_adapter.config or {}
                device_info = (
                    self._camera_adapter.get_device_info()
                    if hasattr(self._camera_adapter, "get_device_info")
                    else {}
                )
                self._detection_adapter.set_camera_info(config, device_info)
        except Exception as e:
            logger.debug("Unable to sync camera info to detection adapter: %s", e)

    def _prompt_configure_camera_once(self) -> None:
        """Show a one-time guidance prompt when auto-detection fails."""
        if self._camera_config_prompted:
            return
        self._camera_config_prompted = True
        QMessageBox.information(
            self,
            "Camera Not Detected",
            "No camera was auto-detected.\n\n"
            "Click 'Configure Camera' to select a device and capture settings.",
        )
    
    def _update_camera_status(self, state: str):
        """Update camera status indicator."""
        if hasattr(self, "btn_config_camera"):
            self.btn_config_camera.setEnabled(self._camera_adapter is not None)
        if hasattr(self, "btn_stop_stream"):
            self.btn_stop_stream.setEnabled(self._stream_active)

        if state == "ready":
            self.lbl_camera_status.setText("● Camera Ready")
            self.lbl_camera_status.setStyleSheet("color: green; font-size: 10px;")
            self.btn_start_stream.setEnabled(not self._stream_active)
            if not self._stream_active:
                self.lbl_camera.setText(
                    "Camera configured and ready.\n"
                    "Click 'Start Stream' to begin live preview."
                )
            return

        self.btn_start_stream.setEnabled(True)

        if state == "inactive":
            self.lbl_camera_status.setText("● Camera Not Started")
            self.lbl_camera_status.setStyleSheet("color: #b8860b; font-size: 10px;")
            if self._saved_camera_config:
                self.lbl_camera.setText(
                    "Saved camera settings are available but inactive.\n"
                    "Use 'Configure Camera' to review the device, then click 'Start Stream'."
                )
            else:
                self.lbl_camera.setText(
                    "Camera access stays off until you choose a device.\n"
                    "Use 'Configure Camera' to set one up, then click 'Start Stream'."
                )
        else:
            self.lbl_camera_status.setText("● No Camera")
            self.lbl_camera_status.setStyleSheet("color: red; font-size: 10px;")
            self.lbl_camera.setText(
                "No camera is active.\n"
                "Use 'Configure Camera' to choose a device before starting the stream."
            )

    def _on_configure_camera(self):
        """Open camera configuration dialog and apply settings."""
        self._ilog.log("button_click", "btn_config_camera", value="clicked")

        if self._camera_adapter is None:
            QMessageBox.warning(self, "Camera Error", "Camera subsystem is not initialized yet.")
            return

        try:
            from src.morphometric import _ensure_morphometric_path

            _ensure_morphometric_path()
            from ui.camera_config_dialog import CameraConfigDialog
        except Exception as e:
            logger.exception("Failed to load camera configuration dialog: %s", e)
            QMessageBox.warning(
                self,
                "Camera Error",
                "Camera configuration UI is unavailable.\nCheck morphometric camera dependencies.",
            )
            return

        was_stream_active = self._stream_active
        was_yolo_active = self._yolo_active

        if self._yolo_active:
            self._on_stop_yolo()
        if self._stream_active:
            self._on_stop_stream()

        dialog_config, preselect_current_camera = self._get_camera_dialog_config()
        dialog = CameraConfigDialog(
            dialog_config,
            parent=self,
            preselect_current_camera=preselect_current_camera,
        )
        if dialog.exec() != QDialog.Accepted:
            if was_stream_active and self._camera_adapter.is_available:
                self._on_start_stream()
                if was_yolo_active:
                    self._on_start_yolo()
            return

        config = dialog.get_config()
        if not config:
            return

        if self._camera_adapter.initialize_with_config(config):
            self._camera_adapter.save_config()
            self._saved_camera_config = dict(config)
            self._update_camera_status("ready")
            self._sync_detection_camera_info()
            QMessageBox.information(self, "Camera Configured", "Camera settings applied successfully.")

            if was_stream_active:
                self._on_start_stream()
                if was_yolo_active:
                    self._on_start_yolo()
        else:
            self._update_camera_status("unavailable")
            QMessageBox.warning(
                self,
                "Camera Error",
                "Failed to open camera with selected settings.",
            )
    
    def _on_start_stream(self):
        """Start camera stream."""
        if self._camera_adapter is None:
            QMessageBox.warning(
                self,
                "Camera Error",
                "Camera subsystem is not initialized yet.",
            )
            return

        if not self._camera_adapter.is_available:
            choice = QMessageBox.question(
                self,
                "Camera Not Started",
                "No camera is active.\n\n"
                "Morphometric only opens a camera after you configure one explicitly.\n\n"
                "Open Camera Configuration now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if choice == QMessageBox.Yes:
                self._on_configure_camera()
            return

        self._timer.start(33)  # ~30fps
        self._stream_active = True
        self.btn_start_stream.setEnabled(False)
        self.btn_stop_stream.setEnabled(True)
        logger.debug("Stream started")
    
    def _on_stop_stream(self):
        """Stop camera stream."""
        self._timer.stop()
        self._stream_active = False
        self.btn_start_stream.setEnabled(True)
        self.btn_stop_stream.setEnabled(False)
        logger.debug("Stream stopped")
    
    def _update_frame(self):
        """Update camera frame display."""
        if self._camera_adapter is None:
            return
        
        success, frame = self._camera_adapter.read_frame()
        if not success or frame is None:
            return
        
        self._last_frame = frame.copy()
        display_frame = frame.copy()
        
        # Draw checkerboard overlay if detected
        if self._detection_adapter and self._detection_adapter.has_checkerboard:
            display_frame = self._detection_adapter.draw_checkerboard_overlay(display_frame)
        
        # Run YOLO if active
        if self._yolo_active and self._detection_adapter and self._detection_adapter.is_model_loaded:
            try:
                results = self._detection_adapter.predict(frame, verbose=False)
                detection = self._detection_adapter.get_primary_detection(results)
                
                if detection:
                    display_frame = self._detection_adapter.draw_detection_overlay(display_frame, detection)
                    self.btn_run_all.setEnabled(True)
                else:
                    self.btn_run_all.setEnabled(False)
            except Exception as e:
                logger.debug("YOLO prediction error: %s", e)
        
        # Display frame
        self._display_frame(display_frame, self.lbl_camera)
    
    def _display_frame(self, frame: np.ndarray, label: QLabel):
        """Display a frame on a QLabel."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        scaled = qimg.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(QPixmap.fromImage(scaled))
    
    def _on_detect_checkerboard(self):
        """Detect checkerboard for calibration."""
        if self._camera_adapter is None or self._detection_adapter is None:
            return
        
        if self._last_frame is None:
            QMessageBox.warning(self, "Error", "No frame captured. Start stream first.")
            return
        
        try:
            rows = int(self.cmb_rows.currentText())
            cols = int(self.cmb_cols.currentText())
            square_size = float(self.cmb_square_size.currentText())
            
            success, info = self._detection_adapter.detect_checkerboard(
                self._last_frame, rows, cols, square_size
            )
            
            if success:
                self.btn_clear_board.setEnabled(True)
                
                # Save any new preset values
                presets = _load_checkerboard_presets()
                changed = False
                if _add_preset_value(presets, "rows", rows):
                    changed = True
                if _add_preset_value(presets, "cols", cols):
                    changed = True
                if _add_preset_value(presets, "square_sizes_mm", square_size):
                    changed = True
                
                if changed:
                    _save_checkerboard_presets(presets)
                    # Update combo boxes with new values
                    self._refresh_checkerboard_combos(presets)
                
                QMessageBox.information(self, "Success", "Checkerboard detected!")
            else:
                QMessageBox.warning(self, "Detection Failed", "Checkerboard not detected.")
                
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {e}")
    
    def _refresh_checkerboard_combos(self, presets: Dict[str, List]):
        """Refresh combo boxes with updated presets while preserving current selection."""
        # Store current values
        curr_rows = self.cmb_rows.currentText()
        curr_cols = self.cmb_cols.currentText()
        curr_square = self.cmb_square_size.currentText()
        
        # Update rows combo
        self.cmb_rows.clear()
        for val in presets.get("rows", []):
            self.cmb_rows.addItem(str(val))
        self.cmb_rows.setCurrentText(curr_rows)
        
        # Update cols combo
        self.cmb_cols.clear()
        for val in presets.get("cols", []):
            self.cmb_cols.addItem(str(val))
        self.cmb_cols.setCurrentText(curr_cols)
        
        # Update square size combo
        self.cmb_square_size.clear()
        for val in presets.get("square_sizes_mm", []):
            self.cmb_square_size.addItem(str(val))
        self.cmb_square_size.setCurrentText(curr_square)
    
    def _on_clear_checkerboard(self):
        """Clear checkerboard calibration."""
        if self._detection_adapter:
            self._detection_adapter.clear_checkerboard()
        self.btn_clear_board.setEnabled(False)
    
    def _on_start_yolo(self):
        """Start YOLO detection."""
        if self._detection_adapter is None or not self._detection_adapter.is_model_loaded:
            QMessageBox.warning(self, "Error", "YOLO model not loaded.")
            return
        
        self._yolo_active = True
        self.btn_start_yolo.setEnabled(False)
        self.btn_stop_yolo.setEnabled(True)
    
    def _on_stop_yolo(self):
        """Stop YOLO detection."""
        self._yolo_active = False
        self.btn_start_yolo.setEnabled(True)
        self.btn_stop_yolo.setEnabled(False)
        self.btn_run_all.setEnabled(False)
    
    # =========================================================================
    # Archive Selection Handlers
    # =========================================================================
    
    def _on_target_changed(self):
        """Handle target (Gallery/Queries) change."""
        if not self._loading_measurement:
            self._clear_edit_context()
        target = self.cmb_target.currentText()
        self.meta_form.set_target(target)
        self._refresh_id_list()
    
    def _refresh_id_list(self):
        """Refresh the ID combo box."""
        target = self.cmb_target.currentText()
        ids = list_ids(target)
        
        self.cmb_id.blockSignals(True)
        self.cmb_id.clear()
        
        # Add items - "New ID" first, then sorted IDs
        all_items = ["➕ New ID…"] + sorted(ids)
        self.cmb_id.addItems(all_items)
        
        # Update completer model with just the IDs (not "New ID")
        self.cmb_id.completer().setModel(self.cmb_id.model())
        
        self.cmb_id.blockSignals(False)
        
        self._on_id_changed()
    
    def _populate_initials(self):
        """Populate initials combobox from saved initials and existing metadata."""
        initials_set = set()
        
        # Load from saved initials file first (these are prioritized)
        saved_initials = _load_saved_initials()
        for init in saved_initials:
            initials_set.add(init.upper())
        
        # Also gather from both Gallery and Queries metadata
        for target in ["Gallery", "Queries"]:
            try:
                csv_paths = ap.metadata_csv_paths_for_read(target)
                rows = read_rows_multi(csv_paths)
                for row in rows:
                    init = row.get('user_initials', '').strip()
                    if init:
                        initials_set.add(init.upper())
            except Exception as e:
                logger.debug("Could not read metadata for %s: %s", target, e)
        
        # Sort but keep saved initials at the front
        all_initials = []
        # Add saved initials first (in order)
        for init in saved_initials:
            if init.upper() in initials_set:
                all_initials.append(init.upper())
                initials_set.discard(init.upper())
        # Add remaining from metadata (sorted)
        all_initials.extend(sorted(initials_set))
        
        # Populate initials combobox
        self.cmb_initials.blockSignals(True)
        self.cmb_initials.clear()
        self.cmb_initials.addItems(all_initials)
        self.cmb_initials.completer().setModel(self.cmb_initials.model())
        self.cmb_initials.setCurrentText("")  # Allow new entries
        self.cmb_initials.blockSignals(False)
    
    def _on_id_changed(self):
        """Handle ID selection change."""
        if not self._loading_measurement:
            self._clear_edit_context()
        is_new = (self.cmb_id.currentIndex() == 0)
        self.edit_new_id.setVisible(is_new)
        
        if is_new:
            # Reset metadata to defaults when creating new entry
            self.meta_form.clear_all()
            self.meta_form.set_id_value("")
        else:
            id_val = self.cmb_id.currentText()
            self.meta_form.set_id_value(id_val)
            # Load existing metadata
            self._load_existing_metadata(id_val)
    
    def _on_new_query(self):
        """Create a new query entry."""
        self._clear_edit_context()
        # Switch to Queries target
        self.cmb_target.setCurrentText("Queries")
        # This triggers _on_target_changed() which refreshes the ID list
        
        # Select "New ID" option (index 0)
        self.cmb_id.setCurrentIndex(0)
        # This triggers _on_id_changed() which clears metadata
        
        # Focus on the new ID input field
        self.edit_new_id.setFocus()
        self.edit_new_id.selectAll()
    
    def _on_reset_metadata(self):
        """Reset all metadata fields to defaults."""
        self.meta_form.clear_all()
    
    def _load_existing_metadata(self, id_val: str):
        """Load existing metadata for an ID."""
        target = self.cmb_target.currentText()
        try:
            csv_paths = ap.metadata_csv_paths_for_read(target)
        except:
            csv_paths = []
        
        if not csv_paths:
            return
        
        id_col = ap.id_column_name(target)
        rows = read_rows_multi(csv_paths)
        latest_map = last_row_per_id(rows, id_col)
        data = latest_map.get(normalize_id_value(id_val), {})
        data[id_col] = id_val
        self.meta_form.populate(data)
    
    def _get_current_id(self) -> str:
        """Get the current ID value."""
        if self.cmb_id.currentIndex() == 0:
            return self.edit_new_id.text().strip()
        return self.cmb_id.currentText().strip()
    
    # =========================================================================
    # Analysis Handlers
    # =========================================================================
    
    def _on_capture(self):
        """Capture current detection."""
        if self._last_frame is None or self._detection_adapter is None:
            return
        
        if not self._detection_adapter.has_checkerboard:
            QMessageBox.warning(self, "Error", "Please detect checkerboard first.")
            return
        
        # Get current detection
        results = self._detection_adapter.predict(self._last_frame, verbose=False)
        detection = self._detection_adapter.get_primary_detection(results)
        
        if detection is None:
            QMessageBox.warning(self, "Error", "No detection found.")
            return
        
        # Correct detection
        corrected = self._detection_adapter.correct_detection(self._last_frame, detection)
        
        if corrected is None:
            QMessageBox.warning(self, "Error", "Failed to correct detection.")
            return
        
        self._clear_edit_context()
        self._corrected_detection = corrected
        # CRITICAL: Store a copy of the frame used for this detection
        # This ensures raw_frame.png matches the corrected_object.png
        self._captured_frame = self._last_frame.copy()
        
        # Store corrected object as RGB for visualization
        if corrected.get('corrected_object') is not None:
            self._set_corrected_object_rgb(corrected['corrected_object'])
            
            # Display the corrected object (without annotations yet)
            self._display_result_fallback()
        
        logger.info("Detection captured")
    
    def _on_analyze(self):
        """Run morphometric analysis."""
        if self._corrected_detection is None or self._analysis_adapter is None:
            return
        
        mask = self._corrected_detection.get('corrected_mask')
        mm_per_pixel = self._corrected_detection.get('mm_per_pixel', 1.0)
        
        if mask is None:
            QMessageBox.warning(self, "Error", "No mask available.")
            return
        
        # Run analysis
        result = self._analysis_adapter.analyze_contour(
            mask,
            mm_per_pixel,
            smoothing_factor=self.slider_smoothing.value(),
            prominence_factor=self.slider_prominence.value() / 100.0,
            distance_factor=self.slider_distance.value()
        )
        
        if result is None:
            QMessageBox.warning(self, "Analysis Error", "Failed to analyze contour.")
            return
        
        # Store visualization state
        self._center = self._analysis_adapter.center
        self._arm_data = self._analysis_adapter.arm_data
        self._ellipse_data = self._analysis_adapter.ellipse_data
        self._morphometrics_data = result
        
        # Update rotation slider range
        num_arms = len(self._analysis_adapter.arm_data)
        self.slider_rotation.setRange(0, max(0, num_arms - 1))
        self.slider_rotation.setValue(0)
        
        # Update visualization with annotations (including polar plot)
        self._update_arm_visualization()
        
        # Auto-populate morphometric fields
        self._auto_populate_morph_fields()
        
        self.btn_save.setEnabled(True)
        
        QMessageBox.information(self, "Analysis Complete", 
                               f"Detected {result.get('num_arms', 0)} arms.\n"
                               f"Area: {result.get('area_mm2', 0):.1f} mm²")
    
    def _update_arm_visualization(self):
        """Update the annotated visualization and polar plot with arm data."""
        rotation = self.slider_rotation.value() if hasattr(self, 'slider_rotation') else 0
        
        # Update the detection result figure with annotations
        if self._corrected_object_rgb is not None and self._center is not None and self._arm_data:
            try:
                from src.morphometric import _ensure_morphometric_path
                _ensure_morphometric_path()
                from morphometrics.visualization import create_morphometrics_visualization
                
                # Create the annotated visualization
                polar_angles, polar_dists, polar_labels, polar_colors = create_morphometrics_visualization(
                    self._result_ax,
                    self._corrected_object_rgb,
                    self._center,
                    self._arm_data,
                    rotation,
                    self._ellipse_data,
                    self._morphometrics_data or {}
                )
                
                self._result_fig.tight_layout(pad=0)
                self._result_canvas.draw()
                
                # Update polar plot with arm labels if available
                if self._polar_canvas_available and hasattr(self, 'polar_canvas'):
                    angles = self._analysis_adapter.angles_sorted
                    distances = self._analysis_adapter.distances_smoothed
                    peaks = self._analysis_adapter.peaks
                    
                    if angles is not None and distances is not None and peaks is not None:
                        # First set the base polar data
                        angles_norm = np.mod(angles, 2 * np.pi)
                        self.polar_canvas.set_data(angles_norm, distances, np.array(peaks))
                        
                        # Then set the arm labels
                        self.polar_canvas.set_arm_labels(
                            polar_angles,
                            polar_dists,
                            polar_labels,
                            polar_colors
                        )
                
                logger.debug("Arm visualization updated with rotation=%d", rotation)
                
            except ImportError as e:
                logger.warning("Visualization module not available: %s", e)
                # Fallback: just display the corrected object without annotations
                self._display_result_fallback()
            except Exception as e:
                logger.exception("Error updating arm visualization: %s", e)
                self._display_result_fallback()
        else:
            self._display_result_fallback()
    
    def _display_result_fallback(self):
        """Display corrected object without annotations (fallback)."""
        if self._corrected_object_rgb is not None:
            self._result_ax.clear()
            self._result_ax.axis('off')
            self._result_ax.imshow(self._corrected_object_rgb)
            self._result_fig.tight_layout(pad=0)
            self._result_canvas.draw()
    
    def _update_polar_plot(self):
        """Update the polar plot with current analysis data (legacy method)."""
        # This is now handled by _update_arm_visualization
        pass
    
    def _on_peaks_changed(self, new_peaks):
        """Handle peak changes from polar canvas."""
        if self._analysis_adapter is None or self._corrected_detection is None:
            return
        
        mm_per_pixel = self._corrected_detection.get('mm_per_pixel', 1.0)
        self._analysis_adapter.update_peaks(new_peaks, mm_per_pixel)
        
        # Update arm data state
        self._arm_data = self._analysis_adapter.arm_data
        
        # Update rotation slider
        num_arms = len(self._analysis_adapter.arm_data)
        self.slider_rotation.setRange(0, max(0, num_arms - 1))
        
        # Update visualization and morph fields
        self._update_arm_visualization()
        self._auto_populate_morph_fields()
    
    def _on_smoothing_changed(self, value):
        """Handle smoothing slider change."""
        self.lbl_smoothing.setText(f"Smoothing: {value}")
        if self._corrected_detection is not None:
            self._on_analyze()
    
    def _on_prominence_changed(self, value):
        """Handle prominence slider change."""
        self.lbl_prominence.setText(f"Prominence: {value / 100.0:.2f}")
        if self._corrected_detection is not None:
            self._on_analyze()
    
    def _on_distance_changed(self, value):
        """Handle distance slider change."""
        self.lbl_distance.setText(f"Distance: {value}")
        if self._corrected_detection is not None:
            self._on_analyze()
    
    def _on_rotation_changed(self, value):
        """Handle rotation slider change."""
        self.lbl_rotation.setText(f"Arm Rotation: {value}")
        # Update visualization with new rotation
        if self._arm_data is not None:
            self._update_arm_visualization()
    
    def _auto_populate_morph_fields(self):
        """Auto-populate morphometric fields in metadata form."""
        if self._analysis_adapter is None:
            logger.debug("_auto_populate_morph_fields: No analysis adapter")
            return
        
        morph = self._analysis_adapter.current_morphometrics
        if morph is None:
            logger.debug("_auto_populate_morph_fields: No morphometrics data")
            return
        
        # Build field values
        from src.morphometric.data_bridge import extract_starboard_fields
        fields = extract_starboard_fields(morph)
        
        logger.info("_auto_populate_morph_fields: Extracted fields: %s", fields)
        
        # Apply to form without overwriting manually entered values
        self.meta_form.apply_values(fields)
        
        logger.debug("_auto_populate_morph_fields: Applied %d morph fields to form", len(fields))
    
    # =========================================================================
    # Run All Handler
    # =========================================================================
    
    def _on_run_all(self):
        """Run Detect → Morph in sequence."""
        from PySide6.QtWidgets import QApplication
        
        # Step 1: Capture Detection
        self.btn_run_all.setText("Detecting...")
        self.btn_run_all.setEnabled(False)
        QApplication.processEvents()
        
        self._on_capture()
        
        if self._corrected_detection is None:
            self.btn_run_all.setText("Run All")
            self.btn_run_all.setEnabled(True)
            return
        
        # Step 2: Run Analysis
        self.btn_run_all.setText("Analyzing...")
        QApplication.processEvents()
        
        self._on_analyze()
        
        self.btn_run_all.setText("Run All")
        self.btn_run_all.setEnabled(True)
    
    # =========================================================================
    # Save Handler
    # =========================================================================
    
    def _on_save(self):
        """Save measurement to starBoard."""
        if self._analysis_adapter is None or self._corrected_detection is None:
            return
        
        # Validate inputs
        id_val = self._get_current_id()
        if not id_val:
            QMessageBox.warning(self, "Error", "Please enter an ID.")
            return
        
        initials = self.cmb_initials.currentText().strip().upper()
        if not initials or len(initials) != 3 or not initials.isalpha():
            QMessageBox.warning(self, "Error", "Please enter 3-letter initials.")
            return
        
        target = self.cmb_target.currentText()
        
        # Get location from metadata form (not from separate combobox)
        row = self.meta_form.collect_row()
        location = row.get('location', '').strip()
        
        try:
            identity_type = "gallery" if target == "Gallery" else "query"
            edit_mode = self._edit_mfolder is not None

            # 1. Save to morphometric storage (new mFolder or in-place overwrite)
            if edit_mode:
                mfolder = Path(self._edit_mfolder)
                ok = self._analysis_adapter.overwrite_measurement(
                    mfolder=mfolder,
                    identity_type=identity_type,
                    identity_id=id_val,
                    location=location,
                    user_initials=initials,
                    user_notes="",
                    arm_rotation=self.slider_rotation.value(),
                )
                if not ok:
                    QMessageBox.warning(self, "Save Error", "Failed to overwrite existing morphometric data.")
                    return
            else:
                # Use the captured frame from detection time, not the live frame.
                frame_to_save = self._captured_frame if self._captured_frame is not None else self._last_frame
                mfolder = self._analysis_adapter.save_measurement(
                    identity_type=identity_type,
                    identity_id=id_val,
                    location=location,
                    user_initials=initials,
                    user_notes="",  # Notes removed from UI
                    raw_frame=frame_to_save,
                    corrected_detection=self._corrected_detection,
                    arm_rotation=self.slider_rotation.value()
                )
                if mfolder is None:
                    QMessageBox.warning(self, "Save Error", "Failed to save morphometric data.")
                    return

            self._current_mfolder = mfolder
            if edit_mode:
                self._set_edit_context(mfolder)
            
            # 2. Copy raw_frame to starBoard archive
            raw_frame_src = mfolder / "raw_frame.png"
            if raw_frame_src.exists():
                y, m, d = qdate_to_ymd(self.date_encounter)
                enc_name = ensure_encounter_name(y, m, d)
                
                # Place image in archive
                report = place_images(
                    ap.root_for(target),
                    id_val,
                    enc_name,
                    [raw_frame_src],
                    move=False  # Copy, don't move
                )
                
                if report.errors:
                    logger.warning("Errors copying to archive: %s", report.errors)
            
            # 3. Append metadata row (reuse row from earlier)
            csv_path, header = ap.metadata_csv_for(target)
            row[ap.id_column_name(target)] = id_val
            
            # Add morph_source_folder
            row['morph_source_folder'] = str(mfolder)
            
            # Debug: log morph fields being saved
            morph_in_row = {k: v for k, v in row.items() if k.startswith('morph_') and v}
            logger.info("_on_save: Morph fields in row: %s", morph_in_row)
            
            # Capture old state before save (for Gallery metadata history)
            old_values = {}
            if target == "Gallery":
                old_values = get_current_metadata_for_gallery(id_val)
            
            append_row(csv_path, header, row)
            
            # Record metadata history for Gallery (using morphometric import action)
            if target == "Gallery":
                record_morphometric_import(
                    gallery_id=id_val,
                    old_values=old_values,
                    new_values=row,
                    mfolder_path=str(mfolder),
                )
            
            # 4. Notify First-order tab to refresh
            self._notify_first_order_refresh()
            self.dataSaved.emit()
            
            # 5. Save initials if new
            if _add_initials(initials):
                self._populate_initials()
                self.cmb_initials.setCurrentText(initials)
            
            # Refresh ID list if new ID
            if self.cmb_id.currentIndex() == 0:
                self._refresh_id_list()
                # Select the new ID
                idx = self.cmb_id.findText(id_val)
                if idx >= 0:
                    self.cmb_id.setCurrentIndex(idx)
            
            if edit_mode:
                QMessageBox.information(
                    self,
                    "Saved",
                    f"Historical measurement updated in place.\n\n"
                    f"Morphometric data updated: {mfolder}\n"
                    f"Metadata row appended to starBoard archive.",
                )
                self.btn_save.setEnabled(True)
            else:
                QMessageBox.information(
                    self,
                    "Saved",
                    f"Data saved!\n\n"
                    f"Morphometric data: {mfolder}\n"
                    f"Raw frame copied to starBoard archive.",
                )

                # Reset for next capture
                self.btn_save.setEnabled(False)
                self._corrected_detection = None

            self._refresh_saved_measurements()
            
        except Exception as e:
            logger.exception("Error saving: %s", e)
            QMessageBox.critical(self, "Save Error", f"An error occurred:\n{e}")
    
    # =========================================================================
    # Cross-Tab Refresh
    # =========================================================================
    
    def _notify_first_order_refresh(self) -> None:
        """
        Best-effort: find a TabFirstOrder in the same QTabWidget and tell it to rebuild/refresh.
        Works without importing TabFirstOrder directly.
        """
        try:
            tabs = None
            for w in self.window().findChildren(QTabWidget):
                tabs = w
                break
            if not tabs:
                return
            for i in range(tabs.count()):
                wid = tabs.widget(i)
                if wid.__class__.__name__ == "TabFirstOrder":
                    # gentle rebuild + refresh
                    if hasattr(wid, "engine") and hasattr(wid.engine, "rebuild"):
                        try:
                            wid.engine.rebuild()
                        except Exception:
                            pass
                    for name in ("_refresh_query_ids", "_refresh_results"):
                        if hasattr(wid, name):
                            try:
                                getattr(wid, name)()
                            except Exception:
                                pass
                    break
        except Exception as e:
            logger.debug("notify_first_order_refresh skipped: %s", e)
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def closeEvent(self, event):
        """Clean up on close."""
        self._timer.stop()
        if self._camera_adapter:
            self._camera_adapter.close()
        event.accept()

