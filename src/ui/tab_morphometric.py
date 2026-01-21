# src/ui/tab_morphometric.py
"""
Morphometric Tab for starBoard.

Integrates webcam-based morphometric measurement functionality from the
starMorphometricTool, allowing live capture and metadata entry.
"""
from __future__ import annotations

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
    QSizePolicy, QCompleter, QFrame,
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
from src.data import archive_paths as ap
from src.data.csv_io import append_row, read_rows_multi, last_row_per_id, normalize_id_value
from src.data.id_registry import list_ids, id_exists
from src.data.metadata_history import (
    record_morphometric_import, get_current_metadata_for_gallery,
)
from src.data.ingest import ensure_encounter_name, place_images
from src.utils.interaction_logger import get_interaction_logger

logger = logging.getLogger("starBoard.ui.tab_morphometric")


def qdate_to_ymd(q) -> tuple:
    """Convert QDate or QDateEdit to (year, month, day)."""
    if hasattr(q, 'date'):
        q = q.date()
    return q.year(), q.month(), q.day()


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
        self._depth_adapter = None
        self._stream_active = False
        self._yolo_active = False
        self._last_frame = None
        self._captured_frame = None  # Frame captured during detection (frozen)
        self._corrected_detection = None
        self._current_mfolder = None
        self._volume_data: Optional[Dict[str, Any]] = None
        self._depth_arrays: Optional[Dict[str, Any]] = None
        
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
            
            # Initialize camera
            if self._camera_adapter.initialize():
                self._update_camera_status(True)
                # Pass camera info to detection adapter for calibration fingerprinting
                if hasattr(self._detection_adapter, 'set_camera_info'):
                    config = self._camera_adapter.config or {}
                    device_info = self._camera_adapter.get_device_info() if hasattr(self._camera_adapter, 'get_device_info') else {}
                    self._detection_adapter.set_camera_info(config, device_info)
            else:
                self._update_camera_status(False)
            
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
        layout.addWidget(sec_camera)
        
        # Archive Selection Section
        sec_archive = CollapsibleSection("Archive Selection", start_collapsed=False)
        sec_archive.setContent(self._build_archive_selection())
        layout.addWidget(sec_archive)
        
        # Metadata Form Section
        sec_metadata = CollapsibleSection("Metadata", start_collapsed=False)
        self.meta_form = MetadataFormV2()
        self.meta_form.set_target("Gallery")
        sec_metadata.setContent(self.meta_form)
        layout.addWidget(sec_metadata, 1)
        
        # Analysis Controls Section
        sec_analysis = CollapsibleSection("Analysis Controls", start_collapsed=True)
        sec_analysis.setContent(self._build_analysis_controls())
        layout.addWidget(sec_analysis)
        
        # Action Buttons Section
        sec_actions = CollapsibleSection("Actions", start_collapsed=False)
        sec_actions.setContent(self._build_action_buttons())
        layout.addWidget(sec_actions)
        
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
        layout.addLayout(status_row)
        
        # Checkerboard config
        form = QFormLayout()
        form.setSpacing(4)
        
        self.edit_rows = QLineEdit("8")
        self.edit_rows.setFixedWidth(60)
        form.addRow("Rows:", self.edit_rows)
        
        self.edit_cols = QLineEdit("10")
        self.edit_cols.setFixedWidth(60)
        form.addRow("Columns:", self.edit_cols)
        
        self.edit_square_size = QLineEdit("25")
        self.edit_square_size.setFixedWidth(60)
        form.addRow("Square (mm):", self.edit_square_size)
        
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
        
        # Run All button
        self.btn_run_all = QPushButton("Run All")
        self.btn_run_all.setToolTip("Run Detect → Morph → Volume in sequence")
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
        
        # Depth/elevation map display
        self.lbl_depth = QLabel("Depth Map")
        self.lbl_depth.setAlignment(Qt.AlignCenter)
        self.lbl_depth.setMinimumSize(250, 250)
        self.lbl_depth.setStyleSheet("QLabel { background-color: #1a1a1a; color: #666; }")
        h_splitter.addWidget(self.lbl_depth)
        
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
    
    def _update_camera_status(self, available: bool):
        """Update camera status indicator."""
        if available:
            self.lbl_camera_status.setText("● Camera Ready")
            self.lbl_camera_status.setStyleSheet("color: green; font-size: 10px;")
            self.btn_start_stream.setEnabled(True)
        else:
            self.lbl_camera_status.setText("● No Camera")
            self.lbl_camera_status.setStyleSheet("color: red; font-size: 10px;")
            self.btn_start_stream.setEnabled(False)
            self.lbl_camera.setText("No camera detected.\nConnect a webcam and restart.")
    
    def _on_start_stream(self):
        """Start camera stream."""
        if self._camera_adapter is None or not self._camera_adapter.is_available:
            QMessageBox.warning(self, "Camera Error", "No camera available.")
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
            rows = int(self.edit_rows.text())
            cols = int(self.edit_cols.text())
            square_size = float(self.edit_square_size.text())
            
            success, info = self._detection_adapter.detect_checkerboard(
                self._last_frame, rows, cols, square_size
            )
            
            if success:
                self.btn_clear_board.setEnabled(True)
                QMessageBox.information(self, "Success", "Checkerboard detected!")
            else:
                QMessageBox.warning(self, "Detection Failed", "Checkerboard not detected.")
                
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {e}")
    
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
        """Populate initials combobox from existing metadata."""
        initials_set = set()
        
        # Gather from both Gallery and Queries metadata
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
        
        # Populate initials combobox
        self.cmb_initials.blockSignals(True)
        self.cmb_initials.clear()
        self.cmb_initials.addItems(sorted(initials_set))
        self.cmb_initials.completer().setModel(self.cmb_initials.model())
        self.cmb_initials.setCurrentText("")  # Allow new entries
        self.cmb_initials.blockSignals(False)
    
    def _on_id_changed(self):
        """Handle ID selection change."""
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
        return self.cmb_id.currentText()
    
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
        
        self._corrected_detection = corrected
        # CRITICAL: Store a copy of the frame used for this detection
        # This ensures raw_frame.png matches the corrected_object.png
        self._captured_frame = self._last_frame.copy()
        
        # Store corrected object as RGB for visualization
        if corrected.get('corrected_object') is not None:
            obj = corrected['corrected_object']
            if len(obj.shape) == 2:
                self._corrected_object_rgb = cv2.cvtColor(obj, cv2.COLOR_GRAY2RGB)
            elif obj.shape[2] == 4:
                self._corrected_object_rgb = cv2.cvtColor(obj, cv2.COLOR_BGRA2RGB)
            else:
                self._corrected_object_rgb = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
            
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
        """Run Detect → Morph → Volume in sequence."""
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
        
        if self._analysis_adapter is None or not self._analysis_adapter.arm_data:
            self.btn_run_all.setText("Run All")
            self.btn_run_all.setEnabled(True)
            return
        
        # Step 3: Estimate Volume (if available)
        from src.morphometric import is_depth_available
        if is_depth_available():
            self.btn_run_all.setText("Volume...")
            QApplication.processEvents()
            self._on_estimate_volume()
        
        self.btn_run_all.setText("Run All")
        self.btn_run_all.setEnabled(True)
    
    # =========================================================================
    # Volume Estimation Handler
    # =========================================================================
    
    def _on_estimate_volume(self):
        """Run depth estimation and volume computation."""
        if self._corrected_detection is None or self._analysis_adapter is None:
            QMessageBox.warning(self, "Error", "Please run analysis first.")
            return
        
        if self._detection_adapter is None or not self._detection_adapter.has_checkerboard:
            QMessageBox.warning(self, "Error", "Checkerboard calibration required.")
            return
        
        if self._last_frame is None:
            QMessageBox.warning(self, "Error", "No frame available.")
            return
        
        try:
            from src.morphometric import is_depth_available, get_depth_adapter
            
            # Check if depth estimation is available
            if not is_depth_available():
                QMessageBox.warning(
                    self, "Depth Not Available",
                    "Depth estimation requires:\n\n"
                    "1. Depth-Anything-V2 directory alongside starMorphometricTool\n"
                    "2. Model checkpoint in Depth-Anything-V2/checkpoints/\n"
                    "3. PyTorch installed\n\n"
                    "Volume estimation is optional and can be skipped."
                )
                return
            
            # Get depth adapter
            if self._depth_adapter is None:
                self._depth_adapter = get_depth_adapter()
            
            # Get required data
            corrected_mask = self._corrected_detection.get('corrected_mask')
            homography = self._corrected_detection.get('homography_matrix')
            mm_per_pixel = self._corrected_detection.get('mm_per_pixel', 1.0)
            
            if corrected_mask is None or homography is None:
                QMessageBox.warning(self, "Error", "Missing detection data for volume estimation.")
                return
            
            # Convert homography to numpy if needed
            if isinstance(homography, list):
                homography = np.array(homography, dtype=np.float32)
            
            # Build checkerboard info from detection adapter
            cb_info = self._detection_adapter.checkerboard_info
            
            # Get camera intrinsics if available
            intrinsics = None
            camera_id = None
            if hasattr(self._detection_adapter, 'get_camera_intrinsics'):
                intrinsics = self._detection_adapter.get_camera_intrinsics()
            if hasattr(self._detection_adapter, '_get_camera_id'):
                camera_id = self._detection_adapter._get_camera_id()
            
            # Run volume estimation - use the captured frame from detection time
            frame_for_volume = self._captured_frame if self._captured_frame is not None else self._last_frame
            result = self._depth_adapter.run_volume_estimation(
                raw_frame=frame_for_volume,
                corrected_mask=corrected_mask,
                checkerboard_info=cb_info,
                homography_matrix=homography,
                mm_per_pixel=mm_per_pixel,
                encoder='vitb',
                input_size=518,
                intrinsics=intrinsics,
                camera_id=camera_id
            )
            
            if not result['success']:
                QMessageBox.warning(
                    self, "Volume Estimation Failed",
                    f"Error: {result.get('error', 'Unknown error')}"
                )
                return
            
            # Store volume data for save
            self._volume_data = result.get('volume_estimation_data')
            
            # Store depth arrays for saving to files later
            calibration_result = result.get('calibration_result') or {}
            self._depth_arrays = {
                'calibrated_depth': calibration_result.get('calibrated_depth'),
                'elevation_map': result.get('elevation_map'),
                'mask': corrected_mask,
                'raw_depth_map': result.get('raw_depth_map'),
            }
            
            # Display elevation visualization
            elevation_viz = result.get('elevation_visualization')
            if elevation_viz is not None:
                self._display_frame(elevation_viz, self.lbl_depth)
            
            # Update morph_volume_mm3 field in form
            volume_mm3 = result.get('volume_mm3', 0)
            volume_ml = result.get('volume_ml', 0)
            self.meta_form.apply_values({'morph_volume_mm3': f"{volume_mm3:.2f}"})
            
            # Show success message with calibration status
            mean_elev = result.get('mean_elevation_mm', 0)
            max_elev = result.get('max_elevation_mm', 0)
            calibration_status = result.get('calibration_status', 'provisional')
            
            # Build message with calibration status
            status_note = ""
            if calibration_status == 'provisional':
                # Get calibration progress if available
                calib_status = self._detection_adapter.get_calibration_status() if hasattr(self._detection_adapter, 'get_calibration_status') else {}
                detection_count = calib_status.get('detection_count', 0)
                min_required = calib_status.get('min_required', 10)
                status_note = f"\n\n⚠️ PROVISIONAL ESTIMATE\nCamera calibration: {detection_count}/{min_required} detections\nVolume will be recomputed when calibration is complete."
            
            QMessageBox.information(
                self, "Volume Estimation Complete",
                f"Volume: {volume_ml:.3f} mL ({volume_mm3:.1f} mm³)\n"
                f"Mean elevation: {mean_elev:.2f} mm\n"
                f"Max elevation: {max_elev:.2f} mm"
                f"{status_note}"
            )
            
            logger.info("Volume estimation complete: %.1f mm³ [%s]", volume_mm3, calibration_status)
            
        except ImportError as e:
            QMessageBox.warning(
                self, "Import Error",
                f"Failed to import depth module: {e}\n\n"
                "Ensure PyTorch and Depth-Anything-V2 are installed."
            )
            logger.exception("Import error in volume estimation")
        except MemoryError:
            QMessageBox.warning(
                self, "Memory Error",
                "Out of memory during volume estimation.\n"
                "Try closing other applications or using a smaller model."
            )
            logger.exception("Memory error in volume estimation")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Volume estimation failed:\n{e}")
            logger.exception("Error in volume estimation")
    
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
            # 1. Save to morphometric storage (mFolder)
            identity_type = "gallery" if target == "Gallery" else "query"
            # Use the captured frame from detection time, not the live frame
            frame_to_save = self._captured_frame if self._captured_frame is not None else self._last_frame
            mfolder = self._analysis_adapter.save_measurement(
                identity_type=identity_type,
                identity_id=id_val,
                location=location,
                user_initials=initials,
                user_notes="",  # Notes removed from UI
                raw_frame=frame_to_save,
                corrected_detection=self._corrected_detection,
                arm_rotation=self.slider_rotation.value(),
                volume_data=self._volume_data,  # Pass volume data if available
                depth_arrays=self._depth_arrays  # Pass depth arrays for file saving
            )
            
            if mfolder is None:
                QMessageBox.warning(self, "Save Error", "Failed to save morphometric data.")
                return
            
            self._current_mfolder = mfolder
            
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
            
            # Refresh ID list if new ID
            if self.cmb_id.currentIndex() == 0:
                self._refresh_id_list()
                # Select the new ID
                idx = self.cmb_id.findText(id_val)
                if idx >= 0:
                    self.cmb_id.setCurrentIndex(idx)
            
            QMessageBox.information(self, "Saved", 
                                   f"Data saved!\n\n"
                                   f"Morphometric data: {mfolder}\n"
                                   f"Raw frame copied to starBoard archive.")
            
            # Reset for next capture (keep volume visualization visible)
            self.btn_save.setEnabled(False)
            self._corrected_detection = None
            self._volume_data = None
            self._depth_arrays = None
            
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

