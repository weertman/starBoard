"""
Camera Configuration Dialog

Provides a user interface for configuring camera settings including
device selection, backend, resolution, and frame rate. Includes live
preview and test functionality.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QPushButton, QLabel, QGroupBox,
    QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

import cv2
import logging
from typing import Optional, Dict, Any

from camera import CameraInterface, get_default_config
from camera.factory import (
    create_camera, enumerate_devices_by_provider,
    get_available_providers, get_provider_info
)
from camera.providers.opencv_camera import OpenCVCamera


# Common resolutions to offer in configuration
COMMON_RESOLUTIONS = [
    (640, 480, "640x480 (VGA)"),
    (800, 600, "800x600 (SVGA)"),
    (1280, 720, "1280x720 (720p)"),
    (1920, 1080, "1920x1080 (1080p)"),
]

# Common frame rates
COMMON_FRAME_RATES = [15, 24, 30, 60]


class CameraConfigDialog(QDialog):
    """
    Dialog for configuring camera settings with live preview.
    """
    
    def __init__(self, current_config: Optional[Dict[str, Any]] = None, parent=None):
        """
        Initialize the camera configuration dialog.
        
        Args:
            current_config: Current camera configuration dict (or None for defaults)
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Camera Configuration")
        self.setMinimumSize(600, 500)
        
        # Store current config
        self.current_config = current_config or get_default_config()
        self.result_config = None
        
        # Preview camera using abstraction
        self.preview_camera: Optional[CameraInterface] = None
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.update_preview)
        
        # Build UI
        self.setup_ui()
        
        # Populate with current values
        self.populate_from_config()
    
    def setup_ui(self):
        """Create the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Settings group
        settings_group = QGroupBox("Camera Settings")
        settings_layout = QFormLayout()
        
        # Camera device dropdown
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(200)
        self.refresh_cameras_button = QPushButton("Refresh")
        self.refresh_cameras_button.setFixedWidth(80)
        self.refresh_cameras_button.clicked.connect(self.refresh_camera_list)
        
        camera_row = QHBoxLayout()
        camera_row.addWidget(self.camera_combo)
        camera_row.addWidget(self.refresh_cameras_button)
        settings_layout.addRow("Camera Device:", camera_row)
        
        # Backend dropdown (for OpenCV cameras)
        self.backend_combo = QComboBox()
        for name, _ in OpenCVCamera.get_available_backends():
            self.backend_combo.addItem(name)
        self.backend_combo.currentIndexChanged.connect(self.on_backend_changed)
        settings_layout.addRow("Backend:", self.backend_combo)
        
        # Resolution dropdown
        self.resolution_combo = QComboBox()
        for width, height, label in COMMON_RESOLUTIONS:
            self.resolution_combo.addItem(label, (width, height))
        settings_layout.addRow("Resolution:", self.resolution_combo)
        
        # FPS dropdown
        self.fps_combo = QComboBox()
        for fps in COMMON_FRAME_RATES:
            self.fps_combo.addItem(f"{fps} fps", fps)
        settings_layout.addRow("Frame Rate:", self.fps_combo)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Preview group
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_label = QLabel("Click 'Test Camera' to preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setMaximumSize(640, 480)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setStyleSheet("QLabel { background-color: #222; color: #888; }")
        preview_layout.addWidget(self.preview_label)
        
        # Test button
        test_layout = QHBoxLayout()
        self.test_button = QPushButton("Test Camera")
        self.test_button.clicked.connect(self.test_camera)
        self.stop_test_button = QPushButton("Stop Preview")
        self.stop_test_button.clicked.connect(self.stop_preview)
        self.stop_test_button.setEnabled(False)
        test_layout.addWidget(self.test_button)
        test_layout.addWidget(self.stop_test_button)
        test_layout.addStretch()
        preview_layout.addLayout(test_layout)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("QLabel { color: #666; }")
        preview_layout.addWidget(self.status_label)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.save_button = QPushButton("Save && Apply")
        self.save_button.clicked.connect(self.accept_config)
        self.save_button.setDefault(True)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        # Initial camera list refresh
        self.refresh_camera_list()
    
    def populate_from_config(self):
        """Populate UI controls from current config"""
        config = self.current_config
        
        # Set backend
        backend = config.get("backend", "Auto")
        index = self.backend_combo.findText(backend)
        if index >= 0:
            self.backend_combo.setCurrentIndex(index)
        
        # Set resolution
        width = config.get("width", 1280)
        height = config.get("height", 720)
        for i in range(self.resolution_combo.count()):
            res = self.resolution_combo.itemData(i)
            if res == (width, height):
                self.resolution_combo.setCurrentIndex(i)
                break
        
        # Set FPS
        fps = config.get("fps", 30)
        for i in range(self.fps_combo.count()):
            if self.fps_combo.itemData(i) == int(fps):
                self.fps_combo.setCurrentIndex(i)
                break
        
        # Camera index will be set after refresh
        self._pending_camera_index = config.get("camera_index", 0)
    
    def refresh_camera_list(self):
        """Refresh the list of available cameras"""
        self.camera_combo.clear()
        
        # Get current backend name and convert to constant
        backend_name = self.backend_combo.currentText()
        backend_const = OpenCVCamera._backend_name_to_constant(backend_name)
        
        # Enumerate cameras
        self.status_label.setText("Searching for cameras...")
        self.status_label.repaint()
        
        cameras = enumerate_devices_by_provider("opencv", backend=backend_const)
        
        if cameras:
            for device in cameras:
                index = device.get("index", 0)
                name = device.get("name", f"Camera {index}")
                self.camera_combo.addItem(name, index)
            self.status_label.setText(f"Found {len(cameras)} camera(s)")
            
            # Try to select the pending camera index
            if hasattr(self, '_pending_camera_index'):
                for i in range(self.camera_combo.count()):
                    if self.camera_combo.itemData(i) == self._pending_camera_index:
                        self.camera_combo.setCurrentIndex(i)
                        break
        else:
            self.camera_combo.addItem("No cameras found", -1)
            self.status_label.setText("No cameras detected")
    
    def on_backend_changed(self):
        """Handle backend selection change"""
        self.stop_preview()
        self.refresh_camera_list()
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get current UI settings as a config dict"""
        camera_index = self.camera_combo.currentData()
        if camera_index is None or camera_index < 0:
            camera_index = 0
        
        resolution = self.resolution_combo.currentData()
        if resolution is None:
            resolution = (1280, 720)
        
        fps = self.fps_combo.currentData()
        if fps is None:
            fps = 30
        
        return {
            "provider": "opencv",
            "device_index": camera_index,
            "backend": self.backend_combo.currentText(),
            "width": resolution[0],
            "height": resolution[1],
            "fps": fps,
        }
    
    def test_camera(self):
        """Test camera with current settings"""
        self.stop_preview()
        
        settings = self.get_current_settings()
        
        if settings["device_index"] < 0:
            QMessageBox.warning(self, "No Camera", "No camera selected.")
            return
        
        self.status_label.setText("Opening camera...")
        self.status_label.repaint()
        
        # Create camera using abstraction
        camera = create_camera(
            "opencv",
            device_index=settings["device_index"],
            backend_name=settings["backend"]
        )
        
        if camera and camera.open():
            # Apply resolution and fps
            camera.set_resolution(settings["width"], settings["height"])
            camera.set_fps(settings["fps"])
            
            self.preview_camera = camera
            self.preview_timer.start(33)  # ~30 fps
            self.test_button.setEnabled(False)
            self.stop_test_button.setEnabled(True)
            
            # Show actual resolution
            actual_w, actual_h = camera.get_resolution()
            self.status_label.setText(f"Preview active: {actual_w}x{actual_h}")
            self.status_label.setStyleSheet("QLabel { color: green; }")
        else:
            if camera:
                camera.close()
            self.status_label.setText("Error: Failed to open camera")
            self.status_label.setStyleSheet("QLabel { color: red; }")
            QMessageBox.warning(self, "Camera Error", "Failed to open camera with selected settings.")
    
    def stop_preview(self):
        """Stop camera preview"""
        self.preview_timer.stop()
        
        if self.preview_camera is not None:
            self.preview_camera.close()
            self.preview_camera = None
        
        self.preview_label.setText("Click 'Test Camera' to preview")
        self.preview_label.setStyleSheet("QLabel { background-color: #222; color: #888; }")
        self.test_button.setEnabled(True)
        self.stop_test_button.setEnabled(False)
        self.status_label.setText("")
        self.status_label.setStyleSheet("QLabel { color: #666; }")
    
    def update_preview(self):
        """Update preview frame"""
        if self.preview_camera is None:
            return
        
        ret, frame = self.preview_camera.read_frame()
        if ret and frame is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            
            # Create QImage
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale to fit preview label
            scaled = q_img.scaled(
                self.preview_label.width(),
                self.preview_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.preview_label.setPixmap(QPixmap.fromImage(scaled))
        else:
            self.stop_preview()
            self.status_label.setText("Camera disconnected")
            self.status_label.setStyleSheet("QLabel { color: red; }")
    
    def accept_config(self):
        """Accept and return configuration"""
        settings = self.get_current_settings()
        
        if settings["device_index"] < 0:
            QMessageBox.warning(self, "No Camera", 
                "No camera selected. Please select a camera or click Cancel.")
            return
        
        # Test that the camera actually works before accepting
        self.status_label.setText("Verifying camera...")
        self.status_label.repaint()
        
        camera = create_camera(
            "opencv",
            device_index=settings["device_index"],
            backend_name=settings["backend"]
        )
        
        if camera and camera.open():
            camera.close()
            self.result_config = settings
            self.stop_preview()
            self.accept()
        else:
            if camera:
                camera.close()
            QMessageBox.warning(self, "Camera Error",
                "Cannot use these settings. Please try different settings.")
    
    def get_config(self) -> Optional[Dict[str, Any]]:
        """Get the resulting configuration after dialog closes"""
        return self.result_config
    
    def closeEvent(self, event):
        """Clean up on dialog close"""
        self.stop_preview()
        super().closeEvent(event)
    
    def reject(self):
        """Handle cancel/close"""
        self.stop_preview()
        super().reject()
