# src/morphometric/camera_adapter.py
"""
Camera Adapter for starBoard Morphometric Integration.

Wraps the starMorphometricTool camera subsystem to provide a clean interface
for the starBoard application.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

from . import _ensure_morphometric_path

logger = logging.getLogger("starBoard.morphometric.camera")


class CameraAdapter:
    """
    Adapter for the starMorphometricTool camera subsystem.
    
    Provides a simplified interface for webcam operations within starBoard.
    Handles camera initialization, configuration, and frame capture.
    """
    
    def __init__(self):
        """Initialize the camera adapter."""
        self._camera = None
        self._config: Optional[Dict[str, Any]] = None
        self._is_initialized = False
        
        # Ensure morphometric tool path is available
        _ensure_morphometric_path()
    
    def initialize(self) -> bool:
        """
        Initialize the camera subsystem with auto-detection.
        
        Returns:
            True if a camera was successfully detected and opened.
        """
        try:
            from camera.factory import auto_detect_with_config, create_camera_from_config
            from camera.config import load_camera_config as load_config
            
            # Try loading saved config first
            saved_config = load_config()
            if saved_config:
                logger.info("Attempting to use saved camera configuration...")
                camera = create_camera_from_config(saved_config)
                if camera and camera.open():
                    self._camera = camera
                    self._config = saved_config
                    self._is_initialized = True
                    logger.info("Camera initialized from saved config")
                    return True
                elif camera:
                    camera.close()
            
            # Auto-detect camera
            logger.info("Auto-detecting camera...")
            camera, config = auto_detect_with_config()
            
            if camera is not None:
                self._camera = camera
                self._config = config
                self._is_initialized = True
                logger.info("Camera auto-detected and initialized")
                return True
            else:
                logger.warning("No camera detected")
                return False
                
        except ImportError as e:
            logger.error("Camera module not available: %s", e)
            return False
        except Exception as e:
            logger.exception("Error initializing camera: %s", e)
            return False
    
    def initialize_with_config(self, config: Dict[str, Any]) -> bool:
        """
        Initialize camera with specific configuration.
        
        Args:
            config: Camera configuration dictionary with keys:
                - provider: Camera provider name
                - device_index: Device index for OpenCV
                - width, height: Resolution
                - fps: Frame rate
        
        Returns:
            True if camera was successfully opened.
        """
        try:
            from camera.factory import create_camera_from_config
            
            # Close existing camera if any
            self.close()
            
            camera = create_camera_from_config(config)
            if camera and camera.open():
                # Apply resolution and fps if specified
                if config.get("width") and config.get("height"):
                    camera.set_resolution(config["width"], config["height"])
                if config.get("fps"):
                    camera.set_fps(config["fps"])
                
                self._camera = camera
                self._config = config
                self._is_initialized = True
                logger.info("Camera initialized with provided config")
                return True
            else:
                if camera:
                    camera.close()
                logger.warning("Failed to open camera with config")
                return False
                
        except Exception as e:
            logger.exception("Error applying camera config: %s", e)
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if camera is available and ready."""
        return self._is_initialized and self._camera is not None and self._camera.is_open()
    
    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Get current camera configuration."""
        return self._config
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Tuple of (success, frame) where frame is BGR numpy array or None.
        """
        if not self.is_available:
            return False, None
        
        try:
            return self._camera.read_frame()
        except Exception as e:
            logger.error("Error reading frame: %s", e)
            return False, None
    
    def get_resolution(self) -> Tuple[int, int]:
        """
        Get current camera resolution.
        
        Returns:
            (width, height) tuple, or (640, 480) if unavailable.
        """
        if not self.is_available:
            return 640, 480
        return self._camera.get_resolution()
    
    def set_resolution(self, width: int, height: int) -> bool:
        """Set camera resolution."""
        if not self.is_available:
            return False
        return self._camera.set_resolution(width, height)
    
    def get_fps(self) -> float:
        """Get current frame rate."""
        if not self.is_available:
            return 30.0
        return self._camera.get_fps() or 30.0
    
    def set_fps(self, fps: float) -> bool:
        """Set target frame rate."""
        if not self.is_available:
            return False
        return self._camera.set_fps(fps)
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get camera device information."""
        if not self.is_available:
            return {"status": "unavailable"}
        return self._camera.get_device_info()
    
    def close(self) -> None:
        """Release camera resources."""
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception as e:
                logger.debug("Error closing camera: %s", e)
            finally:
                self._camera = None
                self._is_initialized = False
    
    def save_config(self) -> bool:
        """
        Save current camera configuration.
        
        Returns:
            True if config was saved successfully.
        """
        if self._config is None:
            return False
        
        try:
            from camera.config import save_camera_config
            save_camera_config(self._config)
            logger.info("Camera config saved")
            return True
        except Exception as e:
            logger.error("Error saving camera config: %s", e)
            return False
    
    @staticmethod
    def enumerate_devices() -> list:
        """
        Enumerate all available camera devices.
        
        Returns:
            List of device info dictionaries.
        """
        try:
            _ensure_morphometric_path()
            from camera.factory import enumerate_all_devices
            return enumerate_all_devices()
        except Exception as e:
            logger.error("Error enumerating devices: %s", e)
            return []
    
    def __enter__(self) -> "CameraAdapter":
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor - ensure camera is released."""
        self.close()

