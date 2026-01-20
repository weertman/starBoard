"""
OpenCV Camera Provider

Implements the CameraInterface for OpenCV-compatible devices (USB webcams).
Supports multiple backends for cross-platform compatibility.
"""

import cv2
import platform
import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from camera.interface import CameraInterface


class OpenCVCamera(CameraInterface):
    """
    Camera provider for OpenCV-compatible devices (USB webcams).
    
    Supports multiple backends:
    - Windows: DirectShow, Media Foundation
    - macOS: AVFoundation
    - Linux: V4L2, GStreamer
    
    Example:
        camera = OpenCVCamera(device_index=0)
        if camera.open():
            success, frame = camera.read_frame()
            camera.close()
    """
    
    # Backend options by operating system
    BACKENDS = {
        "Windows": [
            ("DirectShow", cv2.CAP_DSHOW),
            ("Media Foundation", cv2.CAP_MSMF),
            ("Auto", cv2.CAP_ANY),
        ],
        "Darwin": [  # macOS
            ("AVFoundation", cv2.CAP_AVFOUNDATION),
            ("Auto", cv2.CAP_ANY),
        ],
        "Linux": [
            ("V4L2", cv2.CAP_V4L2),
            ("GStreamer", cv2.CAP_GSTREAMER),
            ("Auto", cv2.CAP_ANY),
        ],
    }
    
    def __init__(
        self,
        device_index: int = 0,
        backend: Optional[int] = None,
        backend_name: Optional[str] = None
    ):
        """
        Initialize OpenCV camera.
        
        Args:
            device_index: Camera device index (0, 1, 2, ...)
            backend: OpenCV backend constant (e.g., cv2.CAP_DSHOW)
            backend_name: Backend name string (e.g., "DirectShow")
                         Used if backend constant not provided
        """
        self._device_index = device_index
        
        # Resolve backend
        if backend is not None:
            self._backend = backend
        elif backend_name is not None:
            self._backend = self._backend_name_to_constant(backend_name)
        else:
            self._backend = self._get_default_backend()
        
        self._cap: Optional[cv2.VideoCapture] = None
    
    # ==========================================================================
    # Backend Utilities
    # ==========================================================================
    
    @staticmethod
    def _get_default_backend() -> int:
        """Get preferred backend for current OS."""
        os_name = platform.system()
        backends = OpenCVCamera.BACKENDS.get(os_name, [("Auto", cv2.CAP_ANY)])
        return backends[0][1]
    
    @staticmethod
    def _backend_name_to_constant(name: str) -> int:
        """Convert backend name to cv2 constant."""
        for os_backends in OpenCVCamera.BACKENDS.values():
            for backend_name, constant in os_backends:
                if backend_name.lower() == name.lower():
                    return constant
        return cv2.CAP_ANY
    
    @staticmethod
    def _backend_constant_to_name(constant: int) -> str:
        """Convert cv2 backend constant to name."""
        for os_backends in OpenCVCamera.BACKENDS.values():
            for backend_name, const in os_backends:
                if const == constant:
                    return backend_name
        return "Auto"
    
    # ==========================================================================
    # Connection Management (CameraInterface)
    # ==========================================================================
    
    def open(self) -> bool:
        """Open connection to the camera."""
        # Close existing connection if any
        if self._cap is not None:
            self.close()
        
        try:
            self._cap = cv2.VideoCapture(self._device_index, self._backend)
            
            if not self._cap.isOpened():
                logging.debug(f"OpenCV camera {self._device_index} failed to open")
                self._cap = None
                return False
            
            # Verify we can actually read a frame
            ret, _ = self._cap.read()
            if not ret:
                logging.debug(f"OpenCV camera {self._device_index} opened but can't read frames")
                self._cap.release()
                self._cap = None
                return False
            
            logging.debug(f"OpenCV camera {self._device_index} opened successfully")
            return True
            
        except Exception as e:
            logging.error(f"Exception opening OpenCV camera: {e}")
            self._cap = None
            return False
    
    def close(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logging.debug(f"Exception releasing camera: {e}")
            finally:
                self._cap = None
    
    def is_open(self) -> bool:
        """Check if camera connection is active."""
        return self._cap is not None and self._cap.isOpened()
    
    # ==========================================================================
    # Frame Capture (CameraInterface)
    # ==========================================================================
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture a single frame from the camera."""
        if not self.is_open():
            return False, None
        
        try:
            return self._cap.read()
        except Exception as e:
            logging.error(f"Exception reading frame: {e}")
            return False, None
    
    # ==========================================================================
    # Configuration (CameraInterface)
    # ==========================================================================
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get current camera resolution."""
        if not self.is_open():
            return (0, 0)
        
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def set_resolution(self, width: int, height: int) -> bool:
        """Set camera resolution."""
        if not self.is_open():
            return False
        
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Log if actual differs from requested
        actual_w, actual_h = self.get_resolution()
        if actual_w != width or actual_h != height:
            logging.info(f"Requested {width}x{height}, camera using {actual_w}x{actual_h}")
        
        return True
    
    def get_fps(self) -> float:
        """Get current frame rate."""
        if not self.is_open():
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)
    
    def set_fps(self, fps: float) -> bool:
        """Set target frame rate."""
        if not self.is_open():
            return False
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        return True
    
    # ==========================================================================
    # Device Information (CameraInterface)
    # ==========================================================================
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the connected device."""
        width, height = self.get_resolution()
        
        # Try to get backend name
        try:
            backend_name = self._cap.getBackendName() if self.is_open() else "N/A"
        except AttributeError:
            backend_name = self._backend_constant_to_name(self._backend)
        
        return {
            "provider": self.get_provider_name(),
            "device_index": self._device_index,
            "backend": backend_name,
            "resolution": (width, height),
            "fps": self.get_fps(),
        }
    
    # ==========================================================================
    # Class Methods (CameraInterface)
    # ==========================================================================
    
    @classmethod
    def enumerate_devices(
        cls,
        backend: Optional[int] = None,
        max_devices: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Enumerate available OpenCV camera devices.
        
        Args:
            backend: Specific backend to use (None for OS default)
            max_devices: Maximum number of device indices to check
            
        Returns:
            List of device info dictionaries
        """
        if backend is None:
            backend = cls._get_default_backend()
        
        devices = []
        for i in range(max_devices):
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        # Get resolution info
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        devices.append({
                            "index": i,
                            "name": f"Camera {i}",
                            "provider": cls.get_provider_name(),
                            "resolution": (width, height),
                        })
                    cap.release()
            except Exception as e:
                logging.debug(f"Error checking camera {i}: {e}")
                continue
        
        return devices
    
    @classmethod
    def get_provider_name(cls) -> str:
        """Get human-readable provider name."""
        return "OpenCV (Webcam)"
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if OpenCV is available (always True - core dependency)."""
        return True
    
    # ==========================================================================
    # OpenCV-Specific Methods
    # ==========================================================================
    
    @classmethod
    def get_available_backends(cls) -> List[Tuple[str, int]]:
        """Get backends available for current OS."""
        os_name = platform.system()
        return cls.BACKENDS.get(os_name, [("Auto", cv2.CAP_ANY)])
    
    def get_backend(self) -> int:
        """Get current backend constant."""
        return self._backend
    
    def get_backend_name(self) -> str:
        """Get current backend name."""
        return self._backend_constant_to_name(self._backend)
    
    @property
    def device_index(self) -> int:
        """Get device index."""
        return self._device_index
    
    @property
    def native_capture(self) -> Optional[cv2.VideoCapture]:
        """
        Access underlying cv2.VideoCapture for advanced usage.
        
        Warning: Use with caution - may break abstraction.
        """
        return self._cap


