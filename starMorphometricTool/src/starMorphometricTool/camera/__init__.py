"""
Camera Package

Provides an abstract camera interface and implementations for various camera types.
This abstraction layer allows the application to work with different camera backends
(OpenCV webcams, Basler, GigE, etc.) through a unified interface.

Usage:
    from camera import CameraInterface, create_camera, auto_detect_camera
    
    # Auto-detect and open first available camera
    camera = auto_detect_camera()
    if camera:
        success, frame = camera.read_frame()
        camera.close()
    
    # Or create specific camera type
    camera = create_camera("opencv", device_index=0)
"""

from camera.interface import CameraInterface
from camera.factory import (
    create_camera,
    auto_detect_camera,
    get_available_providers,
    enumerate_all_devices,
)
from camera.config import (
    load_camera_config,
    save_camera_config,
    get_default_config,
    DEFAULT_CONFIG_PATH,
)

__all__ = [
    # Interface
    "CameraInterface",
    # Factory
    "create_camera",
    "auto_detect_camera",
    "get_available_providers",
    "enumerate_all_devices",
    # Config
    "load_camera_config",
    "save_camera_config",
    "get_default_config",
    "DEFAULT_CONFIG_PATH",
]


