"""
Camera Utilities Module - DEPRECATED

This module is deprecated. Please use the camera package instead:

    # Old way (deprecated):
    from utils.camera_utils import load_camera_config, enumerate_cameras
    
    # New way:
    from camera import load_camera_config
    from camera.factory import enumerate_all_devices

The camera package provides a proper abstraction layer for multiple camera types.
"""

import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "utils.camera_utils is deprecated. Use the 'camera' package instead. "
    "See camera/__init__.py for available exports.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new locations for backward compatibility
from camera.config import (
    load_camera_config,
    save_camera_config,
    get_default_config,
    DEFAULT_CONFIG_PATH,
)

from camera.factory import (
    auto_detect_camera,
    enumerate_all_devices as enumerate_cameras,
    create_camera_from_config as open_camera_from_config,
)

from camera.providers.opencv_camera import OpenCVCamera

# Backward compatibility constants
COMMON_RESOLUTIONS = [
    (640, 480, "640x480 (VGA)"),
    (800, 600, "800x600 (SVGA)"),
    (1280, 720, "1280x720 (720p)"),
    (1920, 1080, "1920x1080 (1080p)"),
]

COMMON_FRAME_RATES = [15, 24, 30, 60]


# Backward compatibility functions

def get_available_backends():
    """
    DEPRECATED: Use OpenCVCamera.get_available_backends() instead.
    """
    warnings.warn(
        "get_available_backends() is deprecated. Use OpenCVCamera.get_available_backends()",
        DeprecationWarning,
        stacklevel=2
    )
    return OpenCVCamera.get_available_backends()


def backend_name_to_constant(name):
    """
    DEPRECATED: Use OpenCVCamera._backend_name_to_constant() instead.
    """
    warnings.warn(
        "backend_name_to_constant() is deprecated. Use OpenCVCamera._backend_name_to_constant()",
        DeprecationWarning,
        stacklevel=2
    )
    return OpenCVCamera._backend_name_to_constant(name)


def initialize_camera_system(config_path=DEFAULT_CONFIG_PATH):
    """
    DEPRECATED: Use camera.factory functions instead.
    
    This function is kept for backward compatibility but will be removed
    in a future version.
    """
    warnings.warn(
        "initialize_camera_system() is deprecated. Use camera.factory.auto_detect_with_config()",
        DeprecationWarning,
        stacklevel=2
    )
    from camera.factory import auto_detect_with_config
    from camera.config import load_camera_config as load_config
    
    # Try saved config first
    saved_config = load_config(config_path)
    if saved_config:
        from camera.factory import create_camera_from_config
        camera = create_camera_from_config(saved_config)
        if camera and camera.open():
            # Return (cap, config, needs_dialog) format for compatibility
            # But camera is now CameraInterface, not cv2.VideoCapture
            return camera, saved_config, False
        elif camera:
            camera.close()
    
    # Auto-detect
    camera, config = auto_detect_with_config()
    if camera:
        return camera, config, False
    
    return None, None, True


def open_camera(camera_index=0, backend=None, width=None, height=None, fps=None):
    """
    DEPRECATED: Use camera.factory.create_camera() instead.
    """
    warnings.warn(
        "open_camera() is deprecated. Use camera.factory.create_camera()",
        DeprecationWarning,
        stacklevel=2
    )
    from camera.factory import create_camera
    
    camera = create_camera("opencv", device_index=camera_index, backend=backend)
    if camera is None:
        return None, False, "Failed to create camera"
    
    if not camera.open():
        return None, False, "Failed to open camera"
    
    if width and height:
        camera.set_resolution(width, height)
    if fps:
        camera.set_fps(fps)
    
    return camera, True, None
