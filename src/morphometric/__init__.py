# src/morphometric/__init__.py
"""
Morphometric module for starBoard - integrates starMorphometricTool functionality.

This module provides webcam-based morphometric measurements for sunflower sea stars.
Components are lazily loaded to avoid startup overhead when the morphometric tab
is not being used.

Usage:
    from src.morphometric import get_camera_adapter, get_detection_adapter, get_analysis_adapter
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger("starBoard.morphometric")

# Path to the starMorphometricTool source
_MORPH_TOOL_ROOT = Path(__file__).parent.parent.parent / "starMorphometricTool"
_MORPH_TOOL_SRC = _MORPH_TOOL_ROOT / "src" / "starMorphometricTool"

# Track whether we've added the path
_path_added = False


def _ensure_morphometric_path() -> bool:
    """
    Ensure the starMorphometricTool source is on the Python path.
    
    Returns:
        True if path was added successfully, False otherwise.
    """
    global _path_added
    if _path_added:
        return True
    
    if not _MORPH_TOOL_SRC.exists():
        logger.warning("starMorphometricTool source not found at %s", _MORPH_TOOL_SRC)
        return False
    
    src_path = str(_MORPH_TOOL_SRC)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        logger.debug("Added morphometric tool path: %s", src_path)
    
    _path_added = True
    return True


def is_available() -> bool:
    """
    Check if the morphometric module dependencies are available.
    
    Returns:
        True if all required dependencies can be imported.
    """
    if not _ensure_morphometric_path():
        return False
    
    try:
        # Test critical imports
        import cv2
        import numpy
        from ultralytics import YOLO
        return True
    except ImportError as e:
        logger.debug("Morphometric dependencies not available: %s", e)
        return False


def get_morphometric_tool_root() -> Path:
    """Get the root path of the starMorphometricTool."""
    return _MORPH_TOOL_ROOT


def get_measurements_root() -> Path:
    """Get the measurements storage root path."""
    return _MORPH_TOOL_ROOT / "measurements"


# Lazy-loaded adapters
_camera_adapter: Optional["CameraAdapter"] = None
_detection_adapter: Optional["DetectionAdapter"] = None
_analysis_adapter: Optional["AnalysisAdapter"] = None
_depth_adapter: Optional["DepthAdapter"] = None


def get_camera_adapter() -> "CameraAdapter":
    """
    Get the camera adapter instance (lazy-loaded).
    
    Returns:
        CameraAdapter instance for webcam operations.
    
    Raises:
        ImportError: If camera dependencies are not available.
    """
    global _camera_adapter
    if _camera_adapter is None:
        _ensure_morphometric_path()
        from .camera_adapter import CameraAdapter
        _camera_adapter = CameraAdapter()
    return _camera_adapter


def get_detection_adapter() -> "DetectionAdapter":
    """
    Get the detection adapter instance (lazy-loaded).
    
    Returns:
        DetectionAdapter instance for YOLO and checkerboard detection.
    
    Raises:
        ImportError: If detection dependencies are not available.
    """
    global _detection_adapter
    if _detection_adapter is None:
        _ensure_morphometric_path()
        from .detection_adapter import DetectionAdapter
        _detection_adapter = DetectionAdapter()
    return _detection_adapter


def get_analysis_adapter() -> "AnalysisAdapter":
    """
    Get the analysis adapter instance (lazy-loaded).
    
    Returns:
        AnalysisAdapter instance for morphometric analysis.
    
    Raises:
        ImportError: If analysis dependencies are not available.
    """
    global _analysis_adapter
    if _analysis_adapter is None:
        _ensure_morphometric_path()
        from .analysis_adapter import AnalysisAdapter
        _analysis_adapter = AnalysisAdapter()
    return _analysis_adapter


def is_depth_available() -> bool:
    """
    Check if depth estimation (Depth-Anything-V2) is available.
    
    This is separate from is_available() because depth estimation
    has additional dependencies (PyTorch, Depth-Anything-V2 model).
    
    Returns:
        True if depth estimation can be used.
    """
    if not _ensure_morphometric_path():
        return False
    
    try:
        from .depth_adapter import DepthAdapter
        return DepthAdapter.is_available()
    except ImportError:
        return False


def get_depth_adapter() -> "DepthAdapter":
    """
    Get the depth adapter instance (lazy-loaded).
    
    Returns:
        DepthAdapter instance for depth estimation and volume computation.
    
    Raises:
        ImportError: If depth dependencies are not available.
    """
    global _depth_adapter
    if _depth_adapter is None:
        _ensure_morphometric_path()
        from .depth_adapter import DepthAdapter
        _depth_adapter = DepthAdapter()
    return _depth_adapter


def clear_adapters() -> None:
    """Clear all cached adapter instances (useful for testing or cleanup)."""
    global _camera_adapter, _detection_adapter, _analysis_adapter, _depth_adapter
    
    if _camera_adapter is not None:
        try:
            _camera_adapter.close()
        except Exception:
            pass
        _camera_adapter = None
    
    if _depth_adapter is not None:
        try:
            _depth_adapter.clear()
        except Exception:
            pass
        _depth_adapter = None
    
    _detection_adapter = None
    _analysis_adapter = None
    logger.debug("Cleared morphometric adapters")


# Type hints for lazy imports
if TYPE_CHECKING:
    from .camera_adapter import CameraAdapter
    from .detection_adapter import DetectionAdapter
    from .analysis_adapter import AnalysisAdapter
    from .depth_adapter import DepthAdapter

