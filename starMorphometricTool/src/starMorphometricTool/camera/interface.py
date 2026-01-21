"""
Camera Interface - Abstract Base Class

Defines the contract that all camera providers must implement.
This abstraction allows the application to work with any camera type
(OpenCV webcams, Basler, GigE, etc.) without modification.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


class CameraInterface(ABC):
    """
    Abstract interface for camera devices.
    
    All camera providers (OpenCV, Basler, GigE, etc.) must implement this interface.
    This allows the application to work with any camera type without modification.
    
    Example:
        camera = OpenCVCamera(device_index=0)
        if camera.open():
            success, frame = camera.read_frame()
            if success:
                # Process frame (BGR numpy array)
                pass
            camera.close()
    """
    
    # ==========================================================================
    # Connection Management
    # ==========================================================================
    
    @abstractmethod
    def open(self) -> bool:
        """
        Open connection to the camera.
        
        Returns:
            bool: True if successfully opened, False otherwise
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Release camera resources.
        
        Should be safe to call multiple times.
        """
        pass
    
    @abstractmethod
    def is_open(self) -> bool:
        """
        Check if camera connection is active.
        
        Returns:
            bool: True if camera is open and ready
        """
        pass
    
    # ==========================================================================
    # Frame Capture
    # ==========================================================================
    
    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Tuple[bool, ndarray|None]: 
                - (True, frame) on success - frame is BGR numpy array
                - (False, None) on failure
        """
        pass
    
    # ==========================================================================
    # Configuration - Resolution
    # ==========================================================================
    
    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """
        Get current camera resolution.
        
        Returns:
            Tuple[int, int]: (width, height) in pixels
        """
        pass
    
    @abstractmethod
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set camera resolution.
        
        Note: The camera may not support the exact resolution requested.
        Use get_resolution() after setting to verify actual resolution.
        
        Args:
            width: Desired width in pixels
            height: Desired height in pixels
            
        Returns:
            bool: True if setting was applied (may differ from requested)
        """
        pass
    
    # ==========================================================================
    # Configuration - Frame Rate
    # ==========================================================================
    
    @abstractmethod
    def get_fps(self) -> float:
        """
        Get current frame rate.
        
        Returns:
            float: Frames per second (0.0 if not available)
        """
        pass
    
    @abstractmethod
    def set_fps(self, fps: float) -> bool:
        """
        Set target frame rate.
        
        Note: The camera may not support the exact FPS requested.
        
        Args:
            fps: Desired frames per second
            
        Returns:
            bool: True if setting was applied
        """
        pass
    
    # ==========================================================================
    # Device Information
    # ==========================================================================
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the connected device.
        
        Returns:
            dict: Device info including at minimum:
                - 'provider': Camera provider name
                - 'device_index' or 'device_id': Device identifier
                - 'resolution': Current (width, height)
                - 'fps': Current frame rate
        """
        pass
    
    # ==========================================================================
    # Class Methods - Device Discovery
    # ==========================================================================
    
    @classmethod
    @abstractmethod
    def enumerate_devices(cls, **kwargs) -> List[Dict[str, Any]]:
        """
        List all available devices of this camera type.
        
        This is a class method so it can be called without instantiation.
        
        Args:
            **kwargs: Provider-specific options
            
        Returns:
            List of device info dictionaries, each containing at minimum:
                - 'index' or 'id': Device identifier for creating camera
                - 'name': Human-readable device name
                - 'provider': Provider name
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_provider_name(cls) -> str:
        """
        Get human-readable name for this camera provider.
        
        Returns:
            str: Provider name (e.g., "OpenCV (Webcam)", "Basler", "GigE Vision")
        """
        pass
    
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """
        Check if this camera provider's dependencies are installed.
        
        For example, BaslerCamera would check if pypylon is installed.
        
        Returns:
            bool: True if provider can be used
        """
        pass
    
    # ==========================================================================
    # Optional Methods (default implementations)
    # ==========================================================================
    
    def __enter__(self) -> "CameraInterface":
        """Context manager entry - opens camera."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes camera."""
        self.close()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<{self.__class__.__name__} open={self.is_open()}>"


