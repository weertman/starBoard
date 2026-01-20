"""
Camera Factory

Provides a registry of camera providers and factory functions for creating
camera instances. This enables the application to work with multiple camera
types through a unified interface.
"""

import logging
from typing import Dict, List, Type, Optional, Any

from camera.interface import CameraInterface
from camera.providers.opencv_camera import OpenCVCamera


# =============================================================================
# Provider Registry
# =============================================================================

# Registry mapping provider names to their classes
_PROVIDERS: Dict[str, Type[CameraInterface]] = {
    "opencv": OpenCVCamera,
    # Future providers:
    # "basler": BaslerCamera,
    # "gige": GigECamera,
}


def register_provider(name: str, provider_class: Type[CameraInterface]) -> None:
    """
    Register a new camera provider.
    
    This allows plugins or extensions to add new camera types at runtime.
    
    Args:
        name: Unique provider name (lowercase)
        provider_class: Class implementing CameraInterface
    """
    _PROVIDERS[name.lower()] = provider_class
    logging.debug(f"Registered camera provider: {name}")


def get_provider_class(provider_name: str) -> Optional[Type[CameraInterface]]:
    """
    Get camera provider class by name.
    
    Args:
        provider_name: Provider name (case-insensitive)
        
    Returns:
        Provider class or None if not found
    """
    return _PROVIDERS.get(provider_name.lower())


def get_all_providers() -> Dict[str, Type[CameraInterface]]:
    """Get all registered providers (including unavailable ones)."""
    return _PROVIDERS.copy()


# =============================================================================
# Provider Discovery
# =============================================================================

def get_available_providers() -> List[str]:
    """
    Get list of available (installed) camera providers.
    
    Only returns providers whose dependencies are installed.
    
    Returns:
        List of provider names
    """
    return [
        name for name, provider in _PROVIDERS.items()
        if provider.is_available()
    ]


def get_provider_info() -> List[Dict[str, Any]]:
    """
    Get information about all registered providers.
    
    Returns:
        List of dicts with provider info:
            - name: Provider identifier
            - display_name: Human-readable name
            - available: Whether dependencies are installed
    """
    info = []
    for name, provider in _PROVIDERS.items():
        info.append({
            "name": name,
            "display_name": provider.get_provider_name(),
            "available": provider.is_available(),
        })
    return info


# =============================================================================
# Camera Creation
# =============================================================================

def create_camera(
    provider: str = "opencv",
    **kwargs
) -> Optional[CameraInterface]:
    """
    Create a camera instance.
    
    Args:
        provider: Provider name ("opencv", "basler", "gige", etc.)
        **kwargs: Provider-specific arguments:
            - OpenCV: device_index, backend, backend_name
            - Basler: serial_number, ip_address (future)
            - GigE: ip_address, mac_address (future)
            
    Returns:
        CameraInterface instance or None if provider not found/unavailable
        
    Example:
        camera = create_camera("opencv", device_index=0)
        if camera and camera.open():
            success, frame = camera.read_frame()
            camera.close()
    """
    provider_class = get_provider_class(provider)
    
    if provider_class is None:
        logging.error(f"Unknown camera provider: {provider}")
        return None
    
    if not provider_class.is_available():
        logging.error(f"Camera provider '{provider}' is not available (dependencies not installed)")
        return None
    
    try:
        return provider_class(**kwargs)
    except Exception as e:
        logging.error(f"Error creating camera: {e}")
        return None


def create_camera_from_config(config: Dict[str, Any]) -> Optional[CameraInterface]:
    """
    Create a camera from a configuration dictionary.
    
    Args:
        config: Configuration dict with:
            - provider: Provider name
            - device_index: Device index (for OpenCV)
            - backend: Backend name (for OpenCV)
            - width, height, fps: Optional settings to apply after open
            
    Returns:
        CameraInterface instance or None
    """
    provider = config.get("provider", "opencv")
    
    # Build kwargs based on provider
    if provider == "opencv":
        kwargs = {
            "device_index": config.get("device_index", 0),
            "backend_name": config.get("backend"),
        }
    else:
        # Generic - pass all config except known meta fields
        kwargs = {k: v for k, v in config.items() 
                  if k not in ("provider", "width", "height", "fps")}
    
    return create_camera(provider, **kwargs)


# =============================================================================
# Device Enumeration
# =============================================================================

def enumerate_all_devices() -> List[Dict[str, Any]]:
    """
    Enumerate devices from all available providers.
    
    Returns:
        Combined list of device info from all providers
    """
    all_devices = []
    
    for name, provider in _PROVIDERS.items():
        if provider.is_available():
            try:
                devices = provider.enumerate_devices()
                all_devices.extend(devices)
            except Exception as e:
                logging.warning(f"Error enumerating {name} devices: {e}")
    
    return all_devices


def enumerate_devices_by_provider(provider_name: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Enumerate devices for a specific provider.
    
    Args:
        provider_name: Provider name
        **kwargs: Provider-specific enumeration options
        
    Returns:
        List of device info dictionaries
    """
    provider_class = get_provider_class(provider_name)
    
    if provider_class is None:
        logging.error(f"Unknown provider: {provider_name}")
        return []
    
    if not provider_class.is_available():
        logging.warning(f"Provider {provider_name} is not available")
        return []
    
    try:
        return provider_class.enumerate_devices(**kwargs)
    except Exception as e:
        logging.error(f"Error enumerating {provider_name} devices: {e}")
        return []


# =============================================================================
# Auto-Detection
# =============================================================================

def auto_detect_camera() -> Optional[CameraInterface]:
    """
    Auto-detect and open the first available camera.
    
    Tries each available provider in order, attempting to open the first
    device found for each.
    
    Returns:
        Open CameraInterface instance or None if no camera found
    """
    for provider_name in get_available_providers():
        provider_class = get_provider_class(provider_name)
        
        try:
            devices = provider_class.enumerate_devices()
            
            for device in devices:
                # Get device identifier
                device_id = device.get("index", device.get("id", 0))
                
                # Create and try to open camera
                camera = create_camera(provider_name, device_index=device_id)
                if camera is not None and camera.open():
                    logging.info(
                        f"Auto-detected camera: {provider_name} "
                        f"device {device_id} ({device.get('name', 'Unknown')})"
                    )
                    return camera
                elif camera is not None:
                    camera.close()
                    
        except Exception as e:
            logging.debug(f"Error during auto-detect with {provider_name}: {e}")
            continue
    
    logging.warning("No camera detected by any provider")
    return None


def auto_detect_with_config() -> tuple[Optional[CameraInterface], Optional[Dict[str, Any]]]:
    """
    Auto-detect camera and return both camera instance and its config.
    
    Returns:
        Tuple of (CameraInterface or None, config dict or None)
    """
    camera = auto_detect_camera()
    
    if camera is None:
        return None, None
    
    # Build config from detected camera
    info = camera.get_device_info()
    width, height = camera.get_resolution()
    
    config = {
        "provider": "opencv",  # Currently only OpenCV supported
        "device_index": info.get("device_index", 0),
        "backend": info.get("backend", "Auto"),
        "width": width,
        "height": height,
        "fps": camera.get_fps() or 30,
    }
    
    return camera, config


