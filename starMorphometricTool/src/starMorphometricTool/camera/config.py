"""
Camera Configuration Management

Handles loading, saving, and managing camera configuration.
Provides persistence of user's camera settings between sessions.
"""

import json
import os
import logging
from typing import Dict, Any, Optional

import cv2


# Default config file path (relative to app working directory)
DEFAULT_CONFIG_PATH = "camera_config.json"


# =============================================================================
# Default Configuration
# =============================================================================

def get_default_config() -> Dict[str, Any]:
    """
    Get default camera configuration.
    
    Attempts to detect the camera's native resolution rather than using
    hardcoded values, falling back to 1280x720 if detection fails.
    
    Returns:
        Default configuration dictionary
    """
    config = {
        "provider": "opencv",
        "device_index": 0,
        "backend": "Auto",
        "width": 1280,   # Fallback
        "height": 720,
        "fps": 30,
    }
    
    # Try to get camera's native resolution
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if native_width > 0 and native_height > 0:
                config["width"] = native_width
                config["height"] = native_height
                logging.debug(f"Detected native camera resolution: {native_width}x{native_height}")
            cap.release()
    except Exception as e:
        logging.debug(f"Could not detect native camera resolution: {e}")
    
    return config


# =============================================================================
# Configuration Persistence
# =============================================================================

def load_camera_config(config_path: str = DEFAULT_CONFIG_PATH) -> Optional[Dict[str, Any]]:
    """
    Load camera configuration from file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary or None if file doesn't exist
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logging.debug(f"Loaded camera config from {config_path}")
            
            # Ensure provider field exists (backward compatibility)
            if "provider" not in config:
                config["provider"] = "opencv"
            
            return config
    except json.JSONDecodeError as e:
        logging.warning(f"Invalid JSON in camera config: {e}")
    except Exception as e:
        logging.warning(f"Error loading camera config: {e}")
    
    return None


def save_camera_config(
    config: Dict[str, Any],
    config_path: str = DEFAULT_CONFIG_PATH
) -> bool:
    """
    Save camera configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
        
    Returns:
        True if saved successfully
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logging.debug(f"Saved camera config to {config_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving camera config: {e}")
        return False


def delete_camera_config(config_path: str = DEFAULT_CONFIG_PATH) -> bool:
    """
    Delete camera configuration file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        True if deleted (or didn't exist)
    """
    try:
        if os.path.exists(config_path):
            os.remove(config_path)
            logging.debug(f"Deleted camera config at {config_path}")
        return True
    except Exception as e:
        logging.error(f"Error deleting camera config: {e}")
        return False


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config(config: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate a camera configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["provider", "device_index"]
    
    for field in required_fields:
        if field not in config:
            return False, f"Missing required field: {field}"
    
    # Validate provider
    if not isinstance(config["provider"], str):
        return False, "Provider must be a string"
    
    # Validate device_index
    if not isinstance(config["device_index"], int) or config["device_index"] < 0:
        return False, "Device index must be a non-negative integer"
    
    # Validate optional numeric fields
    if "width" in config:
        if not isinstance(config["width"], int) or config["width"] <= 0:
            return False, "Width must be a positive integer"
    
    if "height" in config:
        if not isinstance(config["height"], int) or config["height"] <= 0:
            return False, "Height must be a positive integer"
    
    if "fps" in config:
        if not isinstance(config["fps"], (int, float)) or config["fps"] <= 0:
            return False, "FPS must be a positive number"
    
    return True, ""


def merge_config_with_defaults(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge user config with defaults, filling in missing values.
    
    Args:
        config: User configuration (may be None or partial)
        
    Returns:
        Complete configuration with defaults for missing values
    """
    defaults = get_default_config()
    
    if config is None:
        return defaults
    
    # Merge, preferring user values
    merged = defaults.copy()
    merged.update(config)
    
    return merged


# =============================================================================
# High-Level Configuration Management
# =============================================================================

def initialize_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Initialize camera configuration.
    
    Loads existing config if available, otherwise creates default.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary (loaded or default)
    """
    config = load_camera_config(config_path)
    
    if config is not None:
        # Validate and merge with defaults to ensure all fields exist
        is_valid, error = validate_config(config)
        if is_valid:
            return merge_config_with_defaults(config)
        else:
            logging.warning(f"Invalid saved config ({error}), using defaults")
    
    return get_default_config()


