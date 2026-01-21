# Adding New Camera Providers

This guide explains how to add support for new camera types (Basler, GigE Vision, FLIR, etc.) to the starMorphometricTool camera abstraction layer.

## Architecture Overview

The camera system uses a **Provider Pattern** where all camera types implement a common `CameraInterface`:

```
┌─────────────────────────────────────────────────────┐
│              Application (UI)                        │
│  - Uses CameraInterface methods only                │
│  - No knowledge of specific camera types            │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              CameraInterface (ABC)                   │
│  - Defines the contract all cameras implement       │
│  - Located in: camera/interface.py                  │
└─────────────────────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ OpenCV   │  │ Basler   │  │  GigE    │
    │ Camera   │  │ Camera   │  │ Camera   │
    └──────────┘  └──────────┘  └──────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `camera/interface.py` | Abstract base class defining the camera contract |
| `camera/factory.py` | Provider registry and camera creation functions |
| `camera/config.py` | Configuration persistence |
| `camera/providers/*.py` | Concrete camera implementations |

---

## Step-by-Step: Adding a New Camera Provider

### Step 1: Create the Provider File

Create a new file in `camera/providers/` for your camera type:

```
camera/providers/your_camera.py
```

### Step 2: Implement the CameraInterface

Your provider class must inherit from `CameraInterface` and implement all abstract methods:

```python
"""
Your Camera Provider

Implements CameraInterface for [Your Camera Type] devices.
"""

import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from camera.interface import CameraInterface


class YourCamera(CameraInterface):
    """
    Camera provider for [Your Camera Type] devices.
    
    Requirements:
        - your_sdk_package (pip install your-sdk-package)
    """
    
    def __init__(self, device_id: str = None, **kwargs):
        """
        Initialize camera.
        
        Args:
            device_id: Device identifier (serial number, IP, etc.)
            **kwargs: Additional provider-specific options
        """
        self._device_id = device_id
        self._camera = None  # Your SDK's camera object
        self._is_open = False
    
    # =========================================================================
    # Connection Management (REQUIRED)
    # =========================================================================
    
    def open(self) -> bool:
        """
        Open connection to the camera.
        
        Returns:
            True if successfully opened
        """
        try:
            # 1. Import your SDK (do this here to allow graceful failure)
            # 2. Find/connect to the camera
            # 3. Configure initial settings
            # 4. Start acquisition if needed
            
            self._is_open = True
            return True
        except Exception as e:
            logging.error(f"Failed to open camera: {e}")
            self._is_open = False
            return False
    
    def close(self) -> None:
        """Release camera resources."""
        if self._camera is not None:
            try:
                # 1. Stop acquisition
                # 2. Disconnect
                # 3. Release resources
                pass
            except Exception as e:
                logging.debug(f"Error closing camera: {e}")
            finally:
                self._camera = None
                self._is_open = False
    
    def is_open(self) -> bool:
        """Check if camera is connected and ready."""
        return self._is_open and self._camera is not None
    
    # =========================================================================
    # Frame Capture (REQUIRED)
    # =========================================================================
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single frame.
        
        Returns:
            (True, BGR numpy array) on success
            (False, None) on failure
            
        IMPORTANT: Frame must be in BGR format (OpenCV convention)
        """
        if not self.is_open():
            return False, None
        
        try:
            # 1. Grab frame from camera
            # 2. Convert to numpy array
            # 3. Ensure BGR format (convert if necessary)
            
            frame = None  # Your frame capture code here
            return True, frame
        except Exception as e:
            logging.error(f"Error reading frame: {e}")
            return False, None
    
    # =========================================================================
    # Configuration (REQUIRED)
    # =========================================================================
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get current (width, height)."""
        if not self.is_open():
            return (0, 0)
        # Return (width, height) from your camera
        return (0, 0)
    
    def set_resolution(self, width: int, height: int) -> bool:
        """Set resolution. Returns True if applied."""
        if not self.is_open():
            return False
        # Apply resolution to your camera
        return True
    
    def get_fps(self) -> float:
        """Get current frame rate."""
        if not self.is_open():
            return 0.0
        # Return FPS from your camera
        return 0.0
    
    def set_fps(self, fps: float) -> bool:
        """Set frame rate. Returns True if applied."""
        if not self.is_open():
            return False
        # Apply FPS to your camera
        return True
    
    # =========================================================================
    # Device Information (REQUIRED)
    # =========================================================================
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about connected device."""
        return {
            "provider": self.get_provider_name(),
            "device_id": self._device_id,
            "resolution": self.get_resolution(),
            "fps": self.get_fps(),
            # Add any provider-specific info
        }
    
    # =========================================================================
    # Class Methods (REQUIRED)
    # =========================================================================
    
    @classmethod
    def enumerate_devices(cls, **kwargs) -> List[Dict[str, Any]]:
        """
        List all available devices of this type.
        
        Returns:
            List of dicts, each containing at minimum:
            - 'id': Device identifier for creating camera
            - 'name': Human-readable name
            - 'provider': Provider name
        """
        devices = []
        
        try:
            # Import SDK and enumerate devices
            # For each device found:
            #   devices.append({
            #       "id": device_serial_or_ip,
            #       "name": f"Device {device_serial_or_ip}",
            #       "provider": cls.get_provider_name(),
            #   })
            pass
        except ImportError:
            logging.debug("SDK not installed, no devices to enumerate")
        except Exception as e:
            logging.error(f"Error enumerating devices: {e}")
        
        return devices
    
    @classmethod
    def get_provider_name(cls) -> str:
        """Human-readable provider name."""
        return "Your Camera Type"
    
    @classmethod
    def is_available(cls) -> bool:
        """
        Check if this provider can be used.
        
        Returns True if the required SDK/dependencies are installed.
        """
        try:
            import your_sdk_package
            return True
        except ImportError:
            return False
```

### Step 3: Register the Provider

Edit `camera/factory.py` to register your provider:

```python
# At the top, import your provider
from camera.providers.your_camera import YourCamera

# Add to the _PROVIDERS dictionary
_PROVIDERS: Dict[str, Type[CameraInterface]] = {
    "opencv": OpenCVCamera,
    "your_camera": YourCamera,  # Add this line
}
```

### Step 4: Update the Providers Package

Edit `camera/providers/__init__.py`:

```python
from camera.providers.opencv_camera import OpenCVCamera
from camera.providers.your_camera import YourCamera  # Add this

__all__ = [
    "OpenCVCamera",
    "YourCamera",  # Add this
]
```

### Step 5: Add Dependencies (Optional)

If your camera requires additional packages, add them to `requirements.txt`:

```
# Optional: For [Your Camera Type] support
your-sdk-package>=1.0.0
```

---

## CameraInterface Method Reference

### Connection Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `open()` | `bool` | Open camera connection. Return `True` on success. |
| `close()` | `None` | Release all resources. Safe to call multiple times. |
| `is_open()` | `bool` | Check if camera is connected and ready. |

### Frame Capture

| Method | Returns | Description |
|--------|---------|-------------|
| `read_frame()` | `(bool, ndarray\|None)` | Capture frame. **Must return BGR format.** |

### Configuration

| Method | Returns | Description |
|--------|---------|-------------|
| `get_resolution()` | `(int, int)` | Get (width, height) in pixels. |
| `set_resolution(w, h)` | `bool` | Set resolution. May not match exactly. |
| `get_fps()` | `float` | Get frame rate. Return 0.0 if unknown. |
| `set_fps(fps)` | `bool` | Set frame rate. |

### Class Methods (Discovery)

| Method | Returns | Description |
|--------|---------|-------------|
| `enumerate_devices(**kwargs)` | `List[Dict]` | List available devices. |
| `get_provider_name()` | `str` | Human-readable name (e.g., "Basler"). |
| `is_available()` | `bool` | Check if SDK is installed. |

---

## Example: Basler Camera Implementation

Here's a skeleton for Basler cameras using pypylon:

```python
"""
Basler Camera Provider

Implements CameraInterface for Basler cameras using pypylon SDK.
"""

import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from camera.interface import CameraInterface


class BaslerCamera(CameraInterface):
    """
    Camera provider for Basler cameras.
    
    Requirements:
        pip install pypylon
    """
    
    def __init__(self, serial_number: str = None, ip_address: str = None):
        self._serial_number = serial_number
        self._ip_address = ip_address
        self._camera = None
        self._converter = None
    
    def open(self) -> bool:
        try:
            from pypylon import pylon
            
            # Get the transport layer factory
            tlf = pylon.TlFactory.GetInstance()
            
            if self._serial_number:
                # Find specific camera by serial
                devices = tlf.EnumerateDevices()
                for device in devices:
                    if device.GetSerialNumber() == self._serial_number:
                        self._camera = pylon.InstantCamera(tlf.CreateDevice(device))
                        break
            else:
                # Use first available camera
                self._camera = pylon.InstantCamera(tlf.CreateFirstDevice())
            
            if self._camera is None:
                return False
            
            self._camera.Open()
            
            # Setup converter for BGR output
            self._converter = pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            
            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            return True
            
        except Exception as e:
            logging.error(f"Failed to open Basler camera: {e}")
            return False
    
    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.StopGrabbing()
                self._camera.Close()
            except:
                pass
            self._camera = None
    
    def is_open(self) -> bool:
        return self._camera is not None and self._camera.IsOpen()
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_open():
            return False, None
        
        try:
            from pypylon import pylon
            
            grab_result = self._camera.RetrieveResult(
                5000, pylon.TimeoutHandling_ThrowException
            )
            
            if grab_result.GrabSucceeded():
                image = self._converter.Convert(grab_result)
                frame = image.GetArray()
                grab_result.Release()
                return True, frame
            
            grab_result.Release()
            return False, None
            
        except Exception as e:
            logging.error(f"Error grabbing frame: {e}")
            return False, None
    
    def get_resolution(self) -> Tuple[int, int]:
        if not self.is_open():
            return (0, 0)
        return (
            self._camera.Width.GetValue(),
            self._camera.Height.GetValue()
        )
    
    def set_resolution(self, width: int, height: int) -> bool:
        if not self.is_open():
            return False
        try:
            self._camera.Width.SetValue(width)
            self._camera.Height.SetValue(height)
            return True
        except:
            return False
    
    def get_fps(self) -> float:
        if not self.is_open():
            return 0.0
        try:
            return self._camera.ResultingFrameRate.GetValue()
        except:
            return 0.0
    
    def set_fps(self, fps: float) -> bool:
        if not self.is_open():
            return False
        try:
            self._camera.AcquisitionFrameRateEnable.SetValue(True)
            self._camera.AcquisitionFrameRate.SetValue(fps)
            return True
        except:
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        info = {
            "provider": self.get_provider_name(),
            "resolution": self.get_resolution(),
            "fps": self.get_fps(),
        }
        if self.is_open():
            info["serial_number"] = self._camera.DeviceInfo.GetSerialNumber()
            info["model"] = self._camera.DeviceInfo.GetModelName()
        return info
    
    @classmethod
    def enumerate_devices(cls, **kwargs) -> List[Dict[str, Any]]:
        devices = []
        try:
            from pypylon import pylon
            
            tlf = pylon.TlFactory.GetInstance()
            for device in tlf.EnumerateDevices():
                devices.append({
                    "id": device.GetSerialNumber(),
                    "name": f"{device.GetModelName()} ({device.GetSerialNumber()})",
                    "provider": cls.get_provider_name(),
                    "serial_number": device.GetSerialNumber(),
                    "model": device.GetModelName(),
                })
        except ImportError:
            pass
        except Exception as e:
            logging.error(f"Error enumerating Basler cameras: {e}")
        return devices
    
    @classmethod
    def get_provider_name(cls) -> str:
        return "Basler"
    
    @classmethod
    def is_available(cls) -> bool:
        try:
            import pypylon
            return True
        except ImportError:
            return False
```

---

## Testing Checklist

Before submitting a new camera provider, verify:

- [ ] `is_available()` returns `False` when SDK not installed (no crash)
- [ ] `is_available()` returns `True` when SDK is installed
- [ ] `enumerate_devices()` works without a camera connected (returns empty list)
- [ ] `enumerate_devices()` finds connected cameras
- [ ] `open()` succeeds with valid camera
- [ ] `open()` returns `False` with invalid/disconnected camera (no crash)
- [ ] `read_frame()` returns BGR format numpy array
- [ ] `read_frame()` returns `(False, None)` when camera not open
- [ ] `close()` can be called multiple times without error
- [ ] `get_resolution()` returns actual camera resolution
- [ ] `set_resolution()` applies settings (verify with `get_resolution()`)
- [ ] `get_device_info()` returns meaningful information
- [ ] Camera works in the main application UI
- [ ] Preview works in camera configuration dialog

---

## Troubleshooting

### Provider not appearing in available cameras

1. Check `is_available()` returns `True`
2. Verify import in `providers/__init__.py`
3. Verify registration in `factory.py`

### Frames not displaying correctly

1. Ensure `read_frame()` returns BGR format (not RGB)
2. Check numpy array dtype is `uint8`
3. Verify array shape is `(height, width, 3)`

### Camera hangs or times out

1. Add timeout handling in `read_frame()`
2. Implement proper cleanup in `close()`
3. Check if camera requires explicit acquisition start

---

## Questions?

Refer to the existing `OpenCVCamera` implementation in `camera/providers/opencv_camera.py` as a working reference.


