"""
OpenCV Camera Provider

Implements the CameraInterface for OpenCV-compatible devices (USB webcams).
Supports multiple backends and codec preferences for cross-platform compatibility.
"""

import cv2
import platform
import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from camera.interface import CameraInterface


def _cv2_const(name: str) -> Optional[int]:
    """Get an OpenCV constant if present in the installed build."""
    return getattr(cv2, name, None)


def _build_backends() -> Dict[str, List[Tuple[str, int]]]:
    """Build backend preferences, filtering out unavailable constants."""
    backend_specs = {
        "Windows": [
            ("DirectShow", "CAP_DSHOW"),
            ("Media Foundation", "CAP_MSMF"),
            ("FFMPEG", "CAP_FFMPEG"),
            ("Auto", "CAP_ANY"),
        ],
        "Darwin": [  # macOS
            ("AVFoundation", "CAP_AVFOUNDATION"),
            ("QTKit", "CAP_QT"),
            ("FFMPEG", "CAP_FFMPEG"),
            ("Auto", "CAP_ANY"),
        ],
        "Linux": [
            ("V4L2", "CAP_V4L2"),
            ("GStreamer", "CAP_GSTREAMER"),
            ("FFMPEG", "CAP_FFMPEG"),
            ("Auto", "CAP_ANY"),
        ],
    }

    out: Dict[str, List[Tuple[str, int]]] = {}
    for os_name, entries in backend_specs.items():
        seen_constants = set()
        resolved: List[Tuple[str, int]] = []
        for label, const_name in entries:
            const_val = _cv2_const(const_name)
            if const_val is None:
                continue
            if const_val in seen_constants:
                continue
            seen_constants.add(const_val)
            resolved.append((label, const_val))
        if not resolved:
            resolved = [("Auto", cv2.CAP_ANY)]
        out[os_name] = resolved
    return out


class OpenCVCamera(CameraInterface):
    """
    Camera provider for OpenCV-compatible devices (USB webcams).

    Supports multiple backends and optional FOURCC codec preference.
    """

    # Backend options by operating system
    BACKENDS = _build_backends()

    BACKEND_ALIASES = {
        "auto": "Auto",
        "any": "Auto",
        "dshow": "DirectShow",
        "directshow": "DirectShow",
        "msmf": "Media Foundation",
        "mediafoundation": "Media Foundation",
        "avfoundation": "AVFoundation",
        "qt": "QTKit",
        "qtkit": "QTKit",
        "v4l2": "V4L2",
        "gstreamer": "GStreamer",
        "ffmpeg": "FFMPEG",
    }

    # Codec preferences. None means "do not force codec".
    CODECS = [
        ("Auto", None),
        ("MJPG", "MJPG"),
        ("YUYV", "YUYV"),
        ("YUY2", "YUY2"),
        ("XVID", "XVID"),
        ("H264", "H264"),
    ]

    def __init__(
        self,
        device_index: int = 0,
        backend: Optional[int] = None,
        backend_name: Optional[str] = None,
        codec: Optional[str] = None,
    ):
        """
        Initialize OpenCV camera.

        Args:
            device_index: Camera device index (0, 1, 2, ...)
            backend: OpenCV backend constant (e.g., cv2.CAP_DSHOW)
            backend_name: Backend name string (e.g., "DirectShow")
            codec: Optional FOURCC codec name (e.g., "MJPG", "YUYV", "H264")
        """
        self._device_index = device_index

        # Resolve backend
        if backend is not None:
            self._backend = backend
        elif backend_name is not None:
            self._backend = self._backend_name_to_constant(backend_name)
        else:
            self._backend = self._get_default_backend()

        self._codec_preference = self._normalize_codec_name(codec)
        self._active_codec = "Auto"
        self._cap: Optional[cv2.VideoCapture] = None

    # ==========================================================================
    # Backend/Codec Utilities
    # ==========================================================================

    @staticmethod
    def _normalize_token(name: str) -> str:
        return (name or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")

    @classmethod
    def _normalize_backend_name(cls, name: Optional[str]) -> str:
        """Normalize free-form backend names to a canonical display name."""
        if not name:
            return "Auto"
        token = cls._normalize_token(name)
        if token.startswith("cap"):
            token = token[3:]

        # Match canonical names from all configured backend labels
        for os_backends in cls.BACKENDS.values():
            for backend_name, _ in os_backends:
                if cls._normalize_token(backend_name) == token:
                    return backend_name

        aliased = cls.BACKEND_ALIASES.get(token)
        if aliased:
            return aliased
        return name.strip()

    @classmethod
    def _get_default_backend(cls) -> int:
        """Get preferred backend for current OS."""
        os_name = platform.system()
        backends = cls.BACKENDS.get(os_name, [("Auto", cv2.CAP_ANY)])
        return backends[0][1]

    @classmethod
    def _backend_name_to_constant(cls, name: str) -> int:
        """Convert backend name/alias to cv2 backend constant."""
        if not name:
            return cv2.CAP_ANY

        # Support explicit CAP_* tokens
        token = cls._normalize_token(name)
        if token.startswith("cap"):
            const_name = name.strip().upper()
            const_val = _cv2_const(const_name)
            if const_val is not None:
                return const_val

        canonical = cls._normalize_backend_name(name)

        # Prefer current OS backend names
        for backend_name, constant in cls.get_available_backends():
            if cls._normalize_token(backend_name) == cls._normalize_token(canonical):
                return constant

        # Fallback: search all OS maps
        for os_backends in cls.BACKENDS.values():
            for backend_name, constant in os_backends:
                if cls._normalize_token(backend_name) == cls._normalize_token(canonical):
                    return constant

        return cv2.CAP_ANY

    @classmethod
    def _backend_constant_to_name(cls, constant: int) -> str:
        """Convert cv2 backend constant to canonical backend name."""
        for backend_name, const in cls.get_available_backends():
            if const == constant:
                return backend_name
        for os_backends in cls.BACKENDS.values():
            for backend_name, const in os_backends:
                if const == constant:
                    return backend_name
        return "Auto"

    @classmethod
    def _normalize_codec_name(cls, codec: Optional[str]) -> str:
        """Normalize codec to known label (or Auto)."""
        if not codec:
            return "Auto"
        token = codec.strip().upper()
        for label, _ in cls.CODECS:
            if label.upper() == token:
                return label
        return "Auto"

    @classmethod
    def _codec_name_to_fourcc(cls, codec_name: str) -> Optional[str]:
        normalized = cls._normalize_codec_name(codec_name)
        for label, fourcc in cls.CODECS:
            if label == normalized:
                return fourcc
        return None

    @staticmethod
    def _fourcc_to_string(fourcc_value: float) -> str:
        """Decode OpenCV FOURCC numeric value into a readable 4-char code."""
        try:
            code = int(fourcc_value)
            if code <= 0:
                return "Auto"
            chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
            text = "".join(c for c in chars if 32 <= ord(c) <= 126).strip()
            return text.upper() if text else "Auto"
        except Exception:
            return "Auto"

    def _read_active_codec(self) -> str:
        """Read current active codec from capture object if available."""
        if not self.is_open():
            return self._codec_preference
        try:
            fourcc = self._cap.get(cv2.CAP_PROP_FOURCC)
            codec_name = self._fourcc_to_string(fourcc)
            return codec_name or self._codec_preference
        except Exception:
            return self._codec_preference

    # ==========================================================================
    # Connection Management (CameraInterface)
    # ==========================================================================

    def open(self) -> bool:
        """Open connection to the camera."""
        if self._cap is not None:
            self.close()

        try:
            self._cap = cv2.VideoCapture(self._device_index, self._backend)

            if not self._cap.isOpened():
                logging.debug(
                    "OpenCV camera %s failed to open (backend=%s)",
                    self._device_index,
                    self.get_backend_name(),
                )
                self._cap = None
                return False

            # Apply preferred codec when configured.
            if self._codec_preference != "Auto":
                self.set_codec(self._codec_preference)

            # Verify we can actually read at least one frame (with warmup)
            ret = False
            for _ in range(5):
                ret, _ = self._cap.read()
                if ret:
                    break

            if not ret:
                logging.debug(
                    "OpenCV camera %s opened but can't read frames (backend=%s)",
                    self._device_index,
                    self.get_backend_name(),
                )
                self._cap.release()
                self._cap = None
                return False

            self._active_codec = self._read_active_codec()
            logging.debug(
                "OpenCV camera %s opened successfully (backend=%s codec=%s)",
                self._device_index,
                self.get_backend_name(),
                self._active_codec,
            )
            return True

        except Exception as e:
            logging.error("Exception opening OpenCV camera: %s", e)
            self._cap = None
            return False

    def close(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logging.debug("Exception releasing camera: %s", e)
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
            logging.error("Exception reading frame: %s", e)
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

        actual_w, actual_h = self.get_resolution()
        if actual_w != width or actual_h != height:
            logging.info("Requested %sx%s, camera using %sx%s", width, height, actual_w, actual_h)

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

    def set_codec(self, codec_name: str) -> bool:
        """
        Set preferred capture codec (FOURCC) for this camera session.

        Args:
            codec_name: Codec label (Auto, MJPG, YUYV, YUY2, XVID, H264)
        """
        self._codec_preference = self._normalize_codec_name(codec_name)
        if not self.is_open():
            self._active_codec = self._codec_preference
            return False

        fourcc = self._codec_name_to_fourcc(self._codec_preference)
        try:
            if fourcc:
                ok = self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            else:
                ok = True
            self._active_codec = self._read_active_codec()
            return bool(ok)
        except Exception as e:
            logging.debug("Failed to set codec '%s': %s", self._codec_preference, e)
            self._active_codec = self._read_active_codec()
            return False

    def get_codec_name(self) -> str:
        """Get active codec label/FOURCC."""
        if self.is_open():
            self._active_codec = self._read_active_codec()
        return self._active_codec or self._codec_preference or "Auto"

    # ==========================================================================
    # Device Information (CameraInterface)
    # ==========================================================================

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the connected device."""
        width, height = self.get_resolution()

        backend_raw = self.get_backend_name()
        if self.is_open():
            try:
                backend_raw = self._cap.getBackendName()
            except Exception:
                backend_raw = self.get_backend_name()

        backend_name = self._normalize_backend_name(backend_raw)
        codec_name = self.get_codec_name()

        return {
            "provider": self.get_provider_name(),
            "device_index": self._device_index,
            "backend": backend_name,
            "backend_raw": backend_raw,
            "codec": codec_name,
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
            backend: Specific backend constant to use
            max_devices: Max camera indices to probe
            all_backends: If True and backend is None, probe all available backends
        """
        all_backends = bool(kwargs.get("all_backends", False))
        if backend is not None:
            backend_candidates = [(cls._backend_constant_to_name(backend), backend)]
        elif all_backends:
            backend_candidates = cls.get_available_backends()
        else:
            default_backend = cls._get_default_backend()
            backend_candidates = [(cls._backend_constant_to_name(default_backend), default_backend)]

        devices_by_index: Dict[int, Dict[str, Any]] = {}
        for backend_name, backend_const in backend_candidates:
            for i in range(max_devices):
                cap = None
                try:
                    cap = cv2.VideoCapture(i, backend_const)
                    if not cap.isOpened():
                        continue

                    ret, _ = cap.read()
                    if not ret:
                        continue

                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    info = {
                        "index": i,
                        "name": f"Camera {i}",
                        "provider": cls.get_provider_name(),
                        "resolution": (width, height),
                        "backend": backend_name,
                    }
                    # Keep first successful backend per index.
                    devices_by_index.setdefault(i, info)
                except Exception as e:
                    logging.debug("Error checking camera %s on %s: %s", i, backend_name, e)
                finally:
                    if cap is not None:
                        cap.release()

        return [devices_by_index[i] for i in sorted(devices_by_index.keys())]

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
        backends = cls.BACKENDS.get(os_name)
        if not backends:
            return [("Auto", cv2.CAP_ANY)]
        return list(backends)

    @classmethod
    def get_available_codecs(cls) -> List[Tuple[str, Optional[str]]]:
        """Get selectable codec labels and FOURCC codes."""
        return list(cls.CODECS)

    @classmethod
    def get_auto_detect_codecs(cls) -> List[str]:
        """Get prioritized codec labels for bounded auto-detection probing."""
        available = [label for label, _ in cls.CODECS]
        priority = ["Auto", "MJPG", "YUYV", "YUY2"]
        return [name for name in priority if name in available]

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


