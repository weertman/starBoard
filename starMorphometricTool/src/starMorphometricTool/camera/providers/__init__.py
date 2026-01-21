"""
Camera Providers Package

Contains concrete implementations of the CameraInterface for various camera types.
"""

from camera.providers.opencv_camera import OpenCVCamera

__all__ = [
    "OpenCVCamera",
]


