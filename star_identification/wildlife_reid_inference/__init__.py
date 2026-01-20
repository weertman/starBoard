"""
Wildlife ReID Inference Module

A lightweight inference module that uses the actual training models and configs.
"""

__version__ = "2.0.0"

# When imported as a package
try:
    from .inference import WildlifeReIDInference
    from .preprocessing import YOLOPreprocessor
except ImportError:
    # When run directly from the directory
    from inference import WildlifeReIDInference
    from preprocessing import YOLOPreprocessor

__all__ = ['WildlifeReIDInference', 'YOLOPreprocessor']