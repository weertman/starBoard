"""
Deep Learning integration for starBoard.

Provides visual similarity ranking using MegaStarID pretrained models.
All features are optional - if PyTorch is not installed, the module
gracefully degrades with DL_AVAILABLE = False.

Verification features require the megastar_identity_verification module.
"""

import logging

log = logging.getLogger("starBoard.dl")

# Check if PyTorch is available
try:
    import torch
    DL_AVAILABLE = True
    TORCH_VERSION = torch.__version__
    
    if torch.cuda.is_available():
        DEVICE = "cuda"
        DEVICE_NAME = torch.cuda.get_device_name(0)
    else:
        DEVICE = "cpu"
        DEVICE_NAME = "CPU"
        
except ImportError:
    DL_AVAILABLE = False
    TORCH_VERSION = None
    DEVICE = None
    DEVICE_NAME = None

# Check if verification model is available
VERIFICATION_AVAILABLE = False
if DL_AVAILABLE:
    try:
        from star_identification.megastar_identity_verification import VerificationModel
        VERIFICATION_AVAILABLE = True
    except ImportError:
        pass


def get_status_message() -> str:
    """Get a human-readable status message for DL availability."""
    if not DL_AVAILABLE:
        return "PyTorch not installed. Run: pip install -r requirements-dl.txt"
    return f"Ready - {DEVICE_NAME} (PyTorch {TORCH_VERSION})"


def get_verification_status_message() -> str:
    """Get a human-readable status message for verification availability."""
    if not DL_AVAILABLE:
        return "PyTorch not installed"
    if not VERIFICATION_AVAILABLE:
        return "Verification module not available"
    return "Verification ready"


def is_available() -> bool:
    """Check if DL features are available."""
    return DL_AVAILABLE


def is_verification_available() -> bool:
    """Check if verification features are available."""
    return VERIFICATION_AVAILABLE







