"""
Fine-tuning module for starBoard.

Provides UI-integrated fine-tuning for both embedding and verification models
using starBoard archive data and optionally external star_dataset.

This module is designed to be compartmentalized and isolated from the rest
of the application to minimize risk of breaking existing functionality.
"""

from .config import FinetuneUIConfig, FinetuneMode, DataSource
from .worker import FinetuneWorker
from .data_bridge import get_data_summary

__all__ = [
    "FinetuneUIConfig",
    "FinetuneMode",
    "DataSource",
    "FinetuneWorker",
    "get_data_summary",
]

