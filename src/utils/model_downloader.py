"""
Auto-download model weights from GitHub Releases on first run.

This module checks for required model files and downloads them from
the GitHub Releases page if they're missing.

Usage:
    python -m src.utils.model_downloader

Or called automatically on app startup via main.py
"""

import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple

# GitHub Release download base URL
RELEASE_TAG = "v1.0-models"
RELEASE_BASE_URL = f"https://github.com/weertman/starBoard/releases/download/{RELEASE_TAG}"

# Model files: (download_name, local_path_relative_to_project_root)
MODEL_FILES: List[Tuple[str, str]] = [
    ("megastarid_default.pth", "star_identification/checkpoints/default/best.pth"),
    ("megastarid_finetune.pth", "star_identification/checkpoints/megastarid/finetune/best.pth"),
    ("verification_circleloss.pth", "star_identification/checkpoints/verification/extended_training/circleloss/nofreeze_inat1_neg0_20260109_050432/best.pth"),
    ("starseg_best.pt", "star_identification/wildlife_reid_inference/starseg_best.pt"),
    ("morphometric_yolo.pt", "starMorphometricTool/models/best.pt"),
    ("depth_anything_v2_vitb.pth", "starMorphometricTool/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth"),
]


def get_project_root() -> Path:
    """Get the project root directory."""
    # This file is at src/utils/model_downloader.py
    # Project root is 3 levels up
    return Path(__file__).parent.parent.parent


def check_missing_models(project_root: Path = None) -> List[Tuple[str, str, str]]:
    """
    Check which model files are missing.
    
    Returns:
        List of (download_name, local_path, full_url) for missing files
    """
    if project_root is None:
        project_root = get_project_root()
    
    missing = []
    for download_name, local_path in MODEL_FILES:
        full_path = project_root / local_path
        if not full_path.exists():
            url = f"{RELEASE_BASE_URL}/{download_name}"
            missing.append((download_name, local_path, url))
    
    return missing


def download_file(url: str, destination: Path, show_progress: bool = True) -> bool:
    """
    Download a file from URL to destination with progress indication.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directories if needed
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up progress tracking
        if show_progress:
            def progress_hook(count, block_size, total_size):
                if total_size > 0:
                    percent = min(100, count * block_size * 100 // total_size)
                    mb_downloaded = (count * block_size) / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r    Progress: {percent:3d}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
                else:
                    mb_downloaded = (count * block_size) / (1024 * 1024)
                    print(f"\r    Downloaded: {mb_downloaded:.1f} MB", end="", flush=True)
        else:
            progress_hook = None
        
        # Download the file
        urllib.request.urlretrieve(url, destination, progress_hook)
        
        if show_progress:
            print()  # New line after progress
        
        return True
        
    except urllib.error.HTTPError as e:
        print(f"\n    HTTP Error {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"\n    URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"\n    Error: {e}")
        return False


def download_models(project_root: Path = None, interactive: bool = True) -> Tuple[int, int]:
    """
    Download missing model files from GitHub Releases.
    
    Args:
        project_root: Project root directory (auto-detected if None)
        interactive: If True, prompt user before downloading
    
    Returns:
        Tuple of (successful_downloads, failed_downloads)
    """
    if project_root is None:
        project_root = get_project_root()
    
    missing = check_missing_models(project_root)
    
    if not missing:
        print("✓ All model files are present.")
        return (0, 0)
    
    # Calculate total size (approximate)
    size_estimates = {
        "megastarid_default.pth": 336,
        "megastarid_finetune.pth": 336,
        "verification_circleloss.pth": 624,
        "starseg_best.pt": 53,
        "morphometric_yolo.pt": 20,
        "depth_anything_v2_vitb.pth": 372,
    }
    total_mb = sum(size_estimates.get(name, 100) for name, _, _ in missing)
    
    print(f"\n{'='*60}")
    print("  starBoard Model Downloader")
    print(f"{'='*60}")
    print(f"\nMissing {len(missing)} model file(s) (~{total_mb} MB total):\n")
    
    for download_name, local_path, _ in missing:
        size = size_estimates.get(download_name, "?")
        print(f"  • {download_name} ({size} MB)")
        print(f"    → {local_path}")
    
    print(f"\nSource: https://github.com/weertman/starBoard/releases/tag/{RELEASE_TAG}")
    
    if interactive:
        print("\nDownload now? [Y/n] ", end="", flush=True)
        try:
            response = input().strip().lower()
            if response and response not in ('y', 'yes'):
                print("Download cancelled. You can run this later with:")
                print("  python -m src.utils.model_downloader")
                return (0, 0)
        except (EOFError, KeyboardInterrupt):
            print("\nDownload cancelled.")
            return (0, 0)
    
    print(f"\nDownloading {len(missing)} file(s)...\n")
    
    successful = 0
    failed = 0
    
    for i, (download_name, local_path, url) in enumerate(missing, 1):
        destination = project_root / local_path
        print(f"[{i}/{len(missing)}] {download_name}")
        print(f"    → {local_path}")
        
        if download_file(url, destination):
            print(f"    ✓ Downloaded successfully")
            successful += 1
        else:
            print(f"    ✗ Download failed")
            failed += 1
        
        print()
    
    print(f"{'='*60}")
    print(f"  Complete: {successful} succeeded, {failed} failed")
    print(f"{'='*60}\n")
    
    return (successful, failed)


def ensure_models(project_root: Path = None) -> bool:
    """
    Ensure all required models are present, downloading if necessary.
    
    This is the main entry point for automatic model checking on app startup.
    Non-interactive - will download automatically if models are missing.
    
    Args:
        project_root: Project root directory (auto-detected if None)
    
    Returns:
        True if all models are present (or were downloaded), False if any are missing
    """
    if project_root is None:
        project_root = get_project_root()
    
    missing = check_missing_models(project_root)
    
    if not missing:
        return True
    
    # Download missing models (non-interactive for app startup)
    successful, failed = download_models(project_root, interactive=True)
    
    return failed == 0


if __name__ == "__main__":
    # When run directly, use interactive mode
    successful, failed = download_models(interactive=True)
    sys.exit(0 if failed == 0 else 1)

