#!/usr/bin/env python3
"""
export_for_reid.py — Export StarBoard confirmed matches to star_dataset_raw format

Transforms the StarBoard gallery archive into the hierarchical folder structure
required for training a temporal re-identification model.

OUTPUT FORMAT:
    star_dataset_raw/
    └── DATASET_NAME/
        └── individual_id/
            └── M_D_YYYY_description/
                └── image_files

USAGE:
    python export_for_reid.py                           # Interactive prompts
    python export_for_reid.py --output ./star_dataset_raw --dataset FHL_STARS_2025
    python export_for_reid.py --dry-run                 # Preview without copying
    python export_for_reid.py --include ariadne,beignet # Export specific individuals only
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MMDDYY_PATTERN = re.compile(r"^(\d{2})_(\d{2})_(\d{2})(?:_(.+))?$")

# ---------------------------------------------------------------------------
# Path resolution (standalone, no StarBoard imports required)
# ---------------------------------------------------------------------------

def find_archive_root() -> Path:
    """Locate the StarBoard archive directory."""
    import os
    
    # Check environment variable first
    env_path = os.getenv("STARBOARD_ARCHIVE_DIR")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p
    
    # Check relative to script location
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "archive",
        script_dir.parent / "archive",
    ]
    
    for c in candidates:
        if c.exists() and (c / "gallery").exists():
            return c
    
    raise FileNotFoundError(
        "Could not locate StarBoard archive. Set STARBOARD_ARCHIVE_DIR or ensure "
        "'archive/gallery/' exists relative to this script."
    )


def get_gallery_root(archive: Path) -> Path:
    return archive / "gallery"


# ---------------------------------------------------------------------------
# Date conversion: MM_DD_YY -> M_D_YYYY
# ---------------------------------------------------------------------------

def parse_encounter_folder(name: str) -> Optional[Tuple[int, int, int, str]]:
    """
    Parse an encounter folder name like '09_08_25' or '09_08_25_dive_one'.
    
    Returns (month, day, year_4digit, description) or None if invalid.
    """
    m = MMDDYY_PATTERN.match(name)
    if not m:
        return None
    
    mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
    description = m.group(4) or ""
    
    # Convert 2-digit year to 4-digit (assumes 2000s)
    year_4digit = 2000 + yy
    
    return (mm, dd, year_4digit, description)


def format_reid_date_folder(month: int, day: int, year: int, description: str) -> str:
    """
    Format date folder for reid training: M_D_YYYY_description
    
    Examples:
        (9, 8, 2025, "") -> "9_8_2025_observation"
        (9, 8, 2025, "dive_one") -> "9_8_2025_dive_one"
    """
    # Remove leading zeros from month/day per spec
    base = f"{month}_{day}_{year}"
    
    # Add description suffix
    if description:
        # Clean description: lowercase, underscores
        desc_clean = description.lower().replace(" ", "_").replace("-", "_")
        return f"{base}_{desc_clean}"
    else:
        # Default description if none provided
        return f"{base}_observation"


# ---------------------------------------------------------------------------
# Individual ID normalization
# ---------------------------------------------------------------------------

def normalize_individual_id(gallery_id: str) -> str:
    """
    Normalize gallery ID to reid format: lowercase with underscores.
    
    StarBoard IDs are already mostly compliant (e.g., 'ariadne', 'big_chungus').
    """
    return gallery_id.lower().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Discovery functions
# ---------------------------------------------------------------------------

def list_gallery_individuals(gallery_root: Path) -> List[str]:
    """List all individual IDs in the gallery (folder-based)."""
    individuals = []
    
    for p in sorted(gallery_root.iterdir()):
        if not p.is_dir():
            continue
        # Skip internal/metadata folders
        if p.name.startswith("_"):
            continue
        if p.name.endswith(".csv") or p.name.endswith(".json"):
            continue
        
        individuals.append(p.name)
    
    return individuals


def list_encounter_folders(individual_path: Path) -> List[Tuple[Path, Tuple[int, int, int, str]]]:
    """
    List valid encounter folders for an individual.
    
    Returns list of (folder_path, (month, day, year, description)).
    """
    encounters = []
    
    for p in sorted(individual_path.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("_"):
            continue
        
        parsed = parse_encounter_folder(p.name)
        if parsed:
            encounters.append((p, parsed))
    
    return encounters


def list_images_in_folder(folder: Path) -> List[Path]:
    """List all image files in a folder (non-recursive)."""
    images = []
    
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)
    
    return images


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------

def build_export_plan(
    gallery_root: Path,
    include_ids: Optional[Set[str]] = None,
    exclude_ids: Optional[Set[str]] = None,
) -> Dict[str, List[Tuple[Path, str, List[Path]]]]:
    """
    Build a mapping of what needs to be exported.
    
    Returns:
        {individual_id: [(src_encounter_path, dest_folder_name, [image_paths]), ...]}
    """
    plan: Dict[str, List[Tuple[Path, str, List[Path]]]] = {}
    
    for individual_id in list_gallery_individuals(gallery_root):
        # Apply filters
        if include_ids and individual_id not in include_ids:
            continue
        if exclude_ids and individual_id in exclude_ids:
            continue
        
        individual_path = gallery_root / individual_id
        encounters = list_encounter_folders(individual_path)
        
        if not encounters:
            continue
        
        normalized_id = normalize_individual_id(individual_id)
        plan[normalized_id] = []
        
        for enc_path, (month, day, year, desc) in encounters:
            dest_folder_name = format_reid_date_folder(month, day, year, desc)
            images = list_images_in_folder(enc_path)
            
            if images:
                plan[normalized_id].append((enc_path, dest_folder_name, images))
    
    return plan


def execute_export(
    plan: Dict[str, List[Tuple[Path, str, List[Path]]]],
    output_root: Path,
    dataset_name: str,
    dry_run: bool = False,
    prefer_png: bool = True,
) -> Tuple[int, int, int, List[str]]:
    """
    Execute the export plan.
    
    Returns: (num_individuals, num_encounters, num_images, errors)
    """
    dataset_path = output_root / dataset_name
    
    num_individuals = 0
    num_encounters = 0
    num_images = 0
    errors: List[str] = []
    
    for individual_id, encounters in plan.items():
        if not encounters:
            continue
        
        num_individuals += 1
        individual_path = dataset_path / individual_id
        
        for src_encounter, dest_folder_name, images in encounters:
            num_encounters += 1
            encounter_path = individual_path / dest_folder_name
            
            if not dry_run:
                encounter_path.mkdir(parents=True, exist_ok=True)
            
            for img_path in images:
                num_images += 1
                
                # Determine destination filename
                dest_filename = img_path.name
                
                # Optionally convert to PNG naming (keeps original format)
                # The spec says PNG preferred but accepts JPG/JPEG
                dest_file = encounter_path / dest_filename
                
                if not dry_run:
                    try:
                        shutil.copy2(img_path, dest_file)
                    except Exception as e:
                        errors.append(f"Failed to copy {img_path} -> {dest_file}: {e}")
    
    return num_individuals, num_encounters, num_images, errors


def print_plan_summary(
    plan: Dict[str, List[Tuple[Path, str, List[Path]]]],
    dataset_name: str,
    verbose: bool = False,
):
    """Print a summary of what will be exported."""
    total_individuals = len([k for k, v in plan.items() if v])
    total_encounters = sum(len(v) for v in plan.values())
    total_images = sum(len(imgs) for encs in plan.values() for _, _, imgs in encs)
    
    print(f"\n{'='*60}")
    print(f"EXPORT PLAN SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset name:     {dataset_name}")
    print(f"Individuals:      {total_individuals}")
    print(f"Encounter folders:{total_encounters}")
    print(f"Total images:     {total_images}")
    print(f"{'='*60}\n")
    
    if verbose:
        for individual_id, encounters in sorted(plan.items()):
            if not encounters:
                continue
            print(f"\n  {individual_id}/")
            for _, dest_folder, images in encounters:
                print(f"    └── {dest_folder}/ ({len(images)} images)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export StarBoard gallery to star_dataset_raw format for ReID training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Export all individuals to default location
    python export_for_reid.py
    
    # Export to specific directory with custom dataset name
    python export_for_reid.py --output ./reid_data --dataset FHL_PYCNO_2025
    
    # Preview what would be exported (no files copied)
    python export_for_reid.py --dry-run --verbose
    
    # Export only specific individuals
    python export_for_reid.py --include ariadne,beignet,margarita
    
    # Export all except certain individuals
    python export_for_reid.py --exclude scramble,test_star
        """,
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./star_dataset_raw"),
        help="Output directory root (default: ./star_dataset_raw)",
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Dataset name (Level 1 folder). Default: auto-generated from date",
    )
    
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Path to StarBoard archive (default: auto-detect or STARBOARD_ARCHIVE_DIR)",
    )
    
    parser.add_argument(
        "--include",
        type=str,
        default=None,
        help="Comma-separated list of individual IDs to include (default: all)",
    )
    
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated list of individual IDs to exclude",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview export without copying files",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed export plan",
    )
    
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    
    args = parser.parse_args()
    
    # Resolve archive location
    try:
        if args.archive:
            archive_root = args.archive.resolve()
            if not archive_root.exists():
                print(f"ERROR: Archive path does not exist: {archive_root}", file=sys.stderr)
                sys.exit(1)
        else:
            archive_root = find_archive_root()
        
        gallery_root = get_gallery_root(archive_root)
        if not gallery_root.exists():
            print(f"ERROR: Gallery not found at: {gallery_root}", file=sys.stderr)
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Using archive: {archive_root}")
    print(f"Gallery root:  {gallery_root}")
    
    # Parse include/exclude filters
    include_ids = None
    if args.include:
        include_ids = {x.strip().lower() for x in args.include.split(",") if x.strip()}
        print(f"Including only: {', '.join(sorted(include_ids))}")
    
    exclude_ids = None
    if args.exclude:
        exclude_ids = {x.strip().lower() for x in args.exclude.split(",") if x.strip()}
        print(f"Excluding: {', '.join(sorted(exclude_ids))}")
    
    # Generate dataset name if not provided
    dataset_name = args.dataset
    if not dataset_name:
        dataset_name = f"STARBOARD_EXPORT_{datetime.now().strftime('%Y%m%d')}"
    
    # Validate dataset name format
    if not re.match(r"^[A-Za-z0-9_]+$", dataset_name):
        print(f"ERROR: Dataset name must be alphanumeric with underscores: {dataset_name}", file=sys.stderr)
        sys.exit(1)
    
    # Build export plan
    print("\nScanning gallery...")
    plan = build_export_plan(gallery_root, include_ids, exclude_ids)
    
    if not plan or not any(plan.values()):
        print("No individuals found to export.")
        sys.exit(0)
    
    # Show plan summary
    print_plan_summary(plan, dataset_name, verbose=args.verbose)
    
    # Output path
    output_path = args.output.resolve()
    full_dataset_path = output_path / dataset_name
    print(f"Output path: {full_dataset_path}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files will be copied.\n")
        return
    
    # Confirmation
    if not args.yes:
        response = input("\nProceed with export? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)
    
    # Execute export
    print("\nExporting...")
    num_individuals, num_encounters, num_images, errors = execute_export(
        plan,
        output_path,
        dataset_name,
        dry_run=False,
    )
    
    # Report results
    print(f"\n{'='*60}")
    print(f"EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Individuals exported: {num_individuals}")
    print(f"Encounter folders:    {num_encounters}")
    print(f"Images copied:        {num_images}")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    print(f"\nOutput: {full_dataset_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

