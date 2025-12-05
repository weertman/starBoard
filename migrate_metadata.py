#!/usr/bin/env python3
"""
Migration script: Convert V1 metadata schema to V2.

This script:
1. Reads old V1 CSV files (gallery_metadata.csv, queries_metadata.csv)
2. Maps fields to the new V2 schema
3. Parses free-text description fields to extract structured data
4. Creates backup of old files
5. Writes new V2 CSV files

Run from the project root:
    python migrate_metadata.py [--dry-run]

Options:
    --dry-run    Preview changes without writing files
"""
from __future__ import annotations

import csv
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

V1_GALLERY_HEADER = [
    "gallery_id", "Last location", "sex", "diameter_cm", "volume_ml",
    "num_apparent_arms", "num_arms", "short_arm_codes", "stripe_descriptions",
    "reticulation_descriptions", "rosette_descriptions", "madreporite_descriptions",
    "disk color", "arm color", "Other_descriptions",
]

V1_QUERIES_HEADER = [
    "query_id", "Last location", "sex", "diameter_cm", "volume_ml",
    "num_apparent_arms", "num_arms", "short_arm_codes", "stripe_descriptions",
    "reticulation_descriptions", "rosette_descriptions", "madreporite_descriptions",
    "disk color", "arm color", "Other_descriptions",
]

V2_GALLERY_HEADER = [
    "gallery_id", "num_apparent_arms", "num_total_arms", "tip_to_tip_size_cm",
    "short_arm_code", "stripe_color", "stripe_order", "stripe_prominence",
    "stripe_extent", "arm_color", "arm_thickness", "central_disc_color",
    "papillae_central_disc_color", "rosette_color", "rosette_prominence",
    "papillae_stripe_color", "madreporite_color", "reticulation_order",
    "overall_color", "location", "unusual_observation", "health_observation",
]

V2_QUERIES_HEADER = [
    "query_id", "num_apparent_arms", "num_total_arms", "tip_to_tip_size_cm",
    "short_arm_code", "stripe_color", "stripe_order", "stripe_prominence",
    "stripe_extent", "arm_color", "arm_thickness", "central_disc_color",
    "papillae_central_disc_color", "rosette_color", "rosette_prominence",
    "papillae_stripe_color", "madreporite_color", "reticulation_order",
    "overall_color", "location", "unusual_observation", "health_observation",
]


# =============================================================================
# COLOR EXTRACTION
# =============================================================================

# Common color terms to look for
COLOR_TERMS = [
    # Basic colors
    "white", "yellow", "orange", "peach", "pink", "red", "maroon", "burgundy",
    "purple", "mauve", "brown", "tan", "black", "lavender", "rose",
    # Compound colors (order matters - longer matches first)
    "burnt orange", "bright orange", "rusty orange", "light brown", "light purple",
    "dark purple", "dark brown", "dark orange", "white yellow", "yellow white",
    "white-yellow", "yellow-white", "brown orange", "brown-orange", "purple maroon",
    "purple-maroon", "pink maroon", "pink-maroon", "burgundy mauve", "burgundy-mauve",
    "mauve burgundy", "mauve-burgundy", "brown mauve", "brown-mauve", "mauve brown",
    "mauve-brown", "maroon purple", "maroon-purple", "maroon pink", "maroon-pink",
    "peach brown", "peach-brown", "light peach", "dark maroon", "deep maroon",
    "faint mauve", "pale yellow", "cherry", "black cherry",
]

# Build regex pattern for colors (longer matches first)
COLOR_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(c) for c in sorted(COLOR_TERMS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)


def extract_colors(text: str) -> List[str]:
    """Extract color terms from text."""
    if not text:
        return []
    matches = COLOR_PATTERN.findall(text.lower())
    # Deduplicate while preserving order
    seen = set()
    result = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def extract_primary_color(text: str) -> str:
    """Extract the first/primary color from text."""
    colors = extract_colors(text)
    return colors[0] if colors else ""


# =============================================================================
# STRIPE PARSING
# =============================================================================

STRIPE_ORDER_MAP = {
    "none": "0",
    "mixed": "1", "mix": "1",
    "irregular": "2", "disorganized": "2", "disorgnized": "2", "disorganzied": "2",
    "disorgazied": "2", "disorangized": "2", "disorgnaized": "2", "idsorganzed": "2",
    "disprgnized": "2", "disoragnized": "2",
    "regular": "3", "reulagr": "3", "regualr": "3",
}

STRIPE_PROMINENCE_MAP = {
    "none": "0",
    "faint": "1", "very faint": "1", "subtle": "1", "very subtle": "1",
    "weak": "1", "obscure": "1",
    "moderate": "2", "moderately": "2", "somewhat": "2", "medium": "2",
    "prominent": "3", "prom": "3", "promenent": "3", "prominemnt": "3",
    "very prominent": "4", "veyr prom": "4", "very prom": "4",
}


def parse_stripe_descriptions(text: str) -> Dict[str, str]:
    """Parse stripe_descriptions field for order, prominence, and color."""
    result = {"stripe_color": "", "stripe_order": "", "stripe_prominence": ""}
    if not text:
        return result
    
    text_lower = text.lower()
    
    # Extract color
    result["stripe_color"] = extract_primary_color(text)
    
    # Extract order (regular/irregular/mixed/disorganized)
    for key, val in STRIPE_ORDER_MAP.items():
        if key in text_lower:
            result["stripe_order"] = val
            break
    
    # Extract prominence
    for key, val in sorted(STRIPE_PROMINENCE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if key in text_lower:
            result["stripe_prominence"] = val
            break
    
    return result


# =============================================================================
# RETICULATION PARSING
# =============================================================================

RETICULATION_ORDER_MAP = {
    "none": "0",
    "mixed": "1", "variable": "1",
    "meandering": "2", "meadnering": "2", "merandering": "2",
    "traintrack": "3", "train track": "3", "train-track": "3",
    "traintack": "3", "triansitonign": "3",
}


def parse_reticulation_descriptions(text: str) -> str:
    """Parse reticulation_descriptions for order pattern."""
    if not text:
        return ""
    
    text_lower = text.lower()
    
    # Check for patterns (longer matches first)
    for key, val in sorted(RETICULATION_ORDER_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if key in text_lower:
            return val
    
    return ""


# =============================================================================
# ROSETTE PARSING
# =============================================================================

ROSETTE_PROMINENCE_MAP = {
    "faint": "0", "very faint": "0", "weak": "0",
    "moderate": "1", "medium": "1",
    "prominent": "2", "prom": "2", "large": "2", "strong": "2",
}


def parse_rosette_descriptions(text: str) -> Dict[str, str]:
    """Parse rosette_descriptions for color and prominence."""
    result = {"rosette_color": "", "rosette_prominence": ""}
    if not text:
        return result
    
    text_lower = text.lower()
    
    # Extract color
    result["rosette_color"] = extract_primary_color(text)
    
    # Extract prominence
    for key, val in sorted(ROSETTE_PROMINENCE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if key in text_lower:
            result["rosette_prominence"] = val
            break
    
    return result


# =============================================================================
# SHORT ARM CODE CONVERSION
# =============================================================================

def convert_short_arm_code(old_code: str) -> str:
    """
    Convert old short arm code format to new format.
    
    Old format: 2**, 3, 10*, 11**, (3), 6(r)
    New format: tiny(2), short(3), small(10), tiny(11)
    
    Mapping:
    - ** or *** = tiny
    - * = small  
    - no asterisk = short
    - (N) = tiny (parentheses indicate very small)
    """
    if not old_code or not old_code.strip():
        return ""
    
    entries = []
    parts = [p.strip() for p in old_code.split(",")]
    
    for part in parts:
        if not part:
            continue
        
        # Skip regenerating arms marked with (r)
        if "(r)" in part.lower():
            continue
        
        # Match patterns like: 2**, 3, 10*, (3), 11***
        match = re.match(r'\(?(\d+)\)?(\*{0,3})', part)
        if not match:
            continue
        
        pos = int(match.group(1))
        stars = match.group(2)
        
        # Determine severity
        if part.startswith("(") and part.rstrip("*").endswith(")"):
            # Parentheses without asterisks = tiny
            severity = "tiny"
        elif len(stars) >= 2:
            severity = "tiny"
        elif len(stars) == 1:
            severity = "small"
        else:
            severity = "short"
        
        entries.append(f"{severity}({pos})")
    
    # Sort by position
    entries.sort(key=lambda x: int(re.search(r'\((\d+)\)', x).group(1)))
    
    return ", ".join(entries)


# =============================================================================
# OVERALL COLOR EXTRACTION
# =============================================================================

def extract_overall_color(other_desc: str, stripe_desc: str = "", rosette_desc: str = "") -> str:
    """
    Try to extract an overall color description.
    Usually mentioned in Other_descriptions with words like "overall", "color", "vibe".
    """
    if not other_desc:
        return ""
    
    text_lower = other_desc.lower()
    
    # Look for explicit "overall" mentions
    overall_match = re.search(r'overall\s+(\w+[\w\s-]*?)(?:,|\.|$|color|vibe)', text_lower)
    if overall_match:
        colors = extract_colors(overall_match.group(1))
        if colors:
            return colors[0]
    
    # Look for "color" mentions
    color_match = re.search(r'(\w+[\w\s-]*?)\s+color', text_lower)
    if color_match:
        colors = extract_colors(color_match.group(1))
        if colors:
            return colors[0]
    
    # Fall back to first color in the text
    return extract_primary_color(other_desc)


# =============================================================================
# ROW MIGRATION
# =============================================================================

def migrate_row(old_row: Dict[str, str], id_col: str) -> Dict[str, str]:
    """Migrate a single row from V1 to V2 schema."""
    new_row = {}
    
    # Direct mappings
    new_row[id_col] = old_row.get(id_col, "")
    new_row["num_apparent_arms"] = old_row.get("num_apparent_arms", "")
    new_row["num_total_arms"] = old_row.get("num_arms", "")  # Renamed
    new_row["tip_to_tip_size_cm"] = old_row.get("diameter_cm", "")  # Renamed
    new_row["location"] = old_row.get("Last location", "")
    new_row["arm_color"] = old_row.get("arm color", "")
    new_row["central_disc_color"] = old_row.get("disk color", "")
    
    # Convert short arm code
    new_row["short_arm_code"] = convert_short_arm_code(old_row.get("short_arm_codes", ""))
    
    # Parse stripe descriptions
    stripe_parsed = parse_stripe_descriptions(old_row.get("stripe_descriptions", ""))
    new_row["stripe_color"] = stripe_parsed["stripe_color"]
    new_row["stripe_order"] = stripe_parsed["stripe_order"]
    new_row["stripe_prominence"] = stripe_parsed["stripe_prominence"]
    new_row["stripe_extent"] = ""  # New field, no V1 equivalent
    
    # Parse rosette descriptions
    rosette_parsed = parse_rosette_descriptions(old_row.get("rosette_descriptions", ""))
    new_row["rosette_color"] = rosette_parsed["rosette_color"]
    new_row["rosette_prominence"] = rosette_parsed["rosette_prominence"]
    
    # Parse reticulation
    new_row["reticulation_order"] = parse_reticulation_descriptions(
        old_row.get("reticulation_descriptions", "")
    )
    
    # Extract madreporite color
    new_row["madreporite_color"] = extract_primary_color(
        old_row.get("madreporite_descriptions", "")
    )
    
    # Extract overall color
    new_row["overall_color"] = extract_overall_color(
        old_row.get("Other_descriptions", ""),
        old_row.get("stripe_descriptions", ""),
        old_row.get("rosette_descriptions", ""),
    )
    
    # Other_descriptions -> unusual_observation (preserve full text)
    new_row["unusual_observation"] = old_row.get("Other_descriptions", "")
    
    # New fields with no V1 equivalent
    new_row["arm_thickness"] = ""
    new_row["papillae_central_disc_color"] = ""
    new_row["papillae_stripe_color"] = ""
    new_row["health_observation"] = ""
    
    return new_row


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read CSV file, return (header, rows)."""
    if not path.exists():
        return [], []
    
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    
    return header, rows


def write_csv(path: Path, header: List[str], rows: List[Dict[str, str]]) -> None:
    """Write CSV file with given header and rows."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            # Ensure all fields are present
            clean_row = {col: row.get(col, "") for col in header}
            writer.writerow(clean_row)


def backup_file(path: Path) -> Path:
    """Create timestamped backup of file."""
    if not path.exists():
        return path
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(f".v1_backup_{timestamp}.csv")
    shutil.copy2(path, backup_path)
    return backup_path


# =============================================================================
# MAIN MIGRATION
# =============================================================================

def migrate_csv(
    old_path: Path,
    new_path: Path,
    old_header: List[str],
    new_header: List[str],
    id_col: str,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Migrate a single CSV file from V1 to V2 schema.
    
    Returns statistics dict.
    """
    stats = {
        "rows_read": 0,
        "rows_written": 0,
        "short_arm_codes_converted": 0,
        "stripe_order_extracted": 0,
        "stripe_prominence_extracted": 0,
        "reticulation_order_extracted": 0,
        "colors_extracted": 0,
    }
    
    if not old_path.exists():
        print(f"  [SKIP] File not found: {old_path}")
        return stats
    
    header, rows = read_csv(old_path)
    stats["rows_read"] = len(rows)
    
    if not rows:
        print(f"  [SKIP] No data rows in: {old_path}")
        return stats
    
    # Migrate rows
    new_rows = []
    for old_row in rows:
        new_row = migrate_row(old_row, id_col)
        new_rows.append(new_row)
        
        # Collect stats
        if new_row.get("short_arm_code"):
            stats["short_arm_codes_converted"] += 1
        if new_row.get("stripe_order"):
            stats["stripe_order_extracted"] += 1
        if new_row.get("stripe_prominence"):
            stats["stripe_prominence_extracted"] += 1
        if new_row.get("reticulation_order"):
            stats["reticulation_order_extracted"] += 1
        if any(new_row.get(f) for f in ["stripe_color", "rosette_color", "madreporite_color", "overall_color"]):
            stats["colors_extracted"] += 1
    
    stats["rows_written"] = len(new_rows)
    
    if dry_run:
        print(f"  [DRY-RUN] Would write {len(new_rows)} rows to: {new_path}")
        # Show sample conversion
        if new_rows:
            print(f"\n  Sample conversion (first row):")
            sample = new_rows[0]
            for key, val in sample.items():
                if val:
                    print(f"    {key}: {val[:60]}{'...' if len(val) > 60 else ''}")
    else:
        # Backup old file
        backup = backup_file(old_path)
        print(f"  [BACKUP] Created: {backup.name}")
        
        # Write new file
        write_csv(new_path, new_header, new_rows)
        print(f"  [WRITE] Wrote {len(new_rows)} rows to: {new_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate starBoard metadata from V1 to V2 schema"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Path to archive directory (default: ./archive)",
    )
    args = parser.parse_args()
    
    # Determine archive directory
    if args.archive_dir:
        archive_root = args.archive_dir
    else:
        archive_root = Path(__file__).parent / "archive"
    
    if not archive_root.exists():
        print(f"ERROR: Archive directory not found: {archive_root}")
        return 1
    
    print("=" * 60)
    print("starBoard Metadata Migration: V1 → V2")
    print("=" * 60)
    print(f"\nArchive directory: {archive_root}")
    print(f"Mode: {'DRY-RUN (no files will be modified)' if args.dry_run else 'LIVE'}")
    print()
    
    total_stats = {
        "rows_read": 0,
        "rows_written": 0,
        "short_arm_codes_converted": 0,
        "stripe_order_extracted": 0,
        "stripe_prominence_extracted": 0,
        "reticulation_order_extracted": 0,
        "colors_extracted": 0,
    }
    
    # Migrate gallery metadata
    print("-" * 40)
    print("Migrating: gallery_metadata.csv")
    print("-" * 40)
    gallery_path = archive_root / "gallery" / "gallery_metadata.csv"
    gallery_stats = migrate_csv(
        old_path=gallery_path,
        new_path=gallery_path,  # Overwrite in place (after backup)
        old_header=V1_GALLERY_HEADER,
        new_header=V2_GALLERY_HEADER,
        id_col="gallery_id",
        dry_run=args.dry_run,
    )
    for k, v in gallery_stats.items():
        total_stats[k] += v
    
    print()
    
    # Migrate queries metadata
    print("-" * 40)
    print("Migrating: queries_metadata.csv")
    print("-" * 40)
    queries_path = archive_root / "queries" / "queries_metadata.csv"
    queries_stats = migrate_csv(
        old_path=queries_path,
        new_path=queries_path,  # Overwrite in place (after backup)
        old_header=V1_QUERIES_HEADER,
        new_header=V2_QUERIES_HEADER,
        id_col="query_id",
        dry_run=args.dry_run,
    )
    for k, v in queries_stats.items():
        total_stats[k] += v
    
    # Print summary
    print()
    print("=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"  Total rows read:              {total_stats['rows_read']}")
    print(f"  Total rows written:           {total_stats['rows_written']}")
    print(f"  Short arm codes converted:    {total_stats['short_arm_codes_converted']}")
    print(f"  Stripe order extracted:       {total_stats['stripe_order_extracted']}")
    print(f"  Stripe prominence extracted:  {total_stats['stripe_prominence_extracted']}")
    print(f"  Reticulation order extracted: {total_stats['reticulation_order_extracted']}")
    print(f"  Colors extracted:             {total_stats['colors_extracted']}")
    
    if args.dry_run:
        print("\n[DRY-RUN] No files were modified. Run without --dry-run to apply changes.")
    else:
        print("\n[DONE] Migration complete. Backup files created with .v1_backup_*.csv suffix.")
    
    return 0


if __name__ == "__main__":
    exit(main())

