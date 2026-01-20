"""
Data bridge for converting starBoard archive data to training format.

This module creates training-compatible datasets from the starBoard archive
without modifying the archive structure. It generates temporary metadata files
that the existing training infrastructure can consume.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from src.data import archive_paths as ap
from src.data.id_registry import list_ids
from src.data.image_index import list_image_files
from src.data.past_matches import build_past_matches_dataset
from src.dl.image_cache import get_cached_images, CACHE_ROOT

log = logging.getLogger("starBoard.dl.finetune.data_bridge")


@dataclass
class ArchiveIdentity:
    """Represents an identity from the archive with its images and metadata."""
    id_str: str
    target: str  # "Gallery" or "Queries"
    outings: Dict[str, List[Path]]  # encounter_name -> list of cached image paths
    
    @property
    def total_images(self) -> int:
        return sum(len(imgs) for imgs in self.outings.values())
    
    @property
    def num_outings(self) -> int:
        return len(self.outings)


def _parse_encounter_date(encounter_name: str) -> Optional[date]:
    """
    Parse date from encounter folder name.
    
    Expected format: M_D_YYYY_description (e.g., 3_23_2024_dock_sighting)
    """
    # Pattern: one or two digits for month and day, four digits for year
    match = re.match(r'^(\d{1,2})_(\d{1,2})_(\d{4})(?:_|$)', encounter_name)
    if match:
        try:
            month = int(match.group(1))
            day = int(match.group(2))
            year = int(match.group(3))
            return date(year, month, day)
        except ValueError:
            pass
    return None


def _get_encounter_for_image(image_path: Path, id_folder: Path) -> str:
    """
    Get the encounter name for an image based on its relative path.
    
    Images are organized as: id_folder/encounter_name/image.ext
    """
    try:
        rel = image_path.relative_to(id_folder)
        parts = rel.parts
        if len(parts) >= 2:
            return parts[0]  # First part is encounter name
    except ValueError:
        pass
    return "unknown_encounter"


def collect_archive_identities(
    include_gallery: bool = True,
    include_queries: bool = True,
    min_images: int = 1,
) -> List[ArchiveIdentity]:
    """
    Collect all identities from the starBoard archive with their cached images.
    
    Args:
        include_gallery: Include Gallery identities
        include_queries: Include Queries identities
        min_images: Minimum number of images required for inclusion
        
    Returns:
        List of ArchiveIdentity objects
    """
    identities: List[ArchiveIdentity] = []
    
    targets = []
    if include_gallery:
        targets.append("Gallery")
    if include_queries:
        targets.append("Queries")
    
    for target in targets:
        for id_str in list_ids(target):
            # Get cached images
            cached = get_cached_images(target, id_str)
            if len(cached) < min_images:
                continue
            
            # Group by encounter (outing)
            # We need to map cached images back to their original encounter structure
            outings: Dict[str, List[Path]] = {}
            
            # Get original images to understand encounter structure
            originals = list_image_files(target, id_str)
            original_stems = {p.stem: p for p in originals}
            
            for cached_path in cached:
                stem = cached_path.stem
                if stem in original_stems:
                    orig = original_stems[stem]
                    # Find the encounter folder by looking at the original path
                    for root in ap.roots_for_read(target):
                        id_folder = root / id_str
                        if id_folder.exists():
                            encounter = _get_encounter_for_image(orig, id_folder)
                            break
                    else:
                        encounter = "unknown"
                else:
                    encounter = "unknown"
                
                if encounter not in outings:
                    outings[encounter] = []
                outings[encounter].append(cached_path)
            
            if outings:
                identities.append(ArchiveIdentity(
                    id_str=id_str,
                    target=target,
                    outings=outings,
                ))
    
    log.info("Collected %d identities from archive", len(identities))
    return identities


def create_archive_metadata_csv(
    output_dir: Path,
    identities: Optional[List[ArchiveIdentity]] = None,
    train_ratio: float = 0.8,
    min_outings_for_eval: int = 2,
    seed: int = 42,
) -> Path:
    """
    Create a star_dataset-compatible metadata CSV from archive identities.
    
    This creates a metadata_temporal.csv that the existing StarDataset class
    can consume directly.
    
    Args:
        output_dir: Directory to write the metadata file
        identities: Pre-collected identities (or None to collect fresh)
        train_ratio: Fraction of outings to use for training
        min_outings_for_eval: Minimum outings needed for train/test split
        seed: Random seed for reproducibility
        
    Returns:
        Path to the created metadata CSV
    """
    if identities is None:
        identities = collect_archive_identities()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata_temporal.csv"
    
    rng = random.Random(seed)
    
    rows = []
    for identity in identities:
        # Create a unified identity string that includes the target
        # to distinguish between Gallery and Queries
        unified_id = f"{identity.target.lower()}__{identity.id_str}"
        
        # Determine if this identity can be used for evaluation
        # (needs at least min_outings_for_eval outings)
        num_outings = identity.num_outings
        negative_only = num_outings < min_outings_for_eval
        
        # Assign outings to train/test split
        outing_names = list(identity.outings.keys())
        
        if not negative_only and num_outings >= min_outings_for_eval:
            # Shuffle and split outings
            rng.shuffle(outing_names)
            n_train = max(1, int(len(outing_names) * train_ratio))
            train_outings = set(outing_names[:n_train])
            test_outings = set(outing_names[n_train:])
            
            # Ensure at least one in each split
            if not test_outings and len(outing_names) > 1:
                test_outings.add(train_outings.pop())
        else:
            # All go to train for negative-only identities
            train_outings = set(outing_names)
            test_outings = set()
        
        # Create rows for each image
        for outing_name, images in identity.outings.items():
            outing_date = _parse_encounter_date(outing_name)
            date_str = outing_date.isoformat() if outing_date else ""
            
            if outing_name in train_outings:
                split = "train"
            elif outing_name in test_outings:
                split = "test"
            else:
                split = "train"  # Default to train
            
            for img_path in images:
                # Use ABSOLUTE path to the cached image
                # This ensures StarDataset can find the images regardless of data_root
                abs_path = img_path.resolve()
                
                rows.append({
                    "path": str(abs_path),
                    "identity": unified_id,
                    "outing": outing_name,
                    "date": date_str,
                    "split": split,
                    "negative_only": "1" if negative_only else "0",
                    "source": "archive",
                })
    
    # Write CSV
    fieldnames = ["path", "identity", "outing", "date", "split", "negative_only", "source"]
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    log.info("Created archive metadata CSV: %s (%d rows)", metadata_path, len(rows))
    return metadata_path


def merge_with_star_dataset(
    archive_metadata: Path,
    star_dataset_root: Path,
    output_dir: Path,
) -> Path:
    """
    Merge archive metadata with external star_dataset metadata.
    
    Args:
        archive_metadata: Path to archive metadata CSV
        star_dataset_root: Path to star_dataset_resized root
        output_dir: Directory to write merged metadata
        
    Returns:
        Path to merged metadata CSV
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "metadata_temporal.csv"
    
    rows = []
    
    # Load archive metadata (paths should already be absolute)
    with open(archive_metadata, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        archive_rows = list(reader)
    
    # Archive paths are already absolute from create_archive_metadata_csv
    for row in archive_rows:
        rows.append(row)
    
    # Load star_dataset metadata
    star_metadata = star_dataset_root / "metadata_temporal.csv"
    if not star_metadata.exists():
        star_metadata = star_dataset_root / "metadata.csv"
    
    if star_metadata.exists():
        with open(star_metadata, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Make paths absolute
                if "path" in row:
                    path = row["path"]
                    if not Path(path).is_absolute():
                        row["path"] = str(star_dataset_root / path)
                elif "folder" in row and "filename" in row:
                    # Alternative format
                    row["path"] = str(star_dataset_root / row["folder"] / row["filename"])
                
                # Mark source
                row["source"] = "star_dataset"
                
                # Ensure required fields
                if "negative_only" not in row:
                    row["negative_only"] = "0"
                
                rows.append(row)
        
        log.info("Merged %d star_dataset rows", len(rows) - len(archive_rows))
    else:
        log.warning("star_dataset metadata not found: %s", star_metadata)
    
    # Write merged CSV
    if rows:
        fieldnames = list(rows[0].keys())
        # Ensure standard fields are first
        standard = ["path", "identity", "outing", "date", "split", "negative_only", "source"]
        fieldnames = standard + [f for f in fieldnames if f not in standard]
        
        with open(merged_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    
    log.info("Created merged metadata CSV: %s (%d rows)", merged_path, len(rows))
    return merged_path


def get_confirmed_matches() -> List[Tuple[str, str]]:
    """
    Get list of confirmed matches (YES verdict) from past_matches.
    
    Returns:
        List of (query_id, gallery_id) tuples for confirmed matches
    """
    try:
        ds = build_past_matches_dataset()
        matches = [
            (rec.query_id, rec.gallery_id)
            for rec in ds.records
            if rec.verdict == "yes"
        ]
        log.info("Found %d confirmed matches", len(matches))
        return matches
    except Exception as e:
        log.warning("Failed to load past matches: %s", e)
        return []


def create_verification_pairs(
    output_dir: Path,
    positive_ratio: float = 0.5,
    pairs_per_split: int = 10000,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """
    Create verification pair CSVs from archive data.
    
    Positive pairs come from:
    1. Confirmed matches (query matched to gallery)
    2. Same identity across different outings
    
    Negative pairs come from:
    1. Different identities
    
    Args:
        output_dir: Directory to write pair files
        positive_ratio: Ratio of positive pairs (default 0.5)
        pairs_per_split: Number of pairs per split
        seed: Random seed
        
    Returns:
        Tuple of (train_pairs_path, val_pairs_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rng = random.Random(seed)
    
    # Collect all identities with cached images
    identities = collect_archive_identities(min_images=2)
    
    # Get confirmed matches
    confirmed = get_confirmed_matches()
    confirmed_set = set(confirmed)
    
    # Build image index: identity -> list of (image_path, outing)
    identity_images: Dict[str, List[Tuple[Path, str]]] = {}
    for ident in identities:
        key = f"{ident.target.lower()}__{ident.id_str}"
        identity_images[key] = []
        for outing, imgs in ident.outings.items():
            for img in imgs:
                identity_images[key].append((img, outing))
    
    # Also create Gallery<->Query identity mapping from confirmed matches
    # A confirmed match means query and gallery are the same individual
    identity_groups: Dict[str, Set[str]] = {}  # group_id -> set of identity keys
    
    group_counter = 0
    for query_id, gallery_id in confirmed:
        q_key = f"queries__{query_id}"
        g_key = f"gallery__{gallery_id}"
        
        # Find existing group for either
        q_group = None
        g_group = None
        for gid, members in identity_groups.items():
            if q_key in members:
                q_group = gid
            if g_key in members:
                g_group = gid
        
        if q_group and g_group and q_group != g_group:
            # Merge groups
            identity_groups[q_group].update(identity_groups[g_group])
            del identity_groups[g_group]
        elif q_group:
            identity_groups[q_group].add(g_key)
        elif g_group:
            identity_groups[g_group].add(q_key)
        else:
            # Create new group
            gid = f"group_{group_counter}"
            group_counter += 1
            identity_groups[gid] = {q_key, g_key}
    
    # Create reverse mapping: identity key -> group
    identity_to_group: Dict[str, str] = {}
    for gid, members in identity_groups.items():
        for m in members:
            identity_to_group[m] = gid
    
    def are_same_individual(key1: str, key2: str) -> bool:
        """Check if two identity keys represent the same individual."""
        if key1 == key2:
            return True
        g1 = identity_to_group.get(key1)
        g2 = identity_to_group.get(key2)
        return g1 is not None and g1 == g2
    
    # Generate pairs
    all_keys = list(identity_images.keys())
    
    def generate_pairs(n_pairs: int) -> List[Dict]:
        pairs = []
        n_positive = int(n_pairs * positive_ratio)
        n_negative = n_pairs - n_positive
        
        # Generate positive pairs (same individual)
        attempts = 0
        while len([p for p in pairs if p["label"] == 1]) < n_positive and attempts < n_pairs * 10:
            attempts += 1
            
            # Strategy 1: Use confirmed matches
            if confirmed and rng.random() < 0.3:
                q_id, g_id = rng.choice(confirmed)
                q_key = f"queries__{q_id}"
                g_key = f"gallery__{g_id}"
                if q_key in identity_images and g_key in identity_images:
                    img1 = rng.choice(identity_images[q_key])[0]
                    img2 = rng.choice(identity_images[g_key])[0]
                    pairs.append({
                        "image_a": str(img1),
                        "image_b": str(img2),
                        "label": 1,
                    })
                    continue
            
            # Strategy 2: Same identity, different outing
            key = rng.choice(all_keys)
            imgs = identity_images[key]
            if len(imgs) >= 2:
                # Try to get images from different outings
                outings = list(set(o for _, o in imgs))
                if len(outings) >= 2:
                    o1, o2 = rng.sample(outings, 2)
                    imgs1 = [p for p, o in imgs if o == o1]
                    imgs2 = [p for p, o in imgs if o == o2]
                    if imgs1 and imgs2:
                        pairs.append({
                            "image_a": str(rng.choice(imgs1)),
                            "image_b": str(rng.choice(imgs2)),
                            "label": 1,
                        })
                        continue
                
                # Fallback: any two images from same identity
                img1, img2 = rng.sample(imgs, 2)
                pairs.append({
                    "image_a": str(img1[0]),
                    "image_b": str(img2[0]),
                    "label": 1,
                })
        
        # Generate negative pairs (different individuals)
        attempts = 0
        while len([p for p in pairs if p["label"] == 0]) < n_negative and attempts < n_pairs * 10:
            attempts += 1
            
            key1, key2 = rng.sample(all_keys, 2)
            
            # Make sure they're not the same individual
            if are_same_individual(key1, key2):
                continue
            
            img1 = rng.choice(identity_images[key1])[0]
            img2 = rng.choice(identity_images[key2])[0]
            
            pairs.append({
                "image_a": str(img1),
                "image_b": str(img2),
                "label": 0,
            })
        
        return pairs
    
    # Generate train and val pairs
    train_pairs = generate_pairs(pairs_per_split)
    val_pairs = generate_pairs(pairs_per_split // 5)  # Smaller validation set
    
    # Shuffle
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    
    # Write CSVs
    train_path = output_dir / "train_pairs.csv"
    val_path = output_dir / "val_pairs.csv"
    
    for path, pairs in [(train_path, train_pairs), (val_path, val_pairs)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_a", "image_b", "label"])
            writer.writeheader()
            writer.writerows(pairs)
    
    log.info("Created verification pairs: train=%d, val=%d", len(train_pairs), len(val_pairs))
    return train_path, val_path


def get_data_summary() -> Dict:
    """
    Get a summary of available data for fine-tuning.
    
    Returns:
        Dictionary with data statistics
    """
    gallery_ids = list_ids("Gallery")
    query_ids = list_ids("Queries")
    
    gallery_with_images = 0
    query_with_images = 0
    total_gallery_images = 0
    total_query_images = 0
    
    for gid in gallery_ids:
        cached = get_cached_images("Gallery", gid)
        if cached:
            gallery_with_images += 1
            total_gallery_images += len(cached)
    
    for qid in query_ids:
        cached = get_cached_images("Queries", qid)
        if cached:
            query_with_images += 1
            total_query_images += len(cached)
    
    confirmed = get_confirmed_matches()
    
    return {
        "gallery_ids": len(gallery_ids),
        "gallery_with_cached_images": gallery_with_images,
        "gallery_images": total_gallery_images,
        "query_ids": len(query_ids),
        "query_with_cached_images": query_with_images,
        "query_images": total_query_images,
        "total_images": total_gallery_images + total_query_images,
        "confirmed_matches": len(confirmed),
        "cache_exists": CACHE_ROOT.exists(),
    }

