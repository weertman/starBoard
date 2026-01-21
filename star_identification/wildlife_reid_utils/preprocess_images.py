#!/usr/bin/env python
"""
Preprocess images for faster training.

Resizes all images to a maximum size on the longest side while preserving aspect ratio.
Creates a new folder structure that mirrors the original.

Usage:
    python -m wildlife_reid_utils.preprocess_images
    python -m wildlife_reid_utils.preprocess_images --size 620 --quality 95
    python -m wildlife_reid_utils.preprocess_images --dataset star --dry-run
"""
import argparse
import os
import shutil
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
from multiprocessing import cpu_count

import pandas as pd
from PIL import Image
from tqdm import tqdm


def get_output_path(
    input_path: Path,
    input_root: Path,
    output_root: Path,
) -> Path:
    """Get the corresponding output path for an input image."""
    relative = input_path.relative_to(input_root)
    return output_root / relative


def resize_image(
    input_path: Path,
    output_path: Path,
    max_size: int = 620,
    quality: int = 95,
) -> Tuple[bool, str, Tuple[int, int], Tuple[int, int]]:
    """
    Resize a single image.
    
    Args:
        input_path: Path to input image
        output_path: Path to save resized image
        max_size: Maximum size on longest side
        quality: JPEG quality (1-100)
        
    Returns:
        (success, message, original_size, new_size)
    """
    try:
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load image
        with Image.open(input_path) as img:
            original_size = img.size  # (width, height)
            
            # Convert to RGB if necessary (handles RGBA, palette, etc.)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Calculate new size preserving aspect ratio
            width, height = img.size
            if max(width, height) <= max_size:
                # Already small enough, just copy/save
                new_size = (width, height)
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
            else:
                # Resize
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                
                new_size = (new_width, new_height)
                
                # Use LANCZOS for high-quality downsampling
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                img_resized.save(output_path, 'JPEG', quality=quality, optimize=True)
        
        return True, "OK", original_size, new_size
        
    except Exception as e:
        return False, str(e), (0, 0), (0, 0)


def _process_one_image(args: Tuple) -> Tuple[str, bool, str, int, int]:
    """Process a single image (for multiprocessing)."""
    input_path_str, output_path_str, max_size, quality = args
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)
    
    # Skip if output already exists and is newer
    if output_path.exists():
        try:
            if output_path.stat().st_mtime >= input_path.stat().st_mtime:
                return input_path_str, True, "skipped", 0, 0
        except:
            pass
    
    success, msg, orig_size, new_size = resize_image(
        input_path, output_path, max_size, quality
    )
    
    input_bytes = input_path.stat().st_size if success else 0
    output_bytes = output_path.stat().st_size if success and output_path.exists() else 0
    
    return input_path_str, success, msg, input_bytes, output_bytes


def process_dataset(
    input_root: Path,
    output_root: Path,
    image_paths: List[Path],
    max_size: int = 620,
    quality: int = 95,
    num_workers: int = 32,
    dry_run: bool = False,
) -> dict:
    """
    Process all images in a dataset.
    
    Args:
        input_root: Root directory of input dataset
        output_root: Root directory for output
        image_paths: List of image paths to process
        max_size: Maximum size on longest side
        quality: JPEG quality
        num_workers: Number of parallel workers
        dry_run: If True, don't actually process
        
    Returns:
        Statistics dict
    """
    stats = {
        'total': len(image_paths),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'total_input_bytes': 0,
        'total_output_bytes': 0,
        'failed_paths': [],
    }
    
    if dry_run:
        # Estimate time
        est_per_image = 0.05  # ~50ms per image with fast SSD
        est_time = len(image_paths) * est_per_image / num_workers
        print(f"[DRY RUN] Would process {len(image_paths)} images")
        print(f"  Input: {input_root}")
        print(f"  Output: {output_root}")
        print(f"  Estimated time: ~{est_time/60:.1f} minutes with {num_workers} workers")
        return stats
    
    # Prepare arguments for multiprocessing
    work_items = []
    for input_path in image_paths:
        output_path = get_output_path(input_path, input_root, output_root)
        work_items.append((str(input_path), str(output_path), max_size, quality))
    
    start_time = time.time()
    
    # Use ProcessPoolExecutor for CPU-bound resize operations
    # Fall back to ThreadPoolExecutor if multiprocessing fails
    try:
        executor_class = ProcessPoolExecutor
        # Limit to CPU count for ProcessPoolExecutor
        effective_workers = min(num_workers, cpu_count())
    except:
        executor_class = ThreadPoolExecutor
        effective_workers = num_workers
    
    print(f"Using {effective_workers} workers ({executor_class.__name__})")
    
    with executor_class(max_workers=effective_workers) as executor:
        # Submit all work
        futures = list(executor.map(_process_one_image, work_items, chunksize=100))
        
        # Process results (already completed since map blocks)
        for result in tqdm(futures, desc="Processing results", total=len(futures)):
            input_path_str, success, msg, in_bytes, out_bytes = result
            
            if msg == "skipped":
                stats['skipped'] += 1
            elif success:
                stats['success'] += 1
                stats['total_input_bytes'] += in_bytes
                stats['total_output_bytes'] += out_bytes
            else:
                stats['failed'] += 1
                stats['failed_paths'].append((input_path_str, msg))
    
    elapsed = time.time() - start_time
    images_per_sec = len(image_paths) / elapsed if elapsed > 0 else 0
    print(f"Completed in {elapsed:.1f}s ({images_per_sec:.0f} images/sec)")
    
    return stats


def find_images(root: Path, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp')) -> List[Path]:
    """Find all images in a directory tree."""
    images = []
    for ext in extensions:
        images.extend(root.rglob(f'*{ext}'))
        images.extend(root.rglob(f'*{ext.upper()}'))
    return sorted(set(images))


def update_metadata_paths(
    csv_path: Path,
    old_root: Path,
    new_root: Path,
) -> int:
    """
    Update paths in a metadata CSV file to point to the new root.
    
    Args:
        csv_path: Path to the CSV file to update
        old_root: Original root path to replace
        new_root: New root path
        
    Returns:
        Number of paths updated
    """
    if not csv_path.exists():
        return 0
    
    df = pd.read_csv(csv_path)
    
    if 'path' not in df.columns:
        return 0
    
    # Convert to absolute paths for reliable replacement
    old_root_abs = str(old_root.resolve())
    new_root_abs = str(new_root.resolve())
    
    updated = 0
    new_paths = []
    
    for path in df['path']:
        if pd.isna(path):
            new_paths.append(path)
            continue
            
        path_str = str(path)
        
        # Handle both forward and backslashes
        old_root_variants = [
            old_root_abs,
            old_root_abs.replace('\\', '/'),
            old_root_abs.replace('/', '\\'),
            str(old_root),
            str(old_root).replace('\\', '/'),
            str(old_root).replace('/', '\\'),
        ]
        
        replaced = False
        for old_variant in old_root_variants:
            if old_variant in path_str:
                new_path = path_str.replace(old_variant, new_root_abs)
                new_paths.append(new_path)
                updated += 1
                replaced = True
                break
        
        if not replaced:
            new_paths.append(path_str)
    
    df['path'] = new_paths
    df.to_csv(csv_path, index=False)
    
    return updated


def process_star_dataset(
    input_root: str = './star_dataset',
    output_root: str = './star_dataset_resized',
    max_size: int = 620,
    quality: int = 95,
    num_workers: int = 8,
    dry_run: bool = False,
) -> dict:
    """Process star_dataset."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    print(f"\n{'='*60}")
    print("Processing star_dataset")
    print(f"{'='*60}")
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Max size: {max_size}px | Quality: {quality}")
    
    # Find all images
    image_paths = find_images(input_root)
    print(f"Found {len(image_paths)} images")
    
    if not image_paths:
        print("No images found!")
        return {}
    
    # Process
    stats = process_dataset(
        input_root=input_root,
        output_root=output_root,
        image_paths=image_paths,
        max_size=max_size,
        quality=quality,
        num_workers=num_workers,
        dry_run=dry_run,
    )
    
    # Copy metadata files and update paths
    if not dry_run:
        for meta_file in ['metadata.csv', 'metadata_temporal.csv']:
            src = input_root / meta_file
            dst = output_root / meta_file
            if src.exists():
                shutil.copy2(src, dst)
                # Update paths in the copied CSV to point to resized images
                updated = update_metadata_paths(dst, input_root, output_root)
                print(f"Copied {meta_file} (updated {updated} paths)")
    
    return stats


def process_wildlife10k(
    input_root: str = './wildlifeReID',
    output_root: str = './wildlifeReID_resized',
    max_size: int = 620,
    quality: int = 95,
    num_workers: int = 8,
    dry_run: bool = False,
) -> dict:
    """Process Wildlife10k dataset."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    print(f"\n{'='*60}")
    print("Processing Wildlife10k")
    print(f"{'='*60}")
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Max size: {max_size}px | Quality: {quality}")
    
    # Find all images
    image_paths = find_images(input_root)
    print(f"Found {len(image_paths)} images")
    
    if not image_paths:
        print("No images found!")
        return {}
    
    # Process
    stats = process_dataset(
        input_root=input_root,
        output_root=output_root,
        image_paths=image_paths,
        max_size=max_size,
        quality=quality,
        num_workers=num_workers,
        dry_run=dry_run,
    )
    
    # Copy metadata file and update paths
    if not dry_run:
        meta_src = input_root / 'metadata.csv'
        meta_dst = output_root / 'metadata.csv'
        if meta_src.exists():
            shutil.copy2(meta_src, meta_dst)
            print(f"Copied metadata.csv")
    
    return stats


def print_stats(stats: dict, name: str):
    """Print processing statistics."""
    if not stats:
        return
    
    print(f"\n{name} Statistics:")
    print(f"  Total images: {stats['total']}")
    print(f"  Processed: {stats['success']}")
    print(f"  Skipped (already done): {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
    
    if stats['total_input_bytes'] > 0:
        input_mb = stats['total_input_bytes'] / (1024 * 1024)
        output_mb = stats['total_output_bytes'] / (1024 * 1024)
        ratio = stats['total_output_bytes'] / stats['total_input_bytes'] * 100
        print(f"  Size: {input_mb:.1f}MB → {output_mb:.1f}MB ({ratio:.1f}%)")
    
    if stats['failed_paths']:
        print(f"\n  Failed files:")
        for path, msg in stats['failed_paths'][:10]:
            print(f"    {path}: {msg}")
        if len(stats['failed_paths']) > 10:
            print(f"    ... and {len(stats['failed_paths']) - 10} more")


def main():
    parser = argparse.ArgumentParser(description='Preprocess images for faster training')
    
    parser.add_argument('--size', type=int, default=620,
                        help='Maximum size on longest side (default: 620)')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPEG quality 1-100 (default: 95)')
    parser.add_argument('--workers', type=int, default=32,
                        help='Number of parallel workers (default: 32, I/O bound so more is better)')
    parser.add_argument('--dataset', type=str, default='both',
                        choices=['star', 'wildlife', 'both'],
                        help='Which dataset to process')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without processing')
    
    # Paths
    parser.add_argument('--star-input', type=str, default='./star_dataset',
                        help='Input path for star_dataset')
    parser.add_argument('--star-output', type=str, default='./star_dataset_resized',
                        help='Output path for star_dataset')
    parser.add_argument('--wildlife-input', type=str, default='./wildlifeReID',
                        help='Input path for Wildlife10k')
    parser.add_argument('--wildlife-output', type=str, default='./wildlifeReID_resized',
                        help='Output path for Wildlife10k')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IMAGE PREPROCESSING")
    print("=" * 60)
    print(f"Target size: {args.size}px (longest side)")
    print(f"JPEG quality: {args.quality}")
    print(f"Workers: {args.workers}")
    
    all_stats = {}
    
    if args.dataset in ['star', 'both']:
        stats = process_star_dataset(
            input_root=args.star_input,
            output_root=args.star_output,
            max_size=args.size,
            quality=args.quality,
            num_workers=args.workers,
            dry_run=args.dry_run,
        )
        all_stats['star_dataset'] = stats
        print_stats(stats, 'star_dataset')
    
    if args.dataset in ['wildlife', 'both']:
        stats = process_wildlife10k(
            input_root=args.wildlife_input,
            output_root=args.wildlife_output,
            max_size=args.size,
            quality=args.quality,
            num_workers=args.workers,
            dry_run=args.dry_run,
        )
        all_stats['Wildlife10k'] = stats
        print_stats(stats, 'Wildlife10k')
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    
    if not args.dry_run:
        print("\nTo use resized images, update your config or use these paths:")
        if args.dataset in ['star', 'both']:
            print(f"  star_dataset_root: '{args.star_output}'")
        if args.dataset in ['wildlife', 'both']:
            print(f"  wildlife_root: '{args.wildlife_output}'")


if __name__ == '__main__':
    main()

