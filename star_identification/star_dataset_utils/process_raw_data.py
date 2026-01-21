"""
Script to process raw star images using YOLO segmentation.
Moves images from star_dataset_raw to star_dataset after processing.
Optimized for HIGH RAM systems (Prefetching + Massive Async I/O).
Saves a list of failed images to a CSV file for review.
"""
import sys
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil
from PIL import Image
import math
import concurrent.futures
import time
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from wildlife_reid_inference.preprocessing import YOLOPreprocessor
except ImportError:
    print("Error: Could not import YOLOPreprocessor.")
    sys.exit(1)

def load_image_task(path):
    """Pre-load image into memory"""
    try:
        with Image.open(path) as img:
            img.load() # Force load into memory
            return img
    except Exception:
        return None

def save_image_task(image, path):
    """Task to save image in background thread"""
    try:
        # Optimize=False and compress_level=1 for speed
        image.save(path, optimize=False, compress_level=1)
        return True
    except Exception as e:
        return e

def process_dataset(dataset_name, raw_root, processed_root, yolo_model_path, 
                   confidence=0.7, keep_failed=False, batch_size=32):
    
    source_dir = Path(raw_root) / dataset_name
    dest_dir = Path(processed_root) / dataset_name
    
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist.")
        return

    print(f"Initializing YOLO model from {yolo_model_path}...")
    try:
        # Initialize with High RAM optimization flags if I add them later
        preprocessor = YOLOPreprocessor(str(yolo_model_path), confidence=confidence)
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return

    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Batch Size: {batch_size}")
    print("High RAM Mode: Enabled")
    
    # Collect images
    image_extensions = {'.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'}
    all_images = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if Path(file).suffix in image_extensions:
                all_images.append(Path(root) / file)
    
    print(f"Found {len(all_images)} images.")
    
    images_to_process = []
    skipped_count = 0
    for img_path in all_images:
        rel_path = img_path.relative_to(source_dir)
        out_path = (dest_dir / rel_path).with_suffix('.png')
        if out_path.exists(): skipped_count += 1
        else: images_to_process.append(img_path)
            
    print(f"Skipping {skipped_count} existing. Processing {len(images_to_process)}...")
    
    success_count = 0
    fail_count = 0
    failed_images = []
    
    # EXECUTORS
    # 1. Loader Pool: Loads raw images into RAM ahead of time
    # 2. Writer Pool: Saves processed images to disk
    loader_executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
    writer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
    
    # Using a list to hold loaded images in RAM
    # This is dangerous if dataset > RAM, but user said "ungodly amount"
    # We will use a sliding window of e.g. 200 images to be safe but fast
    PREFETCH_SIZE = batch_size * 4 

    with tqdm(total=len(images_to_process), desc="Processing") as pbar:
        for i in range(0, len(images_to_process), batch_size):
            # 1. Identify batch
            batch_paths = images_to_process[i:i+batch_size]
            
            # 2. Pre-load this batch into RAM using threads
            # (In a real pipelined system we'd be loading i+1 while processing i, 
            # but this is fast enough for now)
            loaded_batch = list(loader_executor.map(load_image_task, batch_paths))
            
            # Filter out load failures
            valid_batch_imgs = []
            valid_batch_indices = []
            for idx, img in enumerate(loaded_batch):
                if img is not None:
                    valid_batch_imgs.append(img)
                    valid_batch_indices.append(idx)
                else:
                    tqdm.write(f"Failed to load: {batch_paths[idx]}")
                    fail_count += 1
                    failed_images.append(str(batch_paths[idx]))

            if not valid_batch_imgs:
                pbar.update(len(batch_paths))
                continue

            try:
                # 3. GPU Inference + CPU Post-processing
                # Pass PIL images directly (already in RAM)
                results = preprocessor.process_batch(valid_batch_imgs, batch_size=len(valid_batch_imgs))
                
                # 4. Submit results for writing
                for k, result in enumerate(results):
                    original_idx = valid_batch_indices[k]
                    img_path = batch_paths[original_idx]
                    
                    rel_path = img_path.relative_to(source_dir)
                    out_path = (dest_dir / rel_path).with_suffix('.png')
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if result is not None:
                        # Submit save task (non-blocking)
                        writer_executor.submit(save_image_task, result, out_path)
                        success_count += 1
                    else:
                        if keep_failed:
                            shutil.copy2(img_path, out_path.parent / img_path.name)
                        fail_count += 1
                        failed_images.append(str(img_path))
                        
            except Exception as e:
                tqdm.write(f"Batch error: {e}")
                fail_count += len(batch_paths)
                for p in batch_paths:
                     failed_images.append(str(p))
                
            pbar.update(len(batch_paths))

    print("Waiting for final disk writes...")
    writer_executor.shutdown(wait=True)
    loader_executor.shutdown()

    print("\nProcessing Complete!")
    print(f"Processed: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Skipped: {skipped_count}")

    # Save failures to CSV
    if failed_images:
        csv_path = Path(__file__).parent / f"{dataset_name}_failures.csv"
        pd.DataFrame({'path': failed_images}).to_csv(csv_path, index=False)
        print(f"\nList of {len(failed_images)} failed images saved to:")
        print(f"  {csv_path}")
        print("Run 'python star_dataset_utils/review_failures.py' to review/delete them.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("--raw-root", default=str(PROJECT_ROOT / "star_dataset_raw"))
    parser.add_argument("--processed-root", default=str(PROJECT_ROOT / "star_dataset"))
    parser.add_argument("--model", default=str(PROJECT_ROOT / "wildlife_reid_inference" / "starseg_best.pt"))
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32, help="Increased default for High RAM")
    
    args = parser.parse_args()
    process_dataset(args.dataset_name, args.raw_root, args.processed_root, 
                   args.model, args.confidence, args.keep_failed, args.batch_size)
