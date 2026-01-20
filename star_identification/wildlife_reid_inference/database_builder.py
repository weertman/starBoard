"""
Build embedding database for Sunflower Star Police Lineup
Compatible with the demo data structure (outings/individuals)
Updated with consistent preprocessing parameters
"""
import numpy as np
from pathlib import Path
import json
import glob
import os
from typing import Dict, List
import argparse
from tqdm import tqdm

# Handle both package and direct script imports
try:
    from .inference import WildlifeReIDInference
except ImportError:
    from inference import WildlifeReIDInference


# Standard preprocessing parameters that must match the query processing
# CRITICAL: These values MUST be the same when building the database and when
# processing query images, or the embeddings will not be comparable!
STANDARD_CONFIDENCE = 0.7  # Minimum confidence to detect a star
STANDARD_HIGH_CONF_THRESHOLD = 0.9  # Above this, we select by size instead of confidence


def construct_field_data(path_archive, image_exts=['png', 'jpg', 'JPG', 'JPEG', 'jpeg'], verbose=True):
    """
    Construct field data dictionary organized by outing and individual.
    (Copied from demo.py for compatibility)

    Structure:
    {
        'outing_name': {
            'individual_id': {
                'images': [list of image paths],
                'path': folder path,
                'count': number of images
            }
        }
    }
    """
    field_data = {}

    outing_folders = [os.path.join(path_archive, s) for s in os.listdir(path_archive)
                      if os.path.isdir(os.path.join(path_archive, s))]

    for outing_path in outing_folders:
        outing_name = os.path.basename(outing_path)
        field_data[outing_name] = {}

        if verbose:
            print(f'\nOuting: {outing_name}')

        # Get all individual folders in this outing
        individual_folders = [os.path.join(outing_path, s) for s in os.listdir(outing_path)
                              if os.path.isdir(os.path.join(outing_path, s))]

        for ind_path in individual_folders:
            ind_id = os.path.basename(ind_path)

            # Collect all images for this individual
            id_images = []
            for ext in image_exts:
                pattern = os.path.join(ind_path, f'*.{ext}')
                id_images.extend(glob.glob(pattern))
            id_images = sorted(id_images)

            if id_images:  # Only add if there are images
                field_data[outing_name][ind_id] = {
                    'images': id_images,
                    'path': ind_path,
                    'count': len(id_images)
                }

                if verbose:
                    print(f'  Individual {ind_id}: {len(id_images)} images')

    # Summary statistics
    if verbose:
        total_outings = len(field_data)
        total_individuals = sum(len(individuals) for individuals in field_data.values())
        total_images = sum(
            ind_data['count']
            for individuals in field_data.values()
            for ind_data in individuals.values()
        )

        print(f'\n--- Field Data Summary ---')
        print(f'Total outings: {total_outings}')
        print(f'Total individuals: {total_individuals}')
        print(f'Total images: {total_images}')
        print('-' * 25)

    return field_data


def build_database_from_field_data(reid_checkpoint: str, yolo_checkpoint: str,
                                   data_dir: str, output_path: str,
                                   batch_size: int = 32,
                                   include_outing_in_id: bool = True,
                                   confidence: float = None,
                                   high_conf_threshold: float = None):
    """
    Build embedding database from field data structure

    Args:
        reid_checkpoint: Path to ReID model
        yolo_checkpoint: Path to YOLO model
        data_dir: Path to field data archive (with outing/individual structure)
        output_path: Output database path
        batch_size: Batch size for processing
        include_outing_in_id: If True, identity will be "outing__individual"
        confidence: YOLO confidence threshold (uses standard if None)
        high_conf_threshold: High confidence threshold for size selection (uses standard if None)
    """

    # Use standard parameters if not specified
    if confidence is None:
        confidence = STANDARD_CONFIDENCE
    if high_conf_threshold is None:
        high_conf_threshold = STANDARD_HIGH_CONF_THRESHOLD

    # Initialize model
    print("Initializing models...")
    reid_model = WildlifeReIDInference(reid_checkpoint, device='cuda')

    # Set preprocessor with consistent parameters
    print(f"Setting up preprocessor with confidence={confidence}, high_conf_threshold={high_conf_threshold}")
    reid_model.set_preprocessor(yolo_checkpoint, confidence=confidence, high_conf_threshold=high_conf_threshold)

    # Load field data
    print("\nLoading field data structure...")
    field_data = construct_field_data(data_dir, verbose=True)

    if not field_data:
        raise ValueError(f"No field data found in {data_dir}")

    # Extract embeddings
    all_embeddings = []
    all_image_paths = []
    all_identities = []
    all_outings = []
    metadata = {}
    failed_images = []

    image_idx = 0

    # Process each outing
    for outing_name, individuals in field_data.items():
        print(f"\nProcessing outing: {outing_name}")

        # Process each individual in this outing
        for ind_id, ind_data in tqdm(individuals.items(),
                                     desc=f"  Individuals in {outing_name}"):

            # Create identity string
            if include_outing_in_id:
                identity = f"{outing_name}__{ind_id}"
            else:
                identity = ind_id

            # Process images in batches
            image_paths = ind_data['images']

            try:
                # Extract embeddings for this individual
                embeddings = reid_model.embed_images(image_paths,
                                                   batch_size=batch_size,
                                                   preprocess=True)

                # Store results (some images might have failed)
                successful_count = 0
                for i, (embedding, img_path) in enumerate(zip(embeddings, image_paths)):
                    # Check if embedding extraction was successful
                    if embedding is not None and not np.isnan(embedding).any():
                        all_embeddings.append(embedding)
                        all_image_paths.append(img_path)
                        all_identities.append(identity)
                        all_outings.append(outing_name)

                        # Add metadata
                        metadata[image_idx] = {
                            'outing': outing_name,
                            'individual_id': ind_id,
                            'full_identity': identity,
                            'image_index': i,
                            'filename': Path(img_path).name,
                            'total_images_for_individual': ind_data['count']
                        }

                        image_idx += 1
                        successful_count += 1
                    else:
                        failed_images.append(img_path)
                        print(f"    Warning: Failed to process image {Path(img_path).name}")

                if successful_count == 0:
                    print(f"    Warning: All images failed for {ind_id}")

            except Exception as e:
                print(f"    Warning: Failed to process {ind_id}: {e}")
                failed_images.extend(image_paths)
                continue

    if not all_embeddings:
        raise ValueError("No embeddings were successfully extracted!")

    # Stack embeddings
    embeddings_array = np.vstack(all_embeddings)

    print(f"\nEmbeddings shape: {embeddings_array.shape}")

    # Save database
    print(f"Saving database to {output_path}...")
    np.savez(
        output_path,
        embeddings=embeddings_array,
        image_paths=all_image_paths,
        identities=all_identities,
        outings=all_outings,
        metadata=metadata,
        # Save preprocessing parameters for verification
        preprocessing_params={
            'confidence': confidence,
            'high_conf_threshold': high_conf_threshold
        },
        # Save field data structure for reference
        field_data_structure=field_data
    )

    print("Database created successfully!")

    # Print detailed statistics
    unique_identities = len(set(all_identities))
    unique_outings = len(set(all_outings))

    print(f"\nDatabase statistics:")
    print(f"  Total embeddings: {len(embeddings_array)}")
    print(f"  Total outings: {unique_outings}")
    print(f"  Unique individuals: {unique_identities}")
    print(f"  Embedding dimension: {embeddings_array.shape[1]}")
    print(f"  Average images per individual: {len(embeddings_array) / unique_identities:.1f}")
    print(f"  Failed images: {len(failed_images)}")
    print(f"\nPreprocessing parameters used:")
    print(f"  Confidence threshold: {confidence}")
    print(f"  High confidence threshold: {high_conf_threshold}")

    # Per-outing statistics
    print(f"\nPer-outing breakdown:")
    outing_counts = {}
    for outing in all_outings:
        outing_counts[outing] = outing_counts.get(outing, 0) + 1

    for outing, count in sorted(outing_counts.items()):
        print(f"  {outing}: {count} images")

    # Save failed images list if any
    if failed_images:
        failed_path = Path(output_path).with_suffix('.failed.txt')
        with open(failed_path, 'w') as f:
            for img_path in failed_images:
                f.write(f"{img_path}\n")
        print(f"\nList of failed images saved to: {failed_path}")


def build_database_single_outing(reid_checkpoint: str, yolo_checkpoint: str,
                               data_dir: str, output_path: str,
                               batch_size: int = 32,
                               confidence: float = None,
                               high_conf_threshold: float = None):
    """
    Build database from a single outing directory (simpler structure)

    Expected structure:
    data_dir/
        individual_001/
            image1.jpg
            image2.jpg
        individual_002/
            image1.jpg
            ...
    """

    # Use standard parameters if not specified
    if confidence is None:
        confidence = STANDARD_CONFIDENCE
    if high_conf_threshold is None:
        high_conf_threshold = STANDARD_HIGH_CONF_THRESHOLD

    # Initialize model
    print("Initializing models...")
    reid_model = WildlifeReIDInference(reid_checkpoint, device='cuda')

    # Set preprocessor with consistent parameters
    print(f"Setting up preprocessor with confidence={confidence}, high_conf_threshold={high_conf_threshold}")
    reid_model.set_preprocessor(yolo_checkpoint, confidence=confidence, high_conf_threshold=high_conf_threshold)

    # Collect images
    print("Collecting images...")
    data_dir = Path(data_dir)

    all_embeddings = []
    all_image_paths = []
    all_identities = []
    metadata = {}
    failed_images = []

    image_idx = 0

    # Find all individual directories
    individual_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    if not individual_dirs:
        raise ValueError(f"No individual directories found in {data_dir}")

    print(f"Found {len(individual_dirs)} individuals")

    for individual_dir in tqdm(individual_dirs, desc="Processing individuals"):
        identity = individual_dir.name

        # Collect images for this individual
        image_paths = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            image_paths.extend(list(individual_dir.glob(ext)))

        if not image_paths:
            print(f"  Warning: No images found for {identity}")
            continue

        image_paths = [str(img) for img in sorted(image_paths)]

        try:
            # Extract embeddings
            embeddings = reid_model.embed_images(image_paths,
                                               batch_size=batch_size,
                                               preprocess=True)

            # Store results
            successful_count = 0
            for i, (embedding, img_path) in enumerate(zip(embeddings, image_paths)):
                if embedding is not None and not np.isnan(embedding).any():
                    all_embeddings.append(embedding)
                    all_image_paths.append(img_path)
                    all_identities.append(identity)

                    # Add metadata
                    metadata[image_idx] = {
                        'individual_id': identity,
                        'image_index': i,
                        'filename': Path(img_path).name,
                        'total_images_for_individual': len(image_paths)
                    }

                    image_idx += 1
                    successful_count += 1
                else:
                    failed_images.append(img_path)

            if successful_count == 0:
                print(f"  Warning: All images failed for {identity}")

        except Exception as e:
            print(f"  Warning: Failed to process {identity}: {e}")
            failed_images.extend(image_paths)
            continue

    if not all_embeddings:
        raise ValueError("No embeddings were successfully extracted!")

    # Stack embeddings
    embeddings_array = np.vstack(all_embeddings)

    print(f"\nEmbeddings shape: {embeddings_array.shape}")

    # Save database
    print(f"Saving database to {output_path}...")
    np.savez(
        output_path,
        embeddings=embeddings_array,
        image_paths=all_image_paths,
        identities=all_identities,
        metadata=metadata,
        # Save preprocessing parameters
        preprocessing_params={
            'confidence': confidence,
            'high_conf_threshold': high_conf_threshold
        }
    )

    print("Database created successfully!")

    # Print statistics
    unique_identities = len(set(all_identities))
    print(f"\nDatabase statistics:")
    print(f"  Total embeddings: {len(embeddings_array)}")
    print(f"  Unique individuals: {unique_identities}")
    print(f"  Embedding dimension: {embeddings_array.shape[1]}")
    print(f"  Average images per individual: {len(embeddings_array) / unique_identities:.1f}")
    print(f"  Failed images: {len(failed_images)}")
    print(f"\nPreprocessing parameters used:")
    print(f"  Confidence threshold: {confidence}")
    print(f"  High confidence threshold: {high_conf_threshold}")


def main():
    parser = argparse.ArgumentParser(description='Build embedding database from field data')
    parser.add_argument('--reid-checkpoint', type=str, required=True,
                       help='Path to ReID model checkpoint')
    parser.add_argument('--yolo-checkpoint', type=str, required=True,
                       help='Path to YOLO model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing field data (outing/individual structure)')
    parser.add_argument('--output', type=str, default='lineup_database.npz',
                       help='Output database path')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--single-outing', action='store_true',
                       help='Data directory contains single outing (no outing subdirs)')
    parser.add_argument('--no-outing-prefix', action='store_true',
                       help='Do not include outing name in identity')
    parser.add_argument('--confidence', type=float, default=STANDARD_CONFIDENCE,
                       help=f'YOLO confidence threshold (default: {STANDARD_CONFIDENCE})')
    parser.add_argument('--high-conf-threshold', type=float, default=STANDARD_HIGH_CONF_THRESHOLD,
                       help=f'High confidence threshold for size selection (default: {STANDARD_HIGH_CONF_THRESHOLD})')

    args = parser.parse_args()

    if args.single_outing:
        # Simple structure: data_dir/individual/images
        build_database_single_outing(
            reid_checkpoint=args.reid_checkpoint,
            yolo_checkpoint=args.yolo_checkpoint,
            data_dir=args.data_dir,
            output_path=args.output,
            batch_size=args.batch_size,
            confidence=args.confidence,
            high_conf_threshold=args.high_conf_threshold
        )
    else:
        # Full field data structure: data_dir/outing/individual/images
        build_database_from_field_data(
            reid_checkpoint=args.reid_checkpoint,
            yolo_checkpoint=args.yolo_checkpoint,
            data_dir=args.data_dir,
            output_path=args.output,
            batch_size=args.batch_size,
            include_outing_in_id=not args.no_outing_prefix,
            confidence=args.confidence,
            high_conf_threshold=args.high_conf_threshold
        )

if __name__ == '__main__':
    main()