"""
Data preparation with temporal train/test splitting.

Key concepts:
- Temporal session: A unique DATE when observations were made (multiple camera folders
  from the same day are collapsed into one session)
- Multi-session identity: An individual observed on 2+ different DATES (can be temporally evaluated)
- Single-session identity: An individual observed on only 1 date (negative-only during training)

Note: The term "outing" in folder names refers to a photo collection session, but multiple
folders can exist for the same date (e.g., different cameras: JU_PICS, WW_PICS). The splitting
logic groups by DATE, not by folder name.
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, List, Dict
from collections import defaultdict


# =============================================================================
# MANUAL DATE OVERRIDES
# =============================================================================
# For folders where dates cannot be parsed from the folder name, specify the
# actual date here. Format: 'folder_name': datetime(year, month, day)
#
# Use the SAME date for folders that represent different cameras from the
# same field session (e.g., JU, WW, BW suffixes = different photographers).

MANUAL_DATE_OVERRIDES: Dict[str, datetime] = {
    # PWS_2023: All three folders are from the same single field trip in 2023
    # Different suffixes represent different cameras/photographers
    'PWS_2023_JU': datetime(2023, 7, 15),  # Prince William Sound 2023 - Camera JU
    'PWS_2023_WW': datetime(2023, 7, 15),  # Prince William Sound 2023 - Camera WW
    'PWS_2023_BW': datetime(2023, 7, 15),  # Prince William Sound 2023 - Camera BW
}


def parse_date_from_folder(folder_name: str, use_overrides: bool = True) -> Optional[datetime]:
    """
    Parse date from folder name with support for multiple formats.
    
    Supported formats:
    - Manual override (checked first if use_overrides=True)
    - M_D_YYYY_* (e.g., '3_23_2024_dock_sighting')
    - MM_DD_YYYY_* (e.g., '04_06_2024__4-6-brown', '09_11_2021')
    - YYYY-MM-DD (ISO format embedded in folder name)
    
    Returns None if no valid date can be extracted.
    
    Args:
        folder_name: The folder name to parse
        use_overrides: If True, check MANUAL_DATE_OVERRIDES first
    
    Note: We require full date (year, month, day) - partial dates (year-only)
    are not returned as that would incorrectly collapse observations.
    """
    folder_name = folder_name.strip()
    
    # Check manual overrides first
    if use_overrides and folder_name in MANUAL_DATE_OVERRIDES:
        return MANUAL_DATE_OVERRIDES[folder_name]
    
    # Pattern 1: M_D_YYYY or MM_DD_YYYY at start (most common)
    # Matches: 3_23_2024_*, 04_06_2024_*, 09_11_2021, etc.
    match = re.match(r'^(\d{1,2})_(\d{1,2})_(\d{4})', folder_name)
    if match:
        try:
            month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                return datetime(year, month, day)
        except ValueError:
            pass
    
    # Pattern 2: ISO format YYYY-MM-DD anywhere in string
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', folder_name)
    if match:
        try:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                return datetime(year, month, day)
        except ValueError:
            pass
    
    return None


def scan_star_dataset(dataset_root: Path) -> pd.DataFrame:
    """
    Scan star dataset directory structure and build metadata DataFrame.
    
    Expected structure:
        dataset_root/
            DATASET_NAME/
                individual_id/
                    date_folder/
                        image.png
    
    Returns DataFrame with columns:
        - identity: unique identifier (DATASET__individual)
        - path: full path to image
        - dataset: source dataset name
        - individual: individual name within dataset
        - outing: date_folder name (observation session)
        - date: parsed datetime (or NaT if unparseable)
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    records = []
    
    dataset_root = Path(dataset_root)
    
    for dataset_dir in sorted(dataset_root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
            continue
        
        dataset_name = dataset_dir.name
        
        for individual_dir in sorted(dataset_dir.iterdir()):
            if not individual_dir.is_dir():
                continue
            
            individual_name = individual_dir.name
            identity = f"{dataset_name}__{individual_name}"
            
            for outing_dir in sorted(individual_dir.iterdir()):
                if not outing_dir.is_dir():
                    continue
                
                outing_name = outing_dir.name
                parsed_date = parse_date_from_folder(outing_name)
                
                for image_path in sorted(outing_dir.iterdir()):
                    if image_path.suffix in image_extensions:
                        records.append({
                            'identity': identity,
                            'path': str(image_path.absolute()),
                            'dataset': dataset_name,
                            'individual': individual_name,
                            'outing': outing_name,
                            'date': parsed_date,
                        })
    
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        raise ValueError(f"No images found in {dataset_root}")
    
    print(f"Scanned {len(df)} images from {df['identity'].nunique()} identities")
    return df


def compute_identity_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-identity statistics for splitting decisions.
    
    IMPORTANT: This function counts unique DATES, not folder names. Multiple folders
    from the same date (e.g., different cameras) are collapsed into one temporal session.
    
    Returns DataFrame indexed by identity with columns:
        - num_images: total images
        - num_folders: number of unique outing folders
        - num_sessions: number of unique temporal sessions (dates + undated folders)
        - num_dated_sessions: number of unique dates with parsed dates
        - min_date, max_date: date range
        - is_multi_session: True if num_sessions >= 2
    """
    stats = df.groupby('identity').agg({
        'path': 'count',
        'outing': 'nunique',
        'date': ['min', 'max', lambda x: x.notna().sum()]
    }).reset_index()
    
    stats.columns = ['identity', 'num_images', 'num_folders', 'min_date', 'max_date', 'num_dated_images']
    
    # Count unique dates (temporal sessions)
    # This is the key fix: we count unique DATES, not unique folder names
    dated_sessions = df[df['date'].notna()].groupby('identity')['date'].nunique().reset_index()
    dated_sessions.columns = ['identity', 'num_dated_sessions']
    stats = stats.merge(dated_sessions, on='identity', how='left')
    stats['num_dated_sessions'] = stats['num_dated_sessions'].fillna(0).astype(int)
    
    # Count undated folders (each is treated as a separate session conservatively)
    undated_folders = df[df['date'].isna()].groupby('identity')['outing'].nunique().reset_index()
    undated_folders.columns = ['identity', 'num_undated_folders']
    stats = stats.merge(undated_folders, on='identity', how='left')
    stats['num_undated_folders'] = stats['num_undated_folders'].fillna(0).astype(int)
    
    # Total sessions = unique dates + undated folders
    stats['num_sessions'] = stats['num_dated_sessions'] + stats['num_undated_folders']
    
    # An identity is multi-session if it has 2+ unique temporal sessions
    stats['is_multi_session'] = stats['num_sessions'] >= 2
    
    return stats.set_index('identity')


def assign_temporal_splits(
    df: pd.DataFrame,
    train_session_ratio: float = 0.8,
    min_sessions_for_eval: int = 2,
    seed: int = 42
) -> pd.DataFrame:
    """
    Assign temporal train/test splits to the dataset.
    
    IMPORTANT: This function splits by DATE, not by folder name. Multiple folders
    from the same date (e.g., different cameras) are treated as ONE temporal session
    and will all go to the same split.
    
    Split logic:
    - Multi-session identities: earliest dates → train, latest dates → test
    - Single-session identities: all images → train (negative_only=True)
    
    Adds columns to df:
        - split: 'train' or 'test'
        - negative_only: True for single-session identities
        - outing_split: 'train' or 'test' for the outing itself
    
    Args:
        df: DataFrame with 'identity', 'outing', 'date' columns
        train_session_ratio: Fraction of temporal sessions for training (default 0.8)
        min_sessions_for_eval: Minimum unique dates/sessions to be evaluable (default 2)
        seed: Random seed for reproducibility
    
    Returns modified DataFrame.
    """
    np.random.seed(seed)
    
    df = df.copy()
    df['split'] = ''
    df['negative_only'] = False
    df['outing_split'] = ''
    
    # Get identity statistics (now counts unique dates, not folders)
    identity_stats = compute_identity_statistics(df)
    
    multi_session_ids = identity_stats[identity_stats['is_multi_session']].index.tolist()
    single_session_ids = identity_stats[~identity_stats['is_multi_session']].index.tolist()
    
    print(f"\nIdentity breakdown (by unique dates, not folders):")
    print(f"  Multi-session (evaluable): {len(multi_session_ids)}")
    print(f"  Single-session (negative-only): {len(single_session_ids)}")
    
    # Process multi-session identities
    for identity in multi_session_ids:
        id_mask = df['identity'] == identity
        id_data = df[id_mask]
        
        # Get unique dates sorted chronologically
        dated_sessions = sorted([d for d in id_data['date'].dropna().unique()])
        
        # Get undated folders (each treated as separate session, placed at end)
        undated_folders = list(id_data[id_data['date'].isna()]['outing'].unique())
        
        # Combine: dated sessions first (sorted), then undated folders
        # Undated folders go at the end (conservative: more likely to be test)
        all_sessions = dated_sessions + undated_folders
        
        # Calculate split point based on sessions, not folders
        n_train = max(1, int(len(all_sessions) * train_session_ratio))
        if n_train >= len(all_sessions):
            n_train = len(all_sessions) - 1  # Ensure at least 1 test session
        
        train_sessions = set(all_sessions[:n_train])
        test_sessions = set(all_sessions[n_train:])
        
        # Assign splits based on DATE (not folder)
        for idx in id_data.index:
            row = df.loc[idx]
            date_val = row['date']
            folder = row['outing']
            
            # Check if this row's date/folder is in train or test
            if pd.notna(date_val) and date_val in train_sessions:
                df.loc[idx, 'split'] = 'train'
                df.loc[idx, 'outing_split'] = 'train'
            elif pd.notna(date_val) and date_val in test_sessions:
                df.loc[idx, 'split'] = 'test'
                df.loc[idx, 'outing_split'] = 'test'
            elif folder in train_sessions:
                # Undated folder in train
                df.loc[idx, 'split'] = 'train'
                df.loc[idx, 'outing_split'] = 'train'
            elif folder in test_sessions:
                # Undated folder in test
                df.loc[idx, 'split'] = 'test'
                df.loc[idx, 'outing_split'] = 'test'
            else:
                # Fallback (should not happen)
                df.loc[idx, 'split'] = 'train'
                df.loc[idx, 'outing_split'] = 'train'
    
    # Process single-session identities (negative-only)
    for identity in single_session_ids:
        id_mask = df['identity'] == identity
        df.loc[id_mask, 'split'] = 'train'
        df.loc[id_mask, 'negative_only'] = True
        df.loc[id_mask, 'outing_split'] = 'train'
    
    return df


def prepare_temporal_split(
    dataset_root: str,
    output_path: Optional[str] = None,
    train_session_ratio: float = 0.8,
    min_sessions_for_eval: int = 2,
    seed: int = 42,
    force_regenerate: bool = False
) -> pd.DataFrame:
    """
    Prepare star dataset with temporal train/test splitting.
    
    IMPORTANT: Splitting is done by DATE, not by folder name. Multiple folders
    from the same date (e.g., different cameras: JU_PICS, WW_PICS) are treated
    as ONE temporal session and will all go to the same split.
    
    Args:
        dataset_root: Path to star_dataset folder
        output_path: Where to save metadata CSV (default: dataset_root/metadata_temporal.csv)
        train_session_ratio: Fraction of temporal sessions (unique dates) for training
        min_sessions_for_eval: Minimum unique dates to be evaluable (default 2)
        seed: Random seed
        force_regenerate: If True, regenerate even if metadata exists
    
    Returns:
        DataFrame with temporal splits assigned
    """
    dataset_root = Path(dataset_root)
    
    if output_path is None:
        output_path = dataset_root / 'metadata_temporal.csv'
    else:
        output_path = Path(output_path)
    
    # Check for existing metadata
    if output_path.exists() and not force_regenerate:
        print(f"Loading existing temporal metadata from {output_path}")
        df = pd.read_csv(output_path)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        _print_split_statistics(df)
        return df
    
    # Scan dataset
    print(f"Scanning dataset at {dataset_root}...")
    df = scan_star_dataset(dataset_root)
    
    # Assign temporal splits (by DATE, not folder)
    print(f"\nAssigning temporal splits (train_ratio={train_session_ratio}, by unique dates)...")
    df = assign_temporal_splits(
        df,
        train_session_ratio=train_session_ratio,
        min_sessions_for_eval=min_sessions_for_eval,
        seed=seed
    )
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved metadata to {output_path}")
    
    _print_split_statistics(df)
    
    return df


def load_metadata(path: str) -> pd.DataFrame:
    """Load metadata CSV with proper date parsing."""
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


def _print_split_statistics(df: pd.DataFrame):
    """Print detailed split statistics."""
    print("\n" + "=" * 60)
    print("TEMPORAL SPLIT STATISTICS")
    print("=" * 60)
    
    # Overall
    n_train = (df['split'] == 'train').sum()
    n_test = (df['split'] == 'test').sum()
    total = len(df)
    
    print(f"\nOverall:")
    print(f"  Total images: {total}")
    print(f"  Train images: {n_train} ({n_train/total:.1%})")
    print(f"  Test images:  {n_test} ({n_test/total:.1%})")
    
    # Identity breakdown
    train_ids = df[df['split'] == 'train']['identity'].nunique()
    test_ids = df[df['split'] == 'test']['identity'].nunique()
    
    print(f"\nIdentities:")
    print(f"  Total: {df['identity'].nunique()}")
    print(f"  In train: {train_ids}")
    print(f"  In test: {test_ids}")
    
    # Session analysis (unique dates vs folders)
    n_folders = df['outing'].nunique()
    n_dates = df['date'].dropna().nunique()
    print(f"\nTemporal sessions:")
    print(f"  Unique folders: {n_folders}")
    print(f"  Unique dates: {n_dates}")
    if n_folders > n_dates:
        print(f"  Note: {n_folders - n_dates} folders collapsed by same-date grouping")
    
    # Negative-only breakdown
    negative_only = df[df['negative_only'] == True]
    evaluable = df[df['negative_only'] == False]
    
    print(f"\nBy evaluation role:")
    print(f"  Evaluable (multi-date): {evaluable['identity'].nunique()} identities, {len(evaluable)} images")
    print(f"  Negative-only (single-date): {negative_only['identity'].nunique()} identities, {len(negative_only)} images")
    
    # Test set details
    test_df = df[df['split'] == 'test']
    if len(test_df) > 0:
        print(f"\nTest set (for evaluation):")
        print(f"  Images: {len(test_df)}")
        print(f"  Identities: {test_df['identity'].nunique()}")
        print(f"  All from held-out dates: Yes")
    
    # Temporal gap analysis
    if df['date'].notna().any():
        evaluable_ids = evaluable['identity'].unique()
        gaps = []
        
        for identity in evaluable_ids:
            id_data = df[df['identity'] == identity]
            train_dates = id_data[id_data['split'] == 'train']['date'].dropna()
            test_dates = id_data[id_data['split'] == 'test']['date'].dropna()
            
            if len(train_dates) > 0 and len(test_dates) > 0:
                gap = (test_dates.min() - train_dates.max()).days
                if gap > 0:
                    gaps.append(gap)
        
        if gaps:
            print(f"\nTemporal gap (test_date - train_date):")
            print(f"  Mean: {np.mean(gaps):.0f} days")
            print(f"  Min: {np.min(gaps)} days")
            print(f"  Max: {np.max(gaps)} days")
    
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prepare temporal splits for star dataset',
        epilog="""
Note: Splitting is done by DATE, not by folder name. Multiple folders from the
same date (e.g., different cameras: JU_PICS, WW_PICS) are treated as ONE temporal
session and will all go to the same split.
        """
    )
    parser.add_argument('--dataset-root', type=str, default='./star_dataset',
                        help='Path to star_dataset folder')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for metadata CSV')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Fraction of temporal sessions (unique dates) for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration even if metadata exists')
    
    args = parser.parse_args()
    
    prepare_temporal_split(
        dataset_root=args.dataset_root,
        output_path=args.output,
        train_session_ratio=args.train_ratio,
        seed=args.seed,
        force_regenerate=args.force
    )

