"""
Test script to verify the temporal splitting fix before implementing.

This script:
1. Shows the current (buggy) behavior
2. Demonstrates the fixed behavior
3. Compares results
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from collections import defaultdict
import re


# =============================================================================
# CURRENT (BUGGY) IMPLEMENTATION
# =============================================================================

def parse_date_from_folder_OLD(folder_name: str) -> Optional[datetime]:
    """Original parser - only handles M_D_YYYY format."""
    try:
        parts = folder_name.split('_')
        if len(parts) >= 3:
            month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
            return datetime(year, month, day)
    except (ValueError, IndexError):
        pass
    return None


def count_outings_OLD(df: pd.DataFrame, identity: str) -> int:
    """Original: counts folder names as outings."""
    return df[df['identity'] == identity]['outing'].nunique()


# =============================================================================
# FIXED IMPLEMENTATION
# =============================================================================

def parse_date_from_folder_NEW(folder_name: str) -> Optional[datetime]:
    """
    Enhanced date parser that handles multiple formats:
    
    Supported formats:
    - M_D_YYYY_* (e.g., '3_23_2024_dock_sighting')
    - MM_DD_YYYY_* (e.g., '04_06_2024__4-6-brown')
    - YYYY-MM-DD (ISO format in folder name)
    - Contains YYYY embedded (e.g., 'PWS_2023_JU' -> 2023, but day unknown)
    
    Returns None only if no date can be extracted.
    """
    folder_name = folder_name.strip()
    
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
    
    # Pattern 3: Folder name IS just a date (01_01_2021__*)
    # Already covered by Pattern 1
    
    # If no full date found, return None
    # We do NOT return partial dates (year-only) as that would incorrectly
    # collapse all observations from that year
    return None


def count_unique_dates(df: pd.DataFrame, identity: str) -> Tuple[int, int]:
    """
    Count unique temporal sessions for an identity.
    
    Returns:
        (n_dated_sessions, n_undated_folders)
        
    A "session" is a unique date. Multiple folders with the same date
    are collapsed into one session.
    """
    id_df = df[df['identity'] == identity]
    
    # Get unique dates (excluding NaT)
    dates = id_df['date'].dropna().unique()
    n_dated = len(dates)
    
    # Count folders without dates
    undated_folders = id_df[id_df['date'].isna()]['outing'].unique()
    n_undated = len(undated_folders)
    
    return n_dated, n_undated


def assign_temporal_splits_NEW(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    min_sessions_for_eval: int = 2,
    seed: int = 42
) -> pd.DataFrame:
    """
    Fixed temporal splitting that groups by DATE, not folder name.
    
    Key changes from original:
    1. Multiple folders with the same date are treated as ONE temporal session
    2. Splitting is done by date, not by folder
    3. All images from the same date go to the same split
    """
    np.random.seed(seed)
    df = df.copy()
    
    df['split'] = ''
    df['negative_only'] = False
    df['outing_split'] = ''
    
    # Process each identity
    for identity in df['identity'].unique():
        id_mask = df['identity'] == identity
        id_df = df[id_mask]
        
        # Count unique temporal sessions (by DATE, not folder)
        n_dated, n_undated = count_unique_dates(df, identity)
        
        # Total sessions: dated sessions + undated folders (each undated folder = 1 session)
        # This is conservative: if we can't parse the date, we assume each folder is separate
        total_sessions = n_dated + n_undated
        
        if total_sessions < min_sessions_for_eval:
            # Single-session identity -> train only, negative_only
            df.loc[id_mask, 'split'] = 'train'
            df.loc[id_mask, 'negative_only'] = True
            df.loc[id_mask, 'outing_split'] = 'train'
            continue
        
        # Multi-session identity -> temporal split BY DATE
        # Get unique dates sorted chronologically
        dated_sessions = sorted([d for d in id_df['date'].dropna().unique()])
        undated_folders = list(id_df[id_df['date'].isna()]['outing'].unique())
        
        # Combine: dated sessions first (sorted), then undated folders
        # Undated folders go at the end (most conservative: they become test)
        all_sessions = dated_sessions + undated_folders
        
        # Calculate split point
        n_train = max(1, int(len(all_sessions) * train_ratio))
        if n_train >= len(all_sessions):
            n_train = len(all_sessions) - 1  # Ensure at least 1 test session
        
        train_sessions = set(all_sessions[:n_train])
        test_sessions = set(all_sessions[n_train:])
        
        # Assign splits based on DATE (not folder)
        for idx in id_df.index:
            row = df.loc[idx]
            date = row['date']
            folder = row['outing']
            
            # Check if this row's date/folder is in train or test
            if pd.notna(date) and date in train_sessions:
                df.loc[idx, 'split'] = 'train'
                df.loc[idx, 'outing_split'] = 'train'
            elif pd.notna(date) and date in test_sessions:
                df.loc[idx, 'split'] = 'test'
                df.loc[idx, 'outing_split'] = 'test'
            elif folder in train_sessions:
                df.loc[idx, 'split'] = 'train'
                df.loc[idx, 'outing_split'] = 'train'
            elif folder in test_sessions:
                df.loc[idx, 'split'] = 'test'
                df.loc[idx, 'outing_split'] = 'test'
            else:
                # Fallback: if somehow not matched, put in train
                df.loc[idx, 'split'] = 'train'
                df.loc[idx, 'outing_split'] = 'train'
    
    return df


# =============================================================================
# TESTING
# =============================================================================

def test_date_parser():
    """Test the date parser with various formats."""
    test_cases = [
        # (folder_name, expected_date_or_None)
        ('3_23_2024_dock_sighting', datetime(2024, 3, 23)),
        ('04_06_2024__4-6-brown', datetime(2024, 4, 6)),
        ('6_16_2023_JU_PICS', datetime(2023, 6, 16)),
        ('6_16_2023_WW_PICS', datetime(2023, 6, 16)),
        ('09_11_2021', datetime(2021, 9, 11)),
        ('01_01_2022__2022-brood-photos', datetime(2022, 1, 1)),
        ('PWS_2023_JU', None),  # Can't determine day
        ('PWS_2023_WW', None),
        ('PWS_2023_BW', None),
        ('8_9_2023', datetime(2023, 8, 9)),
        ('12_7_2023', datetime(2023, 12, 7)),
    ]
    
    print("=" * 70)
    print("DATE PARSER TEST")
    print("=" * 70)
    
    all_passed = True
    for folder, expected in test_cases:
        result_old = parse_date_from_folder_OLD(folder)
        result_new = parse_date_from_folder_NEW(folder)
        
        status = "✓" if result_new == expected else "✗"
        if result_new != expected:
            all_passed = False
        
        print(f"{status} '{folder}'")
        print(f"    OLD: {result_old}")
        print(f"    NEW: {result_new}")
        print(f"    Expected: {expected}")
    
    print(f"\nAll tests passed: {all_passed}")
    return all_passed


def test_splitting_logic(dataset_root: str):
    """Test the splitting logic on actual data."""
    from temporal_reid.data.prepare import scan_star_dataset
    
    print("\n" + "=" * 70)
    print(f"SPLITTING LOGIC TEST: {dataset_root}")
    print("=" * 70)
    
    # Scan dataset
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return
    
    df = scan_star_dataset(dataset_path)
    
    # Re-parse dates with NEW parser
    print("\nRe-parsing dates with fixed parser...")
    df['date_old'] = df['outing'].apply(parse_date_from_folder_OLD)
    df['date'] = df['outing'].apply(parse_date_from_folder_NEW)
    
    # Compare date parsing
    improved = (df['date'].notna() & df['date_old'].isna()).sum()
    print(f"  Dates newly parsed: {improved}")
    
    # Apply NEW splitting
    print("\nApplying fixed temporal splitting...")
    df_new = assign_temporal_splits_NEW(df.copy(), train_ratio=0.8)
    
    # Compare with OLD logic (simulate by loading existing metadata)
    old_metadata = dataset_path / 'metadata_temporal.csv'
    if old_metadata.exists():
        df_old = pd.read_csv(old_metadata)
        
        print("\n" + "-" * 70)
        print("COMPARISON: OLD vs NEW splitting")
        print("-" * 70)
        
        # Per-dataset comparison
        for ds in sorted(df['dataset'].unique()):
            ds_old = df_old[df_old['dataset'] == ds]
            ds_new = df_new[df_new['dataset'] == ds]
            
            old_train_pct = (ds_old['split'] == 'train').mean() * 100 if len(ds_old) > 0 else 0
            new_train_pct = (ds_new['split'] == 'train').mean() * 100 if len(ds_new) > 0 else 0
            
            # Count unique dates vs folders
            n_folders = ds_new['outing'].nunique()
            n_dates = ds_new['date'].dropna().nunique()
            
            change = "→" if abs(new_train_pct - old_train_pct) < 1 else "⚠️"
            if new_train_pct > old_train_pct + 5:
                change = "✓ FIXED"
            
            print(f"\n{ds}:")
            print(f"  Folders: {n_folders}, Unique dates: {n_dates}")
            print(f"  OLD train%: {old_train_pct:.1f}%")
            print(f"  NEW train%: {new_train_pct:.1f}% {change}")
    
    # Summary statistics
    print("\n" + "-" * 70)
    print("NEW SPLIT SUMMARY")
    print("-" * 70)
    
    n_train = (df_new['split'] == 'train').sum()
    n_test = (df_new['split'] == 'test').sum()
    total = len(df_new)
    
    print(f"Total images: {total}")
    print(f"Train: {n_train} ({n_train/total*100:.1f}%)")
    print(f"Test: {n_test} ({n_test/total*100:.1f}%)")
    
    neg_only = df_new[df_new['negative_only'] == True]
    print(f"Negative-only identities: {neg_only['identity'].nunique()}")
    print(f"Evaluable identities: {df_new['identity'].nunique() - neg_only['identity'].nunique()}")
    
    return df_new


def main():
    # Test date parser
    test_date_parser()
    
    # Test on both datasets
    for dataset in ['star_dataset', 'star_dataset_resized']:
        dataset_path = Path(dataset)
        if dataset_path.exists():
            test_splitting_logic(dataset)
        else:
            print(f"\nSkipping {dataset} (not found)")


if __name__ == '__main__':
    main()

