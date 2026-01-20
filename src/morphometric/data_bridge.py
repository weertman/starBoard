# src/morphometric/data_bridge.py
"""
Data Bridge for starBoard Morphometric Integration.

Converts morphometric tool data to starBoard metadata fields.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger("starBoard.morphometric.data_bridge")


def extract_starboard_fields(morphometrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract starBoard metadata fields from morphometrics data.
    
    Converts morphometrics.json data to the new morph_* fields defined
    in the starBoard annotation schema.
    
    Args:
        morphometrics: Dictionary loaded from morphometrics.json
    
    Returns:
        Dictionary of field_name -> string_value for starBoard CSV
    """
    fields = {}
    
    # morph_num_arms - Number of detected arms
    num_arms = morphometrics.get('num_arms')
    if num_arms is not None:
        fields['morph_num_arms'] = str(int(num_arms))
    
    # morph_area_mm2 - Surface area
    area = morphometrics.get('area_mm2')
    if area is not None:
        fields['morph_area_mm2'] = f"{float(area):.2f}"
    
    # morph_major_axis_mm - Ellipse major axis
    major = morphometrics.get('major_axis_mm')
    if major is not None:
        fields['morph_major_axis_mm'] = f"{float(major):.2f}"
    
    # morph_minor_axis_mm - Ellipse minor axis
    minor = morphometrics.get('minor_axis_mm')
    if minor is not None:
        fields['morph_minor_axis_mm'] = f"{float(minor):.2f}"
    
    # Arm length calculations from arm_data
    arm_data = morphometrics.get('arm_data', [])
    if arm_data:
        arm_lengths = [arm[3] for arm in arm_data]  # arm[3] is length_mm
        
        # morph_mean_arm_length_mm
        mean_length = sum(arm_lengths) / len(arm_lengths)
        fields['morph_mean_arm_length_mm'] = f"{mean_length:.2f}"
        
        # morph_max_arm_length_mm
        max_length = max(arm_lengths)
        fields['morph_max_arm_length_mm'] = f"{max_length:.2f}"
        
        # morph_tip_to_tip_mm - Calculate max diameter between opposing arms
        tip_to_tip = calculate_tip_to_tip_diameter(arm_data)
        if tip_to_tip is not None:
            fields['morph_tip_to_tip_mm'] = f"{tip_to_tip:.2f}"
    
    # morph_volume_mm3 - From volume estimation if available
    volume_est = morphometrics.get('volume_estimation', {})
    if volume_est:
        volume = volume_est.get('volume_mm3')
        if volume is not None:
            fields['morph_volume_mm3'] = f"{float(volume):.2f}"
    
    return fields


def calculate_tip_to_tip_diameter(arm_data: List[List[float]]) -> Optional[float]:
    """
    Calculate maximum tip-to-tip diameter from arm data.
    
    For each arm, finds the most directly opposite arm (closest to 180° away)
    and calculates the distance between their tips.
    
    Args:
        arm_data: List of [arm_number, x_vec, y_vec, length_mm]
    
    Returns:
        Maximum diameter in mm, or None if insufficient data
    """
    if len(arm_data) < 2:
        return None
    
    # Calculate angle of each arm vector
    arm_angles = []
    for arm in arm_data:
        arm_number, x_vec, y_vec, length_mm = arm
        angle = math.atan2(y_vec, x_vec)
        arm_angles.append((arm_number, angle, x_vec, y_vec, length_mm))
    
    max_diameter = 0.0
    
    # For each arm, find the most opposite arm
    for i, (_, arm1_angle, x1, y1, _) in enumerate(arm_angles):
        opposite_angle = (arm1_angle + math.pi) % (2 * math.pi)
        
        min_diff = float('inf')
        opposite_idx = -1
        
        for j, (_, arm2_angle, _, _, _) in enumerate(arm_angles):
            if j == i:
                continue
            
            # Angular difference accounting for circular nature
            diff = abs((arm2_angle - opposite_angle + math.pi) % (2 * math.pi) - math.pi)
            if diff < min_diff:
                min_diff = diff
                opposite_idx = j
        
        if opposite_idx >= 0:
            _, _, x2, y2, _ = arm_angles[opposite_idx]
            # Tip-to-tip distance (vectors are from center)
            diameter = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            max_diameter = max(max_diameter, diameter)
    
    return max_diameter if max_diameter > 0 else None


def load_morphometrics_from_mfolder(mfolder_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load morphometrics.json from an mFolder.
    
    Args:
        mfolder_path: Path to mFolder directory
    
    Returns:
        Parsed morphometrics data or None
    """
    morph_file = mfolder_path / "morphometrics.json"
    if not morph_file.exists():
        logger.warning("morphometrics.json not found in %s", mfolder_path)
        return None
    
    try:
        with open(morph_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error loading morphometrics from %s: %s", morph_file, e)
        return None


def get_raw_frame_path(mfolder_path: Path) -> Optional[Path]:
    """
    Get the raw_frame.png path from an mFolder.
    
    Args:
        mfolder_path: Path to mFolder directory
    
    Returns:
        Path to raw_frame.png or None
    """
    raw_path = mfolder_path / "raw_frame.png"
    if raw_path.exists():
        return raw_path
    return None


def create_starboard_row(
    morphometrics: Dict[str, Any],
    mfolder_path: Path,
    additional_fields: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Create a complete starBoard metadata row from morphometric data.
    
    Args:
        morphometrics: Morphometrics data dictionary
        mfolder_path: Path to source mFolder (for traceability)
        additional_fields: Additional fields to include in the row
    
    Returns:
        Dictionary of field values for starBoard CSV
    """
    # Start with morphometric fields
    row = extract_starboard_fields(morphometrics)
    
    # Add source folder reference
    row['morph_source_folder'] = str(mfolder_path)
    
    # Add location if present
    location = morphometrics.get('location')
    if location:
        row['location'] = location
    
    # Merge additional fields (without overwriting morph_ fields)
    if additional_fields:
        for key, value in additional_fields.items():
            if key not in row or not row[key]:
                row[key] = value
    
    return row


def list_mfolders(
    measurements_root: Path,
    identity_type: Optional[str] = None,
    identity_id: Optional[str] = None
) -> List[Path]:
    """
    List all mFolders, optionally filtered by identity.
    
    Args:
        measurements_root: Root measurements directory
        identity_type: Optional filter for "gallery" or "query"
        identity_id: Optional filter for specific identity
    
    Returns:
        List of mFolder paths sorted by date (newest first)
    """
    mfolders = []
    
    if identity_type:
        type_dirs = [measurements_root / identity_type]
    else:
        type_dirs = [measurements_root / "gallery", measurements_root / "query"]
    
    for type_dir in type_dirs:
        if not type_dir.exists():
            continue
        
        if identity_id:
            id_dirs = [type_dir / identity_id]
        else:
            id_dirs = [d for d in type_dir.iterdir() if d.is_dir()]
        
        for id_dir in id_dirs:
            if not id_dir.exists():
                continue
            
            # Iterate through date folders
            for date_dir in id_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                
                # Find mFolders
                for item in date_dir.iterdir():
                    if item.is_dir() and item.name.startswith("mFolder_"):
                        mfolders.append(item)
    
    # Sort by modification time (newest first)
    mfolders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return mfolders


def get_mfolder_datetime(mfolder_path: Path) -> Optional[str]:
    """
    Extract datetime string from mFolder path.
    
    The path structure is: .../identity_id/mm_dd_yyyy/mFolder_N/
    
    Args:
        mfolder_path: Path to mFolder
    
    Returns:
        Date string in mm_dd_yyyy format or None
    """
    try:
        date_folder = mfolder_path.parent
        return date_folder.name
    except Exception:
        return None


def convert_date_to_encounter_name(date_str: str) -> str:
    """
    Convert morphometric tool date format to starBoard encounter name.
    
    Args:
        date_str: Date in mm_dd_yyyy format (morphometric tool)
    
    Returns:
        Date in mm_dd_yy format (starBoard encounter name)
    """
    try:
        parts = date_str.split('_')
        if len(parts) == 3:
            month, day, year = parts
            # Convert 4-digit year to 2-digit
            if len(year) == 4:
                year = year[2:]
            return f"{month}_{day}_{year}"
    except Exception:
        pass
    return date_str


