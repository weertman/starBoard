# src/morphometric/analysis_adapter.py
"""
Analysis Adapter for starBoard Morphometric Integration.

Wraps morphometric analysis functionality from the starMorphometricTool.
"""
from __future__ import annotations

import logging
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import cv2

from . import _ensure_morphometric_path, get_measurements_root

logger = logging.getLogger("starBoard.morphometric.analysis")


class AnalysisAdapter:
    """
    Adapter for morphometric analysis and data management.
    
    Provides contour analysis, arm detection, and data storage.
    """
    
    def __init__(self):
        """Initialize the analysis adapter."""
        _ensure_morphometric_path()
        
        # Analysis state
        self._current_morphometrics: Optional[Dict[str, Any]] = None
        self._arm_data: List[List[float]] = []
        self._center: Optional[np.ndarray] = None
        self._contour_points: Optional[np.ndarray] = None
        self._angles_sorted: Optional[np.ndarray] = None
        self._distances_smoothed: Optional[np.ndarray] = None
        self._peaks: Optional[np.ndarray] = None
        self._sorted_indices: Optional[np.ndarray] = None
        self._shifted_contour: Optional[np.ndarray] = None
        self._ellipse_data: Optional[Tuple] = None
    
    def analyze_contour(
        self,
        corrected_mask: np.ndarray,
        mm_per_pixel: float,
        smoothing_factor: int = 5,
        prominence_factor: float = 0.05,
        distance_factor: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Perform morphometric analysis on corrected mask.
        
        Args:
            corrected_mask: Binary mask from corrected detection
            mm_per_pixel: Calibration scale factor
            smoothing_factor: Smoothing window for distance profile (1-15)
            prominence_factor: Peak prominence threshold (0.01-1.0)
            distance_factor: Minimum distance between peaks (0-15)
        
        Returns:
            Dictionary with morphometric data or None
        """
        try:
            from morphometrics.analysis import find_arm_tips
            from utils.image_processing import smooth_closed_contour
            
            # Find contours in mask
            contours, _ = cv2.findContours(corrected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                logger.warning("No contours found in mask")
                return None
            
            # Use largest contour
            contour = max(contours, key=cv2.contourArea)
            
            # Calculate area
            area_pixels = cv2.contourArea(contour)
            area_mm2 = area_pixels * (mm_per_pixel ** 2)
            
            # Find center
            M = cv2.moments(contour)
            if M['m00'] == 0:
                logger.warning("Cannot compute center (zero moment)")
                return None
            
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            center = np.array([cx, cy])
            self._center = center
            
            # Smooth contour
            raw_points = contour.reshape(-1, 2)
            smoothed_points = smooth_closed_contour(raw_points, iterations=2)
            self._contour_points = smoothed_points
            
            # Find arm tips
            arm_tips, angles_sorted, distances_smoothed, peaks, sorted_indices, shifted_contour = find_arm_tips(
                smoothed_points, center, smoothing_factor, prominence_factor, distance_factor
            )
            
            # Store analysis state
            self._angles_sorted = angles_sorted
            self._distances_smoothed = distances_smoothed
            self._peaks = peaks
            self._sorted_indices = sorted_indices
            self._shifted_contour = shifted_contour
            
            # Build arm data
            num_arms = len(arm_tips)
            arm_data = []
            for i, tip in enumerate(arm_tips):
                x_vec = tip[0] - center[0]
                y_vec = tip[1] - center[1]
                length_px = np.hypot(x_vec, y_vec)
                length_mm = length_px * mm_per_pixel
                arm_data.append([i + 1, float(x_vec), float(y_vec), float(length_mm)])
            self._arm_data = arm_data
            
            # Fit ellipse if possible
            major_axis_mm = None
            minor_axis_mm = None
            self._ellipse_data = None
            
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (x0, y0), (axis_length1, axis_length2), angle = ellipse
                    
                    if axis_length1 >= axis_length2:
                        major_axis_length = axis_length1
                        minor_axis_length = axis_length2
                    else:
                        major_axis_length = axis_length2
                        minor_axis_length = axis_length1
                        angle += 90
                    
                    major_axis_mm = major_axis_length * mm_per_pixel
                    minor_axis_mm = minor_axis_length * mm_per_pixel
                    self._ellipse_data = (x0, y0, major_axis_length, minor_axis_length, angle)
                except Exception as e:
                    logger.warning("Failed to fit ellipse: %s", e)
            
            # Build morphometrics data
            self._current_morphometrics = {
                'area_mm2': float(area_mm2),
                'num_arms': num_arms,
                'arm_data': arm_data,
                'major_axis_mm': float(major_axis_mm) if major_axis_mm else None,
                'minor_axis_mm': float(minor_axis_mm) if minor_axis_mm else None,
                'contour_coordinates': smoothed_points.tolist(),
                'mm_per_pixel': float(mm_per_pixel),
            }
            
            logger.info("Analysis complete: %d arms, area=%.1f mm²", num_arms, area_mm2)
            return self._current_morphometrics
            
        except Exception as e:
            logger.exception("Error in contour analysis: %s", e)
            return None
    
    def update_peaks(self, new_peaks: np.ndarray, mm_per_pixel: float) -> None:
        """
        Update arm data from interactive peak editing.
        
        Args:
            new_peaks: Array of new peak indices
            mm_per_pixel: Calibration scale factor
        """
        if self._center is None or self._shifted_contour is None or self._sorted_indices is None:
            return
        
        # Get sorted contour in global coords
        sorted_contour_global = self._shifted_contour[self._sorted_indices] + self._center
        
        # Collect peak coordinates
        peak_coords = []
        for idx in new_peaks.astype(int):
            if 0 <= idx < len(sorted_contour_global):
                peak_coords.append(sorted_contour_global[idx])
        
        if not peak_coords:
            self._arm_data = []
            return
        
        peak_coords = np.array(peak_coords)
        
        # Compute angles and sort
        center_x, center_y = self._center
        shifted = peak_coords - [center_x, center_y]
        angles = np.arctan2(shifted[:, 1], shifted[:, 0])
        sort_idx = np.argsort(angles)
        peak_coords_sorted = peak_coords[sort_idx]
        
        # Rebuild arm_data
        new_arm_data = []
        for i, pt in enumerate(peak_coords_sorted):
            x_vec = pt[0] - center_x
            y_vec = pt[1] - center_y
            length_px = np.hypot(x_vec, y_vec)
            length_mm = length_px * mm_per_pixel
            new_arm_data.append([i + 1, float(x_vec), float(y_vec), float(length_mm)])
        
        self._arm_data = new_arm_data
        if self._current_morphometrics:
            self._current_morphometrics['arm_data'] = new_arm_data
            self._current_morphometrics['num_arms'] = len(new_arm_data)
    
    def rotate_arm_numbering(self, rotation: int) -> List[List[float]]:
        """
        Rotate arm numbering.
        
        Args:
            rotation: Number of positions to rotate
        
        Returns:
            Reordered arm data
        """
        if not self._arm_data:
            return []
        
        num_arms = len(self._arm_data)
        rotation = rotation % num_arms if num_arms > 0 else 0
        
        reordered = self._arm_data[rotation:] + self._arm_data[:rotation]
        
        # Renumber
        for i, arm in enumerate(reordered):
            arm[0] = i + 1
        
        return reordered
    
    # =========================================================================
    # Data Properties
    # =========================================================================
    
    @property
    def current_morphometrics(self) -> Optional[Dict[str, Any]]:
        """Get current morphometric data."""
        return self._current_morphometrics
    
    @property
    def arm_data(self) -> List[List[float]]:
        """Get current arm data."""
        return self._arm_data
    
    @property
    def center(self) -> Optional[np.ndarray]:
        """Get center point."""
        return self._center
    
    @property
    def ellipse_data(self) -> Optional[Tuple]:
        """Get ellipse fit data."""
        return self._ellipse_data
    
    @property
    def angles_sorted(self) -> Optional[np.ndarray]:
        """Get sorted angles."""
        return self._angles_sorted
    
    @property
    def distances_smoothed(self) -> Optional[np.ndarray]:
        """Get smoothed distance profile."""
        return self._distances_smoothed
    
    @property
    def peaks(self) -> Optional[np.ndarray]:
        """Get peak indices."""
        return self._peaks
    
    @property
    def sorted_contour_points(self) -> Optional[np.ndarray]:
        """Get sorted contour points in global coords."""
        if self._shifted_contour is None or self._sorted_indices is None or self._center is None:
            return None
        return self._shifted_contour[self._sorted_indices] + self._center
    
    # =========================================================================
    # Derived Measurements
    # =========================================================================
    
    def get_arm_lengths(self) -> List[float]:
        """Get list of arm lengths in mm."""
        return [arm[3] for arm in self._arm_data]
    
    def get_mean_arm_length(self) -> Optional[float]:
        """Get mean arm length in mm."""
        lengths = self.get_arm_lengths()
        if not lengths:
            return None
        return sum(lengths) / len(lengths)
    
    def get_max_arm_length(self) -> Optional[float]:
        """Get maximum arm length in mm."""
        lengths = self.get_arm_lengths()
        if not lengths:
            return None
        return max(lengths)
    
    def get_tip_to_tip_diameter(self) -> Optional[float]:
        """
        Calculate maximum tip-to-tip diameter.
        
        Returns:
            Maximum diameter between opposing arm tips in mm.
        """
        if len(self._arm_data) < 2 or self._center is None:
            return None
        
        import math
        
        # Calculate angle of each arm
        arm_angles = []
        for arm in self._arm_data:
            arm_num, x_vec, y_vec, length_mm = arm
            angle = math.atan2(y_vec, x_vec)
            arm_angles.append((arm_num, angle, x_vec, y_vec, length_mm))
        
        max_diameter = 0.0
        
        # For each arm, find most opposite arm and calculate diameter
        for i, (_, angle1, x1, y1, _) in enumerate(arm_angles):
            opposite_angle = (angle1 + math.pi) % (2 * math.pi)
            
            min_diff = float('inf')
            best_j = -1
            
            for j, (_, angle2, _, _, _) in enumerate(arm_angles):
                if i == j:
                    continue
                diff = abs((angle2 - opposite_angle + math.pi) % (2 * math.pi) - math.pi)
                if diff < min_diff:
                    min_diff = diff
                    best_j = j
            
            if best_j >= 0:
                _, _, x2, y2, _ = arm_angles[best_j]
                # Tip-to-tip distance through center
                diameter = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                max_diameter = max(max_diameter, diameter)
        
        return max_diameter if max_diameter > 0 else None
    
    # =========================================================================
    # Data Saving
    # =========================================================================
    
    def save_measurement(
        self,
        identity_type: str,
        identity_id: str,
        location: str,
        user_initials: str,
        user_notes: str,
        raw_frame: np.ndarray,
        corrected_detection: Dict[str, Any],
        arm_rotation: int = 0,
        volume_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """
        Save all measurement data to mFolder structure.
        
        Args:
            identity_type: "gallery" or "query"
            identity_id: Individual identifier
            location: Location string
            user_initials: User initials (3 chars)
            user_notes: Additional notes
            raw_frame: Original camera frame
            corrected_detection: Corrected detection data from DetectionAdapter
            arm_rotation: Arm rotation offset
            volume_data: Optional volume estimation data
        
        Returns:
            Path to created mFolder or None on failure
        """
        if self._current_morphometrics is None:
            logger.error("No morphometric data to save")
            return None
        
        try:
            root_dir = get_measurements_root()
            
            # Build directory structure
            id_folder = root_dir / identity_type / identity_id
            id_folder.mkdir(parents=True, exist_ok=True)
            
            # Date folder
            date_str = datetime.now().strftime("%m_%d_%Y")
            date_folder = id_folder / date_str
            date_folder.mkdir(exist_ok=True)
            
            # Find next mFolder number
            existing = [d.name for d in date_folder.iterdir() if d.is_dir() and d.name.startswith("mFolder_")]
            numbers = [int(n.replace("mFolder_", "")) for n in existing if n.replace("mFolder_", "").isdigit()]
            next_num = max(numbers) + 1 if numbers else 1
            
            mfolder = date_folder / f"mFolder_{next_num}"
            mfolder.mkdir()
            
            # Save raw frame
            raw_path = mfolder / "raw_frame.png"
            cv2.imwrite(str(raw_path), raw_frame)
            
            # Save corrected images
            if corrected_detection.get('corrected_mask') is not None:
                mask = corrected_detection['corrected_mask']
                cv2.imwrite(str(mfolder / "corrected_mask.png"), mask * 255)
            
            if corrected_detection.get('corrected_object') is not None:
                cv2.imwrite(str(mfolder / "corrected_object.png"), corrected_detection['corrected_object'])
            
            # Save combined checkerboard + object image
            if corrected_detection.get('corrected_frame') is not None and corrected_detection.get('corrected_object') is not None:
                try:
                    corrected_frame = corrected_detection['corrected_frame']
                    corrected_object = corrected_detection['corrected_object']
                    corrected_mask = corrected_detection['corrected_mask']
                    
                    # Resize if needed
                    if corrected_frame.shape != corrected_object.shape:
                        co_resized = cv2.resize(corrected_object, (corrected_frame.shape[1], corrected_frame.shape[0]))
                        cm_resized = cv2.resize(corrected_mask, (corrected_frame.shape[1], corrected_frame.shape[0]))
                    else:
                        co_resized = corrected_object
                        cm_resized = corrected_mask
                    
                    alpha_mask = cm_resized.astype(float) / 255.0
                    alpha_mask = np.stack([alpha_mask] * 3, axis=2)
                    combined = ((1.0 - alpha_mask) * corrected_frame + alpha_mask * co_resized).astype(np.uint8)
                    cv2.imwrite(str(mfolder / "checkerboard_with_object.png"), combined)
                except Exception as e:
                    logger.warning("Failed to save combined image: %s", e)
            
            # Save corrected detection JSON
            detection_info = {
                'class_id': corrected_detection.get('class_id'),
                'class_name': 'Pycnopodia_helianthoides',
                'real_world_coordinate': corrected_detection.get('real_world_coordinate'),
                'homography_matrix': corrected_detection.get('homography_matrix'),
                'corrected_polygon': corrected_detection.get('corrected_polygon'),
                'mm_per_pixel': corrected_detection.get('mm_per_pixel'),
                'location': location,
                'identity_type': identity_type,
                'identity_id': identity_id,
                'mask_path': str(mfolder / "corrected_mask.png"),
                'object_path': str(mfolder / "corrected_object.png"),
                'raw_frame_path': str(raw_path),
            }
            
            with open(mfolder / "corrected_detection.json", 'w') as f:
                json.dump(self._convert_numpy_types(detection_info), f, indent=4)
            
            # Apply rotation to arm data
            rotated_arm_data = self.rotate_arm_numbering(arm_rotation)
            
            # Build morphometrics data
            morph_data = dict(self._current_morphometrics)
            morph_data['arm_data'] = rotated_arm_data
            morph_data['arm_rotation'] = arm_rotation
            morph_data['user_initials'] = user_initials.upper()
            morph_data['user_notes'] = user_notes
            morph_data['location'] = location
            morph_data['identity_type'] = identity_type
            morph_data['identity_id'] = identity_id
            
            if volume_data:
                morph_data['volume_estimation'] = volume_data
            
            with open(mfolder / "morphometrics.json", 'w') as f:
                json.dump(self._convert_numpy_types(morph_data), f, indent=4)
            
            logger.info("Saved measurement to %s", mfolder)
            return mfolder
            
        except Exception as e:
            logger.exception("Error saving measurement: %s", e)
            return None
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        else:
            return obj
    
    @staticmethod
    def load_morphometrics(mfolder_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load morphometrics data from an mFolder.
        
        Args:
            mfolder_path: Path to mFolder
        
        Returns:
            Morphometrics data dictionary or None
        """
        morph_path = mfolder_path / "morphometrics.json"
        if not morph_path.exists():
            return None
        
        try:
            with open(morph_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error loading morphometrics: %s", e)
            return None


