"""
Camera Calibration Module for Metric Depth Estimation.

Accumulates checkerboard detections over time to estimate camera intrinsics,
enabling accurate metric depth/volume measurements.
"""

import json
import logging
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import cv2

logger = logging.getLogger("starBoard.calibration")

# Default paths relative to starMorphometricTool root
DEFAULT_CALIBRATION_DIR = "calibration"
CALIBRATION_FILE = "camera_calibration.json"
DETECTION_HISTORY_FILE = "detection_history.json"

# Reliability thresholds
MIN_DETECTIONS = 10
MAX_REPROJECTION_ERROR_1080P = 1.0  # pixels
MAX_REPROJECTION_ERROR_720P = 0.7   # pixels
MIN_POSE_DIVERSITY = 0.1


def _get_calibration_dir() -> Path:
    """Get the calibration directory path."""
    # Find starMorphometricTool root
    current = Path(__file__).parent
    while current.name != "starMorphometricTool" and current.parent != current:
        current = current.parent
    
    if current.name == "starMorphometricTool":
        calib_dir = current / DEFAULT_CALIBRATION_DIR
    else:
        # Fallback to current directory
        calib_dir = Path.cwd() / DEFAULT_CALIBRATION_DIR
    
    calib_dir.mkdir(parents=True, exist_ok=True)
    return calib_dir


class CameraCalibrationManager:
    """
    Manages camera intrinsics calibration from accumulated checkerboard detections.
    
    Key features:
    - Accumulates detections per camera (identified by fingerprint + resolution)
    - Estimates intrinsics using cv2.calibrateCamera when enough detections available
    - Gates reliability based on detection count, reprojection error, and pose diversity
    - Does NOT use square_size_mm for intrinsics (prevents user typos from corrupting K)
    """
    
    def __init__(self, calibration_dir: Optional[Path] = None):
        """
        Initialize the calibration manager.
        
        Args:
            calibration_dir: Directory for storing calibration data.
                           Defaults to starMorphometricTool/calibration/
        """
        self._calibration_dir = calibration_dir or _get_calibration_dir()
        self._calibration_file = self._calibration_dir / CALIBRATION_FILE
        self._detection_file = self._calibration_dir / DETECTION_HISTORY_FILE
        
        # In-memory caches
        self._calibrations: Dict[str, Dict[str, Any]] = {}
        self._detections: List[Dict[str, Any]] = []
        
        # Load existing data
        self._load_calibrations()
        self._load_detections()
    
    # =========================================================================
    # Camera ID Generation
    # =========================================================================
    
    @staticmethod
    def get_camera_id(config: Dict[str, Any], device_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a stable camera identifier from config and device info.
        
        Args:
            config: Camera configuration with 'width', 'height', etc.
            device_info: Optional device info with 'name', 'vendor_id', etc.
        
        Returns:
            Stable camera ID string like "camera_abc123_1920x1080"
        """
        # Get resolution
        width = config.get('width', 0)
        height = config.get('height', 0)
        
        # Build a stable name from device info
        if device_info:
            name = device_info.get('name', '')
            vendor = device_info.get('vendor_id', '')
            product = device_info.get('product_id', '')
            device_str = f"{name}_{vendor}_{product}"
        else:
            # Fallback to config-based identifier
            provider = config.get('provider', 'unknown')
            device_index = config.get('device_index', 0)
            backend = config.get('backend', '')
            device_str = f"{provider}_{device_index}_{backend}"
        
        # Create a short hash for readability
        hash_input = device_str.encode('utf-8')
        short_hash = hashlib.md5(hash_input).hexdigest()[:8]
        
        return f"camera_{short_hash}_{width}x{height}"
    
    # =========================================================================
    # Detection Management
    # =========================================================================
    
    def add_detection(
        self,
        camera_id: str,
        image_size: Tuple[int, int],
        board_dims: Tuple[int, int],
        image_points: np.ndarray,
        homography: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Add a checkerboard detection to the history.
        
        Args:
            camera_id: Camera identifier from get_camera_id()
            image_size: (width, height) of the image
            board_dims: (cols-1, rows-1) internal corners
            image_points: Nx2 array of refined corner positions in pixels
            homography: Optional homography matrix for diversity scoring
        
        Returns:
            Dict with detection info and current calibration status
        """
        # Compute pose diversity score from homography
        diversity_score = self._compute_diversity_score(homography) if homography is not None else 0.0
        
        detection = {
            'camera_id': camera_id,
            'image_size': list(image_size),
            'board_dims': list(board_dims),
            'image_points': image_points.reshape(-1, 2).tolist(),
            'timestamp': datetime.now().isoformat(),
            'homography_diversity_score': float(diversity_score)
        }
        
        self._detections.append(detection)
        self._save_detections()
        
        # Get detection count for this camera
        camera_detections = self.get_detections_for_camera(camera_id)
        count = len(camera_detections)
        
        logger.info(f"Added detection for {camera_id} ({count} total)")
        
        # Check if we should update calibration
        result = {
            'detection_count': count,
            'is_reliable': False,
            'should_update': count >= MIN_DETECTIONS and count % 5 == 0  # Update every 5 detections
        }
        
        if result['should_update']:
            calib_result = self.update_calibration(camera_id)
            result['is_reliable'] = calib_result.get('is_reliable', False)
            result['reprojection_error'] = calib_result.get('reprojection_error')
        
        return result
    
    def get_detections_for_camera(self, camera_id: str) -> List[Dict[str, Any]]:
        """Get all detections for a specific camera."""
        return [d for d in self._detections if d['camera_id'] == camera_id]
    
    def _compute_diversity_score(self, H: np.ndarray) -> float:
        """
        Compute a pose diversity score from homography.
        
        Higher scores indicate more perspective distortion (better for calibration).
        """
        if H is None:
            return 0.0
        
        try:
            # Normalize homography
            H_norm = H / H[2, 2]
            
            # The third row indicates perspective distortion
            # For fronto-parallel views, H[2,0] and H[2,1] are near zero
            perspective_strength = np.sqrt(H_norm[2, 0]**2 + H_norm[2, 1]**2)
            
            return float(perspective_strength * 1000)  # Scale for readability
        except Exception:
            return 0.0
    
    # =========================================================================
    # Calibration
    # =========================================================================
    
    def update_calibration(self, camera_id: str) -> Dict[str, Any]:
        """
        Update camera calibration using accumulated detections.
        
        Uses cv2.calibrateCamera with unit-scale object points
        (independent of square_size_mm to prevent user errors from corrupting K).
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            Dict with calibration results including is_reliable flag
        """
        detections = self.get_detections_for_camera(camera_id)
        
        if len(detections) < MIN_DETECTIONS:
            logger.info(f"Not enough detections for {camera_id}: {len(detections)}/{MIN_DETECTIONS}")
            return {'is_reliable': False, 'reason': 'insufficient_detections'}
        
        # Prepare calibration data
        object_points_list = []
        image_points_list = []
        image_size = None
        
        for det in detections:
            board_dims = tuple(det['board_dims'])
            img_pts = np.array(det['image_points'], dtype=np.float32)
            
            # Generate object points in UNIT scale (not mm!)
            # This makes K independent of square_size_mm
            obj_pts = np.zeros((board_dims[0] * board_dims[1], 3), np.float32)
            obj_pts[:, :2] = np.mgrid[0:board_dims[0], 0:board_dims[1]].T.reshape(-1, 2)
            
            object_points_list.append(obj_pts)
            image_points_list.append(img_pts)
            
            if image_size is None:
                image_size = tuple(det['image_size'])
        
        if image_size is None:
            return {'is_reliable': False, 'reason': 'no_valid_detections'}
        
        try:
            # Run OpenCV calibration
            ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_points_list,
                image_points_list,
                image_size,
                None,
                None,
                flags=cv2.CALIB_FIX_ASPECT_RATIO
            )
            
            reprojection_error = ret
            
            # Check pose diversity
            diversity_scores = [d.get('homography_diversity_score', 0) for d in detections]
            diversity_variance = np.var(diversity_scores) if len(diversity_scores) > 1 else 0
            
            # Determine if reliable
            max_error = MAX_REPROJECTION_ERROR_1080P if image_size[1] >= 1080 else MAX_REPROJECTION_ERROR_720P
            is_reliable = (
                reprojection_error < max_error and
                len(detections) >= MIN_DETECTIONS and
                diversity_variance > MIN_POSE_DIVERSITY * 0.01  # Scaled threshold
            )
            
            # Store calibration
            calibration = {
                'camera_id': camera_id,
                'image_size': list(image_size),
                'K': K.tolist(),
                'dist_coeffs': dist_coeffs.flatten().tolist(),
                'reprojection_error': float(reprojection_error),
                'num_detections': len(detections),
                'diversity_variance': float(diversity_variance),
                'is_reliable': is_reliable,
                'last_updated': datetime.now().isoformat(),
                'version': self._calibrations.get(camera_id, {}).get('version', 0) + 1
            }
            
            self._calibrations[camera_id] = calibration
            self._save_calibrations()
            
            logger.info(f"Calibration updated for {camera_id}: "
                       f"reprojection_error={reprojection_error:.4f}px, "
                       f"is_reliable={is_reliable}, "
                       f"num_detections={len(detections)}")
            
            return calibration
            
        except cv2.error as e:
            logger.error(f"OpenCV calibration failed for {camera_id}: {e}")
            return {'is_reliable': False, 'reason': f'opencv_error: {e}'}
        except Exception as e:
            logger.exception(f"Calibration failed for {camera_id}")
            return {'is_reliable': False, 'reason': str(e)}
    
    def get_intrinsics(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """
        Get camera intrinsics if available.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            Dict with K, dist_coeffs, is_reliable, version, etc. or None
        """
        return self._calibrations.get(camera_id)
    
    def is_reliable(self, camera_id: str) -> bool:
        """Check if camera has reliable calibration."""
        calib = self._calibrations.get(camera_id)
        return calib is not None and calib.get('is_reliable', False)
    
    def get_calibration_status(self, camera_id: str) -> Dict[str, Any]:
        """
        Get detailed calibration status for UI display.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            Dict with detection_count, min_required, is_reliable, etc.
        """
        detections = self.get_detections_for_camera(camera_id)
        calib = self._calibrations.get(camera_id)
        
        return {
            'camera_id': camera_id,
            'detection_count': len(detections),
            'min_required': MIN_DETECTIONS,
            'is_reliable': calib.get('is_reliable', False) if calib else False,
            'reprojection_error': calib.get('reprojection_error') if calib else None,
            'version': calib.get('version', 0) if calib else 0,
            'last_updated': calib.get('last_updated') if calib else None
        }
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def _load_calibrations(self) -> None:
        """Load calibrations from disk."""
        if self._calibration_file.exists():
            try:
                with open(self._calibration_file, 'r') as f:
                    data = json.load(f)
                self._calibrations = data.get('cameras', {})
                logger.debug(f"Loaded {len(self._calibrations)} camera calibrations")
            except Exception as e:
                logger.warning(f"Failed to load calibrations: {e}")
                self._calibrations = {}
        else:
            self._calibrations = {}
    
    def _save_calibrations(self) -> None:
        """Save calibrations to disk."""
        try:
            data = {'cameras': self._calibrations}
            with open(self._calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self._calibrations)} camera calibrations")
        except Exception as e:
            logger.error(f"Failed to save calibrations: {e}")
    
    def _load_detections(self) -> None:
        """Load detection history from disk."""
        if self._detection_file.exists():
            try:
                with open(self._detection_file, 'r') as f:
                    data = json.load(f)
                self._detections = data.get('detections', [])
                logger.debug(f"Loaded {len(self._detections)} detection records")
            except Exception as e:
                logger.warning(f"Failed to load detections: {e}")
                self._detections = []
        else:
            self._detections = []
    
    def _save_detections(self) -> None:
        """Save detection history to disk."""
        try:
            data = {'detections': self._detections}
            with open(self._detection_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self._detections)} detection records")
        except Exception as e:
            logger.error(f"Failed to save detections: {e}")


# Module-level singleton instance
_manager_instance: Optional[CameraCalibrationManager] = None


def get_calibration_manager() -> CameraCalibrationManager:
    """Get the global calibration manager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = CameraCalibrationManager()
    return _manager_instance


__all__ = [
    'CameraCalibrationManager',
    'get_calibration_manager',
    'MIN_DETECTIONS',
    'MAX_REPROJECTION_ERROR_1080P',
    'MAX_REPROJECTION_ERROR_720P',
    'MIN_POSE_DIVERSITY',
]
