# src/morphometric/detection_adapter.py
"""
Detection Adapter for starBoard Morphometric Integration.

Wraps YOLO detection and checkerboard calibration functionality from
the starMorphometricTool.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import cv2

from . import _ensure_morphometric_path, get_morphometric_tool_root

logger = logging.getLogger("starBoard.morphometric.detection")


class DetectionAdapter:
    """
    Adapter for YOLO detection and checkerboard calibration.
    
    Provides object detection (star segmentation) and checkerboard-based
    calibration for perspective correction and measurement.
    """
    
    def __init__(self):
        """Initialize the detection adapter."""
        self._yolo_model = None
        self._checkerboard_info: Optional[Dict[str, Any]] = None
        self._camera_config: Optional[Dict[str, Any]] = None
        self._camera_device_info: Optional[Dict[str, Any]] = None
        self._calibration_manager = None
        
        # Ensure morphometric tool path is available
        _ensure_morphometric_path()
    
    def set_camera_info(
        self,
        config: Dict[str, Any],
        device_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Set camera configuration for calibration fingerprinting.
        
        Should be called when camera is initialized or changes.
        
        Args:
            config: Camera configuration with width, height, etc.
            device_info: Optional device info with name, vendor_id, etc.
        """
        self._camera_config = config
        self._camera_device_info = device_info
        logger.debug("Camera info set for detection adapter")
    
    def _get_calibration_manager(self):
        """Lazy-load the calibration manager."""
        if self._calibration_manager is None:
            try:
                from calibration import get_calibration_manager
                self._calibration_manager = get_calibration_manager()
            except ImportError:
                logger.warning("Calibration module not available")
        return self._calibration_manager
    
    def _get_camera_id(self) -> Optional[str]:
        """Get camera ID for current camera config."""
        if self._camera_config is None:
            return None
        try:
            from camera.config import get_camera_fingerprint
            return get_camera_fingerprint(self._camera_config, self._camera_device_info)
        except ImportError:
            logger.warning("Could not get camera fingerprint")
            return None
    
    def load_yolo_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the YOLO model for star detection.
        
        Args:
            model_path: Optional path to model file. Uses default if not specified.
        
        Returns:
            True if model loaded successfully.
        """
        try:
            from detection.yolo_handler import load_yolo_model
            
            if model_path is None:
                # Default model path
                model_path = str(get_morphometric_tool_root() / "models" / "best.pt")
            
            self._yolo_model = load_yolo_model(model_path)
            logger.info("YOLO model loaded from %s", model_path)
            return True
            
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            return False
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if YOLO model is loaded."""
        return self._yolo_model is not None
    
    @property
    def has_checkerboard(self) -> bool:
        """Check if checkerboard calibration is available."""
        return self._checkerboard_info is not None
    
    @property
    def checkerboard_info(self) -> Optional[Dict[str, Any]]:
        """Get current checkerboard calibration info."""
        return self._checkerboard_info
    
    # =========================================================================
    # Checkerboard Detection
    # =========================================================================
    
    def detect_checkerboard(
        self,
        frame: np.ndarray,
        rows: int,
        cols: int,
        square_size_mm: float
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Detect checkerboard in frame and compute calibration.
        
        Args:
            frame: BGR image from camera
            rows: Number of squares in rows
            cols: Number of squares in columns
            square_size_mm: Size of each square in millimeters
        
        Returns:
            Tuple of (success, calibration_info)
        """
        try:
            from detection.checkerboard import find_checkerboard, compute_checkerboard_homography, calculate_mm_per_pixel
            
            # Board dims are internal corners = squares - 1
            board_dims = (cols - 1, rows - 1)
            
            found, corners, corners_refined = find_checkerboard(frame, board_dims)
            
            if not found:
                logger.warning("Checkerboard not detected")
                return False, None
            
            # Compute homography
            image_points = corners_refined.reshape(-1, 2)
            H, obj_pts = compute_checkerboard_homography(image_points, board_dims, square_size_mm)
            
            if H is None:
                logger.error("Failed to compute homography")
                return False, None
            
            # Calculate mm per pixel
            mm_per_pixel = calculate_mm_per_pixel(H, image_points, obj_pts)
            
            if mm_per_pixel is None:
                logger.error("Failed to calculate mm/pixel")
                return False, None
            
            # Store calibration info
            self._checkerboard_info = {
                'dims': board_dims,
                'corners': corners,
                'corners_refined': corners_refined,
                'image_points': image_points,
                'object_points': obj_pts,
                'square_size': square_size_mm,
                'homography': H,
                'mm_per_pixel': mm_per_pixel,
                'calibration_frame': frame.copy(),
            }
            
            logger.info("Checkerboard detected: %dx%d, mm/px=%.4f", 
                       board_dims[0], board_dims[1], mm_per_pixel)
            
            # Record detection for camera intrinsics calibration
            # Note: square_size_mm is NOT stored in detection records (prevents user errors)
            self._record_detection_for_calibration(frame, board_dims, image_points, H)
            
            return True, self._checkerboard_info
            
        except Exception as e:
            logger.exception("Error detecting checkerboard: %s", e)
            return False, None
    
    def clear_checkerboard(self) -> None:
        """Clear checkerboard calibration."""
        self._checkerboard_info = None
        logger.debug("Checkerboard calibration cleared")
    
    def _record_detection_for_calibration(
        self,
        frame: np.ndarray,
        board_dims: Tuple[int, int],
        image_points: np.ndarray,
        homography: np.ndarray
    ) -> None:
        """
        Record checkerboard detection for camera intrinsics calibration.
        
        This accumulates detections over time to enable accurate metric
        depth estimation once enough detections are collected.
        
        Args:
            frame: The image frame (for size)
            board_dims: (cols-1, rows-1) internal corners
            image_points: Nx2 array of corner positions in pixels
            homography: Homography matrix (for pose diversity scoring)
        """
        camera_id = self._get_camera_id()
        if camera_id is None:
            logger.debug("No camera ID available, skipping calibration record")
            return
        
        manager = self._get_calibration_manager()
        if manager is None:
            logger.debug("Calibration manager not available")
            return
        
        try:
            h, w = frame.shape[:2]
            result = manager.add_detection(
                camera_id=camera_id,
                image_size=(w, h),
                board_dims=board_dims,
                image_points=image_points,
                homography=homography
            )
            
            count = result.get('detection_count', 0)
            is_reliable = result.get('is_reliable', False)
            
            if is_reliable:
                logger.info(f"Camera {camera_id} calibration is now reliable ({count} detections)")
            else:
                logger.debug(f"Recorded detection for {camera_id} ({count} detections)")
                
        except Exception as e:
            logger.warning(f"Failed to record detection for calibration: {e}")
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """
        Get the current camera calibration status.
        
        Returns:
            Dict with detection_count, min_required, is_reliable, etc.
        """
        camera_id = self._get_camera_id()
        if camera_id is None:
            return {
                'camera_id': None,
                'detection_count': 0,
                'min_required': 10,
                'is_reliable': False,
                'status_message': 'No camera configured'
            }
        
        manager = self._get_calibration_manager()
        if manager is None:
            return {
                'camera_id': camera_id,
                'detection_count': 0,
                'min_required': 10,
                'is_reliable': False,
                'status_message': 'Calibration system unavailable'
            }
        
        status = manager.get_calibration_status(camera_id)
        
        # Add human-readable status message
        if status['is_reliable']:
            status['status_message'] = 'Camera calibrated (reliable)'
        elif status['detection_count'] >= status['min_required']:
            status['status_message'] = f"Calibrating... (error too high)"
        else:
            remaining = status['min_required'] - status['detection_count']
            status['status_message'] = f"Need {remaining} more detections"
        
        return status
    
    def get_camera_intrinsics(self) -> Optional[Dict[str, Any]]:
        """
        Get camera intrinsics if available.
        
        Returns:
            Dict with K, dist_coeffs, is_reliable, or None
        """
        camera_id = self._get_camera_id()
        if camera_id is None:
            return None
        
        manager = self._get_calibration_manager()
        if manager is None:
            return None
        
        return manager.get_intrinsics(camera_id)
    
    def draw_checkerboard_overlay(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Draw checkerboard corners overlay on frame.
        
        Args:
            frame: BGR image to draw on
            alpha: Transparency of overlay (0-1)
        
        Returns:
            Frame with checkerboard overlay
        """
        if self._checkerboard_info is None:
            return frame
        
        overlay = frame.copy()
        cv2.drawChessboardCorners(
            overlay,
            self._checkerboard_info['dims'],
            self._checkerboard_info['corners_refined'],
            True
        )
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # =========================================================================
    # YOLO Detection
    # =========================================================================
    
    def predict(self, frame: np.ndarray, verbose: bool = False):
        """
        Run YOLO prediction on frame.
        
        Args:
            frame: BGR image
            verbose: Whether to print YOLO output
        
        Returns:
            YOLO results object
        """
        if not self.is_model_loaded:
            raise RuntimeError("YOLO model not loaded")
        
        return self._yolo_model.predict(frame, verbose=verbose)
    
    def get_primary_detection(self, results) -> Optional[Dict[str, Any]]:
        """
        Extract primary detection from YOLO results.
        
        Args:
            results: YOLO results object
        
        Returns:
            Dictionary with detection info or None
        """
        try:
            from detection.yolo_handler import select_primary_detection
            return select_primary_detection(results)
        except Exception as e:
            logger.error("Error getting primary detection: %s", e)
            return None
    
    def detect_star(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Convenience method to detect star in frame.
        
        Args:
            frame: BGR image
        
        Returns:
            Primary detection info or None
        """
        if not self.is_model_loaded:
            return None
        
        results = self.predict(frame, verbose=False)
        return self.get_primary_detection(results)
    
    def draw_detection_overlay(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detection mask overlay on frame.
        
        Args:
            frame: BGR image
            detection: Detection info from get_primary_detection
            color: BGR color for contour
            thickness: Line thickness
        
        Returns:
            Frame with detection overlay
        """
        if detection is None:
            return frame
        
        result = frame.copy()
        mask = detection.get('mask')
        
        if mask is not None:
            # Resize mask if needed
            h, w = frame.shape[:2]
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Draw contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color, thickness)
            
            # Draw label if available
            box = detection.get('box')
            conf = detection.get('confidence')
            cls_id = detection.get('class_id')
            
            if box is not None and conf is not None:
                x1, y1 = int(box[0]), int(box[1])
                label = f"cls={cls_id}, conf={conf:.2f}"
                cv2.putText(result, label, (x1, max(0, y1 - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result
    
    # =========================================================================
    # Corrected Detection (with homography)
    # =========================================================================
    
    def correct_detection(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Apply perspective correction to detection using checkerboard homography.
        
        Args:
            frame: Original BGR frame
            detection: Detection info from get_primary_detection
        
        Returns:
            Dictionary with corrected detection data or None
        """
        if self._checkerboard_info is None:
            logger.warning("No checkerboard calibration available")
            return None
        
        if detection is None:
            return None
        
        try:
            from utils.image_processing import warp_points
            
            H = self._checkerboard_info['homography']
            obj_pts = self._checkerboard_info['object_points']
            mm_per_pixel = self._checkerboard_info['mm_per_pixel']
            
            # Compute output size
            max_x = int(obj_pts[:, 0].max()) + 10
            max_y = int(obj_pts[:, 1].max()) + 10
            
            # Warp the frame
            corrected_frame = cv2.warpPerspective(frame, H, (max_x, max_y))
            
            # Get mask and resize to frame size
            mask = detection['mask']
            h, w = frame.shape[:2]
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Find contours in original mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            
            # Use largest contour
            contour = max(contours, key=cv2.contourArea)
            
            # Warp contour points
            contour_pts = contour.reshape(-1, 2).astype(np.float32)
            warped_contour = warp_points(contour_pts, H)
            
            # Create corrected mask
            corrected_mask = np.zeros((max_y, max_x), dtype=np.uint8)
            warped_contour_int = np.round(warped_contour).astype(np.int32)
            cv2.fillPoly(corrected_mask, [warped_contour_int], 255)
            
            # Create corrected object
            corrected_object = cv2.bitwise_and(corrected_frame, corrected_frame, mask=corrected_mask)
            
            # Calculate center coordinates
            M = cv2.moments(corrected_mask)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                real_world_coord = [cx * mm_per_pixel, cy * mm_per_pixel]
            else:
                real_world_coord = [None, None]
            
            return {
                'class_id': detection.get('class_id'),
                'confidence': detection.get('confidence'),
                'corrected_mask': (corrected_mask // 255).astype(np.uint8),
                'corrected_object': corrected_object,
                'corrected_frame': corrected_frame,
                'corrected_polygon': warped_contour_int.reshape(-1, 2).tolist(),
                'real_world_coordinate': real_world_coord,
                'homography_matrix': H.tolist(),
                'mm_per_pixel': mm_per_pixel,
                'original_frame': frame.copy(),
            }
            
        except Exception as e:
            logger.exception("Error correcting detection: %s", e)
            return None
    
    @property
    def model_names(self) -> Dict[int, str]:
        """Get class names from loaded model."""
        if self._yolo_model is not None:
            return self._yolo_model.names
        return {}


