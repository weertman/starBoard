# src/morphometric/depth_adapter.py
"""
Depth Adapter for starBoard Morphometric Integration.

Wraps Depth-Anything-V2 depth estimation and volume computation functionality
from the starMorphometricTool.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2

from . import _ensure_morphometric_path, get_morphometric_tool_root

logger = logging.getLogger("starBoard.morphometric.depth")


class DepthAdapter:
    """
    Adapter for depth estimation and volume computation.
    
    Uses Depth-Anything-V2 model for monocular depth estimation,
    calibrated with checkerboard data for volume calculation.
    """
    
    def __init__(self):
        """Initialize the depth adapter."""
        _ensure_morphometric_path()
        
        self._last_result: Optional[Dict[str, Any]] = None
        self._elevation_visualization: Optional[np.ndarray] = None
    
    @staticmethod
    def is_available() -> bool:
        """
        Check if depth estimation dependencies are available.
        
        Returns:
            True if Depth-Anything-V2 and dependencies are installed.
        """
        try:
            _ensure_morphometric_path()
            import torch
            from depth import load_depth_model
            
            # Check if Depth-Anything-V2 directory exists
            # It can be inside starMorphometricTool or alongside it
            possible_paths = [
                get_morphometric_tool_root() / "Depth-Anything-V2",  # Inside starMorphometricTool
                get_morphometric_tool_root().parent / "Depth-Anything-V2",  # Alongside starMorphometricTool
            ]
            
            da_found = False
            for da_root in possible_paths:
                if da_root.exists():
                    da_found = True
                    logger.debug("Depth-Anything-V2 found at %s", da_root)
                    break
            
            if not da_found:
                logger.debug("Depth-Anything-V2 directory not found in any expected location")
                return False
            
            return True
        except ImportError as e:
            logger.debug("Depth dependencies not available: %s", e)
            return False
    
    @staticmethod
    def get_device_info() -> Dict[str, str]:
        """
        Get information about available compute devices.
        
        Returns:
            Dict with 'device' and 'name' keys.
        """
        try:
            _ensure_morphometric_path()
            from depth import get_device
            device, name = get_device()
            return {"device": device, "name": name}
        except Exception as e:
            return {"device": "cpu", "name": f"CPU (error: {e})"}
    
    def run_volume_estimation(
        self,
        raw_frame: np.ndarray,
        corrected_mask: np.ndarray,
        checkerboard_info: Dict[str, Any],
        homography_matrix: np.ndarray,
        mm_per_pixel: float,
        encoder: str = 'vitb',
        input_size: int = 518,
        mfolder_path: Optional[Path] = None,
        intrinsics: Optional[Dict[str, Any]] = None,
        camera_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete volume estimation pipeline.
        
        Args:
            raw_frame: Original BGR image (before homography correction)
            corrected_mask: Binary segmentation mask (in corrected coordinates)
            checkerboard_info: Dict with 'corners', 'dims', 'square_size'
            homography_matrix: 3x3 homography matrix for perspective correction
            mm_per_pixel: Scale factor from checkerboard calibration
            encoder: Depth model size ('vits', 'vitb', 'vitl')
            input_size: Model input resolution (higher = more detail, more memory)
            mfolder_path: Optional path to save depth data files
            intrinsics: Optional camera intrinsics (K, dist_coeffs, is_reliable)
            camera_id: Optional camera identifier
        
        Returns:
            Dict with:
                - success: bool
                - volume_mm3: float (if successful)
                - volume_ml: float (if successful) 
                - mean_elevation_mm: float
                - max_elevation_mm: float
                - elevation_map: np.ndarray
                - elevation_visualization: np.ndarray (BGR image)
                - calibration_status: 'reliable' or 'provisional'
                - error: str (if failed)
        """
        result = {
            'success': False,
            'volume_mm3': None,
            'volume_ml': None,
            'mean_elevation_mm': None,
            'max_elevation_mm': None,
            'elevation_map': None,
            'elevation_visualization': None,
            'depth_result': None,
            'calibration_result': None,
            'calibration_status': 'provisional',
            'calibration_camera_id': camera_id,
            'calibration_version': None,
            'error': None
        }
        
        try:
            from depth import run_volume_estimation_pipeline, save_depth_data, clear_model_cache
            from depth.volume_estimation import create_volume_estimation_data
            
            # Prepare checkerboard info in expected format
            # The pipeline looks for 'checkerboard_corners' first, so set that
            corners_data = checkerboard_info.get('image_points')
            if corners_data is None:
                corners_data = checkerboard_info.get('corners_refined')
            if corners_data is None:
                corners_data = checkerboard_info.get('corners')
            
            # Ensure corners are reshaped properly
            if corners_data is not None and hasattr(corners_data, 'reshape'):
                corners_data = corners_data.reshape(-1, 2)
            
            cb_info = {
                'checkerboard_corners': corners_data,  # Primary key the pipeline expects
                'checkerboard_dims': checkerboard_info.get('dims'),
                'checkerboard_square_size': checkerboard_info.get('square_size'),
                'dims': checkerboard_info.get('dims'),
                'square_size': checkerboard_info.get('square_size'),
            }
            
            # Get mask shape for warping
            mask_shape = corrected_mask.shape
            
            logger.info("Starting volume estimation pipeline (encoder=%s, input_size=%d)", encoder, input_size)
            
            # Run the pipeline with intrinsics if available
            pipeline_result = run_volume_estimation_pipeline(
                raw_image=raw_frame,
                mask=corrected_mask,
                checkerboard_info=cb_info,
                mm_per_pixel=mm_per_pixel,
                homography_matrix=homography_matrix.tolist() if isinstance(homography_matrix, np.ndarray) else homography_matrix,
                mask_shape=mask_shape,
                encoder=encoder,
                input_size=input_size,
                intrinsics=intrinsics,
                camera_id=camera_id
            )
            
            if not pipeline_result['success']:
                result['error'] = pipeline_result.get('error', 'Unknown pipeline error')
                logger.error("Volume estimation failed: %s", result['error'])
                return result
            
            # Extract results
            depth_result = pipeline_result['depth_result']
            calibration_result = pipeline_result['calibration_result']
            volume_result = pipeline_result['volume_result']
            
            result['depth_result'] = depth_result
            result['calibration_result'] = calibration_result
            
            # Capture calibration status from pipeline
            result['calibration_status'] = pipeline_result.get('calibration_status', 'provisional')
            result['calibration_camera_id'] = pipeline_result.get('calibration_camera_id', camera_id)
            result['calibration_version'] = pipeline_result.get('calibration_version')
            result['calibration_method'] = pipeline_result.get('calibration_method')
            result['raw_depth_map'] = pipeline_result.get('raw_depth_map')
            
            # Volume data
            volume_mm3 = volume_result.get('volume_mm3', 0)
            result['volume_mm3'] = float(volume_mm3)
            result['volume_ml'] = float(volume_mm3 / 1000.0)
            result['mean_elevation_mm'] = float(volume_result.get('mean_elevation_mm', 0))
            result['max_elevation_mm'] = float(volume_result.get('max_elevation_mm', 0))
            result['elevation_map'] = volume_result.get('elevation_map')
            result['surface_area_mm2'] = float(volume_result.get('surface_area_mm2', 0))
            
            # Create elevation visualization
            elevation_map = volume_result.get('elevation_map')
            if elevation_map is not None:
                result['elevation_visualization'] = self._create_elevation_visualization(
                    elevation_map, volume_mm3, corrected_mask
                )
                self._elevation_visualization = result['elevation_visualization']
            
            # Build checkerboard detection data for saving (for recomputation)
            checkerboard_detection = {
                'image_points': corners_data.tolist() if hasattr(corners_data, 'tolist') else corners_data,
                'board_dims': list(checkerboard_info.get('dims', [])),
                'square_size': checkerboard_info.get('square_size'),
            }
            
            # Save depth data if mfolder provided
            if mfolder_path is not None:
                try:
                    saved_files = save_depth_data(
                        str(mfolder_path),
                        calibration_result.get('calibrated_depth'),
                        volume_result.get('elevation_map'),
                        volume_result,
                        calibration_result,
                        mask=corrected_mask,
                        raw_depth_map=result.get('raw_depth_map'),
                        checkerboard_detection=checkerboard_detection
                    )
                    result['saved_files'] = saved_files
                except Exception as e:
                    logger.warning("Failed to save depth data files: %s", e)
            
            # Create volume estimation data for morphometrics.json
            result['volume_estimation_data'] = create_volume_estimation_data(
                volume_result, calibration_result, depth_result, encoder,
                calibration_status=result['calibration_status'],
                camera_id=result['calibration_camera_id'],
                calibration_version=result['calibration_version'],
                calibration_method=result['calibration_method']
            )
            
            result['success'] = True
            status = result['calibration_status']
            logger.info("Volume estimation complete: %.1f mm³ (%.3f mL) [%s]", 
                       volume_mm3, volume_mm3 / 1000, status)
            
            # Store for later access
            self._last_result = result
            
        except ImportError as e:
            result['error'] = f"Depth module not available: {e}"
            logger.error(result['error'])
        except MemoryError:
            result['error'] = "Out of memory. Try closing other applications or using a smaller model (vits)."
            logger.error(result['error'])
        except Exception as e:
            result['error'] = f"Volume estimation error: {e}"
            logger.exception(result['error'])
        finally:
            # Clear model cache to free memory
            try:
                from depth import clear_model_cache
                clear_model_cache()
            except Exception:
                pass
        
        return result
    
    def _create_elevation_visualization(
        self,
        elevation_map: np.ndarray,
        volume_mm3: float,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create a colorized elevation visualization.
        
        Args:
            elevation_map: Elevation values in mm
            volume_mm3: Volume for annotation
            mask: Optional mask to apply
        
        Returns:
            BGR visualization image
        """
        # Normalize elevation for visualization
        elev_max = elevation_map.max() if elevation_map.max() > 0 else 1
        elev_normalized = np.clip(elevation_map / elev_max, 0, 1)
        
        # Apply colormap
        elev_colored = cv2.applyColorMap((elev_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Apply mask if provided
        if mask is not None:
            mask_resized = mask
            if mask.shape != elevation_map.shape:
                mask_resized = cv2.resize(
                    mask.astype(np.uint8),
                    (elevation_map.shape[1], elevation_map.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            mask_3ch = np.stack([mask_resized > 0] * 3, axis=-1)
            elev_colored = np.where(mask_3ch, elev_colored, 0)
        else:
            # Use elevation > 0 if no mask
            elev_colored = np.where(elevation_map[..., np.newaxis] > 0, elev_colored, 0)
        
        # Add volume text annotation (smaller font for compact display)
        volume_ml = volume_mm3 / 1000.0
        text = f"{volume_ml:.2f}mL"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        text_x = 5
        text_y = text_size[1] + 5
        
        # Draw background rectangle
        cv2.rectangle(
            elev_colored,
            (text_x - 2, text_y - text_size[1] - 2),
            (text_x + text_size[0] + 2, text_y + 2),
            (0, 0, 0), -1
        )
        
        # Draw text
        cv2.putText(elev_colored, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        return elev_colored
    
    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        """Get the last volume estimation result."""
        return self._last_result
    
    @property
    def elevation_visualization(self) -> Optional[np.ndarray]:
        """Get the last elevation visualization image."""
        return self._elevation_visualization
    
    def clear(self) -> None:
        """Clear cached results."""
        self._last_result = None
        self._elevation_visualization = None

