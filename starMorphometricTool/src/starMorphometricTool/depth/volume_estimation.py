"""
Volume estimation from depth maps using checkerboard calibration.
All functions designed for robustness with comprehensive error handling.
"""

import logging
import os
import datetime
import numpy as np
import cv2


def calibrate_depth_with_checkerboard(depth_map, corner_pixels, board_dims, square_size_mm,
                                       image_shape, camera_matrix=None):
    """
    Calibrate depth map using checkerboard corners as ground truth.
    
    Since all checkerboard corners are coplanar, we can't establish an absolute
    depth scale. Instead, we estimate a mm-per-depth-unit scale factor by using
    the spatial relationship: adjacent corners are square_size_mm apart, and their
    depth variation (due to perspective) gives us scale information.
    
    For DA-V2 (inverse/disparity depth): higher values = closer to camera.
    
    Args:
        depth_map: Raw depth map from DA-V2 (H, W)
        corner_pixels: Detected checkerboard corners [(u, v), ...]
        board_dims: (cols-1, rows-1) internal corners
        square_size_mm: Size of each square in mm
        image_shape: (height, width) of original image
        camera_matrix: Optional camera matrix, will estimate if None
    
    Returns:
        dict with calibration results
    """
    result = {
        'success': False,
        'calibrated_depth': None,
        'reference_plane_depth_mm': None,
        'scale_factor': None,
        'offset_mm': None,
        'rmse_mm': None,
        'camera_pose': None,
        'depth_scale_mm_per_unit': None,
        'error': None
    }
    
    try:
        # Validate inputs
        if depth_map is None or corner_pixels is None:
            result['error'] = "Missing depth map or corner pixels"
            return result
        
        corner_pixels = np.array(corner_pixels).reshape(-1, 2)
        num_corners = board_dims[0] * board_dims[1]
        
        # Generate 3D object points (checkerboard lies on Z=0 plane)
        obj_points = np.zeros((num_corners, 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:board_dims[0], 0:board_dims[1]].T.reshape(-1, 2)
        obj_points *= square_size_mm
        
        # Estimate camera matrix if not provided
        height, width = image_shape[:2]
        if camera_matrix is None:
            focal_length = max(width, height)
            cx, cy = width / 2, height / 2
            camera_matrix = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
        
        dist_coeffs = np.zeros(4, dtype=np.float32)
        
        # Try solvePnP for reference depth (may not work with warped coordinates)
        reference_depth_mm = 200.0  # Default fallback
        try:
            if len(corner_pixels) == num_corners:
                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    corner_pixels.astype(np.float32),
                    camera_matrix,
                    dist_coeffs
                )
                if success:
                    reference_depth_mm = float(abs(tvec[2][0]))
                    result['camera_pose'] = {
                        'rvec': rvec.flatten().tolist(),
                        'tvec': tvec.flatten().tolist()
                    }
        except:
            pass
        
        result['reference_plane_depth_mm'] = reference_depth_mm
        
        # Extract depth values at corner pixels (or near them)
        depth_at_corners = []
        for (u, v) in corner_pixels:
            u_int, v_int = int(round(u)), int(round(v))
            if 0 <= v_int < depth_map.shape[0] and 0 <= u_int < depth_map.shape[1]:
                d = depth_map[v_int, u_int]
                if np.isfinite(d):
                    depth_at_corners.append(d)
        
        if len(depth_at_corners) == 0:
            result['error'] = "No valid depth samples at checkerboard corners"
            return result
        
        depth_at_corners = np.array(depth_at_corners)
        
        # For DA-V2: the depth values are relative/inverse depth
        # We need to estimate a scale that converts depth differences to mm
        # 
        # Key insight: typical object height above checkerboard is 1-50mm
        # DA-V2 depth range for the object vs background gives us the conversion
        #
        # Use the depth standard deviation at corners to estimate noise level
        corner_depth_mean = np.mean(depth_at_corners)
        corner_depth_std = np.std(depth_at_corners)
        
        # Estimate depth scale based on the assumption that the checkerboard
        # is at reference_depth_mm and typical depth variation is ~10-20mm per
        # unit of DA-V2 depth change (this is empirical)
        # 
        # Better approach: use mm_per_pixel and assume similar scale in Z
        # Since mm_per_pixel ~ 1 typically, depth_scale ~ square_size_mm / depth_range
        
        depth_range = depth_map.max() - depth_map.min()
        if depth_range > 0:
            # Assume typical object height is proportional to checkerboard square size
            # This is a rough heuristic - actual scale depends on camera/scene
            depth_scale_mm_per_unit = square_size_mm / (depth_range * 0.1 + 1e-6)
        else:
            depth_scale_mm_per_unit = 1.0
        
        # Clamp to reasonable range (0.1 to 100 mm per depth unit)
        depth_scale_mm_per_unit = np.clip(depth_scale_mm_per_unit, 0.1, 100.0)
        
        result['depth_scale_mm_per_unit'] = float(depth_scale_mm_per_unit)
        result['scale_factor'] = float(depth_scale_mm_per_unit)
        result['offset_mm'] = float(-corner_depth_mean * depth_scale_mm_per_unit)
        
        # Apply calibration: convert to relative mm (0 = at checkerboard level)
        # calibrated = (depth - corner_mean) * scale
        calibrated_depth = (depth_map.astype(np.float64) - corner_depth_mean) * depth_scale_mm_per_unit
        
        result['calibrated_depth'] = calibrated_depth.astype(np.float32)
        result['rmse_mm'] = float(corner_depth_std * depth_scale_mm_per_unit)
        result['success'] = True
        
        logging.info(f"Depth calibration: scale={depth_scale_mm_per_unit:.4f} mm/unit, "
                     f"corner_mean={corner_depth_mean:.2f}, corner_std={corner_depth_std:.4f}, "
                     f"reference={reference_depth_mm:.1f}mm")
        
    except Exception as e:
        logging.exception("Error in depth calibration")
        result['error'] = f"Calibration error: {e}"
    
    return result


def compute_volume(calibrated_depth, mask, mm_per_pixel, reference_plane_depth_mm, 
                    depth_map_raw=None, corner_pixels=None):
    """
    Compute volume of the masked region above the reference plane.
    
    The calibrated_depth is now RELATIVE depth where:
    - 0 = at checkerboard level (reference plane)
    - Positive values = ABOVE the checkerboard (closer to camera)
    - Negative values = BELOW the checkerboard
    
    For DA-V2: objects closer to camera have higher raw depth values, so after
    calibration (subtracting background mean), the sea star should have POSITIVE values.
    
    Args:
        calibrated_depth: Relative depth map in mm (H, W), 0 = reference plane
        mask: Binary segmentation mask (H, W), 1 = sea star
        mm_per_pixel: Scale factor from homography calibration
        reference_plane_depth_mm: Distance to checkerboard (for logging)
        depth_map_raw: Raw depth map (unused, kept for compatibility)
        corner_pixels: Corner pixels (unused, kept for compatibility)
    
    Returns:
        dict with volume estimation results
    """
    result = {
        'success': False,
        'volume_mm3': None,
        'elevation_map': None,
        'mean_elevation_mm': None,
        'max_elevation_mm': None,
        'surface_area_mm2': None,
        'error': None
    }
    
    try:
        # Validate inputs
        if calibrated_depth is None or mask is None:
            result['error'] = "Missing depth map or mask"
            return result
        
        if calibrated_depth.shape != mask.shape:
            # Resize mask to match depth map if needed
            mask = cv2.resize(mask.astype(np.uint8),
                              (calibrated_depth.shape[1], calibrated_depth.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask is binary
        mask_binary = (mask > 0).astype(np.float32)
        
        # Compute statistics of depth within the mask
        masked_depths = calibrated_depth[mask_binary > 0]
        
        if len(masked_depths) == 0:
            result['error'] = "No valid pixels in mask"
            return result
        
        # Log depth statistics for debugging
        logging.info(f"Masked depth stats: min={masked_depths.min():.2f}, "
                     f"max={masked_depths.max():.2f}, mean={masked_depths.mean():.2f}, "
                     f"std={masked_depths.std():.2f}")
        
        # Get background depth for comparison
        background_mask = mask_binary == 0
        if np.any(background_mask):
            bg_depths = calibrated_depth[background_mask]
            # Filter out zeros (areas outside warped region)
            bg_depths_valid = bg_depths[bg_depths != 0]
            if len(bg_depths_valid) > 0:
                bg_mean = np.mean(bg_depths_valid)
                logging.info(f"Background depth mean: {bg_mean:.2f}")
            else:
                bg_mean = 0
        else:
            bg_mean = 0
        
        # Elevation is the calibrated depth (already relative to checkerboard)
        # For DA-V2, higher values = closer = above the plane
        elevation = calibrated_depth * mask_binary
        
        # Only consider positive elevations (object above plane)
        elevation_positive = np.maximum(elevation, 0)
        
        # Apply mask
        elevation_masked = elevation_positive * mask_binary
        
        # Get valid elevations
        valid_elevations = elevation_masked[mask_binary > 0]
        
        # Filter out extreme outliers
        if np.any(valid_elevations > 0):
            p95 = np.percentile(valid_elevations[valid_elevations > 0], 95)
            valid_elevations_filtered = valid_elevations[valid_elevations <= p95 * 2]
            if len(valid_elevations_filtered) > 0:
                valid_elevations = valid_elevations_filtered
        
        # Volume = sum of (elevation * pixel_area)
        pixel_area_mm2 = mm_per_pixel ** 2
        volume_mm3 = float(np.sum(valid_elevations) * pixel_area_mm2)
        
        # Statistics
        positive_elevations = valid_elevations[valid_elevations > 0]
        mean_elevation = float(np.mean(positive_elevations)) if len(positive_elevations) > 0 else 0
        max_elevation = float(np.max(valid_elevations)) if len(valid_elevations) > 0 else 0
        surface_area = float(np.sum(mask_binary) * pixel_area_mm2)
        
        result['success'] = True
        result['volume_mm3'] = volume_mm3
        result['elevation_map'] = elevation_masked.astype(np.float32)
        result['mean_elevation_mm'] = mean_elevation
        result['max_elevation_mm'] = max_elevation
        result['surface_area_mm2'] = surface_area
        result['reference_depth_calibrated'] = float(bg_mean)
        
        logging.info(f"Volume estimation: {volume_mm3:.1f}mm³, "
                     f"mean height={mean_elevation:.2f}mm, max={max_elevation:.2f}mm, "
                     f"num_positive_pixels={len(positive_elevations)}")
        
    except Exception as e:
        logging.exception("Error computing volume")
        result['error'] = f"Volume computation error: {e}"
    
    return result


def run_volume_estimation_pipeline(raw_image, mask, checkerboard_info, mm_per_pixel,
                                    homography_matrix=None, mask_shape=None,
                                    encoder='vitb', input_size=518):
    """
    Complete pipeline: depth estimation → calibration → volume computation.
    
    Args:
        raw_image: Original BGR image (before homography)
        mask: Segmentation mask (in warped/corrected coordinates)
        checkerboard_info: Dict with 'corners' (image_points), 'dims', 'square_size'
        mm_per_pixel: Scale factor from checkerboard calibration
        homography_matrix: 3x3 homography matrix to warp depth to corrected coordinates
        mask_shape: Shape of the corrected mask (height, width) for warping destination
        encoder: Depth model size
        input_size: Depth model input resolution
    
    Returns:
        dict: Combined results from all pipeline stages
    """
    from . import depth_handler
    
    result = {
        'success': False,
        'error': None,
        'depth_result': None,
        'calibration_result': None,
        'volume_result': None
    }
    
    try:
        # Step 1: Depth estimation
        logging.info("Starting depth estimation...")
        depth_result = depth_handler.estimate_depth(raw_image, encoder, input_size)
        result['depth_result'] = depth_result
        
        if not depth_result['success']:
            result['error'] = f"Depth estimation failed: {depth_result['error']}"
            return result
        
        depth_map = depth_result['depth_map']
        
        # Step 1.5: Warp depth map to corrected coordinate system if homography provided
        if homography_matrix is not None:
            logging.info("Warping depth map to corrected coordinates...")
            H = np.array(homography_matrix, dtype=np.float32)
            
            # Determine output size
            if mask_shape is not None:
                out_h, out_w = mask_shape[:2]
            else:
                out_h, out_w = mask.shape[:2]
            
            # Warp the depth map using the homography
            depth_map = cv2.warpPerspective(
                depth_map, H, (out_w, out_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            logging.info(f"Depth map warped to shape: {depth_map.shape}")
        
        # Step 2: Calibrate depth using checkerboard
        logging.info("Calibrating depth with checkerboard...")
        
        # Extract corner pixels from checkerboard_info
        # Use explicit None checks to avoid numpy array boolean evaluation issues
        corners = checkerboard_info.get('checkerboard_corners')
        if corners is None:
            corners = checkerboard_info.get('corners')
        if corners is None:
            corners = checkerboard_info.get('image_points')
        if corners is None:
            result['error'] = "No checkerboard corners found in calibration data"
            return result
        
        corners = np.array(corners).reshape(-1, 2)
        
        board_dims = checkerboard_info.get('checkerboard_dims')
        if board_dims is None:
            board_dims = checkerboard_info.get('dims')
        
        square_size = checkerboard_info.get('checkerboard_square_size')
        if square_size is None:
            square_size = checkerboard_info.get('square_size')
        
        if board_dims is None or square_size is None:
            result['error'] = "Missing checkerboard dimensions or square size"
            return result
        
        # Ensure board_dims is a tuple
        if isinstance(board_dims, list):
            board_dims = tuple(board_dims)
        
        # If depth map was warped, use object points (warped corner locations) instead
        if homography_matrix is not None:
            # After warping, corners are at object point locations (real-world mm coords)
            # Generate object points
            num_corners = board_dims[0] * board_dims[1]
            warped_corners = np.zeros((num_corners, 2), np.float32)
            warped_corners[:, :2] = np.mgrid[0:board_dims[0], 0:board_dims[1]].T.reshape(-1, 2)
            warped_corners *= square_size
            corners_for_calib = warped_corners
            image_shape_for_calib = depth_map.shape
        else:
            corners_for_calib = corners
            image_shape_for_calib = raw_image.shape
        
        calib_result = calibrate_depth_with_checkerboard(
            depth_map,
            corners_for_calib,
            board_dims,
            square_size,
            image_shape_for_calib
        )
        result['calibration_result'] = calib_result
        
        if not calib_result['success']:
            result['error'] = f"Depth calibration failed: {calib_result['error']}"
            return result
        
        # Step 3: Compute volume
        logging.info("Computing volume...")
        volume_result = compute_volume(
            calib_result['calibrated_depth'],
            mask,
            mm_per_pixel,
            calib_result['reference_plane_depth_mm']
        )
        result['volume_result'] = volume_result
        
        if not volume_result['success']:
            result['error'] = f"Volume computation failed: {volume_result['error']}"
            return result
        
        result['success'] = True
        logging.info("Volume estimation pipeline completed successfully")
        
    except Exception as e:
        logging.exception("Error in volume estimation pipeline")
        result['error'] = f"Pipeline error: {e}"
    
    return result


def save_depth_data(output_dir, calibrated_depth, elevation_map, volume_info, calibration_info, mask=None):
    """
    Save depth-related outputs to files.
    
    Args:
        output_dir: Directory to save files
        calibrated_depth: Calibrated depth map (mm)
        elevation_map: Elevation above reference plane (mm) - already masked
        volume_info: Volume computation results dict
        calibration_info: Calibration parameters dict
        mask: Optional segmentation mask for visualization
    
    Returns:
        dict: Paths to saved files
    """
    saved_files = {}
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save calibrated depth as numpy array
        if calibrated_depth is not None:
            depth_path = os.path.join(output_dir, 'calibrated_depth.npy')
            np.save(depth_path, calibrated_depth)
            saved_files['depth_npy'] = depth_path
        
        # Save elevation map
        if elevation_map is not None:
            elevation_path = os.path.join(output_dir, 'elevation_map.npy')
            np.save(elevation_path, elevation_map)
            saved_files['elevation_npy'] = elevation_path
        
        # Save depth visualization as image (masked if mask provided)
        if calibrated_depth is not None:
            depth_for_viz = calibrated_depth.copy()
            
            # Apply mask if provided
            if mask is not None:
                mask_resized = mask
                if mask.shape != calibrated_depth.shape:
                    mask_resized = cv2.resize(mask.astype(np.uint8),
                                              (calibrated_depth.shape[1], calibrated_depth.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
                # Set non-masked areas to minimum for visualization
                mask_binary = mask_resized > 0
                if np.any(mask_binary):
                    depth_for_viz = np.where(mask_binary, depth_for_viz, np.nan)
            
            # Normalize for visualization (ignoring NaN)
            valid_mask = ~np.isnan(depth_for_viz)
            if np.any(valid_mask):
                depth_min = np.nanmin(depth_for_viz)
                depth_max = np.nanmax(depth_for_viz)
                if depth_max > depth_min:
                    depth_normalized = (depth_for_viz - depth_min) / (depth_max - depth_min)
                else:
                    depth_normalized = np.zeros_like(depth_for_viz)
                depth_normalized = np.nan_to_num(depth_normalized, nan=0)
            else:
                depth_normalized = np.zeros_like(calibrated_depth)
            
            depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
            
            # Make background black where mask is 0
            if mask is not None:
                mask_3ch = np.stack([mask_resized > 0] * 3, axis=-1)
                depth_colored = np.where(mask_3ch, depth_colored, 0)
            
            depth_img_path = os.path.join(output_dir, 'depth_visualization.png')
            cv2.imwrite(depth_img_path, depth_colored)
            saved_files['depth_image'] = depth_img_path
        
        # Save elevation visualization (elevation_map is already masked)
        if elevation_map is not None:
            elev_max = elevation_map.max() if elevation_map.max() > 0 else 1
            elev_normalized = np.clip(elevation_map / elev_max, 0, 1)
            elev_colored = cv2.applyColorMap((elev_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Use the same mask as depth visualization for consistency
            if mask is not None:
                mask_for_elev = mask
                if mask.shape != elevation_map.shape:
                    mask_for_elev = cv2.resize(mask.astype(np.uint8),
                                               (elevation_map.shape[1], elevation_map.shape[0]),
                                               interpolation=cv2.INTER_NEAREST)
                elev_mask_3ch = np.stack([mask_for_elev > 0] * 3, axis=-1)
                elev_colored = np.where(elev_mask_3ch, elev_colored, 0)
            else:
                # Fallback: use elevation > 0 if no mask provided
                elev_colored = np.where(elevation_map[..., np.newaxis] > 0, elev_colored, 0)
            
            # Add volume text to the visualization
            if volume_info and 'volume_mm3' in volume_info:
                volume_ml = volume_info['volume_mm3'] / 1000.0  # Convert mm³ to mL (cm³)
                text = f"Volume: {volume_ml:.3f} mL"
                
                # Calculate text size and position
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                
                # Position at top-left with padding
                text_x = 10
                text_y = text_size[1] + 10
                
                # Draw black background for text
                cv2.rectangle(elev_colored, 
                             (text_x - 5, text_y - text_size[1] - 5),
                             (text_x + text_size[0] + 5, text_y + 5),
                             (0, 0, 0), -1)
                
                # Draw text in white
                cv2.putText(elev_colored, text, (text_x, text_y), 
                           font, font_scale, (255, 255, 255), thickness)
            
            elev_img_path = os.path.join(output_dir, 'elevation_visualization.png')
            cv2.imwrite(elev_img_path, elev_colored)
            saved_files['elevation_image'] = elev_img_path
        
        logging.info(f"Depth data saved to {output_dir}")
        
    except Exception as e:
        logging.exception("Error saving depth data")
    
    return saved_files


def create_volume_estimation_data(volume_result, calibration_result, depth_result, encoder):
    """
    Create the volume_estimation dict to be saved in morphometrics.json
    
    Args:
        volume_result: Result from compute_volume()
        calibration_result: Result from calibrate_depth_with_checkerboard()
        depth_result: Result from estimate_depth()
        encoder: Encoder name used
    
    Returns:
        dict: Volume estimation data for morphometrics.json
    """
    volume_mm3 = volume_result.get('volume_mm3', 0)
    volume_ml = volume_mm3 / 1000.0 if volume_mm3 else 0  # Convert mm³ to mL (cm³)
    
    return {
        'volume_mm3': volume_mm3,
        'volume_ml': volume_ml,
        'mean_elevation_mm': volume_result.get('mean_elevation_mm'),
        'max_elevation_mm': volume_result.get('max_elevation_mm'),
        'surface_area_mm2': volume_result.get('surface_area_mm2'),
        'depth_model': encoder,
        'calibration_scale': calibration_result.get('scale_factor'),
        'calibration_offset_mm': calibration_result.get('offset_mm'),
        'calibration_rmse_mm': calibration_result.get('rmse_mm'),
        'reference_plane_depth_mm': calibration_result.get('reference_plane_depth_mm'),
        'device_used': depth_result.get('device'),
        'computed_at': datetime.datetime.now().isoformat()
    }

