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
    Calibrate depth map using checkerboard corners and solvePnP as Z reference.
    
    Uses the camera-to-checkerboard distance from solvePnP combined with DA-V2's
    inverse depth values to establish a proper metric scale.
    
    For DA-V2 (inverse/disparity depth): higher values = closer to camera.
    
    Calibration approach:
    - solvePnP gives camera distance to checkerboard (reference_depth_mm)
    - DA-V2 values at corners give corner_depth_mean
    - Calibration constant k = reference_depth_mm * corner_depth_mean
    - Real depth = k / da_v2_value (inverse relationship)
    - Height above plane = reference_depth_mm - real_depth
    
    Args:
        depth_map: Raw depth map from DA-V2 (H, W)
        corner_pixels: Detected checkerboard corners [(u, v), ...]
        board_dims: (cols-1, rows-1) internal corners
        square_size_mm: Size of each square in mm
        image_shape: (height, width) of original image
        camera_matrix: Optional camera matrix, will estimate if None
    
    Returns:
        dict with calibration results including calibrated_depth (height above plane in mm)
    """
    result = {
        'success': False,
        'calibrated_depth': None,
        'reference_plane_depth_mm': None,
        'calibration_constant_k': None,
        'scale_factor': None,  # Kept for backward compatibility
        'offset_mm': None,
        'rmse_mm': None,
        'camera_pose': None,
        'depth_scale_mm_per_unit': None,  # Deprecated, no longer used
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
        
        # For DA-V2: the depth values are INVERSE/DISPARITY depth
        # Higher values = closer to camera
        # 
        # Proper calibration using solvePnP reference depth:
        # - We know the camera-to-checkerboard distance (reference_depth_mm) from solvePnP
        # - We know the DA-V2 value at the checkerboard (corner_depth_mean)
        # - For inverse depth: real_depth = k / da_v2_value
        # - Calibration constant k = reference_depth_mm * corner_depth_mean
        # - Height above plane = reference_depth_mm - real_depth
        
        corner_depth_mean = np.mean(depth_at_corners)
        corner_depth_std = np.std(depth_at_corners)
        
        # Compute calibration constant using inverse depth relationship
        # k = reference_depth * corner_mean, so that: reference_depth = k / corner_mean
        calibration_constant_k = reference_depth_mm * corner_depth_mean
        
        # Convert DA-V2 inverse depth to real depth in mm
        # real_depth = k / da_v2_value
        # Avoid division by zero for very small depth values
        depth_map_safe = np.maximum(depth_map.astype(np.float64), 1e-6)
        real_depth_map = calibration_constant_k / depth_map_safe
        
        # Height above the reference plane (checkerboard)
        # Objects closer to camera have higher DA-V2 values -> smaller real_depth -> positive height
        calibrated_depth = reference_depth_mm - real_depth_map
        
        # Store calibration parameters
        result['calibration_constant_k'] = float(calibration_constant_k)
        result['scale_factor'] = float(calibration_constant_k)  # Keep for backward compatibility
        result['depth_scale_mm_per_unit'] = None  # No longer used
        result['offset_mm'] = 0.0  # Reference plane is at 0
        result['calibrated_depth'] = calibrated_depth.astype(np.float32)
        
        # Estimate RMSE based on corner depth variation
        # After calibration, corners should ideally all be at 0 (on the plane)
        corner_heights = reference_depth_mm - (calibration_constant_k / np.maximum(depth_at_corners, 1e-6))
        result['rmse_mm'] = float(np.std(corner_heights))
        result['success'] = True
        
        logging.info(f"Depth calibration (inverse): k={calibration_constant_k:.2f}, "
                     f"corner_mean={corner_depth_mean:.2f}, corner_std={corner_depth_std:.4f}, "
                     f"reference={reference_depth_mm:.1f}mm, rmse={result['rmse_mm']:.2f}mm")
        
    except Exception as e:
        logging.exception("Error in depth calibration")
        result['error'] = f"Calibration error: {e}"
    
    return result


def calibrate_depth_with_intrinsics(
    depth_map: np.ndarray,
    image_points: np.ndarray,
    board_dims: tuple,
    square_size_mm: float,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    checkerboard_mask: np.ndarray = None
) -> dict:
    """
    Calibrate depth map using known camera intrinsics.
    
    This is the preferred calibration method when camera intrinsics (K) are available
    from accumulated checkerboard detections. It uses:
    1. solvePnP to get the checkerboard pose (R, t)
    2. Ray-plane intersection to compute true depth Z_plane(u,v) at board pixels
    3. Fitting Z = a/(d+b) to map DA-V2 output to metric depth
    
    Args:
        depth_map: Raw depth map from DA-V2 (H, W)
        image_points: Checkerboard corners in ORIGINAL image pixels (Nx2)
        board_dims: (cols-1, rows-1) internal corners
        square_size_mm: Size of each square in mm (used for metric pose)
        K: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients (1x5 or similar)
        checkerboard_mask: Optional mask of checkerboard region for sampling
    
    Returns:
        dict with:
            - success: bool
            - calibrated_depth: Height above plane in mm (H, W)
            - fit_params: (a, b) for Z = a/(d+b) mapping
            - fit_residual: RMS fitting error in mm
            - plane_depth_mm: Distance to checkerboard plane
            - error: Error message if failed
    """
    result = {
        'success': False,
        'calibrated_depth': None,
        'fit_params': None,
        'fit_residual': None,
        'plane_depth_mm': None,
        'calibration_method': 'intrinsics',
        'error': None
    }
    
    try:
        # Validate inputs
        if depth_map is None or image_points is None or K is None:
            result['error'] = "Missing required inputs"
            return result
        
        K = np.array(K, dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.array(dist_coeffs, dtype=np.float64).flatten()
        image_points = np.array(image_points, dtype=np.float32).reshape(-1, 2)
        
        num_corners = board_dims[0] * board_dims[1]
        if len(image_points) != num_corners:
            result['error'] = f"Expected {num_corners} corners, got {len(image_points)}"
            return result
        
        # Generate 3D object points in mm (checkerboard lies on Z=0 plane)
        obj_points = np.zeros((num_corners, 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:board_dims[0], 0:board_dims[1]].T.reshape(-1, 2)
        obj_points *= square_size_mm
        
        # Solve for camera pose
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            image_points,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            result['error'] = "solvePnP failed"
            return result
        
        # Get rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Checkerboard plane in camera coordinates:
        # The plane is at Z=0 in object space, transformed by [R|t]
        # Plane normal in camera space: n = R @ [0, 0, 1]^T = R[:, 2]
        # Plane point in camera space: p0 = t (origin of object space)
        plane_normal = R[:, 2]
        plane_point = tvec.flatten()
        
        # Distance from camera to plane along camera Z axis
        plane_depth_mm = float(tvec[2, 0])
        result['plane_depth_mm'] = plane_depth_mm
        
        logging.info(f"Checkerboard pose: distance={plane_depth_mm:.1f}mm")
        
        # Sample depth values at checkerboard corners
        h, w = depth_map.shape[:2]
        depth_samples = []
        z_plane_samples = []
        
        for i, (u, v) in enumerate(image_points):
            u_int, v_int = int(round(u)), int(round(v))
            if 0 <= v_int < h and 0 <= u_int < w:
                d = depth_map[v_int, u_int]
                if np.isfinite(d) and d > 0:
                    # Compute ray direction for this pixel
                    # ray = K^-1 @ [u, v, 1]^T (normalized)
                    pixel_homog = np.array([u, v, 1.0])
                    ray_dir = np.linalg.inv(K) @ pixel_homog
                    ray_dir = ray_dir / np.linalg.norm(ray_dir)
                    
                    # Ray-plane intersection: find t where ray intersects plane
                    # Plane equation: n · (p - p0) = 0
                    # Ray: p = t * ray_dir
                    # Solution: t = (n · p0) / (n · ray_dir)
                    denom = np.dot(plane_normal, ray_dir)
                    if abs(denom) > 1e-6:
                        t = np.dot(plane_normal, plane_point) / denom
                        if t > 0:  # Plane is in front of camera
                            z_plane = t * ray_dir[2]  # Z component of intersection
                            depth_samples.append(d)
                            z_plane_samples.append(z_plane)
        
        if len(depth_samples) < 4:
            result['error'] = f"Not enough valid samples: {len(depth_samples)}"
            return result
        
        depth_samples = np.array(depth_samples)
        z_plane_samples = np.array(z_plane_samples)
        
        # Fit inverse model: Z = a / (d + b)
        # Rearranged: Z * (d + b) = a  =>  Z*d + Z*b = a
        # Linear least squares: [Z*d, Z] @ [1, b]^T = a  ... not quite linear
        # 
        # Better: fit d = a/Z - b (linear in a, b given Z)
        # Or use nonlinear fit for Z = a/(d+b)
        
        # Simple approach: assume b ≈ 0 and fit a = Z * d directly
        # Then refine with nonlinear if needed
        
        # Initial estimate: a = mean(Z * d), b = 0
        a_init = np.mean(z_plane_samples * depth_samples)
        
        # Refine with scipy if available, otherwise use initial
        try:
            from scipy.optimize import least_squares
            
            def residual(params):
                a, b = params
                z_pred = a / (depth_samples + b)
                return z_pred - z_plane_samples
            
            result_opt = least_squares(
                residual,
                [a_init, 0.0],
                bounds=([0, -100], [np.inf, 100])
            )
            a, b = result_opt.x
            
        except ImportError:
            # Fallback: use initial estimate
            a, b = a_init, 0.0
            logging.debug("scipy not available, using simple depth fit")
        
        result['fit_params'] = (float(a), float(b))
        
        # Compute fit residual
        z_fitted = a / (depth_samples + b)
        fit_residual = np.sqrt(np.mean((z_fitted - z_plane_samples) ** 2))
        result['fit_residual'] = float(fit_residual)
        
        logging.info(f"Depth fit: a={a:.2f}, b={b:.4f}, residual={fit_residual:.2f}mm")
        
        # Apply calibration to full depth map
        depth_safe = np.maximum(depth_map.astype(np.float64), 1e-6)
        z_metric = a / (depth_safe + b)
        
        # Height above plane = plane_depth - metric_depth
        # Objects closer to camera have smaller Z, so positive height
        calibrated_depth = plane_depth_mm - z_metric
        
        result['calibrated_depth'] = calibrated_depth.astype(np.float32)
        result['success'] = True
        
        logging.info(f"Depth calibration (intrinsics): plane={plane_depth_mm:.1f}mm, "
                     f"fit_residual={fit_residual:.2f}mm")
        
    except Exception as e:
        logging.exception("Error in intrinsics-based depth calibration")
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
                                    encoder='vitb', input_size=518,
                                    intrinsics=None, camera_id=None):
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
        intrinsics: Optional dict with 'K', 'dist_coeffs', 'is_reliable', 'version'
                   from camera calibration. If reliable, uses proper metric calibration.
        camera_id: Optional camera identifier for tracking calibration source
    
    Returns:
        dict: Combined results from all pipeline stages, including:
            - raw_depth_map: Unprocessed DA-V2 output (always included for recomputation)
            - calibration_status: 'reliable' or 'provisional'
            - calibration_camera_id: Camera used for calibration
            - calibration_version: Version of camera calibration used
    """
    from . import depth_handler
    
    result = {
        'success': False,
        'error': None,
        'depth_result': None,
        'calibration_result': None,
        'volume_result': None,
        'raw_depth_map': None,  # Always stored for later recomputation
        'calibration_status': 'provisional',
        'calibration_camera_id': camera_id,
        'calibration_version': None,
        'calibration_method': None
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
        
        # ALWAYS store raw depth map for later recomputation
        result['raw_depth_map'] = depth_map.copy()
        
        # Step 1.5: Warp depth map to corrected coordinate system if homography provided
        depth_map_warped = depth_map
        if homography_matrix is not None:
            logging.info("Warping depth map to corrected coordinates...")
            H = np.array(homography_matrix, dtype=np.float32)
            
            # Determine output size
            if mask_shape is not None:
                out_h, out_w = mask_shape[:2]
            else:
                out_h, out_w = mask.shape[:2]
            
            # Warp the depth map using the homography
            depth_map_warped = cv2.warpPerspective(
                depth_map, H, (out_w, out_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            logging.info(f"Depth map warped to shape: {depth_map_warped.shape}")
        
        # Extract checkerboard info
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
        
        # Step 2: Calibrate depth
        # Choose calibration method based on intrinsics availability
        use_intrinsics = (
            intrinsics is not None and 
            intrinsics.get('is_reliable', False) and 
            intrinsics.get('K') is not None
        )
        
        if use_intrinsics:
            # Use proper metric calibration with known camera intrinsics
            logging.info("Calibrating depth with known camera intrinsics...")
            
            K = np.array(intrinsics['K'])
            dist_coeffs = np.array(intrinsics.get('dist_coeffs', [0, 0, 0, 0, 0]))
            
            # Use ORIGINAL image corners (not warped) for solvePnP
            calib_result = calibrate_depth_with_intrinsics(
                depth_map,  # Use un-warped depth for proper geometry
                corners,    # Original image pixel coordinates
                board_dims,
                square_size,
                K,
                dist_coeffs
            )
            
            if calib_result['success']:
                result['calibration_status'] = 'reliable'
                result['calibration_version'] = intrinsics.get('version')
                result['calibration_method'] = 'intrinsics'
                
                # If we have homography, warp the calibrated depth
                if homography_matrix is not None:
                    calibrated_warped = cv2.warpPerspective(
                        calib_result['calibrated_depth'], H, (out_w, out_h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    )
                    calib_result['calibrated_depth'] = calibrated_warped
            else:
                logging.warning(f"Intrinsics calibration failed: {calib_result.get('error')}, falling back to heuristic")
                use_intrinsics = False
        
        if not use_intrinsics:
            # Fall back to heuristic calibration (provisional)
            logging.info("Calibrating depth with heuristic method (provisional)...")
            
            # If depth map was warped, use object points (warped corner locations)
            if homography_matrix is not None:
                num_corners = board_dims[0] * board_dims[1]
                warped_corners = np.zeros((num_corners, 2), np.float32)
                warped_corners[:, :2] = np.mgrid[0:board_dims[0], 0:board_dims[1]].T.reshape(-1, 2)
                warped_corners *= square_size
                corners_for_calib = warped_corners
                image_shape_for_calib = depth_map_warped.shape
                depth_for_calib = depth_map_warped
            else:
                corners_for_calib = corners
                image_shape_for_calib = raw_image.shape
                depth_for_calib = depth_map
            
            calib_result = calibrate_depth_with_checkerboard(
                depth_for_calib,
                corners_for_calib,
                board_dims,
                square_size,
                image_shape_for_calib
            )
            
            result['calibration_status'] = 'provisional'
            result['calibration_method'] = 'heuristic'
        
        result['calibration_result'] = calib_result
        
        if not calib_result['success']:
            result['error'] = f"Depth calibration failed: {calib_result['error']}"
            return result
        
        # Step 3: Compute volume
        logging.info("Computing volume...")
        reference_depth = calib_result.get('reference_plane_depth_mm') or calib_result.get('plane_depth_mm', 0)
        
        volume_result = compute_volume(
            calib_result['calibrated_depth'],
            mask,
            mm_per_pixel,
            reference_depth
        )
        result['volume_result'] = volume_result
        
        if not volume_result['success']:
            result['error'] = f"Volume computation failed: {volume_result['error']}"
            return result
        
        result['success'] = True
        status_str = result['calibration_status']
        logging.info(f"Volume estimation pipeline completed successfully (status: {status_str})")
        
    except Exception as e:
        logging.exception("Error in volume estimation pipeline")
        result['error'] = f"Pipeline error: {e}"
    
    return result


def save_depth_data(output_dir, calibrated_depth, elevation_map, volume_info, calibration_info, 
                    mask=None, raw_depth_map=None, checkerboard_detection=None):
    """
    Save depth-related outputs to files.
    
    Args:
        output_dir: Directory to save files
        calibrated_depth: Calibrated depth map (mm)
        elevation_map: Elevation above reference plane (mm) - already masked
        volume_info: Volume computation results dict
        calibration_info: Calibration parameters dict
        mask: Optional segmentation mask for visualization
        raw_depth_map: Optional raw DA-V2 output (for later recomputation)
        checkerboard_detection: Optional dict with image_points, board_dims, square_size
                               (for later recomputation)
    
    Returns:
        dict: Paths to saved files
    """
    import json
    
    saved_files = {}
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # ALWAYS save raw depth map if provided (critical for recomputation)
        if raw_depth_map is not None:
            raw_depth_path = os.path.join(output_dir, 'raw_depth.npy')
            np.save(raw_depth_path, raw_depth_map)
            saved_files['raw_depth_npy'] = raw_depth_path
            logging.debug(f"Saved raw depth map to {raw_depth_path}")
        
        # Save checkerboard detection info for recomputation
        if checkerboard_detection is not None:
            cb_path = os.path.join(output_dir, 'checkerboard_detection.json')
            # Convert numpy arrays to lists for JSON serialization
            cb_data = {}
            for key, value in checkerboard_detection.items():
                if isinstance(value, np.ndarray):
                    cb_data[key] = value.tolist()
                else:
                    cb_data[key] = value
            with open(cb_path, 'w') as f:
                json.dump(cb_data, f, indent=2)
            saved_files['checkerboard_detection'] = cb_path
            logging.debug(f"Saved checkerboard detection to {cb_path}")
        
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


def create_volume_estimation_data(volume_result, calibration_result, depth_result, encoder,
                                   calibration_status='provisional', camera_id=None, 
                                   calibration_version=None, calibration_method=None):
    """
    Create the volume_estimation dict to be saved in morphometrics.json
    
    Args:
        volume_result: Result from compute_volume()
        calibration_result: Result from calibrate_depth_with_checkerboard() or calibrate_depth_with_intrinsics()
        depth_result: Result from estimate_depth()
        encoder: Encoder name used
        calibration_status: 'reliable' or 'provisional'
        camera_id: Camera identifier used for calibration
        calibration_version: Version of camera calibration used (for recomputation tracking)
        calibration_method: 'intrinsics' or 'heuristic'
    
    Returns:
        dict: Volume estimation data for morphometrics.json
    """
    volume_mm3 = volume_result.get('volume_mm3', 0)
    volume_ml = volume_mm3 / 1000.0 if volume_mm3 else 0  # Convert mm³ to mL (cm³)
    
    # Get calibration-specific values depending on method used
    if calibration_method == 'intrinsics':
        fit_params = calibration_result.get('fit_params')
        calibration_k = fit_params[0] if fit_params else None
        plane_depth = calibration_result.get('plane_depth_mm')
        fit_residual = calibration_result.get('fit_residual')
    else:
        calibration_k = calibration_result.get('calibration_constant_k')
        plane_depth = calibration_result.get('reference_plane_depth_mm')
        fit_residual = calibration_result.get('rmse_mm')
    
    return {
        'volume_mm3': volume_mm3,
        'volume_ml': volume_ml,
        'mean_elevation_mm': volume_result.get('mean_elevation_mm'),
        'max_elevation_mm': volume_result.get('max_elevation_mm'),
        'surface_area_mm2': volume_result.get('surface_area_mm2'),
        'depth_model': encoder,
        'calibration_status': calibration_status,
        'calibration_camera_id': camera_id,
        'calibration_version': calibration_version,
        'calibration_method': calibration_method,
        'calibration_constant_k': calibration_k,
        'calibration_rmse_mm': fit_residual,
        'reference_plane_depth_mm': plane_depth,
        'raw_depth_file': 'raw_depth.npy',  # Always saved for recomputation
        'device_used': depth_result.get('device'),
        'computed_at': datetime.datetime.now().isoformat()
    }


def recompute_volumes_for_camera(camera_id: str, measurements_root: str, 
                                  calibration_manager=None) -> dict:
    """
    Recompute volumes for all measurements using a specific camera.
    
    This function finds all measurements with provisional calibration status
    that have raw depth data saved, and recomputes their volumes using the
    current (hopefully now reliable) camera intrinsics.
    
    Args:
        camera_id: Camera identifier to recompute volumes for
        measurements_root: Root directory containing measurement folders
        calibration_manager: Optional CameraCalibrationManager instance
                           (will create one if not provided)
    
    Returns:
        dict with:
            - recomputed: List of successfully recomputed measurement paths
            - skipped: List of skipped measurements (no raw data or already reliable)
            - failed: List of failed recomputations with error messages
            - intrinsics_used: The intrinsics used for recomputation
    """
    import json
    import glob
    
    result = {
        'recomputed': [],
        'skipped': [],
        'failed': [],
        'intrinsics_used': None
    }
    
    # Get calibration manager
    if calibration_manager is None:
        try:
            from calibration import get_calibration_manager
            calibration_manager = get_calibration_manager()
        except ImportError:
            logging.error("Calibration module not available")
            return result
    
    # Get intrinsics for this camera
    intrinsics = calibration_manager.get_intrinsics(camera_id)
    if intrinsics is None or not intrinsics.get('is_reliable', False):
        logging.warning(f"No reliable intrinsics available for camera {camera_id}")
        return result
    
    result['intrinsics_used'] = {
        'camera_id': camera_id,
        'version': intrinsics.get('version'),
        'reprojection_error': intrinsics.get('reprojection_error')
    }
    
    K = np.array(intrinsics['K'])
    dist_coeffs = np.array(intrinsics.get('dist_coeffs', [0, 0, 0, 0, 0]))
    
    logging.info(f"Recomputing volumes for camera {camera_id} using intrinsics v{intrinsics.get('version')}")
    
    # Find all morphometrics.json files
    pattern = os.path.join(measurements_root, '**', 'morphometrics.json')
    morphometric_files = glob.glob(pattern, recursive=True)
    
    for morph_path in morphometric_files:
        mfolder = os.path.dirname(morph_path)
        
        try:
            # Load morphometrics.json
            with open(morph_path, 'r') as f:
                morph_data = json.load(f)
            
            # Check if this measurement uses our camera and is provisional
            vol_est = morph_data.get('volume_estimation', {})
            
            if vol_est.get('calibration_camera_id') != camera_id:
                result['skipped'].append({
                    'path': mfolder,
                    'reason': 'different_camera'
                })
                continue
            
            if vol_est.get('calibration_status') == 'reliable':
                result['skipped'].append({
                    'path': mfolder,
                    'reason': 'already_reliable'
                })
                continue
            
            # Check for raw depth data
            raw_depth_path = os.path.join(mfolder, 'raw_depth.npy')
            cb_detection_path = os.path.join(mfolder, 'checkerboard_detection.json')
            
            if not os.path.exists(raw_depth_path):
                result['skipped'].append({
                    'path': mfolder,
                    'reason': 'no_raw_depth'
                })
                continue
            
            if not os.path.exists(cb_detection_path):
                result['skipped'].append({
                    'path': mfolder,
                    'reason': 'no_checkerboard_detection'
                })
                continue
            
            # Load raw depth and checkerboard detection
            raw_depth = np.load(raw_depth_path)
            
            with open(cb_detection_path, 'r') as f:
                cb_detection = json.load(f)
            
            image_points = np.array(cb_detection['image_points'], dtype=np.float32)
            board_dims = tuple(cb_detection['board_dims'])
            square_size = cb_detection['square_size']
            
            # Recompute calibration with new intrinsics
            calib_result = calibrate_depth_with_intrinsics(
                raw_depth,
                image_points,
                board_dims,
                square_size,
                K,
                dist_coeffs
            )
            
            if not calib_result['success']:
                result['failed'].append({
                    'path': mfolder,
                    'error': calib_result.get('error', 'Unknown calibration error')
                })
                continue
            
            # Load mask for volume computation
            mask_path = os.path.join(mfolder, 'corrected_mask.png')
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                result['failed'].append({
                    'path': mfolder,
                    'error': 'No mask file found'
                })
                continue
            
            # Compute new volume
            mm_per_pixel = morph_data.get('mm_per_pixel', 1.0)
            volume_result = compute_volume(
                calib_result['calibrated_depth'],
                mask,
                mm_per_pixel,
                calib_result['plane_depth_mm']
            )
            
            if not volume_result['success']:
                result['failed'].append({
                    'path': mfolder,
                    'error': volume_result.get('error', 'Volume computation failed')
                })
                continue
            
            # Update morphometrics.json
            old_volume = vol_est.get('volume_mm3', 0)
            new_volume = volume_result['volume_mm3']
            
            morph_data['volume_estimation'] = {
                'volume_mm3': new_volume,
                'volume_ml': new_volume / 1000.0,
                'mean_elevation_mm': volume_result.get('mean_elevation_mm'),
                'max_elevation_mm': volume_result.get('max_elevation_mm'),
                'surface_area_mm2': volume_result.get('surface_area_mm2'),
                'depth_model': vol_est.get('depth_model', 'vitb'),
                'calibration_status': 'reliable',
                'calibration_camera_id': camera_id,
                'calibration_version': intrinsics.get('version'),
                'calibration_method': 'intrinsics',
                'calibration_constant_k': calib_result['fit_params'][0] if calib_result.get('fit_params') else None,
                'calibration_rmse_mm': calib_result.get('fit_residual'),
                'reference_plane_depth_mm': calib_result.get('plane_depth_mm'),
                'raw_depth_file': 'raw_depth.npy',
                'device_used': vol_est.get('device_used'),
                'computed_at': datetime.datetime.now().isoformat(),
                'recomputed_from_version': vol_est.get('calibration_version'),
                'previous_volume_mm3': old_volume
            }
            
            # Save updated morphometrics.json
            with open(morph_path, 'w') as f:
                json.dump(morph_data, f, indent=4)
            
            # Save updated elevation map and visualization
            save_depth_data(
                mfolder,
                calib_result['calibrated_depth'],
                volume_result['elevation_map'],
                volume_result,
                calib_result,
                mask=mask
            )
            
            result['recomputed'].append({
                'path': mfolder,
                'old_volume_mm3': old_volume,
                'new_volume_mm3': new_volume,
                'change_percent': ((new_volume - old_volume) / old_volume * 100) if old_volume > 0 else None
            })
            
            logging.info(f"Recomputed {mfolder}: {old_volume:.1f} -> {new_volume:.1f} mm³")
            
        except Exception as e:
            result['failed'].append({
                'path': mfolder,
                'error': str(e)
            })
            logging.exception(f"Error recomputing {mfolder}")
    
    logging.info(f"Recomputation complete: {len(result['recomputed'])} recomputed, "
                 f"{len(result['skipped'])} skipped, {len(result['failed'])} failed")
    
    return result

