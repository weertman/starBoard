"""
YOLO-based preprocessing for wildlife images
Updated with instance segmentation support
Optimized with Crop-First strategy and Parallel Processing
"""
import numpy as np
from pathlib import Path
from typing import Union, Optional, List
from PIL import Image
import threading
import cv2
import concurrent.futures

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLO preprocessing unavailable.")


class YOLOPreprocessor:
    """Thread-safe YOLO preprocessor for sunflower stars with segmentation"""

    def __init__(self, model_path: str, confidence: float = 0.7, high_conf_threshold: float = 0.9):
        """
        Initialize YOLO preprocessor

        Args:
            model_path: Path to YOLO model (.pt file)
            confidence: Minimum confidence threshold for detection
            high_conf_threshold: Threshold for high-confidence detections (for size-based selection)
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics is required for YOLO preprocessing. Install with: pip install ultralytics")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at {model_path}")

        # Thread safety lock
        self._lock = threading.Lock()
        
        # Thread pool for post-processing
        # Limit workers to avoid OOM with large images
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        # Load model
        with self._lock:
            self.model = YOLO(str(self.model_path))
            self.confidence = confidence
            self.high_conf_threshold = high_conf_threshold

            # Get target class (assuming it's the first or only class)
            self.class_names = self.model.names
            print(f"YOLO model loaded with classes: {list(self.class_names.values())}")
            print(f"Detection confidence threshold: {confidence}")
            print(f"High confidence threshold for size selection: {high_conf_threshold}")

    def _process_result(self, result) -> Optional[Image.Image]:
        """
        Internal helper to process a single YOLO result
        Optimized to crop *before* masking to save memory and CPU time.
        """
        if len(result) == 0: 
            return None
        
        # Check if we have masks (instance segmentation)
        if not hasattr(result, 'masks') or result.masks is None:
            return None

        masks = result.masks
        if masks is None or (hasattr(masks, 'data') and len(masks.data) == 0):
            return None

        # Get original image from result (numpy array BGR)
        if result.orig_img is None:
            return None
        
        img_h, img_w = result.orig_img.shape[:2]

        # Get confidence scores and find best detection
        if result.boxes is not None and len(result.boxes) > 0:
            confidences = result.boxes.conf.cpu().numpy()
            
            # Filter by high confidence threshold
            high_conf_indices = np.where(confidences >= self.high_conf_threshold)[0]

            if len(high_conf_indices) > 0:
                # Among high confidence detections, select the largest mask
                # Note: This is an approximation using the mask data size, 
                # which is faster than decoding the full mask
                mask_areas = []
                for idx in high_conf_indices:
                    mask_data = masks.data[idx].cpu().numpy()
                    area = np.sum(mask_data > 0.5)
                    mask_areas.append(area)

                largest_area_idx = high_conf_indices[np.argmax(mask_areas)]
                best_idx = largest_area_idx
            else:
                # Fallback to highest confidence
                best_idx = confidences.argmax()
        else:
            best_idx = 0
            
        # === OPTIMIZATION: CROP FIRST ===
        # Instead of resizing the mask to the full 24MP image size,
        # we get the bounding box, crop the image, and then generate the mask for the crop.
        
        # Get bbox (x1, y1, x2, y2)
        box = result.boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = box.astype(int)
        
        # Add padding
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_w, x2 + padding)
        y2 = min(img_h, y2 + padding)
        
        # Crop width/height
        w = x2 - x1
        h = y2 - y1
        
        if w <= 0 or h <= 0:
            return None
            
        # Crop the original image (BGR)
        # Copying here is important to ensure contiguous memory
        crop_bgr = result.orig_img[y1:y2, x1:x2].copy()
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        
        # Get the polygon segments for the mask
        # masks.xy returns the segments in original image coordinates
        try:
            segments = masks.xy[best_idx]
        except AttributeError:
            # Fallback if xy not available (older versions)
            # This path is slower but safer
            return self._process_result_full(result, best_idx, img_w, img_h)

        if len(segments) == 0:
             return self._process_result_full(result, best_idx, img_w, img_h)

        # Shift segments to crop coordinates
        segments[:, 0] -= x1
        segments[:, 1] -= y1
        
        # Create binary mask for the crop
        mask_crop = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_crop, [segments.astype(np.int32)], 1)
        
        # Apply mask to crop
        segmented = np.zeros((h, w, 4), dtype=np.uint8)
        segmented[:, :, :3] = crop_rgb
        segmented[:, :, 3] = mask_crop * 255
        
        # Create PIL Image
        segmented_pil = Image.fromarray(segmented, mode='RGBA')
        
        # Final tight crop to mask content (removing the extra padding if it was empty)
        bbox = segmented_pil.getbbox()
        if bbox:
            segmented_pil = segmented_pil.crop(bbox)

        # Paste onto white background
        final_img = Image.new('RGB', segmented_pil.size, (255, 255, 255))
        final_img.paste(segmented_pil, mask=segmented_pil.split()[3])

        return final_img

    def _process_result_full(self, result, best_idx, img_w, img_h) -> Optional[Image.Image]:
        """Fallback method using full-image processing (slower/more memory)"""
        mask_data = result.masks.data[best_idx].cpu().numpy()
        mask_data = cv2.resize(mask_data, (img_w, img_h))
        binary_mask = (mask_data > 0.5).astype(np.uint8)
        
        img_rgb = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
        
        segmented = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        segmented[:, :, :3] = img_rgb
        segmented[:, :, 3] = binary_mask * 255
        
        segmented_pil = Image.fromarray(segmented, mode='RGBA')
        bbox = segmented_pil.getbbox()
        if bbox:
            segmented_pil = segmented_pil.crop(bbox)
            
        final_img = Image.new('RGB', segmented_pil.size, (255, 255, 255))
        final_img.paste(segmented_pil, mask=segmented_pil.split()[3])
        return final_img

    def process_image(self, image: Union[str, Image.Image]) -> Optional[Image.Image]:
        """Process single image"""
        with self._lock:
            try:
                results = self.model(image, conf=self.confidence, verbose=False)
                if len(results) == 0: return None
                return self._process_result(results[0])
            except Exception as e:
                print(f"Error: {e}")
                return None

    def process_batch(self, images: List[Union[str, Image.Image]], batch_size: int = 8) -> List[Optional[Image.Image]]:
        """
        Process multiple images with thread safety and parallel post-processing
        """
        if not isinstance(images, list): images = [images]
        results_list = []
        
        # Helper to prepare input paths
        def prepare_input(img_list):
            clean = []
            for i in img_list:
                if isinstance(i, Path): clean.append(str(i))
                else: clean.append(i)
            return clean

        for i in range(0, len(images), batch_size):
            chunk = images[i:i+batch_size]
            chunk_input = prepare_input(chunk)
            
            # 1. GPU Inference (Sequential / Batched)
            try:
                with self._lock:
                    batch_results = self.model(chunk_input, conf=self.confidence, verbose=False, stream=False)
            except Exception as e:
                print(f"Batch inference error: {e}")
                results_list.extend([None] * len(chunk))
                continue

            # 2. CPU Post-processing (Parallel)
            # Map the _process_result function over the results
            try:
                processed_chunk = list(self.executor.map(self._process_result, batch_results))
                results_list.extend(processed_chunk)
            except Exception as e:
                print(f"Batch processing error: {e}")
                results_list.extend([None] * len(chunk))

        return results_list
