"""
Depth estimation handler using Depth-Anything-V2
Designed for robustness: graceful fallbacks, OOM protection, cross-platform support
"""

import logging
import os
import sys
import gc
import numpy as np

# Global model cache to avoid reloading
_depth_model = None
_model_config = None


def get_device():
    """
    Detect the best available device with fallback chain.
    Returns tuple of (device_string, device_name_for_display)
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Check CUDA memory
            try:
                torch.cuda.empty_cache()
                free_memory = torch.cuda.get_device_properties(0).total_memory
                if free_memory > 1e9:  # At least 1GB
                    return 'cuda', f'CUDA ({torch.cuda.get_device_name(0)})'
            except Exception as e:
                logging.warning(f"CUDA available but error checking memory: {e}")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps', 'Apple MPS'
        
        return 'cpu', 'CPU'
    except ImportError:
        return 'cpu', 'CPU (torch not available)'


def clear_model_cache():
    """Release the cached model to free memory."""
    global _depth_model, _model_config
    
    if _depth_model is not None:
        del _depth_model
        _depth_model = None
        _model_config = None
        
        # Force garbage collection
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        logging.info("Depth model cache cleared")


def load_depth_model(encoder='vitb', checkpoint_path=None):
    """
    Load Depth-Anything-V2 model with caching and error handling.
    
    Args:
        encoder: Model size ('vits', 'vitb', 'vitl')
        checkpoint_path: Path to model weights. If None, searches default locations.
    
    Returns:
        tuple: (model, device) or (None, error_message)
    """
    global _depth_model, _model_config
    
    # Return cached model if same config
    if _depth_model is not None and _model_config == encoder:
        device, _ = get_device()
        return _depth_model, device
    
    # Clear any existing model first
    clear_model_cache()
    
    try:
        import torch
        
        # Add Depth-Anything-V2 to path
        # Try multiple possible locations
        possible_da_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Depth-Anything-V2'),
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'Depth-Anything-V2'),
            os.path.join(os.getcwd(), 'Depth-Anything-V2'),
        ]
        
        da_path = None
        for path in possible_da_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                da_path = abs_path
                break
        
        if da_path is None:
            return None, "Depth-Anything-V2 directory not found. Please ensure it exists in the project root."
        
        if da_path not in sys.path:
            sys.path.insert(0, da_path)
        
        from depth_anything_v2.dpt import DepthAnythingV2
        
        device, device_name = get_device()
        logging.info(f"Loading depth model on {device_name}")
        
        # Model configurations
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        if encoder not in model_configs:
            return None, f"Unknown encoder: {encoder}. Use 'vits', 'vitb', or 'vitl'"
        
        # Find checkpoint
        if checkpoint_path is None:
            # Search default locations
            search_paths = [
                os.path.join(da_path, 'checkpoints', f'depth_anything_v2_{encoder}.pth'),
                os.path.join(os.path.dirname(__file__), 'checkpoints', f'depth_anything_v2_{encoder}.pth'),
                os.path.join(os.getcwd(), 'checkpoints', f'depth_anything_v2_{encoder}.pth'),
            ]
            for path in search_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    break
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            return None, f"Checkpoint not found for encoder '{encoder}'. Please download from HuggingFace and place in Depth-Anything-V2/checkpoints/"
        
        # Load model
        model = DepthAnythingV2(**model_configs[encoder])
        
        # Load weights with map_location for cross-platform compatibility
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        # Cache the model
        _depth_model = model
        _model_config = encoder
        
        logging.info(f"Depth model loaded successfully: {encoder} on {device_name}")
        return model, device
        
    except ImportError as e:
        return None, f"Missing dependency: {e}. Ensure torch and depth_anything_v2 are installed."
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            clear_model_cache()
            return None, "Out of memory. Try a smaller model (vits) or close other applications."
        return None, f"Runtime error loading model: {e}"
    except Exception as e:
        logging.exception("Error loading depth model")
        return None, f"Error loading depth model: {e}"


def estimate_depth(image, encoder='vitb', input_size=518):
    """
    Run depth estimation on an image with full error handling.
    
    Args:
        image: BGR image as numpy array (H, W, 3)
        encoder: Model size to use
        input_size: Input resolution for model (higher = more detail, more memory)
    
    Returns:
        dict: {
            'success': bool,
            'depth_map': numpy array (H, W) of depth values, or None
            'error': error message if failed, or None
            'device': device used for inference
        }
    """
    result = {
        'success': False,
        'depth_map': None,
        'error': None,
        'device': None
    }
    
    try:
        import torch
        
        # Validate input
        if image is None or image.size == 0:
            result['error'] = "Invalid input image"
            return result
        
        # Load model
        model, device = load_depth_model(encoder)
        if model is None:
            result['error'] = device  # device contains error message when model is None
            return result
        
        result['device'] = device
        
        # Run inference with memory protection
        try:
            with torch.no_grad():
                # Reduce input size if on CPU to prevent slowness
                actual_input_size = input_size
                if device == 'cpu' and input_size > 392:
                    logging.info("Reducing input size for CPU inference")
                    actual_input_size = 392
                
                depth = model.infer_image(image, actual_input_size)
                
                # Ensure we return numpy array on CPU
                if isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
                
                result['depth_map'] = depth.astype(np.float32)
                result['success'] = True
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Try with smaller input size
                clear_model_cache()
                gc.collect()
                
                if device != 'cpu':
                    logging.warning("OOM on GPU, retrying with smaller input on CPU")
                    try:
                        import torch
                        # Reload model on CPU
                        model, _ = load_depth_model(encoder)
                        if model is not None:
                            model = model.to('cpu')
                            with torch.no_grad():
                                depth = model.infer_image(image, 392)
                                if isinstance(depth, torch.Tensor):
                                    depth = depth.cpu().numpy()
                                result['depth_map'] = depth.astype(np.float32)
                                result['success'] = True
                                result['device'] = 'cpu (fallback)'
                    except Exception as fallback_error:
                        result['error'] = f"Out of memory, fallback also failed: {fallback_error}"
                else:
                    result['error'] = "Out of memory on CPU. Try reducing input_size or image resolution."
            else:
                result['error'] = f"Inference error: {e}"
                
    except ImportError as e:
        result['error'] = f"Missing dependency: {e}"
    except Exception as e:
        logging.exception("Unexpected error in depth estimation")
        result['error'] = f"Unexpected error: {e}"
    
    return result


def get_available_encoders():
    """Return list of available encoder options with descriptions."""
    return [
        ('vits', 'Small (24.8M params) - Fastest, lowest memory'),
        ('vitb', 'Base (97.5M params) - Balanced'),
        ('vitl', 'Large (335.3M params) - Best quality, high memory'),
    ]


