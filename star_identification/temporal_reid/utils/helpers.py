"""Helper utilities."""
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = 'cuda') -> torch.device:
    """Get torch device, falling back to CPU if CUDA unavailable."""
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')
    return torch.device(device_str)


def count_parameters(model: torch.nn.Module) -> tuple:
    """Count model parameters.
    
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable



