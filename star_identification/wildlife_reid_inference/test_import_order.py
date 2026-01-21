"""
Test script to diagnose why training works but GUI fails
"""
import sys
from pathlib import Path

print("Python version:", sys.version)
print("\nTest 1: Import transformers directly (like training)")
try:
    from transformers import SwinModel
    print("✓ Success: transformers imported directly")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

print("\nTest 2: Import after PySide6 (like GUI)")
try:
    from PySide6.QtWidgets import QApplication
    from transformers import SwinModel
    print("✓ Success: transformers imported after PySide6")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

print("\nTest 3: Check torch._dynamo directly")
try:
    import torch._dynamo
    print("✓ Success: torch._dynamo imported")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

print("\nTest 4: Import your actual model")
sys.path.append(str(Path(__file__).parent.parent))
try:
    from src.models import create_model
    print("✓ Success: Your model imported")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")

print("\nTest 5: Check torch compile settings")
import torch
print(f"Torch version: {torch.__version__}")
print(f"Dynamo enabled: {torch._dynamo.config.enabled}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Check environment variables that might affect dynamo
import os
relevant_env_vars = [
    'PYTORCH_DISABLE_LIBRARY_INIT',
    'TORCH_COMPILE_DEBUG',
    'TORCHDYNAMO_DISABLE',
    'PYTORCH_SKIP_LIBRARY_CHECK'
]
print("\nRelevant environment variables:")
for var in relevant_env_vars:
    value = os.environ.get(var, "Not set")
    print(f"  {var}: {value}")