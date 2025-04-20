import torch

import torch
import numpy as np

print(f"PyTorch: {torch.__version__}")  # Should be ≥2.3.0
print(f"NumPy: {np.__version__}")       # Should be 2.2.4
print(f"CUDA: {torch.version.cuda}")    # Should match your system CUDA
print(f"CUDA Available: {torch.cuda.is_available()}")  # Should be True