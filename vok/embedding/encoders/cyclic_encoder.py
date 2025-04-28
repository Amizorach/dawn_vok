import torch
import numpy as np
from datetime import datetime

# --- Cyclic Feature Encoder ---
class CyclicEncoder:
    """
    Encodes cyclic (angle-based) features using harmonics.
    """
    @staticmethod
    def encode(value: float, dim: int) -> torch.Tensor:
        assert dim >= 2, "Cyclic encoding requires at least 2 dimensions"
        radians = np.radians(value)
        embedding = []
        num_harmonics = dim // 2
        for n in range(1, num_harmonics + 1):
            embedding.append(np.sin(n * radians))
            embedding.append(np.cos(n * radians))
        return torch.tensor(embedding[:dim], dtype=torch.float32)

# Example usage:
if __name__ == "__main__":
    # Cyclic
    print("CyclicEncoder:", CyclicEncoder.encode(180.0, 4))
    # Linear
   