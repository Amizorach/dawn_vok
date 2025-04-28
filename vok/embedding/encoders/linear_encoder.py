import torch
import numpy as np
from datetime import datetime



# --- Linear Feature Encoder ---
class LinearEncoder:
    """
    Encodes linear (scalar) features into multiple views:
    raw, normalized, log, z-score, etc.
    """
    @staticmethod
    def normalize(value: float, min_val: float, max_val: float) -> float:
        midpoint = (min_val + max_val) / 2
        half_range = (max_val - min_val) / 2
        return (value - midpoint) / half_range

    @staticmethod
    def encode(value: float, dim: int, normalize_cfg: dict = None) -> torch.Tensor:
        views = [value]
        if dim >= 2 and normalize_cfg:
            views.append(LinearEncoder.normalize(value, normalize_cfg['min'], normalize_cfg['max']))
        if dim >= 3:
            views.append(np.log1p(abs(value)) * np.sign(value))
        if dim >= 4 and normalize_cfg:
            mean = normalize_cfg.get('mean', 0.0)
            std = normalize_cfg.get('std', 1.0)
            z = (value - mean) / std if std > 0 else 0.0
            views.append(z)
            views.append(mean)
            views.append(std)
            views.append(np.log1p(abs(value)) * np.sign(value))
            views.append(np.log10(abs(value)) * np.sign(value))
        return torch.tensor(views[:dim], dtype=torch.float32)


class RangeEncoder:
    """
    Encodes a range of values into multiple views:
    raw, normalized, log, z-score, etc.
    """
    

# Example usage:
if __name__ == "__main__":
    # Cyclic
    # Linear
    print("LinearEncoder:", LinearEncoder.encode(12.5, 6, {'min':0.0,'max':24.0,'mean':12.0,'std':2.5}))
    # Event Dynamics
    