import torch
import numpy as np

class GeneralRangeEncoder:
    """
    Encodes two scalar points into a fixed-length feature vector up to 16 dims.
    The first two dimensions are the raw inputs, allowing exact decoding.
    """

    MAX_DIMS = 16
    MIN_DIMS = 2
    eps = 1e-6

    def __init__(self, normalize_cfg: dict = None):
        """
        normalize_cfg: dict with keys 'min', 'max' to normalize values into [-1,1].
        """
        self.norm = normalize_cfg or {}

        self.transforms = [
            # 1. raw point1
            lambda p1, p2: p1,
            # 2. raw point2
            lambda p1, p2: p2,
            # 3. normalized point1
            lambda p1, p2: p1,
            # 4. normalized point2
            lambda p1, p2: p2,
            # 5. difference
            lambda p1, p2: p2 - p1,
            # 6. scaled difference
            lambda p1, p2: (p2 - p1) / (abs(p1) + self.eps),
            # 7. ratio
            lambda p1, p2: p2 / (p1 + self.eps),
            # 8. signed log diff
            lambda p1, p2: np.log1p(abs(p2 - p1)) * np.sign(p2 - p1),
            # 9. mean of the two
            lambda p1, p2: (p1 + p2) / 2,
            # 10. absolute difference
            lambda p1, p2: abs(p2 - p1),
            # 11. normalized squared point1
            lambda p1, p2: p1**2,
            # 12. normalized squared point2
            lambda p1, p2: p2**2,
            # 13. sign of difference
            lambda p1, p2: np.sign(p2 - p1),
            # 14. signed sqrt of diff
            lambda p1, p2: np.sign(p2 - p1) * np.sqrt(abs(p2 - p1)),
            # 15. min(point1, point2)
            lambda p1, p2: min(p1, p2),
            # 16. max(point1, point2)
            lambda p1, p2: max(p1, p2),
        ]

    def _normalize(self, v: float) -> float:
        if 'min' in self.norm and 'max' in self.norm:
            lo, hi = self.norm['min'], self.norm['max']
            mid = (lo + hi) / 2
            half = (hi - lo) / 2
            return (v - mid) / (half if half != 0 else 1)
        elif 'div' in self.norm:
            return v / self.norm['div']
        elif 'mul' in self.norm:
            return v * self.norm['mul']
        return v

    def encode(self, point1: float, point2: float, dim: int) -> torch.Tensor:
        """
        Returns a vector of length `dim` with the first `dim` transform outputs.
        """
        if not (self.MIN_DIMS <= dim <= self.MAX_DIMS):
            raise ValueError(f"dim must be between {self.MIN_DIMS} and {self.MAX_DIMS}")
        point1 = self._normalize(point1)
        point2 = self._normalize(point2)
        feats = [fn(point1, point2) for fn in self.transforms[:dim]]
        return torch.tensor(feats, dtype=torch.float32)

    def decode(self, embedding: torch.Tensor) -> (float, float):
        """
        Restores the original point1 and point2 from the first two dimensions
        of the embedding.
        """
        if embedding.ndim != 1 or embedding.shape[0] < 2:
            raise ValueError("Embedding must be a 1D tensor with at least 2 dimensions")
        point1 = embedding[0].item()
        point2 = embedding[1].item()
        return point1, point2


# Example usage:
if __name__ == "__main__":
    cfg = {'min': 0.0, 'max': 100.0}
    encoder = GeneralRangeEncoder(normalize_cfg=cfg)
    emb = encoder.encode(10.0, 30.0, dim=16)
    p1, p2 = encoder.decode(emb)
    print("Embedding:", emb)
    print("Decoded:", p1, p2)
