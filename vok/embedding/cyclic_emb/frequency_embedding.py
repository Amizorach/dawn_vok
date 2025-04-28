import numpy as np

class FrequencyEmbedding:
    def __init__(self, base_unit: float = 60.0):
        self.base_unit = base_unit

    def encode(self, seconds: float, dim_size: int = 64) -> np.ndarray:
        seconds=float(seconds)
        seconds = max(seconds, 1.0)
        raw = np.clip(seconds / 86400.0, 0.0, 1.0)
        log_f = np.log10(seconds)
        log_ratio = np.log10(seconds / self.base_unit)
        bucket = np.clip(int(np.log2(seconds)), 0, 31) / 31.0

        sincos = []
        for f in [10, 100]:
            sincos.append(np.sin(seconds / f))
            sincos.append(np.cos(seconds / f))

        ret = np.array([raw, log_f / 10, log_ratio / 10, bucket] + sincos)
        if ret.shape[0] < dim_size:
            ret = np.concatenate([ret, np.zeros(dim_size - ret.shape[0])])
        return ret

    def decode(self, embedding: np.ndarray) -> float:
        raw = embedding[0]
        seconds = np.clip(raw, 0.0, 1.0) * 86400.0
        return seconds


if __name__ == "__main__":
    fe = FrequencyEmbedding()
    print(fe.encode(120))
    print(fe.decode(fe.encode(120)))
