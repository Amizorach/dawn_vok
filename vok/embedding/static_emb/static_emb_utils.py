import hashlib
import numpy as np
from datetime import datetime



class StaticEmbeddingUtils:
    sensor_dim = 16
    source_dim = 16
    time_dim = 32
    freq_dim = 4
    cmd_dim = 4
    # def __init__(self):
    #     self.sensor_dim = 16
    #     self.source_dim = 16
    #     self.time_dim = 32
    #     self.freq_dim = 4

    @classmethod
    def _hash_to_float32(cls, input_str, dim):
        # Create a SHA256 hash and get its digest.
        hash_digest = hashlib.sha256(input_str.encode("utf-8")).digest()
        # Interpret the digest as an array of unsigned 8-bit integers.
        # We take the first 'dim' bytes.
        byte_array = np.frombuffer(hash_digest, dtype=np.uint8)[:dim]
        # Scale each byte to the range [0,1] by dividing by 255.
        float_array = byte_array.astype(np.float32) / 255.0
        return float_array


    @classmethod
    def encode_sensor_type(cls, sensor_type):
        return cls._hash_to_float32(sensor_type, cls.sensor_dim)

    @classmethod
    def encode_source_id(cls, source_id):
        return cls._hash_to_float32(source_id, cls.source_dim)

    @classmethod
    def encode_timestamp(cls, ts):
        if isinstance(ts, datetime):
            dt = ts
        else:
            dt = datetime.fromtimestamp(ts)
        hour = dt.hour + dt.minute / 60.0
        dow = dt.weekday()
        doy = dt.timetuple().tm_yday
        return np.array([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365),
            np.cos(2 * np.pi * doy / 365),
            ts / 1e9,                           # scaled timestamp
            (ts % 86400) / 86400                  # fraction of day
        ], dtype=np.float32)

    @classmethod
    def encode_duration(cls, start_ts, end_ts):
        duration = max(end_ts - start_ts, 0.0)
        return np.array([
            duration,
            np.log1p(duration),
            1.0 / duration if duration > 0 else 0.0,
            1.0 if duration == 0 else 0.0
        ], dtype=np.float32)

    @classmethod
    def encode_time_range(cls, start_ts, end_ts):
        start_enc = cls.encode_timestamp(start_ts)    # 8D
        end_enc = cls.encode_timestamp(end_ts)          # 8D
        duration_enc = cls.encode_duration(start_ts, end_ts)  # 4D
        midpoint_ts = (start_ts + end_ts) / 2
        mid_enc = cls.encode_timestamp(midpoint_ts)     # 8D
        return np.concatenate([start_enc, end_enc, duration_enc, mid_enc], dtype=np.float32)  # 32D

    @classmethod
    def encode_frequency(cls, freq_seconds):
        # Frequency is treated as a scalar interval (in seconds)
        duration = max(freq_seconds, 1e-6)  # avoid divide-by-zero
        return np.array([
            duration,
            1.0 / duration,
            np.log1p(duration),
            1.0  # bias term
        ], dtype=np.float32)

  
    @classmethod
    def encode_cmd(cls, cmd):
        return cls._hash_to_float32(cmd, cls.cmd_dim)

    @classmethod
    def encode_data_context(cls, sensor_type, source_id, start_ts, end_ts, freq_seconds):
        return np.concatenate([
            cls.encode_sensor_type(sensor_type),
            cls.encode_source_id(source_id),
            cls.encode_time_range(start_ts, end_ts),
            cls.encode_frequency(freq_seconds)
        ], dtype=np.float32)

    @classmethod
    def encode_context_cmd(cls, sensor_type, source_id, start_ts, end_ts, freq_seconds, cmd):
        return np.concatenate([
            cls.encode_data_context(sensor_type, source_id, start_ts, end_ts, freq_seconds),
            cls.encode_cmd(cmd)
        ], dtype=np.float32)
    