import numpy as np
from datetime import datetime, timedelta


class CyclicEncoder:
    """
    Encodes scalar values on a circular domain into 2D vectors using sine and cosine.

    For a given period, maps any value v to (sin(2π * (v mod period) / period),
    cos(2π * (v mod period) / period)).
    """
    def __init__(self, period: float):
        if period <= 0:
            raise ValueError("period must be positive")
        self.period = float(period)

    def encode(self, value: float) -> np.ndarray:
        """
        Encode the given numeric value into a 2D cyclic representation.

        :param value: The scalar value to encode (can be outside [0, period]).
        :return: A numpy array of shape (2,) with [sin(angle), cos(angle)].
        """
        angle = 2 * np.pi * ((value % self.period) / self.period)
        return np.array([np.sin(angle), np.cos(angle)], dtype=float)


class TimestampEncoder:
    """
    Encodes datetime objects into a fixed-length numeric vector capturing both global
    progression (linear and logarithmic) and periodic components at various granularities.
    """
    def __init__(self,
                 base_datetime: datetime = datetime(2021, 1, 1),
                 max_span_seconds: float = 5 * 356 * 24 * 60 * 60,
                 num_log_freqs: int = 10):
        """
        :param base_datetime: Reference epoch for encoding deltas.
        :param max_span_seconds: Duration (in seconds) corresponding to normalized value 1.0.
        :param num_log_freqs: Number of log-spaced frequencies for multi-scale periodic encoding.
        """
        if max_span_seconds <= 0:
            raise ValueError("max_span_seconds must be positive")
        self.base_datetime = base_datetime
        self.max_span_seconds = float(max_span_seconds)
        self.base_seconds = (self.base_datetime - datetime(2021, 1, 1)).total_seconds()/self.max_span_seconds
        self.base_timestamp = self.base_datetime.timestamp()
        # Prepare log-spaced frequencies between 1s and max_span_seconds
        self.frequencies = np.logspace(0,
                                       np.log10(self.max_span_seconds),
                                       num=num_log_freqs,
                                       base=10.0)

    def encode(self, dt: datetime) -> np.ndarray:
        """
        Convert a datetime into a feature vector:
        1) Linear normalized delta (0..1)
        2) Log10-normalized delta (0..1)
        3) Two-component sin/cos for second, minute, hour, weekday, day-of-year
        4) Multi-scale sin/cos for log-spaced time frequencies

        :param dt: datetime to encode
        :return: 1D numpy array of floats
        """
        # Compute delta in seconds from base
        delta_seconds = dt.timestamp() - self.base_timestamp
        # print(f"dt: {dt}")
        # print(f"delta_seconds: {delta_seconds}")
        # print(f"base_timestamp: {delta_seconds/self.max_span_seconds}")
        # Clip to [0, max_span]
        # clipped = np.clip(delta_seconds, 0.0, self.max_span_seconds)

        # 1) Linear normalization
        norm = delta_seconds / self.max_span_seconds
        # print(f"norm: {norm}, {dt}")
        # 2) Logarithmic normalization
        log_norm = norm

        # 3) Periodic features for standard time units
        sec = dt.second
        minute = dt.minute
        hour = dt.hour
        weekday = dt.weekday()  # Monday=0
        day_of_year = dt.timetuple().tm_yday

        periodic = []
        for unit, period in [
            (sec, 60),
            (minute, 60),
            (hour, 24),
            (weekday, 7),
            (day_of_year, 365)
        ]:
            angle = 2 * np.pi * unit / period
            periodic.append(np.sin(angle))
            periodic.append(np.cos(angle))

        # 4) Multi-scale periodic features
        multi_scale = []
        for freq in self.frequencies:
            angle = delta_seconds / freq  # use clipped to avoid negative
            multi_scale.append(np.sin(angle))
            multi_scale.append(np.cos(angle))

        # Concatenate all features into one vector
        features = np.concatenate(
            ([norm, log_norm], periodic, multi_scale)
        )
        return features


class TimestampDecoder:
    """
    Decodes a normalized scalar back into a datetime by linear mapping.
    """
    def __init__(self,
                 base_datetime: datetime = datetime(2021, 1, 1),
                 max_span_seconds: float = 5 * 356 * 24 * 60 * 60):
        if max_span_seconds <= 0:
            raise ValueError("max_span_seconds must be positive")
        self.base_datetime = base_datetime
        self.max_span_seconds = float(max_span_seconds)

    def decode(self, norm: float) -> datetime:
        """
        Convert a normalized value in [0,1] back to a datetime.

        :param norm: Normalized scalar (will be clipped to [0,1])
        :return: Decoded datetime
        """
        # Clip input
        clipped = float(np.clip(norm, 0.0, 1.0))
        # Map back to seconds and add to base
        delta = clipped * self.max_span_seconds
        return self.base_datetime + timedelta(seconds=delta)
