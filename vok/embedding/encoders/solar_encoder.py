import torch
import numpy as np
from datetime import datetime

from dawn_vok.vok.embedding.encoders.linear_encoder import LinearEncoder

# --- Event Dynamics Encoder ---
class SolarEncoder:
    """
    Encodes sunrise/sunset dynamics and related daylength features.
    """
    def __init__(self, reference_time: datetime):
        self.ref_time = reference_time

    @staticmethod
    def _norm_time(dt: datetime, ref: datetime) -> float:
        seconds = (dt - ref).total_seconds()
        # Normalize over a 50-year range to [-1, 1]
        return LinearEncoder.normalize(seconds, 0.0, 86400 * 365 * 50)

    def encode(self,
               sunrise: datetime,
               sunset: datetime,
               prev_daylength: float,
               next_daylength: float,
               daylength: float,
               day_of_year: int,
               dim: int) -> torch.Tensor:
        feats = []
        # 1-2: normalized timestamps
        feats.append(self._norm_time(sunrise, self.ref_time))
        feats.append(self._norm_time(sunset, self.ref_time))
        # 3: normalized daylength
        feats.append(LinearEncoder.normalize(daylength, 0.0, 24.0))
        # 4-5: derivatives
        feats.append((daylength - prev_daylength) / 24.0)
        feats.append((next_daylength - daylength) / 24.0)
        # 6: daylight ratio
        feats.append(daylength / 24.0)

        # Extended dimensions
        noon = datetime.combine(sunrise.date(), datetime.min.time()).replace(hour=12)
        if dim >= 7:
            feats.append((noon - sunrise).total_seconds() / 3600.0 / 12.0)
        if dim >= 8:
            feats.append((sunset - noon).total_seconds() / 3600.0 / 12.0)
        if dim >= 9:
            midpoint = (sunset - sunrise).total_seconds() / 2 + (sunrise - noon).total_seconds()
            feats.append(midpoint / 3600.0 / 12.0)
        if dim >= 10:
            sunrise_off = (noon - sunrise).total_seconds() / 3600.0
            sunset_off  = (sunset - noon).total_seconds() / 3600.0
            feats.append((sunset_off - sunrise_off) / 12.0)
        if dim >= 11:
            feats.append(((next_daylength - prev_daylength) / 2) / 24.0)
        if dim >= 12:
            std_win = np.std([prev_daylength, daylength, next_daylength])
            feats.append(std_win / 24.0)
        if dim >= 13:
            feats.append(np.cos(2 * np.pi * day_of_year / 365.0))
        if dim >= 14:
            feats.append(np.sin(2 * np.pi * day_of_year / 365.0))
        if dim >= 15:
            feats.append(np.log1p(daylength))
        if dim >= 16:
            feats.append(daylength / 24.0)

        return torch.tensor(feats[:dim], dtype=torch.float32)


# Example usage:
if __name__ == "__main__":

    ref = datetime(2000,1,1)
    ede = SolarEncoder(ref)
    example = ede.encode(
        sunrise=datetime(2023,6,21,4,0),
        sunset=datetime(2023,6,21,18,0),
        prev_daylength=14.0,
        next_daylength=14.2,
        daylength=14.1,
        day_of_year=172,
        dim=16
    )
    print("EventDynamicsEncoder:", example)
