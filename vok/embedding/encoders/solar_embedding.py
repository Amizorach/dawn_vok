# solar_time_encoder.py

from pprint import pprint
import torch
from datetime import datetime

from dawn_vok.vok.calculators.solar_calculator import SolarCalculator
from dawn_vok.vok.embedding.encoders.linear_encoder import LinearEncoder
from dawn_vok.vok.embedding.encoders.cyclic_encoder import CyclicEncoder
from dawn_vok.vok.embedding.encoders.solar_encoder import SolarEncoder


class SolarTimeEncoder:
    """
    Encodes astronomical time features plus raw decode fields into a fixed-size vector.
    The first three dimensions are reserved for decoding:
      [ timestamp/10, latitude/100, longitude/100 ]
    Supports 32- and 64-dim core embeddings via built-in feature maps.
    """
# {'daylength_hours': -14.408828258333333,
# {'day_of_year': 114,
#  'daylength_hours': -10.479612115833334,
#  'equation_of_time_min': np.float64(-3.2149689531731696),
#  'julian_day': np.float64(2460789.5101062963),
#  'latitude': 37.7749,
#  'local_solar_time': np.float64(16.11562281588622),
#  'longitude': -122.4194,
#  'moon_azimuth_deg': np.float64(275.43127046217967),
#  'moon_elevation_deg': np.float64(-22.028206404688206),
#  'moon_phase_angle_deg': np.float64(52.90958419365518),
#  'solar_azimuth_deg': np.float64(262.69138109181534),
#  'solar_elevation_deg': np.float64(30.360388278391664),
#  'sunrise_utc': '13:21',
#  'sunset_utc': '02:53',
#  'timestamp': 1745442804.31609}
   
    _default_maps = {
        32: {
            'timestamp':             {'type': 'linear',        'dim': 1},
            'latitude':              {'type': 'linear',        'dim': 1},
            'longitude':             {'type': 'linear',        'dim':1},
            'day_of_year':           {'type': 'linear',        'dim': 1,
                                      'normalize': {'min': 1, 'max': 365}},
            'solar_elevation_deg':   {'type': 'cyclic',        'dim': 4},
            'solar_azimuth_deg':     {'type': 'cyclic',        'dim': 2},
            'local_solar_time':      {'type': 'cyclic',        'dim': 4},
            'moon_phase_angle_deg':  {'type': 'cyclic',        'dim': 2},
           

            'daylength_hours':       {'type': 'linear',        'dim': 3,
                                      'normalize': {'min': 0, 'max': 24, 'mean': 12, 'std': 2.5}},
            'equation_of_time_min':  {'type': 'linear',        'dim': 2,
                                      'normalize': {'min': -16, 'max': 16}},
            'moon_elevation_deg':    {'type': 'linear',        'dim': 2,
                                      'normalize': {'min': -90, 'max': 90}},
            'moon_azimuth_deg':      {'type': 'linear',        'dim': 1,
                                      'normalize': {'min': 0, 'max': 360}},
            'julian_day':            {'type': 'linear',        'dim': 2,
                                      'normalize': {'min': 2451545, 'max': 2461545}},

            'solar_dynamics':   {'type': 'solar_dynamics', 'dim': 12,
                                      'params': {'reference_time': datetime(2000,1,1)}},
        },
        64: {
            'solar_elevation_deg':   {'type': 'cyclic',        'dim': 6},
            'solar_azimuth_deg':     {'type': 'cyclic',        'dim': 4},
            'local_solar_time':      {'type': 'cyclic',        'dim': 6},
            'moon_phase_angle_deg':  {'type': 'cyclic',        'dim': 4},

            'daylength_hours':       {'type': 'linear',        'dim': 4,
                                      'normalize': {'min': 0, 'max': 24, 'mean': 12, 'std': 2.5}},
            'equation_of_time_min':  {'type': 'linear',        'dim': 3,
                                      'normalize': {'min': -16, 'max': 16}},
            'moon_elevation_deg':    {'type': 'linear',        'dim': 3,
                                      'normalize': {'min': -90, 'max': 90}},
            'moon_azimuth_deg':      {'type': 'linear',        'dim': 2,
                                      'normalize': {'min': 0, 'max': 360}},
            'julian_day':            {'type': 'linear',        'dim': 3,
                                      'normalize': {'min': 2451545, 'max': 2461545}},
            'solar_declination':     {'type': 'linear',        'dim': 2,
                                      'normalize': {'min': -23.44, 'max': 23.44}},

            'solar_dynamics':   {'type': 'solar_dynamics', 'dim': 24,
                                      'params': {'reference_time': datetime(2000,1,1)}},
        }
    }

    def __init__(self, core_dim: int = 32, feature_map: dict = None):
        # core_dim = 32 or 64 for the astronomical features
        if feature_map is not None:
            self.feature_map = feature_map
        else:
            if core_dim not in self._default_maps:
                raise ValueError("Only 32 or 64 core dimensions supported")
            self.feature_map = self._default_maps[core_dim]

        self.core_dim = core_dim
        self.solar_calc = SolarCalculator()

        # Prepare EventDynamicsEncoder if needed
        edm_cfg = next((cfg for cfg in self.feature_map.values() if cfg['type']=='solar'), None)
        ref = None
        if edm_cfg:
            ref = edm_cfg['params']['reference_time']
        self.solar_encoder = SolarEncoder(ref)

    def encode(self, dt: datetime, lat: float, lon: float) -> torch.Tensor:
        # 1) Decoding slice: raw inputs normalized
        ts_norm  = dt.timestamp() / 10.0
        lat_norm = lat / 100.0
        lon_norm = lon / 100.0
        decode_slice = torch.tensor([ts_norm, lat_norm, lon_norm], dtype=torch.float32)

        # 2) Core astronomical features
        feats = self.solar_calc.load(dt, lat, lon)
        embeds = []
        pprint(feats)
        pprint(self.feature_map)
        sc = self.solar_calc.load(dt, lat, lon)
        pprint(sc)
        exit()
        for key, cfg in self.feature_map.items():
            typ = cfg['type']
            dim = cfg['dim']

            if typ == 'cyclic':
                embeds.append(CyclicEncoder.encode(feats[key], dim))
            elif typ == 'linear':
                embeds.append(LinearEncoder.encode(feats[key], dim, cfg.get('normalize')))
            elif typ == 'solar_dynamics':
                # Assumes feats includes 'sunrise_dt', 'sunset_dt', 'prev_daylength', 'next_daylength', 'day_of_year'
                embeds.append(
                    self.solar_encoder.encode(
                        sunrise      = feats['sunrise_utc'],
                        sunset       = feats['sunset_utc'],
                        # prev_daylength = feats['prev_daylength'],
                        # next_daylength = feats['next_daylength'],
                        daylength    = feats['daylength_hours'],
                        day_of_year  = feats['day_of_year'],
                        dim          = dim
                    )
                )
            else:
                raise ValueError(f"Unknown feature type '{typ}'")

        core_embedding = torch.cat(embeds)

        # 3) Final embedding: [ decode_slice | core_embedding ]
        return torch.cat([decode_slice, core_embedding])

if __name__ == "__main__":
    encoder = SolarTimeEncoder(core_dim=64)
    dt = datetime.now()
    lat = 37.7749
    lon = -122.4194
    embedding = encoder.encode(dt, lat, lon)
    print(embedding)