# solar_calculator.py

import math
from pprint import pprint
from skyfield.api import load as skyfield_load, Topos
from skyfield import almanac
from datetime import datetime, timedelta
import numpy as np

class SolarCalculator:
    """
    Computes astronomical time features for a given datetime and location.
    """

    def __init__(self):
        # Load timescale and ephemeris once
        self.ts = skyfield_load.timescale()
        self.eph = skyfield_load('de421.bsp')  # JPL DE421 ephemeris
        self.sun = self.eph['sun']
        self.moon = self.eph['moon']
        self.earth = self.eph['earth']

    def get_daylength(self, date, f_sun):
        t0 = self.ts.utc(date.year, date.month, date.day)
        t1 = self.ts.utc(date.year, date.month, date.day + 1)
        times, _ = almanac.find_discrete(t0, t1, f_sun)
        sunrise = times[0].utc_datetime()
        sunset  = times[1].utc_datetime()
        dl_secs = (sunset - sunrise).total_seconds()
        return {
            'dl_secs': dl_secs,
            'sunrise_min': sunrise.hour * 60 + sunrise.minute,
            'sunset_min' : sunset.hour  * 60 + sunset.minute
        }

    def calc_daylength(self, date, f_sun,
                       prev=True, next=True, trend=7,
                       climatology_dl_by_doy: dict = None):
        """
        Returns a dict of features for `date`:
          - raw daylength & times
          - cyclical encodings for DOY and DOW
          - pct changes vs. prev/next day
          - slope over ±trend days
          - anomaly vs. climatology if provided
        """
        # 1. Base metrics for today
        base = self.get_daylength(date, f_sun)
        feats = {
            'dl_secs': base['dl_secs'],
            'sunrise_min': base['sunrise_min'],
            'sunset_min' : base['sunset_min']
        }

        # 2. Cyclical date features
        doy = date.timetuple().tm_yday
        dow = date.weekday()
        feats['sin_doy'] = math.sin(2*math.pi * doy/365)
        feats['cos_doy'] = math.cos(2*math.pi * doy/365)
        # feats['sin_dow'] = math.sin(2*math.pi * dow/7)
        # feats['cos_dow'] = math.cos(2*math.pi * dow/7)

        # 3. Percent changes vs. yesterday/tomorrow
        if prev:
            prev = self.get_daylength(date - timedelta(days=1), f_sun)
            feats['pct_dl_vs_prev']      = (base['dl_secs'] - prev['dl_secs']) / prev['dl_secs']
            feats['pct_sunrise_vs_prev'] = (base['sunrise_min'] - prev['sunrise_min']) / prev['sunrise_min']
            feats['pct_sunset_vs_prev']  = (base['sunset_min']  - prev['sunset_min']) / prev['sunset_min']
        if next:
            nxt = self.get_daylength(date + timedelta(days=1), f_sun)
            feats['pct_dl_vs_next']      = (nxt['dl_secs'] - base['dl_secs']) / base['dl_secs']
            feats['pct_sunrise_vs_next'] = (nxt['sunrise_min'] - base['sunrise_min']) / base['sunrise_min']
            feats['pct_sunset_vs_next']  = (nxt['sunset_min']  - base['sunset_min']) / base['sunset_min']

        # 4. Trend slope over ±trend days
        if trend > 0:
            prev_t = self.get_daylength(date - timedelta(days=trend), f_sun)
            next_t = self.get_daylength(date + timedelta(days=trend), f_sun)
            # per-day slope
            feats['slope_dl_secs']      = (next_t['dl_secs']      - prev_t['dl_secs'])      / (2 * trend)
            feats['slope_sunrise_min']  = (next_t['sunrise_min']  - prev_t['sunrise_min'])  / (2 * trend)
            feats['slope_sunset_min']   = (next_t['sunset_min']   - prev_t['sunset_min'])   / (2 * trend)

        # 5. Climatology anomaly (if provided)
        if climatology_dl_by_doy is not None:
            climo = climatology_dl_by_doy.get(doy)
            if climo is not None:
                feats['dl_anomaly_secs'] = base['dl_secs'] - climo
        pprint(feats)
        feats = self.normalize_features(feats)
        pprint(feats)
        return feats
    @staticmethod
    def make_scaler(divisor: float):
        return lambda x: x / divisor
    
    def normalize_features(self, feats) -> dict:
        """
        Normalize daylength-related features into roughly [-1,1] ranges,
        using domain-informed bounds:
        - dl_secs:    fraction of 24h (0–86400s)
        - sunrise/sunset: fraction of 24h (0–1440min)
        - pct_changes: typical daily % changes ~±10%
        - slopes:     typical daily slopes ~±300s/day for dl, ±10min/day for times
        - sin/cos:    already in [-1,1]
        """
        # expected maximum absolute values
        norm_map = {
    # fraction of full day
            'dl_secs':      SolarCalculator.make_scaler(86400.0),
            'sunrise_min':  SolarCalculator.make_scaler(1440.0),
            'sunset_min':   SolarCalculator.make_scaler(1440.0),
            # percent changes (assume ±10% max)
            'pct_dl_vs_prev':      SolarCalculator.make_scaler(0.10),
            'pct_dl_vs_next':      SolarCalculator.make_scaler(0.10),
            'pct_sunrise_vs_prev': SolarCalculator.make_scaler(0.10),
            'pct_sunrise_vs_next': SolarCalculator.make_scaler(0.10),
            'pct_sunset_vs_prev':  SolarCalculator.make_scaler(0.10),
            'pct_sunset_vs_next':  SolarCalculator.make_scaler(0.10),
            # slopes (dl in secs/day, times in min/day)
            'slope_dl_secs':      SolarCalculator.make_scaler(300.0),
            'slope_sunrise_min':  SolarCalculator.make_scaler(10.0),
            'slope_sunset_min':   SolarCalculator.make_scaler(10.0),
                # leave cyclical encodings untouched (identity)
                # 'sin_doy': lambda x: x,
                # 'cos_doy': lambda x: x,
        }
       

        out = {}
        for key, value in feats.items():
            if key in norm_map:
                out[key] = norm_map[key](value)
            else:
                out[key] = value
        
        pprint(out)
       

        return out

    
    def get_daylength1(self, date, f_sun,):
        t0 = self.ts.utc(date.year, date.month, date.day)
        t1 = self.ts.utc(date.year, date.month, date.day + 1)
        times, events = almanac.find_discrete(t0, t1, f_sun)
        sunrise = times[0].utc_datetime()
        sunset = times[1].utc_datetime()
        print('sunrise:', sunrise)
        print('sunset:', sunset)
        ret = {}
        ret['date'] = date
        ret['daylength_seconds'] = (sunset - sunrise).total_seconds()
        ret['sunrise_minute_of_day']    = sunrise.hour * 60 + sunrise.minute
        ret['sunset_minute_of_day']     = sunset.hour * 60 + sunset.minute
        return ret
    
    def calc_daylength1(self, date, f_sun, prev=True, next=True, trend=7) -> float:
        """
        Calculate and return the daylength in hours
        """
        ret = {}
        dl = self.get_daylength(date, f_sun)
       
        prev_dl = None
        next_dl = None
        prev_trend = None
        next_trend = None
        if prev:
            prev_dl = self.get_daylength(date - timedelta(days=1), f_sun)
            ret['prev_diff_daylength_seconds'] = prev_dl['daylength_seconds'] - dl['daylength_seconds']
            ret['prev_diff_sunrise_minute_of_day'] = prev_dl['sunrise_minute_of_day'] - dl['sunrise_minute_of_day']
            ret['prev_diff_sunset_minute_of_day'] = prev_dl['sunset_minute_of_day'] - dl['sunset_minute_of_day']
        if next:
            next_dl = self.get_daylength(date + timedelta(days=1), f_sun)
            ret['next_diff_daylength_seconds'] = next_dl['daylength_seconds'] - dl['daylength_seconds']
            ret['next_diff_sunrise_minute_of_day'] = next_dl['sunrise_minute_of_day'] - dl['sunrise_minute_of_day']
            ret['next_diff_sunset_minute_of_day'] = next_dl['sunset_minute_of_day'] - dl['sunset_minute_of_day']
        if trend> 0:
            prev_trend = self.get_daylength(date - timedelta(days=trend), f_sun)
            next_trend = self.get_daylength(date + timedelta(days=trend), f_sun)
            ret[f'prev_trend_{trend}_daylength_seconds'] = prev_trend['daylength_seconds'] - dl['daylength_seconds']
            ret[f'next_trend_{trend}_daylength_seconds'] = next_trend['daylength_seconds'] - dl['daylength_seconds']
            ret[f'prev_trend_{trend}_diff_sunrise_minute_of_day'] = prev_trend['sunrise_minute_of_day'] - dl['sunrise_minute_of_day']
            ret[f'next_trend_{trend}_diff_sunrise_minute_of_day'] = next_trend['sunrise_minute_of_day'] - dl['sunrise_minute_of_day']
            ret[f'prev_trend_{trend}_diff_sunset_minute_of_day'] = prev_trend['sunset_minute_of_day'] - dl['sunset_minute_of_day']
            ret[f'next_trend_{trend}_diff_sunset_minute_of_day'] = next_trend['sunset_minute_of_day'] - dl['sunset_minute_of_day']
        print('prev_dl:', prev_dl)
        print('next_dl:', next_dl)
        print('prev_trend:', prev_trend)
        print('next_trend:', next_trend)
        ret = self.normalize_features(ret)
       
        # if prev or trend > 0:
        #     if prev or trend == 1:
        #         prev = self.calc_daylength(date - timedelta(days=1), f_sun, prev=False, next=False)
        #     else:
        #         prev = self.calc_daylength(date - timedelta(days=trend), f_sun, prev=False, next=False)
        #     ret['prev_daylength_seconds'] = prev['daylength_seconds']
        #     ret['prev_daylength_seconds'] = (prev_sunset - prev_sunrise).total_seconds()
        #     ret['prev_daylength_hours'] = ret['prev_daylength_seconds'] / 3600.0
        #     print('prev_sunrise:', prev_sunrise)
        #     print('prev_sunset:', prev_sunset)
        #     print('prev_daylength_seconds:', ret['prev_daylength_seconds'])
        #     print('prev_daylength_hours:', ret['prev_daylength_hours'])
        # if next or trend > 0:
        #     t0 = self.ts.utc(date.year, date.month, date.day+trend)
        #     t1 = self.ts.utc(date.year, date.month, date.day+2*trend)
        # print(ret)
        return ret

    def load(self, dt: datetime, lat: float, lon: float) -> dict:
        """
        Calculate and return a dictionary of astronomical features:
          -timestamp
          - latitude
          - longitude
          - solar_elevation_deg
          - solar_azimuth_deg
          - julian_day
          - equation_of_time_min
          - local_solar_time
          - daylength_hours, sunrise_utc, sunset_utc
          - moon_elevation_deg
          - moon_azimuth_deg
          - moon_phase_angle_deg
        """
        result = {}

        # 1) Prepare location and time
        location = Topos(latitude_degrees=lat, longitude_degrees=lon)
        observer = self.earth + location
        t = self.ts.utc(dt.year, dt.month, dt.day,
                        dt.hour, dt.minute, dt.second)

        # 2) Compute Local True Solar Time (LTST)
        astrometric = observer.at(t).observe(self.sun).apparent()
        ra_sun, _, _ = astrometric.radec()
        gast = t.gast                                  # Greenwich Apparent Sidereal Time (hours)
        last = (gast + lon/15.0) % 24                  # Local Apparent Sidereal Time
        sun_ha = (last * 15 - ra_sun.degrees) % 360    # Hour angle (deg)
        ltst = ((sun_ha/15) + 12) % 24                 # True solar time (hours)
        result['local_solar_time'] = ltst

        # 3) Compute Equation of Time: LMST − LTST (in minutes)
        lmst = (dt.hour + dt.minute/60 + dt.second/3600 + lon/15.0) % 24
        result['equation_of_time_min'] = (lmst - ltst) * 60

        # 4) Solar position
        alt_sun, az_sun, _ = astrometric.altaz()
        result['solar_elevation_deg'] = alt_sun.degrees
        result['solar_azimuth_deg']  = az_sun.degrees

        # 5) Julian day (Terrestrial Time)
        result['julian_day'] = t.tt

        # 6) Sunrise, sunset, and daylength
        # Pass `location` (Topos), not `observer`
        f_sun = almanac.sunrise_sunset(self.eph, location)
        self.calc_daylength(dt, f_sun)
        t0 = self.ts.utc(dt.year, dt.month, dt.day)
        t1 = self.ts.utc(dt.year, dt.month, dt.day + 5)
        times, events = almanac.find_discrete(t0, t1, f_sun)
        print(times)
        sunrises = [ti.utc_datetime() for ti, e in zip(times, events) if e == 1]
        sunsets  = [ti.utc_datetime() for ti, e in zip(times, events) if e == 0]
        print(sunrises)
        print(sunsets)
      
        if sunrises and sunsets:
            duration = (sunsets[0] - sunrises[0]).total_seconds() / 3600.0
            result['daylength_hours'] = duration
            result['sunrise_utc']    = sunrises[0].time().isoformat(timespec='minutes')
            result['sunset_utc']     = sunsets[0].time().isoformat(timespec='minutes')
        else:
            result['daylength_hours'] = 0.0
            result['sunrise_utc']     = None
            result['sunset_utc']      = None

        # 7) Moon position
        astmo_moon = observer.at(t).observe(self.moon).apparent()
        alt_moon, az_moon, _ = astmo_moon.altaz()
        result['moon_elevation_deg'] = alt_moon.degrees
        result['moon_azimuth_deg']   = az_moon.degrees
        # result['prev_daylength_hours'] = result['daylength_hours']
        # result['next_daylength_hours'] = result['daylength_hours']
        # 8) Moon phase angle (elongation)
        sun_vec  = self.earth.at(t).observe(self.sun).apparent().position.km
        moon_vec = self.earth.at(t).observe(self.moon).apparent().position.km
        phase_angle = np.arccos(
            np.dot(sun_vec, moon_vec) /
            (np.linalg.norm(sun_vec) * np.linalg.norm(moon_vec))
        )
        result['day_of_year'] = dt.timetuple().tm_yday
        result['moon_phase_angle_deg'] = np.degrees(phase_angle)
        result['timestamp'] = dt.timestamp()
        result['latitude'] = lat
        result['longitude'] = lon
        ret = self.calc_daylength(dt, f_sun)
        pprint(ret)
        result.update(ret)
        return result


if __name__ == "__main__":
    solar_calculator = SolarCalculator()
    dt = datetime(2024,11, 1, 12, 0, 0)
    lat, lon = 32.04, 34.78
    result = solar_calculator.load(dt, lat, lon)
    pprint(result)
