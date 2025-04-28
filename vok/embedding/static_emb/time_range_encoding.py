import math
import random
import numpy as np
from datetime import datetime, timedelta

import torch

from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.vok.embedding.static_emb.frequency_encoding import FrequencyEncoding
from dawn_vok.vok.v_objects.vok_object import VOKObject



class BaseEncoderValues:
    encoder_configs = {
        "time_stamp_encoder": {
            "min_year": 1990,
            "max_year": 2029,
            "info_map": {
                0: "year_norm",
                1: "month_norm",
                2: "month_sin",
                3: "month_cos",
                4: "day_norm",
                5: "day_sin",
                6: "day_cos",
                7: "hour_norm",
                8: "hour_sin",
                9: "hour_cos",
                10: "minute_norm",
                11: "minute_sin",
                12: "minute_cos",
                13: "since_start_norm",
                14: "since_start_sin",
                15: "since_start_cos",
            }
        }
    }
    
class VOKStructuredEncoder(VOKObject):
    @classmethod
    def get_db_name(cls):
        return 'embeddings'
    
    @classmethod
    def get_collection_name(cls):
        return 'structured_encoders'
    
    def __init__(self, uid=None, info_map=None, config=None):
        super().__init__(uid=uid, obj_type="structured_encoder", name="structured_encoder")
        self.config = config or self.get_config()
        self.info_map = info_map or self.config.get("info_map", {})
    
    def to_dict(self):
        ret = super().to_dict()
        ret["info_map"] = self.info_map
        return ret
    
    def populate_from_dict(self, d):
        super().populate_from_dict(d)
        self.info_map = d["info_map"]

    def encode(self, dt):
        raise NotImplementedError("Subclasses must implement this method")
    
    def decode(self, v):
        raise NotImplementedError("Subclasses must implement this method")
    
    

class TimeStampEncoder(VOKStructuredEncoder):

    @classmethod
    def get_config(cls):
        return {
            "min_year": 1990,
            "max_year": 2029,
            "info_map": {
                0: "year_norm",
                1: "month_norm",
                2: "month_sin",
                3: "month_cos",
                4: "day_norm",
                5: "day_sin",
                6: "day_cos",
                7: "hour_norm",
                8: "hour_sin",
                9: "hour_cos",
                10: "minute_norm",
                11: "minute_sin",
                12: "minute_cos",
                13: "since_start_norm",
                14: "since_start_sin",
                15: "since_start_cos",
            }
        }
    
    def __init__(self):
        uid = "time_stamp_encoder"  
        super().__init__(uid=uid)
        self.min_year = self.config["min_year"]
        self.max_year = self.config["max_year"]
        self.year_range = self.max_year - self.min_year
        self.tot_days = (self.max_year - self.min_year) * 365 + (self.max_year - self.min_year) // 4
        self.tot_hours = self.tot_days * 24
        self.tot_minutes = self.tot_hours * 60
       
    def encode(self, dt):
        if dt is not None:
            dt = DictUtils.parse_datetime_direct(dt)
        ts= datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
        # 2) Build a feature vector 'x' with informative features
        x = []
        # --- Feature Engineering ---
        # Feature 0: Normalized Year
        # Scale year to be roughly between 0 and 1
        norm_year = (ts.year - self.min_year) / self.year_range
        # Already converting to tensor here, this is fine:
        x.append(torch.tensor(norm_year, dtype=torch.float32))
        # Features 1, 2: Cyclical Month (0-11)
        # Using ts.month - 1 to get a 0-based index for the 12 months
        month_norm = (ts.month - 1) / 12.0 
        month_angle_float = 2 * torch.pi * month_norm

        # Convert float angle to tensor BEFORE sin/cos
        month_angle_tensor = torch.tensor(month_angle_float, dtype=torch.float32)

        x.append(torch.tensor(month_norm, dtype=torch.float32))
        x.append(torch.sin(month_angle_tensor))
        x.append(torch.cos(month_angle_tensor))
        # Features 3, 4: Cyclical Day of Month (0-30 approx)
        # Using ts.day - 1 for 0-based index. Normalizing by 31 is an approximation.
        day_norm = (ts.day - 1) / 31.0
        day_angle_float = 2 * torch.pi * day_norm
        # Convert float angle to tensor BEFORE sin/cos
        day_angle_tensor = torch.tensor(day_angle_float, dtype=torch.float32)
        x.append(torch.tensor(day_norm, dtype=torch.float32))
        x.append(torch.sin(day_angle_tensor))
        x.append(torch.cos(day_angle_tensor))
        # Features 5, 6: Cyclical Hour (0-23)
        hour_norm = ts.hour / 24.0
        hour_angle_float = 2 * torch.pi * hour_norm
        # Convert float angle to tensor BEFORE sin/cos
        hour_angle_tensor = torch.tensor(hour_angle_float, dtype=torch.float32)
        x.append(torch.tensor(hour_norm, dtype=torch.float32))
        x.append(torch.sin(hour_angle_tensor))
        x.append(torch.cos(hour_angle_tensor))
        # Features 7, 8: Cyclical Minute (0-59)

        # Features 7, 8: Cyclical Minute (0-59)
        minute_norm = ts.minute / 60.0
        minute_angle_float = 2 * torch.pi * minute_norm
        # Convert float angle to tensor BEFORE sin/cos
        minute_angle_tensor = torch.tensor(minute_angle_float, dtype=torch.float32)
        x.append(torch.tensor(minute_norm, dtype=torch.float32))
        x.append(torch.sin(minute_angle_tensor))
        x.append(torch.cos(minute_angle_tensor))

        tot_days_since_start = (ts.year - self.min_year) * 365 + (ts.year - self.min_year) // 4 + (ts.month - 1) * 31 + ts.day - 1
        tot_minutes_since_start = tot_days_since_start * 24 * 60 + ts.hour * 60 + ts.minute
        norm_minutes_since_start = tot_minutes_since_start / self.tot_minutes
        total_minutes_angle = torch.tensor(2 * torch.pi * norm_minutes_since_start, dtype=torch.float32)
        x.append(torch.tensor(norm_minutes_since_start, dtype=torch.float32))
        x.append(torch.sin(total_minutes_angle))
        x.append(torch.cos(total_minutes_angle))
        
        v = torch.stack(x, dim=0)
        return v
        
    def decode(self, v):
        v = {self.info_map[i]: float(v[i]) for i in range(len(v))}
        y = v["year_norm"] * self.year_range + self.min_year
        m = v["month_norm"] * 12 + 1
        d = v["day_norm"] * 31 + 1
        h = v["hour_norm"] * 24
        mi = v["minute_norm"] * 60
        dt = datetime(round(y), round(m), round(d), round(h), round(mi))
        return dt
    
    def decode_batch_logits(self, logits: dict, minute_bin_size: int):
        """
        logits: dict of tensors, each of shape [B, num_classes] for heads
        minute_bin_size: the bin size used when training (so we can invert the minute bin)
        Returns: list of datetime objects of length B
        """
        # 1) pick predicted class indices for each head
        preds = { head: torch.argmax(logits[head], dim=1).cpu().numpy()
                  for head in ["year","month","day","hour","minute"]
                  if head in logits }

        B = next(iter(preds.values())).shape[0]
        out = []
        for i in range(B):
            # 2) build the “normalized feature” vector v of length len(info_map)
            v = torch.zeros(len(self.info_map), dtype=torch.float32)
            # year_norm lives at index where info_map==“year_norm” (should be 0)
            v[0] = (preds["year"][i]) / self.year_range
            # month_norm at index 1
            v[1] = (preds["month"][i]) / 12.0
            # day_norm at index 4
            v[4] = (preds["day"][i]) / 31.0
            # hour_norm at index 7
            v[7] = (preds["hour"][i]) / 24.0
            # minute_norm at index 10
            v[10] = (preds["minute"][i] * minute_bin_size) / 60.0

            # 3) decode single vector
            dt = self.decode(v)
            out.append(dt)
        return out
# Our previously defined rich time range encoding class.
class RichTimeRangeEncoding:
    def __init__(self, start, end=None, frequency=None):
        """
        Initialize with a start datetime, an optional end datetime, and an optional frequency.
        If end is None, the time range is considered to be a point (i.e. start == end).
        
        Args:
            start (datetime): The starting datetime.
            end (datetime, optional): The ending datetime. Defaults to start.
            frequency (seconds): The recording frequency (e.g., every 10 minutes).
            
        """
        self.start = start
        self.end = end if end is not None else start
        self.frequency = frequency or 60*10
        self.max_year = 2029
        self.min_year = 1990
        self.year_range = self.max_year - self.min_year
        self.map = {
            "start_timestamp": 0,
            "middle_timestamp": 1,
            "end_timestamp": 2,
            "freq_feature": 3,
            "start_enc": 4,
            "middle_enc": 5,
            "end_enc": 6,
            "duration": 7,
            "type_flag": 8
        }

    def datetime_encode(self, dt):
        if dt is not None:
            dt = DictUtils.parse_datetime_str(dt)
        ts= datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
        # 2) Build a feature vector 'x' with informative features
        x = torch.zeros(self.input_dim, dtype=torch.float32) # Use float32

        # --- Feature Engineering ---
        i =0
        # Feature 0: Normalized Year
        # Scale year to be roughly between 0 and 1
        norm_year = (ts.year - self.min_year) / self.year_range
        # Already converting to tensor here, this is fine:
        x[i] = torch.tensor(norm_year, dtype=torch.float32)
        i += 1
        # Features 1, 2: Cyclical Month (0-11)
        # Using ts.month - 1 to get a 0-based index for the 12 months
        month_norm = (ts.month - 1) / 12.0 
        month_angle_float = 2 * torch.pi * month_norm

        # Convert float angle to tensor BEFORE sin/cos
        month_angle_tensor = torch.tensor(month_angle_float, dtype=torch.float32)

        x[i] = torch.tensor(month_norm, dtype=torch.float32)
        i += 1
        x[i] = torch.sin(month_angle_tensor)
        i += 1
        x[i] = torch.cos(month_angle_tensor)
        i += 1
        # Features 3, 4: Cyclical Day of Month (0-30 approx)
        # Using ts.day - 1 for 0-based index. Normalizing by 31 is an approximation.
        day_norm = (ts.day - 1) / 31.0
        day_angle_float = 2 * torch.pi * day_norm
        # Convert float angle to tensor BEFORE sin/cos
        day_angle_tensor = torch.tensor(day_angle_float, dtype=torch.float32)
        x[i] = torch.tensor(day_norm, dtype=torch.float32)
        i += 1
        x[i] = torch.sin(day_angle_tensor)
        i += 1
        x[i] = torch.cos(day_angle_tensor)
        i += 1
        # Features 5, 6: Cyclical Hour (0-23)
        hour_norm = ts.hour / 24.0
        hour_angle_float = 2 * torch.pi * hour_norm
        # Convert float angle to tensor BEFORE sin/cos
        hour_angle_tensor = torch.tensor(hour_angle_float, dtype=torch.float32)
        x[i] = torch.tensor(hour_norm, dtype=torch.float32)
        i += 1
        x[i] = torch.sin(hour_angle_tensor)
        i += 1
        x[i] = torch.cos(hour_angle_tensor)
        i += 1

        # Features 7, 8: Cyclical Minute (0-59)
        minute_norm = ts.minute / 60.0
        minute_angle_float = 2 * torch.pi * minute_norm
        # Convert float angle to tensor BEFORE sin/cos
        minute_angle_tensor = torch.tensor(minute_angle_float, dtype=torch.float32)
        x[i] = torch.tensor(minute_norm, dtype=torch.float32)
        i += 1
        x[i] = torch.sin(minute_angle_tensor)
        i += 1
        x[i] = torch.cos(minute_angle_tensor)
        
        return x
    def cyclic_encode(self, dt):
        """
        Encode a datetime into cyclic features for time-of-day, day-of-week, and day-of-year.
        
        Args:
            dt (datetime): The datetime to encode.
            
        Returns:
            np.ndarray: A 6-dimensional vector containing 
                        [time_sin, time_cos, weekday_sin, weekday_cos, day_year_sin, day_year_cos].
        """
        # Time-of-day: convert hour (including minutes and seconds) to a cyclic angle.
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        hour_angle = 2 * np.pi * hour / 24.0
        time_sin = np.sin(hour_angle)
        time_cos = np.cos(hour_angle)
        
        # Day-of-week: Monday=0, Sunday=6 mapped to a circle.
        weekday = dt.weekday()
        weekday_angle = 2 * np.pi * weekday / 7.0
        weekday_sin = np.sin(weekday_angle)
        weekday_cos = np.cos(weekday_angle)
        
        # Day-of-year: map the day of year (1-366) to a cycle (using 365.25 for leap years)
        day_of_year = dt.timetuple().tm_yday
        day_year_angle = 2 * np.pi * day_of_year / 365.25
        day_year_sin = np.sin(day_year_angle)
        day_year_cos = np.cos(day_year_angle)
        
        return np.array([time_sin, time_cos,
                         weekday_sin, weekday_cos,
                         day_year_sin, day_year_cos])
    
    def normalize_timestamp(self, timestamp):
      
        return (timestamp - self.min_timestamp) / (self.max_timestamp - self.min_timestamp)
    
    def decode_timestamp(self, normalized_timestamp):
        return normalized_timestamp * (self.max_timestamp - self.min_timestamp) + self.min_timestamp
    

    def get_encoding(self):
        return self.datetime_encode(self.start)
    # def update_map(self, map, key, index):
    #     map[key] = index
    #     else:
    #         map[key] = map[list(map.keys())[-1]] + 1
    #     return map
    def get_encoding(self):
        """
        Create the rich time-range embedding.
        
        This embedding includes:
          - Normalized timestamps (3 dimensions)
          - Cyclic encoding for the start time (6 dimensions)
          - Cyclic encoding for the middle of the range (6 dimensions)
          - Cyclic encoding for the end time (6 dimensions)
          - Duration in seconds (1 dimension)
          - Frequency feature (log-transformed, 1 dimension)
          - A type flag: [1, 0] for a point (zero duration) or [0, 1] for a nonzero range (2 dimensions)
        
        Total dimensions: 6 + 6 + 6 + 1 + 1 + 2 = 22.
        """
        # Encode start and end times
        start_timestamp = self.normalize_timestamp(float(self.start.timestamp()))
        end_timestamp = self.normalize_timestamp(float(self.end.timestamp()))
        start_enc = self.cyclic_encode(self.start)
        end_enc = self.cyclic_encode(self.end)
        
        # Compute the middle time as the average (works even when start == end)
        middle_timestamp = self.start.timestamp() + (self.end.timestamp() - self.start.timestamp()) / 2.0
        middle_dt = datetime.fromtimestamp(middle_timestamp)
        middle_enc = self.cyclic_encode(middle_dt)
        middle_timestamp = self.normalize_timestamp(middle_timestamp)
        # Compute the duration of the time range (in seconds)
        duration = (self.end.timestamp() - self.start.timestamp())/(60*60*24*365)
     
        # Encode the frequency:


        freq_feature = FrequencyEncoding().encode(self.frequency, dim_size=None)
        #Define a type flag: [1, 0] for a point (duration zero) and [0, 1] for a range.
        flags= np.array([1, 0]) if duration == 0 else np.array([0, 1])
        # Concatenate all features into one vector
        normalized_timestamps = [start_timestamp, middle_timestamp, end_timestamp]
        # log_timestamp = [np.log10(start_timestamp), np.log10(middle_timestamp), np.log10(end_timestamp)]
        offset = 0
        
        # keys = list(self.map.keys())
        # offset = 0
        # for i, feature in enumerate(encoding):
        #     if isinstance(feature, float):
        #         offset += 1
        #     else:
        #         offset += len(feature)
        #     self.map[keys[i]] = offset
        # print(self.map)
        encoding = [normalized_timestamps, freq_feature, start_enc, middle_enc, end_enc, [duration],  flags]
        encoding = np.concatenate(encoding)
        return encoding

    def decode(self, encoding):
        start_timestamp = self.decode_timestamp(self.map["start_timestamp"])
        middle_timestamp = self.decode_timestamp(self.map["middle_timestamp"])
        end_timestamp = self.decode_timestamp(self.map["end_timestamp"])
        start_dt = datetime.fromtimestamp(start_timestamp)
        middle_dt = datetime.fromtimestamp(middle_timestamp)
        end_dt = datetime.fromtimestamp(end_timestamp)
      
        frequency = FrequencyEncoding().decode(encoding[self.map["freq_feature"]:self.map["start_enc"]])
        ret = {
            "start_dt": start_dt,
            "middle_dt": middle_dt,
            "end_dt": end_dt,
            "frequency": frequency,
            "duration": start_timestamp - end_timestamp
        }
        return ret

        
# New class that combines location with time to produce a "Location Time Embedding."
class LocationTimeEmbedding:
    def __init__(self, lat, lon, start_time, frequency=None):
        """
        Initialize with a geographic location (lat, lon) and a start time.
        For now, we use only the start time (i.e. a point) from our time range encoding.
        
        Args:
            lat (float): Latitude in decimal degrees.
            lon (float): Longitude in decimal degrees.
            start_time (datetime): The starting time (assumed to be in GMT).
            frequency (timedelta, optional): The recording frequency.
        """
        self.lat = lat
        self.lon = lon
        self.start_time = start_time
        self.frequency = frequency
        # Use the rich time range encoding for the start time (as a point).
        self.time_encoding = RichTimeRangeEncoding(start=start_time, end=start_time, frequency=frequency)
        
    def get_solar_position(self, dt, lat, lon):
        """
        Approximate the solar position (altitude and azimuth) for a given datetime and location.
        
        This implementation computes:
          - The solar declination based on the day of the year.
          - The local solar time correction (assuming GMT input and a rough correction using longitude).
          - The hour angle, then solar altitude using the formula:
            
              sin(alt) = sin(lat) * sin(decl) + cos(lat) * cos(decl) * cos(hour_angle)
            
          - An approximate solar azimuth.
        
        Args:
            dt (datetime): The datetime (in GMT).
            lat (float): Latitude in decimal degrees.
            lon (float): Longitude in decimal degrees.
        
        Returns:
            (float, float): A tuple of (solar_altitude, solar_azimuth) in radians.
        """
        # Convert lat and lon to radians.
        lat_rad = math.radians(lat)
        
        # Day of year
        N = dt.timetuple().tm_yday
        
        # Approximate solar declination (in degrees) and convert to radians.
        decl_deg = 23.45 * math.sin(math.radians(360.0 * (284 + N) / 365.0))
        decl = math.radians(decl_deg)
        
        # Compute local solar time.
        # A rough correction: each degree of longitude corresponds to 4 minutes.
        time_correction = lon / 15.0  # hours (15° per hour)
        local_time = dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + time_correction
        
        # Hour angle (in degrees): difference from solar noon (12:00 local time)
        hour_angle_deg = 15 * (local_time - 12)
        hour_angle = math.radians(hour_angle_deg)
        
        # Solar altitude calculation.
        sin_alt = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle)
        alt = math.asin(sin_alt)
        
        # Solar azimuth calculation.
        # Using an approximate formula; note that more robust methods exist.
        cos_az = (math.sin(decl) - math.sin(lat_rad) * math.sin(alt)) / (math.cos(lat_rad) * math.cos(alt))
        # Clamp cos_az to avoid domain errors.
        cos_az = max(min(cos_az, 1), -1)
        az = math.acos(cos_az)
        # Adjust azimuth based on the hour angle.
        if hour_angle > 0:
            az = 2 * math.pi - az
        return alt, az

    def get_embedding(self):
        """
        Combine the time embedding (using the start time) with location-dependent solar features.
        
        The final embedding is the concatenation of:
          - The rich time range encoding for the start time (22 dimensions).
          - Raw solar altitude and azimuth (2 dimensions).
          - The sine and cosine of the solar altitude and azimuth (4 dimensions),
            so that similar solar positions are close in the embedding space.
        
        Returns:
            np.ndarray: The combined Location Time Embedding.
        """
        # Get time embedding (for the start time, treated as a point).
        time_enc = self.time_encoding.get_encoding()
        
        # Compute solar position from the start time and location.
        alt, az = self.get_solar_position(self.start_time, self.lat, self.lon)
        
        # Compute sine and cosine features for solar altitude and azimuth.
        solar_features = np.array([alt, az])
        solar_sin_cos = np.array([np.sin(alt), np.cos(alt), np.sin(az), np.cos(az)])
        
        # Combine all features.
        combined_embedding = np.concatenate([time_enc, solar_features, solar_sin_cos])
        return combined_embedding

# Example usage:
if __name__ == "__main__":
    time_stamp_encoder = TimeStampEncoder()
    encoding = time_stamp_encoder.encode(datetime(2021, 3, 12, 17, 22))
    print(encoding)
    print(len(encoding))
    decoded = time_stamp_encoder.decode(encoding)
    print(decoded)
    # start_time = datetime(2021, 1, 1)
    # end_time = datetime(2021, 1, 21, 1, 0, 0)
    # time_range_encoding = RichTimeRangeEncoding(start=start_time, end=end_time)
    # encoding = time_range_encoding.get_encoding()
    # print(encoding)
    # print(len(encoding))
    # decoded = time_range_encoding.decode(encoding)


    # # # Example location: San Francisco (latitude, longitude) and current GMT time.
    # lat = 37.7749
    # lon = -122.4194
    # start_time = datetime.utcnow()  # Using GMT (UTC) time.
    # frequency = timedelta(minutes=10)  # Example recording frequency.
    
    # loc_time_embed = LocationTimeEmbedding(lat=lat, lon=lon, start_time=start_time, frequency=frequency)
    # embedding = loc_time_embed.get_embedding()
    
    # print("Location Time Embedding:\n", embedding)
