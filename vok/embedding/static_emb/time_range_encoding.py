import math
import numpy as np
from datetime import datetime, timedelta

# Our previously defined rich time range encoding class.
class RichTimeRangeEncoding:
    def __init__(self, start, end=None, frequency=None):
        """
        Initialize with a start datetime, an optional end datetime, and an optional frequency.
        If end is None, the time range is considered to be a point (i.e. start == end).
        
        Args:
            start (datetime): The starting datetime.
            end (datetime, optional): The ending datetime. Defaults to start.
            frequency (timedelta, optional): The recording frequency (e.g., every 10 minutes).
        """
        self.start = start
        self.end = end if end is not None else start
        self.frequency = frequency  # frequency should be a timedelta

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
    
    def get_encoding(self):
        """
        Create the rich time-range embedding.
        
        This embedding includes:
          - Cyclic encoding for the start time (6 dimensions)
          - Cyclic encoding for the middle of the range (6 dimensions)
          - Cyclic encoding for the end time (6 dimensions)
          - Duration in seconds (1 dimension)
          - Frequency feature (log-transformed, 1 dimension)
          - A type flag: [1, 0] for a point (zero duration) or [0, 1] for a nonzero range (2 dimensions)
        
        Total dimensions: 6 + 6 + 6 + 1 + 1 + 2 = 22.
        """
        # Encode start and end times
        start_enc = self.cyclic_encode(self.start)
        end_enc = self.cyclic_encode(self.end)
        
        # Compute the middle time as the average (works even when start == end)
        middle_timestamp = self.start.timestamp() + (self.end.timestamp() - self.start.timestamp()) / 2.0
        middle_dt = datetime.fromtimestamp(middle_timestamp)
        middle_enc = self.cyclic_encode(middle_dt)
        
        # Compute the duration of the time range (in seconds)
        duration = (self.end - self.start).total_seconds()
        
        # Encode the frequency:
        if self.frequency is not None:
            freq_seconds = self.frequency.total_seconds()
            freq_feature = np.log(freq_seconds + 1)  # +1 to avoid log(0)
        else:
            freq_feature = 0.0
        
        # Define a type flag: [1, 0] for a point (duration zero) and [0, 1] for a range.
        type_flag = np.array([1, 0]) if duration == 0 else np.array([0, 1])
        
        # Concatenate all features into one vector.
        encoding = np.concatenate([start_enc, middle_enc, end_enc, np.array([duration, freq_feature]), type_flag])
        return encoding

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
    # Example location: San Francisco (latitude, longitude) and current GMT time.
    lat = 37.7749
    lon = -122.4194
    start_time = datetime.utcnow()  # Using GMT (UTC) time.
    frequency = timedelta(minutes=10)  # Example recording frequency.
    
    loc_time_embed = LocationTimeEmbedding(lat=lat, lon=lon, start_time=start_time, frequency=frequency)
    embedding = loc_time_embed.get_embedding()
    
    print("Location Time Embedding:\n", embedding)
