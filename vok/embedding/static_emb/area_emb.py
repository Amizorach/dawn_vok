import numpy as np
from datetime import datetime

class AreaEmbedding:
    def __init__(self, lat_min, lat_max, lon_min, lon_max, timestamp=None):
        """
        Initialize the AreaEmbedding with a bounding box defined by lat_min, lat_max,
        lon_min, and lon_max. If no timestamp is provided, the current time is used.
        """
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.timestamp = timestamp if timestamp is not None else datetime.now()
        
        # Compute the centroid of the area for positional encoding.
        self.centroid_lat = (lat_min + lat_max) / 2.0
        self.centroid_lon = (lon_min + lon_max) / 2.0
        
        # Compute the spatial extent (range) of the area.
        self.lat_range = lat_max - lat_min
        self.lon_range = lon_max - lon_min
        
    def get_fourier_embedding(self, num_frequencies=4):
        """
        Create a Fourier embedding for the centroid of the area.
        This uses sine and cosine functions at multiple frequencies to encode the position.
        """
        # Convert the centroid to radians.
        lat_rad = np.radians(self.centroid_lat)
        lon_rad = np.radians(self.centroid_lon)
        
        freqs = 2.0 ** np.arange(num_frequencies)
        
        # Wrap the scalar results in np.array to get one-dimensional arrays.
        lat_features = np.array([np.sin(2 * np.pi * freq * lat_rad) for freq in freqs] +
                                [np.cos(2 * np.pi * freq * lat_rad) for freq in freqs])
        
        lon_features = np.array([np.sin(2 * np.pi * freq * lon_rad) for freq in freqs] +
                                [np.cos(2 * np.pi * freq * lon_rad) for freq in freqs])
        
        return np.concatenate([lat_features, lon_features])
    
    def get_extent_embedding(self):
        """
        Encode the spatial extent (the range in latitude and longitude) of the area.
        """
        # This can be normalized or processed further if needed.
        return np.array([self.lat_range, self.lon_range])
    
    def get_gis_embedding(self):
        """
        Retrieve and encode aggregated GIS data for the area.
        
        This dummy implementation returns average elevation, slope, and aspect for the area,
        along with a timestamp to indicate when the data was valid.
        """
        # Dummy aggregated GIS data.
        avg_elevation = 150.0  # For example: average elevation in meters.
        avg_slope = 3.0        # For example: average slope in degrees.
        avg_aspect = 60.0      # For example: average aspect in degrees.
        
        # Convert the timestamp to a numeric value (seconds since epoch).
        timestamp_numeric = self.timestamp.timestamp()
        
        return np.array([avg_elevation, avg_slope, avg_aspect, timestamp_numeric])
    
    def get_land_cover_embedding(self):
        """
        Retrieve and encode aggregated land cover data for the area.
        
        This dummy implementation returns a distribution over five land cover types (summing to 1)
        and appends a timestamp. In practice, you would derive this data from a dynamic dataset.
        """
        # Dummy distribution for land cover types.
        land_cover_distribution = np.array([0.3, 0.25, 0.15, 0.2, 0.1])
        timestamp_numeric = self.timestamp.timestamp()
        
        return np.concatenate([land_cover_distribution, [timestamp_numeric]])
    
    def combine_embeddings(self):
        """
        Combine the embeddings (Fourier encoding, spatial extent, aggregated GIS, and land cover)
        into a single unified embedding. A type flag [0, 1] is appended to indicate an area embedding.
        """
        fourier = self.get_fourier_embedding()
        extent = self.get_extent_embedding()
        gis = self.get_gis_embedding()
        land_cover = self.get_land_cover_embedding()
        
        # Type flag: [0, 1] indicates an area embedding (as opposed to [1, 0] for a point location).
        type_flag = np.array([0, 1])
        
        combined_embedding = np.concatenate([fourier, extent, gis, land_cover, type_flag])
        return combined_embedding

# Example usage:
if __name__ == "__main__":
    # Define an area bounding box (for example, a section of San Francisco)
    lat_min, lat_max = 37.75, 37.80
    lon_min, lon_max = -122.45, -122.40
    
    area_embed = AreaEmbedding(lat_min, lat_max, lon_min, lon_max)
    combined_area_embedding = area_embed.combine_embeddings()
    
    print("Combined Area Embedding:\n", combined_area_embedding)
