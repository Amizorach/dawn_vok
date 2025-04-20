import ee
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import time # For potential retries or delays if needed

class GEETempTimeSeries:
    """
    A class to fetch and process temperature time series data from Google Earth Engine
    for a central point or a grid around it.
    """

    def __init__(self, latitude, longitude, year, month, project_id='ee-amizorach'):
        """
        Initializes the class with location, time period, and optional GEE project ID.

        Args:
            latitude (float): Latitude of the center point.
            longitude (float): Longitude of the center point.
            year (int): The target year.
            month (int): The target month (1-12).
            project_id (str, optional): Your GEE project ID. Defaults to None (will try auto-detect).
        """
        self.latitude = latitude
        self.longitude = longitude
        self.year = year
        self.month = month
        self.project_id = project_id
        self.point = ee.Geometry.Point(self.longitude, self.latitude)
        self._calculate_dates() # Calculate start/end dates

        self.ee_initialized = self._initialize_ee()
        if not self.ee_initialized:
            raise RuntimeError("Google Earth Engine could not be initialized. Check authentication and project ID.")

        self.grid_points_fc = None # To store the grid FeatureCollection if created
        self.grid_point_ids = None # To store the IDs of grid points
        self.results = {} # Dictionary to store results (DataFrames)

        print(f"Initialized GEETempTimeSeries for Location: ({self.latitude:.4f}, {self.longitude:.4f})")
        print(f"Time Period: {self.start_date_str} to {self.end_date.strftime('%Y-%m-%d')}")


    def _initialize_ee(self):
        """Initializes the Earth Engine library."""
        try:
            # Using default credentials or Service Account if configured
            credentials = ee.ServiceAccountCredentials(None, key_path=None)
            ee.Initialize(credentials=credentials, project=self.project_id, opt_url='https://earthengine-highvolume.googleapis.com')
            print("Earth Engine Initialized using default credentials.")
            return True
        except Exception as e:
            print(f"Initialization with default creds failed: {e}. Trying authentication flow...")
            try:
                ee.Authenticate() # Trigger authentication if needed
                ee.Initialize(project=self.project_id)
                print("Earth Engine Initialized after authentication.")
                return True
            except Exception as auth_e:
                print(f"Authentication or subsequent initialization failed: {auth_e}")
                return False

    def _calculate_dates(self):
        """Calculates start and end date strings for the specified month and year."""
        self.start_date = datetime.date(self.year, self.month, 1)
        # Find the next month's first day, then subtract one day
        next_month_start = (self.start_date.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        self.end_date = next_month_start - datetime.timedelta(days=1)

        # Format dates as strings for GEE (end date is exclusive in GEE filterDate)
        self.start_date_str = self.start_date.strftime('%Y-%m-%d')
        self.end_date_str = (self.end_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    def _create_grid(self, grid_dim=5, spacing_meters=1000):
        """
        Generates an NxN grid of ee.Feature points centered on the instance's point.

        Args:
            grid_dim (int): The dimension of the grid (e.g., 5 for 5x5).
            spacing_meters (float): The approximate spacing between grid points in meters.

        Returns:
            ee.FeatureCollection: A collection of grid points with 'row', 'col', 'point_id' properties.
                                  Returns None if calculation fails. Also sets self.grid_points_fc.
        """
        if grid_dim % 2 == 0:
            print("Warning: Even grid dimension provided. Center point might be offset.")

        features = []
        point_ids = []
        center_offset = grid_dim // 2

        # Approximate degree conversion (less accurate away from equator)
        lat_rad = math.radians(self.latitude)
        deg_lat_spacing = spacing_meters / 111000.0
        deg_lon_spacing = spacing_meters / (111320.0 * math.cos(lat_rad)) if math.cos(lat_rad) > 1e-6 else deg_lat_spacing

        print(f"Creating {grid_dim}x{grid_dim} grid with ~{spacing_meters}m spacing...")
        print(f"Approx Deg Spacing: Lat={deg_lat_spacing:.6f}, Lon={deg_lon_spacing:.6f}")

        for r in range(grid_dim):
            for c in range(grid_dim):
                # Calculate offsets from center (dx, dy) in grid units
                dx = c - center_offset
                dy = r - center_offset # dy positive upwards (North)

                # Calculate coords
                pt_lon = self.longitude + dx * deg_lon_spacing
                pt_lat = self.latitude + dy * deg_lat_spacing # Use +dy for North

                point_id = f'row_{r}_col_{c}' # Unique ID based on row/col index
                point_ids.append(point_id)

                # Create EE objects
                ee_point = ee.Geometry.Point(pt_lon, pt_lat)
                feature = ee.Feature(ee_point, {'row': r, 'col': c, 'point_id': point_id})
                # GEE needs an ID for reduceRegions features, set system:index or use point_id
                feature = feature.set({'system:index': point_id})
                features.append(feature)

        self.grid_points_fc = ee.FeatureCollection(features)
        self.grid_point_ids = point_ids
        print(f"Generated {len(features)} grid points.")
        return self.grid_points_fc

    def _extract_time_series_single_point(self, image_collection, band_name, scale, property_name='value'):
        """(Internal) Extracts time series for the central point."""
        def extract_value(image):
            value = image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=self.point,
                scale=scale,
                maxPixels=1
            ).get(band_name)
            return ee.Feature(None).set({
                'date': image.date().format('YYYY-MM-dd HH:mm:ss'),
                property_name: value
            }).copyProperties(image, ['system:index', 'system:time_start'])

        time_series = image_collection.map(extract_value)
        print(f"Requesting single point '{property_name}' data at {scale}m scale from GEE...")
        try:
            info = time_series.getInfo()
            print("Data received.")
            return info.get('features') # Use .get for safer access
        except ee.EEException as e:
            print(f"Error retrieving single point data from GEE: {e}")
            return None

    def _extract_time_series_grid(self, image_collection, band_name, scale):
        """
        (Internal) Extracts time series for all points in a pre-generated grid.

        Args:
            image_collection (ee.ImageCollection): Scaled collection to sample from.
            band_name (str): The name of the band containing the value.
            scale (int): The scale (resolution in meters) to sample at.

        Returns:
            list: A list of features (from .getInfo()), where each feature represents
                  a time step and contains a 'date' property and a 'point_values'
                  dictionary mapping point_id to its value for that date.
                  Returns None on error.
        """
        if self.grid_points_fc is None or self.grid_point_ids is None:
             print("Error: Grid has not been generated. Call create_grid() first.")
             return None

        grid_fc = self.grid_points_fc # Use the stored grid

        def process_image(image):
            # Sample the image at all grid points simultaneously
            # reduceRegions is generally preferred over sampleRegions for reducers like 'first'
            sampled_fc = image.reduceRegions(
                collection=grid_fc,
                reducer=ee.Reducer.first().setOutputs([band_name]), # Ensure output name matches band
                scale=scale
                # tileScale=4 # Optional: can sometimes help with large computations
            )

            # sampled_fc now has features for each grid point, with the sampled value
            # We need to convert this into a dictionary {point_id: value} for this image's timestamp

            # Efficient way to create a dictionary server-side:
            # Use reduceColumns to get lists of IDs and values, then ee.Dictionary.fromLists
            keys = sampled_fc.aggregate_array('point_id') # Get the point_id property we added
            values = sampled_fc.aggregate_array(band_name) # Get the value (reducer output)

            point_values_dict = ee.Dictionary.fromLists(keys, values)

            # Return a single feature for this image's timestamp
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd HH:mm:ss'),
                'point_values': point_values_dict # Dictionary of point_id -> value
            }).copyProperties(image, ['system:index', 'system:time_start'])

        # Map the processing function over the image collection
        grid_time_series = image_collection.map(process_image)

        # Retrieve the results
        print(f"Requesting grid data ({len(self.grid_point_ids)} points) at {scale}m scale from GEE...")
        print("Note: This can take significantly longer and consume more resources.")
        try:
            info = grid_time_series.getInfo()
            print("Grid data received.")
            return info.get('features') # Use .get for safer access
        except ee.EEException as e:
            print(f"Error retrieving grid data from GEE: {e}")
            print("This might be due to computational limits or memory constraints. Try a smaller grid, shorter time period, or export.")
            return None


    def _process_features_to_df(self, features, value_key, title_prefix):
        """(Internal) Processes feature list (single point) into a DataFrame."""
        if not features:
            print(f"No features received for {title_prefix}.")
            return None
        dates, values, null_count = [], [], 0
        for feature in features:
            props = feature.get('properties', {})
            value = props.get(value_key)
            if props and value is not None:
                try: dt_obj = datetime.datetime.strptime(props['date'], '%Y-%m-%d %H:%M:%S')
                except ValueError: dt_obj = datetime.datetime.strptime(props['date'].split(' ')[0], '%Y-%m-%d')
                dates.append(dt_obj)
                values.append(value)
            else: null_count += 1
        if not dates: return None
        df = pd.DataFrame({'datetime': pd.to_datetime(dates), value_key: values})
        df = df.sort_values(by='datetime').set_index('datetime')
        print(f"\n--- Processed {title_prefix} Data ({len(df)} valid, {null_count} null points) ---")
        print(df.head())
        return df

    def _process_grid_features(self, features, grid_point_ids, value_base_key):
        """
        (Internal) Processes the complex grid feature list into a dictionary of DataFrames.

        Args:
            features (list): The list of features from _extract_time_series_grid.
            grid_point_ids (list): List of expected point IDs (e.g., 'row_0_col_0').
            value_base_key (str): Base key for column names in output DataFrames.

        Returns:
            dict: A dictionary where keys are point_ids and values are pandas DataFrames
                  containing the time series for that point. Returns None if processing fails.
        """
        if not features or not grid_point_ids:
            print("No grid features or point IDs to process.")
            return None

        # Initialize dictionary to hold lists for each point's time series
        results_dict = {point_id: {'dates': [], 'values': []} for point_id in grid_point_ids}
        processed_dates = 0

        for feature in features: # Each feature is a time step
            props = feature.get('properties')
            if not props: continue

            point_values = props.get('point_values') # This is the dict {point_id: value}
            date_str = props.get('date')

            if not point_values or not date_str: continue

            # Parse date once per time step
            try: dt_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            except ValueError: dt_obj = datetime.datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d')

            processed_dates += 1
            # Populate the data for each grid point for this time step
            for point_id in grid_point_ids:
                value = point_values.get(point_id) # Use .get for safety
                if value is not None:
                    results_dict[point_id]['dates'].append(dt_obj)
                    results_dict[point_id]['values'].append(value)

        if processed_dates == 0:
             print("No valid dates found in grid features.")
             return None

        # Convert lists to DataFrames
        df_dict = {}
        total_valid_points = 0
        for point_id, data in results_dict.items():
            if data['dates']:
                col_name = f"{value_base_key}_{point_id}"
                df = pd.DataFrame({'datetime': pd.to_datetime(data['dates']), col_name: data['values']})
                df = df.sort_values(by='datetime').set_index('datetime')
                df_dict[point_id] = df
                total_valid_points += len(df)
            else:
                 df_dict[point_id] = None # Or an empty DataFrame: pd.DataFrame(columns=[f"{value_base_key}_{point_id}"])

        print(f"\n--- Processed Grid Data ---")
        print(f"Total valid data points across all {len(grid_point_ids)} grid cells: {total_valid_points}")

        # Calculate center ID more clearly and robustly
        center_point_id = None # Default to None
        center_index = -1 # Default index
        if grid_point_ids:
            grid_dim_sqrt = math.sqrt(len(grid_point_ids))
            # Check if the number of points is a perfect square
            if grid_dim_sqrt == int(grid_dim_sqrt) and len(grid_point_ids) > 0:
                grid_dim = int(grid_dim_sqrt)
                if grid_dim % 2 != 0: # Ensure odd dimension for a clear center
                    center_index = grid_dim // 2
                    # Construct the ID using the calculated index
                    center_point_id = f'row_{center_index}_col_{center_index}'
                else:
                    print("Warning: Grid dimension is even, cannot determine unique center point ID.")
            else:
                print("Warning: Grid size is not a perfect square, cannot determine center point ID.")

        # Optionally print head of the center point's DataFrame if available
        if center_point_id and center_point_id in df_dict and df_dict[center_point_id] is not None:
            print(f"\nHead of center point ({center_point_id}) DataFrame:")
            print(df_dict[center_point_id].head())
        elif center_index != -1: # We knew the center index but didn't find data
            print(f"\nCenter point (row_{center_index}_col_{center_index}) data not available or empty.")
        else:
            print("\nCould not determine or find data for a center grid point.")


        return df_dict # Return the dictionary of DataFrames


    # --- Public Methods for Data Fetching ---

    def fetch_modis_lst(self, resolution=1000, data_key_suffix=''):
        """
        Fetches MODIS LST for the central point at specified resolution (native or aggregated).

        Args:
            resolution (int): Target resolution in meters (e.g., 1000 for native, 4000 for aggregated).
            data_key_suffix (str): Optional suffix for the key in self.results dictionary.
        """
        print(f"\nFetching MODIS LST (Daytime) at {resolution}m scale...")
        lst_dataset_id = 'MODIS/061/MOD11A1'
        lst_band = 'LST_Day_1km'
        lst_native_scale = 1000
        value_key = f'LST_Celsius_{resolution}m{data_key_suffix}'
        title_prefix = f'MODIS LST ({resolution}m{data_key_suffix})'

        # Prepare base 1km scaled collection
        base_lst_collection = ee.ImageCollection(lst_dataset_id) \
            .filterDate(self.start_date_str, self.end_date_str) \
            .filterBounds(self.point).select(lst_band)

        def scale_modis_lst(image):
            scaled = image.multiply(0.02).subtract(273.15).rename('LST_Celsius')
            return scaled.copyProperties(image, image.propertyNames())
        lst_collection_scaled_1km = base_lst_collection.map(scale_modis_lst)

        collection_to_sample = lst_collection_scaled_1km
        band_to_sample = 'LST_Celsius' # Band name after scaling
        scale_to_sample = lst_native_scale # Default to native

        # Aggregate if requested resolution is different from native
        if resolution != lst_native_scale:
            print(f"Aggregating from {lst_native_scale}m to {resolution}m resolution...")
            band_to_sample = value_key # Use the final key as the band name

            def aggregate_lst(image):
                aggregated = image.reduceResolution(
                    reducer=ee.Reducer.mean(), maxPixels=256 # Increased maxPixels
                ).reproject(crs=image.projection(), scale=resolution)
                return aggregated.rename(band_to_sample).copyProperties(image, image.propertyNames())

            collection_to_sample = lst_collection_scaled_1km.map(aggregate_lst)
            scale_to_sample = resolution # Sample the aggregated collection at target scale

        # Extract time series for the central point
        features = self._extract_time_series_single_point(
            collection_to_sample, band_to_sample, scale_to_sample, property_name=value_key
        )
        # Process and store result
        self.results[value_key] = self._process_features_to_df(features, value_key, title_prefix)


    def fetch_era5_air_temp(self, data_key_suffix=''):
        """Fetches ERA5 Daily Mean Air Temperature for the central point."""
        print("\nFetching ERA5 Daily Mean Air Temperature (~28km Native)...")
        era5_dataset_id = 'ECMWF/ERA5/DAILY'
        era5_band = 'mean_2m_air_temperature'
        era5_native_scale = 27830 # Approx native resolution
        value_key = f'AirTemp_Celsius_ERA5{data_key_suffix}'
        title_prefix = f'ERA5 Air Temp (~28km{data_key_suffix})'

        era5_collection = ee.ImageCollection(era5_dataset_id) \
            .filterDate(self.start_date_str, self.end_date_str) \
            .filterBounds(self.point).select(era5_band)

        def scale_era5_temp(image):
            scaled = image.subtract(273.15).rename(value_key) # Rename to final key
            return scaled.copyProperties(image, image.propertyNames())
        era5_collection_scaled = era5_collection.map(scale_era5_temp)

        features = self._extract_time_series_single_point(
            era5_collection_scaled, value_key, era5_native_scale, property_name=value_key
        )
        self.results[value_key] = self._process_features_to_df(features, value_key, title_prefix)

    def fetch_modis_lst_grid(self, grid_dim=5, spacing_meters=1000, data_key='MODIS_LST_Grid_1km'):
        """
        Fetches MODIS LST 1km for an NxN grid of points.

        Args:
            grid_dim (int): Dimension of the grid (e.g., 5 for 5x5).
            spacing_meters (float): Approximate spacing between grid points.
            data_key (str): Key to store the results dictionary under in self.results.
        """
        print(f"\nFetching MODIS LST (Daytime) for {grid_dim}x{grid_dim} Grid at ~{spacing_meters}m spacing (1km native)...")
        lst_dataset_id = 'MODIS/061/MOD11A1'
        lst_band = 'LST_Day_1km'
        lst_native_scale = 1000
        value_base_key = 'LST_Celsius' # Base name for processing

        # Ensure grid is created
        if self.grid_points_fc is None or self.grid_points_fc.size().getInfo() != grid_dim*grid_dim:
             self._create_grid(grid_dim=grid_dim, spacing_meters=spacing_meters)
             if self.grid_points_fc is None: return # Grid creation failed

        # Prepare base 1km scaled collection
        base_lst_collection = ee.ImageCollection(lst_dataset_id) \
            .filterDate(self.start_date_str, self.end_date_str) \
            .filterBounds(self.grid_points_fc.geometry()).select(lst_band)

        def scale_modis_lst(image):
            # Rename band AFTER scaling for clarity in extraction
            scaled = image.multiply(0.02).subtract(273.15).rename(value_base_key)
            return scaled.copyProperties(image, image.propertyNames())
        lst_collection_scaled_1km = base_lst_collection.map(scale_modis_lst)

        # Extract grid time series
        grid_features = self._extract_time_series_grid(
            lst_collection_scaled_1km,
            value_base_key, # Name of the band in the collection
            lst_native_scale
        )

        # Process and store result (dictionary of DataFrames)
        self.results[data_key] = self._process_grid_features(grid_features, self.grid_point_ids, value_base_key)


    # --- Public Methods for Accessing and Plotting Data ---

    def get_data(self, data_key):
        """
        Retrieves processed data (DataFrame or dict of DataFrames) from results.

        Args:
            data_key (str): The key used when fetching the data (e.g., 'LST_Celsius_1000m').

        Returns:
            pandas.DataFrame or dict or None: The requested data, or None if not found.
        """
        return self.results.get(data_key)

    def plot_time_series(self, data_keys=None, title=None, **kwargs):
        """
        Plots one or more time series stored in self.results on a single graph.

        Args:
            data_keys (list, optional): List of keys from self.results to plot.
                                       If None, plots all available single-point DataFrames.
            title (str, optional): Custom title for the plot.
            **kwargs: Additional keyword arguments passed to ax.plot().
        """
        print("\n--- Generating Combined Plot ---")
        if data_keys is None:
            # Plot all results that are DataFrames (i.e., single point results)
            data_keys = [k for k, v in self.results.items() if isinstance(v, pd.DataFrame)]

        if not data_keys:
            print("No data available to plot.")
            return

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_count = 0

        for key in data_keys:
            df = self.get_data(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                 # Assuming DataFrame has one column besides index
                 value_col = df.columns[0]
                 ax.plot(df.index, df[value_col], marker='.', linestyle='-',
                         markersize=5, label=key, **kwargs)
                 plot_count += 1
            elif df is None:
                 print(f"Warning: Data for key '{key}' not found or is None.")
            else:
                 print(f"Warning: Data for key '{key}' is not a plottable DataFrame (found type: {type(df)}). Skipping.")


        if plot_count > 0:
            plot_title = title if title else f'Temperature Comparison at ({self.latitude:.3f}, {self.longitude:.3f})'
            ax.set_title(f'{plot_title}\n{self.start_date.strftime("%Y-%m-%d")} to {self.end_date.strftime("%Y-%m-%d")}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Temperature (°C)')
            ax.legend()
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=15))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            plt.show()
        else:
            print("No valid data series found to generate a plot.")

    def plot_grid_time_series(self, grid_data_key, title=None, alpha=0.6, plot_mean=False):
        """
        Plots all time series from a grid result dictionary on a single graph.

        Args:
            grid_data_key (str): The key in self.results holding the dictionary of grid DataFrames.
            title (str, optional): Custom title for the plot.
            alpha (float): Transparency level for individual grid point lines.
            plot_mean (bool): If True, also calculates and plots the mean across all grid points.
        """
        print(f"\n--- Generating Plot for Grid Data: {grid_data_key} ---")
        grid_data_dict = self.get_data(grid_data_key)

        if not isinstance(grid_data_dict, dict):
            print(f"Error: Data for key '{grid_data_key}' is not a dictionary as expected for grid results.")
            return

        valid_dfs = {k: df for k, df in grid_data_dict.items() if isinstance(df, pd.DataFrame) and not df.empty}

        if not valid_dfs:
            print("No valid DataFrames found within the grid data dictionary to plot.")
            return

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_count = 0

        # Plot individual grid time series
        print(f"Plotting {len(valid_dfs)} individual time series from the grid...")
        all_series_for_mean = [] # Collect series for mean calculation if needed
        for point_id, df in valid_dfs.items():
            value_col = df.columns[0]
            ax.plot(df.index, df[value_col], linestyle='-', alpha=alpha, linewidth=1, label=None) # No individual labels for legend
            if plot_mean:
                 all_series_for_mean.append(df[value_col])
            plot_count += 1

        # Plot mean if requested
        if plot_mean and all_series_for_mean:
             # Concatenate series, aligning by index, then calculate row-wise mean
             mean_series = pd.concat(all_series_for_mean, axis=1).mean(axis=1)
             ax.plot(mean_series.index, mean_series, color='black', linestyle='--', linewidth=2, label='Grid Mean')
             print("Plotted grid mean.")

        plot_title = title if title else f'Grid Time Series Comparison ({grid_data_key}) at ({self.latitude:.3f}, {self.longitude:.3f})'
        ax.set_title(f'{plot_title}\n{self.start_date.strftime("%Y-%m-%d")} to {self.end_date.strftime("%Y-%m-%d")}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (°C)')
        if plot_mean: # Only show legend if mean is plotted
             ax.legend()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # !! Define Parameters !!
    center_lat = 40.055489
    center_lon = -113.41842
    data_year = 2023
    data_month = 7 # July
    gee_project = 'ee-amizorach' # Optional: Replace with your GEE project ID if needed

    try:
        # 1. Initialize the class
        temp_fetcher = GEETempTimeSeries(
            latitude=center_lat,
            longitude=center_lon,
            year=data_year,
            month=data_month,
            project_id=gee_project
        )

        # 2. Fetch data for single points (native and aggregated LST, ERA5)
        #temp_fetcher.fetch_modis_lst(resolution=1000) # Fetch 1km LST
        #temp_fetcher.fetch_modis_lst(resolution=4000) # Fetch 4km Aggregated LST
        #temp_fetcher.fetch_era5_air_temp()             # Fetch ERA5

        # 3. Plot the single point results together
        #temp_fetcher.plot_time_series()

        # 4. Fetch data for the 5x5 grid (using MODIS 1km LST)
        grid_data_key = 'MODIS_LST_5x5_Grid_1km'
        temp_fetcher.fetch_modis_lst_grid(
            grid_dim=5,                 # 5x5 grid
            spacing_meters=1000,        # ~1km spacing (matches native MODIS)
            data_key=grid_data_key
        )

        # 5. Plot the grid results
        grid_data = temp_fetcher.get_data(grid_data_key)
        if grid_data:
             temp_fetcher.plot_grid_time_series(grid_data_key=grid_data_key, plot_mean=True)

        # You can access specific results like this:
        # modis_1km_df = temp_fetcher.get_data('LST_Celsius_1000m')
        # grid_results_dict = temp_fetcher.get_data('MODIS_LST_5x5_Grid_1km')
        # if grid_results_dict:
        #     center_point_df = grid_results_dict.get('row_2_col_2') # Assuming 5x5 grid

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        import traceback
        traceback.print_exc()