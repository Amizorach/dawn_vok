import os

import pandas as pd
import logging

from dawn_vok.utils.dir_utils import DirUtils 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataProvider:
    def __init__(self, provider_id, provider_name):
        self.provider_id = provider_id
        self.provider_name = provider_name

    def to_dict(self):
        return {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name
        }
    
    def populate_from_dict(self, data_dict):
        self.provider_id = data_dict.get("provider_id")
        self.provider_name = data_dict.get("provider_name")

    # def prepare_df(self):
    #     raise NotImplementedError("Subclasses must implement this method")

    # def get_data(self):
    #     raise NotImplementedError("Subclasses must implement this method")
    
    


class SynopticProvider(DataProvider):
    synoptic_sensor_dict = {
        'air_temp_high_24_hour': 'air_temp_high_24_hour_set_1',
        'air_temp_high_6_hour': 'air_temp_high_6_hour_set_1',
        'air_temp_low_24_hour': 'air_temp_low_24_hour_set_1',
        'air_temp_low_6_hour': 'air_temp_low_6_hour_set_1',
        'ceiling': 'ceiling_set_1',
        'cloud_layer_1': 'cloud_layer_1_set_1d',
        'cloud_layer_1_code': 'cloud_layer_1_code_set_1',
        'cloud_layer_2': 'cloud_layer_2_set_1d',
        'cloud_layer_2_code': 'cloud_layer_2_code_set_1',
        'cloud_layer_3': 'cloud_layer_3_set_1d',
        'cloud_layer_3_code': 'cloud_layer_3_code_set_1',
        'datetime': 'Date_Time', # Target standard name for the date column
        'dew_point': 'dew_point_temperature_set_1d',
        'dew_point_temperature': 'dew_point_temperature_set_1d', # Allow alias
        'heat_index': 'heat_index_set_1d',
        'humidity': 'relative_humidity_set_1',
        'metar': 'metar_set_1',
        'peak_wind_direction': 'peak_wind_direction_set_1',
        'peak_wind_speed': 'peak_wind_speed_set_1',
        'precip_accum_24_hour': 'precip_accum_24_hour_set_1',
        'precip_accum_one_hour': 'precip_accum_one_hour_set_1',
        'precip_accum_six_hour': 'precip_accum_six_hour_set_1',
        'precip_accum_three_hour': 'precip_accum_three_hour_set_1',
        'pressure': 'pressure_set_1d',
        'pressure_change_code': 'pressure_change_code_set_1',
        'pressure_tendency': 'pressure_tendency_set_1',
        'sea_level_pressure': 'sea_level_pressure_set_1d',
        'station_id': 'Station_ID', # Target standard name for station identifier
        'temperature': 'air_temp_set_1',
        'visibility': 'visibility_set_1',
        'volt': 'volt_set_1',
        'weather_cond_code': 'weather_cond_code_set_1',
        'weather_condition': 'weather_condition_set_1d',
        'weather_summary': 'weather_summary_set_1d',
        'wind_cardinal_direction': 'wind_cardinal_direction_set_1d',
        'wind_chill': 'wind_chill_set_1d',
        'wind_direction': 'wind_direction_set_1',
        'wind_gust': 'wind_gust_set_1',
        'wind_speed': 'wind_speed_set_1'
    }
    
    # Define which of the *standard* column names are expected to be numeric
    _numeric_columns = [
        'air_temp_high_24_hour', 'air_temp_high_6_hour', 'air_temp_low_24_hour',
        'air_temp_low_6_hour', 'ceiling', 'dew_point', 'heat_index', 'humidity', 
        'peak_wind_direction', 'peak_wind_speed', 'precip_accum_24_hour', 
        'precip_accum_one_hour', 'precip_accum_six_hour', 'precip_accum_three_hour',
        'pressure', 'pressure_tendency', 'sea_level_pressure', 'temperature',
        'visibility', 'volt', 'wind_chill', 'wind_direction', 'wind_gust', 'wind_speed'
    ]
    
    # Define essential *standard* columns expected after renaming
    _required_columns = ['station_id', 'datetime']


    def __init__(self):
        super().__init__(provider_id="pr_synoptic", provider_name="Synoptic")

    def prepare_df(self, df):
        """
        Prepares the raw Synoptic DataFrame by renaming columns according
        to synoptic_sensor_dict, converting data types, and setting the index.

        Args:
            df: The raw pandas DataFrame loaded from a Synoptic CSV file.

        Returns:
            The prepared DataFrame with standardized columns and types, 
            or None if preparation fails (e.g., missing essential columns).
        """
        if df is None or df.empty:
            logging.warning("Input DataFrame is empty or None. Skipping preparation.")
            return None
            
        logging.info("Starting DataFrame preparation...")
        
        # 1. Rename columns: Create a map from original (Synoptic) names to standard names
        # The rename function needs {original_name: new_name}
        rename_map = {v: k for k, v in self.synoptic_sensor_dict.items() if v in df.columns}
        
        if not rename_map:
             logging.warning("No columns to rename based on the provided dictionary and DataFrame columns.")
             # Decide if we should proceed or return None/original df
        
        df = df.rename(columns=rename_map)
        logging.info(f"Renamed columns. Current columns: {df.columns.tolist()}")

        # 2. Check for essential columns *after* renaming
        missing_required = [col for col in self._required_columns if col not in df.columns]
        if missing_required:
            logging.error(f"Missing essential columns after renaming: {missing_required}")
            # Fail preparation if essential columns are missing
            return None 

        # 3. Convert data types
        # Convert datetime column
        if 'datetime' in df.columns:
            # Attempt conversion, turning errors into NaT (Not a Time)
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            
            # Check how many rows were affected by conversion errors
            invalid_dates = df['datetime'].isnull().sum()
            if invalid_dates > 0:
                 logging.warning(f"{invalid_dates} rows had invalid datetime format and were set to NaT.")
                 # Option: Drop rows with invalid dates
                 original_rows = len(df)
                 df.dropna(subset=['datetime'], inplace=True)
                 logging.warning(f"Dropped {original_rows - len(df)} rows due to invalid datetime format.")
                 if df.empty:
                     logging.error("DataFrame is empty after dropping rows with invalid dates.")
                     return None
        else:
            # This check is technically redundant due to _required_columns check, but good practice
            logging.error("Cannot proceed without 'datetime' column.")
            return None

        # Convert numeric columns
        for col in self._numeric_columns:
            if col in df.columns:
                # Only attempt conversion if the column exists
                original_dtype = df[col].dtype
                # Convert to numeric, turning errors into NaN (Not a Number)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Log if NaNs were introduced and the original type wasn't already numeric (float/int)
                if df[col].isnull().any() and not pd.api.types.is_numeric_dtype(original_dtype): 
                    logging.warning(f"Column '{col}' contained non-numeric values. Coerced to NaN.")
        
        # 4. Set index (Optional but often useful for time series)
        if 'datetime' in df.columns:
             # Set the 'datetime' column as the DataFrame index
             df.set_index('datetime', inplace=True)
             # Sort the DataFrame by the new index (time)
             df.sort_index(inplace=True)
             logging.info("Set 'datetime' as index and sorted.")

        logging.info("DataFrame preparation complete.")
        # Use logging.debug for verbose output like head()
        logging.debug(f"Prepared DataFrame head:\n{df.head()}")
        
        return df
    
    def get_data_from_file(self, file_path):
        """
        Loads data from a Synoptic CSV file located relative to the provider's raw data directory.

        Args:
            file_path: Relative path to the CSV file (e.g., 'data_4.csv').

        Returns:
            A pandas DataFrame containing the raw data, or None if loading fails.
        """
        try:
            # Construct the full path to the file
            raw_dir = DirUtils.get_raw_data_dir(path='provider/raw/synoptic')
            raw_file = os.path.join(raw_dir, file_path)
            logging.info(f"Attempting to load data from: {raw_file}")

            # Check if file exists before attempting to read
            if not os.path.exists(raw_file):
                raise FileNotFoundError(f"File not found at calculated path: {raw_file}")

            # Read the CSV, ignoring lines starting with '#'
            # Consider adding 'low_memory=False' if getting DtypeWarning for large files with mixed types
            df = pd.read_csv(raw_file, encoding='utf-8', comment='#')
            
            logging.info(f"Successfully loaded data from {raw_file}. Shape: {df.shape}")
            # Use logging.debug for verbose output
            logging.debug(f"Raw DataFrame head:\n{df.head()}")
            logging.debug(f"Raw DataFrame columns: {df.columns.tolist()}")
            return df

        except FileNotFoundError as e:
            # Log specific file not found error
            logging.error(e)
            return None
        except pd.errors.EmptyDataError:
            # Log error if the file is empty
            logging.error(f"File is empty: {raw_file}")
            return None
        except pd.errors.ParserError as e:
            # Log error if CSV parsing fails
            logging.error(f"Error parsing CSV file {raw_file}: {e}")
            return None
        except Exception as e:
            # Log any other unexpected errors during file loading
            logging.error(f"An unexpected error occurred while loading {raw_file}: {e}", exc_info=True) # exc_info=True logs stack trace
            return None



if __name__ == "__main__":
    provider = SynopticProvider()
    df = provider.get_data_from_file(file_path='C0933.2025-04-16.csv')
    df = provider.prepare_df(df)