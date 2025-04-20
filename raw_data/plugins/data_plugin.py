from datetime import datetime
import json
import os
import pprint

import pandas as pd

from dawn_vok.utils.dir_utils import DirUtils

class NormalizeRange:
    sensor_ranges = {
        'temperature': {'min': -5, 'max': 50},
        'humidity': {'min': 0, 'max': 100},
        'rainfall': {'min': 0, 'max': 100},
        'grass_temperature': {'min': -5, 'max': 50},
        'max_temperature': {'min': -5, 'max': 50},
        'min_temperature': {'min': -5, 'max': 50},
        'diffused_radiation': {'min': 0, 'max': 1000},
        'global_radiation': {'min': 0, 'max': 1000},
        'direct_radiation': {'min': 0, 'max': 1000},
        'wind_direction': {'min': 0, 'max': 360},
        'gust_wind_direction': {'min': 0, 'max': 360},
        'wind_speed': {'min': 0, 'max': 100},
        'max_1_minute_wind_speed': {'min': 0, 'max': 100},
        'max_10_minute_wind_speed': {'min': 0, 'max': 100},
        'time_ending_max_10_minute_wind_speed': {'min': 0, 'max': 100},
        'gust_wind_speed': {'min': 0, 'max': 100},
        'standard_deviation_wind_direction': {'min': 0, 'max': 360},
        'wet_temperature': {'min': -5, 'max': 50}
    }
   

class DataPlugin:
        
    def __init__(self, provider_id, data_dir, pickle_dir):
        self.provider_id = provider_id
        self.data_dir = data_dir
        self.pickle_dir = pickle_dir

    def load_csv(self,  file_name):
        """
        Load a CSV file into a pandas DataFrame.

        Parameters:
        file_path (str): The path to the CSV file.

        Returns:
        pd.DataFrame: The loaded DataFrame.
        """
        path = DirUtils.get_raw_data_path(file_name=file_name, path=self.data_dir)
        try:
            self.df = pd.read_csv(path, encoding='utf-8', comment='#')
            return self.df
        except Exception as e:
            print(f"Failed to load file: {e}")
            return None
        
    def load_excel(self,  file_name):
        """
        Load an Excel file into a pandas DataFrame.

        Parameters:
        file_name (str): The name of the Excel file to load.

        Returns:
        pd.DataFrame: The loaded DataFrame.
        """
        path = DirUtils.get_raw_data_path(file_name=file_name, path=self.data_dir)
        try:
            self.df = pd.read_excel(path)
            return self.df
        except Exception as e:
            print(f"Failed to load file: {e}")
            return None
            
    def load_raw_data(self, file_name):
        if file_name.endswith('.csv'):
            return self.load_csv(file_name)
        elif file_name.endswith('.xlsx'):
            return self.load_excel(file_name)
        else:
            raise ValueError(f"Unsupported file type: {file_name}")


    def normalize_data(self, df, ndf, column, normalize_range):
        """
        Linearly normalize column values using provided min/max.
        Formula: (value - min) / (max - min)
        
        Special case: -5 remains unchanged.
        """
        
        if ndf is None:
            ndf = pd.DataFrame()
        min_val = normalize_range.get('min', 0)
        max_val = normalize_range.get('max', 100)
        range_span = max_val - min_val

        def normalize(x):
            if x == -5:
                return -5
            return (x - min_val) / range_span

        ndf[column] = df[column].apply(normalize)
        return ndf
    
   
    
class IMSDataPlugin(DataPlugin):
    # col_names ={'Station':'station', 'Date & Time (UTC)':'datetime', 'Relative humidity (%)':'humidity', 'Temperature (°C)':'temperature', 'Maximum temperature (°C)':'max_temperature', 'Minimum temperature (°C)':'min_temperature', 
    #         'Grass temperature (°C)':'grass_temperature', 'Rainfall (mm)':'rainfall'}
    col_names ={'Station':'station', 'Date & Time (UTC)':'datetime', 'Relative humidity (%)':'humidity', 'Temperature (°C)':'temperature', 'Maximum temperature (°C)':'max_temperature', 'Minimum temperature (°C)':'min_temperature', 
            'Grass temperature (°C)':'grass_temperature', 'Rainfall (mm)':'rainfall',
            'Diffused radiation (W/m^2)':'diffused_radiation', 'Global radiation (W/m^2)':'global_radiation', 'Direct radiation (W/m^2)':'direct_radiation',
            'Wind direction (°)':'wind_direction', 'Gust wind direction (°)':'gust_wind_direction', 'Wind speed (m/s)':'wind_speed',
            'Maximum 1 minute wind speed (m/s)':'max_1_minute_wind_speed', 'Maximum 10 minutes wind speed (m/s)':'max_10_minute_wind_speed',
            'Time ending maximum 10 minutes wind speed (hhmm)':'time_ending_max_10_minute_wind_speed',
            'Gust wind speed (m/s)':'gust_wind_speed', 'Standard deviation wind direction (°)':'standard_deviation_wind_direction',
            'Wet Temperature (°C)':'wet_temperature'
            }
    used_columns = ['temperature', 'humidity', 'rainfall', 'grass_temperature', 'max_temperature', 'min_temperature', 'diffused_radiation', 'global_radiation', 'direct_radiation', 'wind_direction', 'gust_wind_direction', 'wind_speed', 'max_1_minute_wind_speed', 'max_10_minute_wind_speed', 'time_ending_max_10_minute_wind_speed', 'gust_wind_speed', 'standard_deviation_wind_direction', 'wet_temperature']
    def __init__(self):
        super().__init__('ims', 'provider/raw/ims', 'provider/pickle/ims')
        self.data_dir = 'provider/raw/ims'
        self.pickle_dir = 'provider/pickle/ims'
        
     
    def set_columns(self, df):
        df.rename(columns=self.col_names, inplace=True)
        return df
    
    def set_index(self, df, full_range=True):
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')
        df.set_index('datetime', inplace=True, drop=False)
        if full_range:
            full_range = pd.date_range(
                start=df.index.min().floor('D'),
                end=df.index.max().ceil('D') - pd.Timedelta(minutes=10),
                freq='10min'
            )
            df = df[~df.index.duplicated(keep='first')]

            df = df.reindex(full_range)
        return df
            
    def prepare_data(self, file_name):
        df = self.load_raw_data(file_name)
        df = self.set_columns(df)
        df = self.set_index(df)
        df = self.clean_data(df)
        # df = self.save_pickle(df, file_name)
        return df
    
    def prepare_pickle(self, df, ndf):
        if ndf is None:
            ndf = pd.DataFrame()
        for column in self.used_columns:
            if column in df.columns:
                ndf = self.normalize_data(df=df, ndf=ndf, column=column, normalize_range=NormalizeRange.sensor_ranges.get(column))
        return ndf
    
    def clean_data(self, df):
        df.fillna(-5, inplace=True)
        station_id = df['station'].iloc[0]
        df.drop(columns=['station'], inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df['station'] = station_id
        return df


    
    def create_pickles(self, df, file_name, by_column=True, start_date = datetime(2022, 1, 1)):
        plist = []
        vlist = []
        if start_date is not None and start_date < df.index.min():
            full_range = pd.date_range(
                start=start_date,
                end=df.index.max().ceil('D') - pd.Timedelta(minutes=10),
                freq='10min'
            )
            df = df.reindex(full_range)
        df.fillna(-5, inplace=True)
        #we try to load the pickle file if it exists
        if by_column:
            if file_name.endswith('.pkl'):
                file_name = file_name.replace('.pkl', '')
            for column in df.columns:
                path = DirUtils.get_raw_data_path(file_name=f'{file_name}_{column}.pkl', path=self.pickle_dir)
                if os.path.exists(path):
                    #if the file exists, we load it
                    #and join the new data to the existing data
                    vdf = pd.read_pickle(path)
                    vdf = vdf.reindex(vdf.index.union(df.index))
                    # then update vdf’s column with df’s values wherever they exist
                    vdf.update(df[[column]])
                    #they should join them by the index overiding data if the data exists
                    # vdf = vdf.join(df[[column]], how='outer')
                else:
                    vdf = df[[column]]
                vdf.to_pickle(path)
                plist.append(path)
                vlist.append(vdf)
        else:
            if not file_name.endswith('.pkl'):
                file_name = file_name + '.pkl'
            path = DirUtils.get_raw_data_path(file_name=file_name, path=self.pickle_dir)
            if os.path.exists(path):
                #if the file exists, we load it
                #and join the new data to the existing data
                vdf = pd.read_pickle(path)
                vdf = vdf.reindex(vdf.index.union(df.index))
                vdf.update(df)
                # vdf = vdf.join(df, how='outer')
            else:
                vdf = df
            df.to_pickle(path)
            plist.append(path)
            vlist.append(df)
        return plist, vlist

    def create_data_provider(self, file_name, pickle_file_name=None, station_di=None):
        from dawn_vok.raw_data.time_series.ts_dp import IMSTimeSeriesDataProvider

        df = self.prepare_data(file_name)
        station_id = df['station'].iloc[0].replace(' ', '_').lower()
        sensor_type = df.columns[1]
        station_info = station_di.get(station_id, None)
        if station_info is None:
            raise ValueError(f'Station {station_id} not found')
        source_id = f'ims_{station_info["name"]}'
        pprint.pp(station_info)
        ndf = self.prepare_pickle(df=df, ndf=None)
        pickle_file_name = f'{source_id}' if pickle_file_name is None else pickle_file_name
        plist, vlist = self.create_pickles(df=ndf, file_name=pickle_file_name, by_column=False)
        for p, v in zip(plist, vlist):
            sensor_type = v.columns[0]

            dp = IMSTimeSeriesDataProvider(source_id=source_id, 
                                           station_info=station_info,
                                           sensor_type=sensor_type, raw_data_file_name=file_name, pickle_file_name=p,
                                       raw_data_path=self.data_dir, pickle_path=self.pickle_dir, start_date=df.index.min(), end_date=df.index.max(), frequency=10*60)
            dp.populate_from_dict(v.to_dict())
            dp.save_to_db()
        return dp

class SynopticDataPlugin(DataPlugin):
    # 'Station_ID', 'Date_Time', 'altimeter_set_1', 'air_temp_set_1',
    #    'relative_humidity_set_1', 'wind_speed_set_1', 'wind_direction_set_1',
    #    'wind_gust_set_1', 'precip_accum_24_hour_set_1',
    #    'precip_accum_since_local_midnight_set_1', 'wind_chill_set_1d',
    #    'wind_cardinal_direction_set_1d', 'heat_index_set_1d',
    #    'dew_point_temperature_set_1d', 'pressure_set_1d',
    #    'sea_level_pressure_set_1d'],
    col_names = {'Station_ID':'station', 'Date_Time':'datetime', 'altimeter_set_1':'altimeter', 'air_temp_set_1':'temperature', 'relative_humidity_set_1':'humidity', 'wind_speed_set_1':'wind_speed', 'wind_direction_set_1':'wind_direction', 'wind_gust_set_1':'gust_wind_speed', 'precip_accum_24_hour_set_1':'precip_accum_24_hour', 'precip_accum_since_local_midnight_set_1':'precip_accum_since_local_midnight', 'wind_chill_set_1d':'wind_chill', 'wind_cardinal_direction_set_1d':'wind_cardinal_direction', 'heat_index_set_1d':'heat_index', 'dew_point_temperature_set_1d':'dew_point_temperature', 'pressure_set_1d':'pressure', 'sea_level_pressure_set_1d':'sea_level_pressure'}
    used_columns = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'gust_wind_speed',  'wind_chill', 'wind_cardinal_direction', 'heat_index', 'dew_point_temperature', 'pressure', 'sea_level_pressure']
    def __init__(self):
        super().__init__('synoptic', 'provider/raw/synoptic', 'provider/pickle/synoptic')
        self.data_dir = 'provider/raw/synoptic'
        self.pickle_dir = 'provider/pickle/synoptic'

    def prepare_data(self, file_name):
        df = self.load_raw_data(file_name)
        print(df.columns)
        df = self.set_columns(df)
        df = self.set_index(df)
        df = self.clean_data(df)
        return df
    
    def set_columns(self, df):
        df.rename(columns=self.col_names, inplace=True)
        return df
    
    def set_index(self, df):
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%dT%H:%M:%SZ')
        df.set_index('datetime', inplace=True, drop=False)
        return df
    
    def clean_data(self, df):
        df.fillna(-5, inplace=True)
        station_id = df['station'].iloc[0]
        df.drop(columns=['station'], inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df['station'] = station_id
        return df
    
    def create_pickles(self, df, file_name, by_column=True, start_date = datetime(2022, 1, 1)):
        plist = []
        vlist = []
        if start_date is not None and start_date < df.index.min():
            full_range = pd.date_range(
                start=start_date,
                end=df.index.max().ceil('D') - pd.Timedelta(minutes=10),
                freq='10min'
            )
            df = df.reindex(full_range)
        df.fillna(-5, inplace=True)
        if by_column:
            if file_name.endswith('.pkl'):
                file_name = file_name.replace('.pkl', '')
            for column in df.columns:
                path = DirUtils.get_raw_data_path(file_name=f'{file_name}_{column}.pkl', path=self.pickle_dir)
                if os.path.exists(path):
                    vdf = pd.read_pickle(path)
                    vdf = vdf.reindex(vdf.index.union(df.index))
                    vdf.update(df[[column]])
                else:
                    vdf = df[[column]]
                vdf.to_pickle(path) 
                plist.append(path)
                vlist.append(vdf)
        else:
            if not file_name.endswith('.pkl'):
                file_name = file_name + '.pkl'
            path = DirUtils.get_raw_data_path(file_name=file_name, path=self.pickle_dir)
            if os.path.exists(path):
                vdf = pd.read_pickle(path)
                vdf = vdf.reindex(vdf.index.union(df.index))
                vdf.update(df)
            else:
                vdf = df
            df.to_pickle(path)      
            plist.append(path)
            vlist.append(df)
        return plist, vlist 
    
    def create_data_provider(self, file_name, pickle_file_name=None, station_di=None, station_id=None):

        df = self.prepare_data(file_name)
        sensor_type = df.columns[1]
        if station_id is None:
            station_id = file_name.replace('.csv', '').replace('.xlsx', '').replace(' ', '_').replace('-', '_').replace('.', '_').lower()
        source_id = f'synoptic_{station_id}_{sensor_type}'
        
        pickle_file_name = f'{source_id}' if pickle_file_name is None else pickle_file_name
        plist, vlist = self.create_pickles(df=df, file_name=pickle_file_name)
        # for p, v in zip(plist, vlist):
        #     sensor_type = v.columns[0]  
        #     dp = SynopticTimeSeriesDataProvider(source_id=source_id, 
        #                                         station_info=station_info,
        #                                         sensor_type=sensor_type, raw_data_file_name=file_name, pickle_file_name=p,
        #                                     raw_data_path=self.data_dir, pickle_path=self.pickle_dir, start_date=df.index.min(), end_date=df.index.max(), frequency=10*60)
        #     dp.populate_from_dict(v.to_dict())
        #     dp.save_to_db()
        return plist
            

if __name__ == '__main__':
    plugin = IMSDataPlugin()
    # df = plugin.prepare_data('data_202504081943.csv')

    # print(df)
    # ndf = plugin.prepare_pickle(df=df, ndf=None)
    # plugin.save_pickle(ndf, 'data_paran.pkl')
    # print(ndf)
    station_di_path = DirUtils.get_raw_data_path('ims_stations.json', 'provider/raw/ims')
    with open(station_di_path, 'r') as f:
        station_di = json.load(f)
    station_di = {d['name']: d for d in station_di}
    # pprint.pp(station_di.keys())
    # dp = plugin.create_data_provider(file_name='data_202504081943.csv', station_di=station_di)
    # pprint.pp(dp.to_dict())
    plugin.create_data_provider(file_name='ims_afeq_78.csv', station_di=station_di)
    # plugin = SynopticDataPlugin()
    # plugin.create_data_provider(file_name='C0933.2025-04-16.csv')
    #     df = plugin.prepare_data('C0933.2025-04-16.csv')
    # pprint.pp(df)
    # plist, vlist = plugin.create_pickles(df=df, file_name='C0933.2025-04-16')
    # pprint.pp(plist)
    # pprint.pp(vlist)
