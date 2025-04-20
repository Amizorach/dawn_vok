import json
import os
import pprint
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.raw_data.plugins.data_plugin import IMSDataPlugin
from dawn_vok.utils.dir_utils import DirUtils


class DSource:
    @classmethod
    def get_collection_name(cls):
        return 'source_md'
    
    @classmethod
    def get_db_name(cls):
        return 'meta_data'
    
    def __init__(self, source_id, provider_id, source_name, source_type, location=None, height=None, structure_type=None, active_since=None, data_since=None):
        self.source_id = source_id
        self.provider_id = provider_id
        self.source_name = source_name
        self.source_type = source_type
        self.location = location or DLocation()
        self.structure_type = structure_type or 'unknown'
        self.height = height or 0
        self.active_since = active_since or None
        self.data_since = data_since or None
        self.data_map = [0]*10

    def to_dict(self):
        return {
            'source_id': self.source_id,
            'provider_id': self.provider_id,
            'source_name': self.source_name,
            'source_type': self.source_type,
            'location': self.location.to_dict(),
            'structure_type': self.structure_type,
            'height': self.height,
            'active_since': self.active_since,
            'data_since': self.data_since,
            'data_map': self.data_map
        }
    
    def populate_from_dict(self, data_dict):
        self.source_id = data_dict.get('source_id', self.source_id)
        self.source_name = data_dict.get('source_name', self.source_name)
        self.source_type = data_dict.get('source_type', self.source_type)
        self.location.populate_from_dict(data_dict.get('location', self.location.to_dict()))
        self.structure_type = data_dict.get('structure_type', self.structure_type)
        self.height = data_dict.get('height', self.height)
        self.active_since = data_dict.get('active_since', self.active_since)
        self.data_since = data_dict.get('data_since', self.data_since)
        self.data_map = data_dict.get('data_map', self.data_map)
        return self
    
    def save_to_db(self):
        collection = MongoUtils.get_collection(self.get_db_name(), self.get_collection_name())
        collection.update_one({'_id': self.source_id}, {'$set': self.to_dict()}, upsert=True)
        return self
    
class DSensorSource(DSource):
    def __init__(self, source_id, provider_id, source_name, source_type, sensor_type_list, sensor_info_id_list, location=None):
        super().__init__(source_id, provider_id, source_name, source_type, location)
        self.sensor_type_list = sensor_type_list or []
        self.sensor_info_id_list = sensor_info_id_list or []
    
    def to_dict(self):
        data_dict = super().to_dict()
        data_dict['sensor_type_list'] = self.sensor_type_list
        data_dict['sensor_info_id_list'] = self.sensor_info_id_list
        return data_dict
    
    def populate_from_dict(self, data_dict):
        super().populate_from_dict(data_dict)
        self.sensor_type_list = data_dict.get('sensor_type_list', self.sensor_type_list)
        self.sensor_info_id_list = data_dict.get('sensor_info_id_list', self.sensor_info_id_list)
        return self
    
class IMSSensorSource(DSensorSource):
    def __init__(self, source_id, source_name, source_type, sensor_type_list=None, sensor_info_id_list=None, location=None, station_info=None):
        super().__init__(source_id, 'ims', source_name, source_type, sensor_type_list, sensor_info_id_list, location)
        self.sensor_type_list = sensor_type_list or ['temperature', 'humidity', 'wind_speed', 'wind_direction']
        self.sensor_info_id_list = sensor_info_id_list or ['temperature', 'humidity', 'wind_speed', 'wind_direction']
        self.ims_api_id = None
        self.ims_api_rain_id = None

        self.populate_from_file(station_info)
        
    def populate_from_file(self, station_info):
        if station_info is None:
            file_path = pt = DirUtils.get_raw_data_path('ims_stations.json', 'provider/raw/ims')

            with open(file_path, 'r') as file:
                data_dict = json.load(file)
            station_info = next((item for item in data_dict if item['name'] == self.source_name), None)
        if station_info is None:
            raise ValueError(f'Station {self.source_name} not found')
        pprint.pp(station_info)
        self.location.update_location(lat=station_info.get('lat', None), lon=station_info.get('lon', None), alt=station_info.get('alt', None))
        self.active_since = station_info.get('date', None)
        self.structure_type = station_info.get('shelter', None)
        if isinstance(self.structure_type, str):
            self.structure_type = 10
        elif isinstance(self.structure_type, int):
            self.structure_type = self.structure_type
        else:
            self.structure_type = -1
        self.height = station_info.get('height', 0) 
        if isinstance(self.height, str):
            self.height = 0
        elif isinstance(self.height, int):
            self.height = self.height
        else:
            self.height = 0
        self.sensor_type_list = station_info.get('sensor_type_list', self.sensor_type_list)
        self.sensor_info_id_list = station_info.get('sensor_info_id_list', self.sensor_info_id_list)
        self.ims_api_id = station_info.get('api_id', None)
        self.ims_api_rain_id = station_info.get('api_rain_id', None)
        self.source_id = f'ims_{self.source_name}_{self.ims_api_id}'

    def to_dict(self):
        data_dict = super().to_dict()
        data_dict['sensor_type_list'] = self.sensor_type_list
        data_dict['sensor_info_id_list'] = self.sensor_info_id_list
        data_dict['ims_api_id'] = self.ims_api_id
        data_dict['ims_api_rain_id'] = self.ims_api_rain_id
        return data_dict
    
    def populate_from_dict(self, data_dict):
        super().populate_from_dict(data_dict)
        self.sensor_type_list = data_dict.get('sensor_type_list', self.sensor_type_list)
        self.sensor_info_id_list = data_dict.get('sensor_info_id_list', self.sensor_info_id_list)
        self.ims_api_id = data_dict.get('ims_api_id', self.ims_api_id)
        self.ims_api_rain_id = data_dict.get('ims_api_rain_id', self.ims_api_rain_id)
        return self
    
    
    

class DLocation:
    def __init__(self, lat=None, lon=None, alt=None):
        self.lat = lat or -5
        self.lon = lon or -5
        self.alt = alt or -500
        self.lat = self.clean_coordinate(self.lat)
        self.lon = self.clean_coordinate(self.lon)
        
    def to_dict(self):
        return {
            'lat': self.lat,
            'lon': self.lon,
            'alt': self.alt
        }
    
    def clean_coordinate(self, val):
        if val == None or val == -5:
            return -5   
        try:
            # If it's a string, strip spaces and remove 'º' if present
            if isinstance(val, str):
                val = val.strip().replace('º', '')
            return float(val)
        except (ValueError, TypeError):
            return None  # or np.nan if you prefer


    def populate_from_dict(self, data_dict):
        self.lat = data_dict.get('lat', self.lat)
        self.lon = data_dict.get('lon', self.lon)
        self.alt = data_dict.get('alt', self.alt)
        return self
    
    def update_location(self, lat=None, lon=None, alt=None):
        self.lat = lat or self.lat
        self.lon = lon or self.lon
        self.alt = alt or self.alt
        self.lat = self.clean_coordinate(self.lat)
        self.lon = self.clean_coordinate(self.lon)
        return self

class DataProvider:
    @classmethod
    def get_collection_name(cls):
        return 'time_series_providers'
    
    @classmethod
    def load_from_db(cls, dp_id):
        collection = MongoUtils.get_collection(cls.get_db_name(), cls.get_collection_name())
        data_dict = collection.find_one({'_id': dp_id})
        return cls.from_dict(data_dict)
    
    @classmethod
    def get_db_name(cls):
        return 'raw_data'
    
    def __init__(self, dp_id, provider_id=None, source_id=None, raw_data_path=None, raw_data_file_name=None, pickle_path=None, pickle_file_name=None):
        self.dp_id = dp_id
        self.provider_id = provider_id
        self.source_id = source_id
        self.db_name = self.get_db_name()
        self.collection_name = self.get_collection_name()
        self.raw_data_path = raw_data_path
        self.raw_data_file_name = raw_data_file_name
        self.pickle_path = pickle_path
        self.pickle_file_name = pickle_file_name
        self.location = DLocation()

    def to_dict(self):
        return {
            '_id': self.dp_id,
            'dp_id': self.dp_id,
            'provider_id': self.provider_id,
            'source_id': self.source_id,
            'raw_data_path': self.raw_data_path,
            'raw_data_file_name': self.raw_data_file_name,
            'pickle_path': self.pickle_path,
            'pickle_file_name': self.pickle_file_name,
            'location': self.location.to_dict()
        }
    
    def set_location(self, lat=None, lon=None, alt=None):
        self.location.update_location(lat=lat, lon=lon, alt=alt)
        return self
    
    def populate_from_dict(self, data_dict):
        self.dp_id = data_dict.get('_id', self.dp_id)
        self.provider_id = data_dict.get('provider_id', self.provider_id)
        self.source_id = data_dict.get('source_id', self.source_id)
        self.raw_data_path = data_dict.get('raw_data_path', self.raw_data_path)
        self.raw_data_file_name = data_dict.get('raw_data_file_name', self.raw_data_file_name)
        self.pickle_path = data_dict.get('pickle_path', self.pickle_path)
        self.pickle_file_name = data_dict.get('pickle_file_name', self.pickle_file_name)
        self.location.populate_from_dict(data_dict.get('location', self.location.to_dict()))
        return self
    def save_to_db(self):
        if self.dp_id is None:
            raise ValueError('dp_id is required')
        collection = MongoUtils.get_collection(self.db_name, self.collection_name)
        collection.update_one({'_id': self.dp_id}, {'$set': self.to_dict()}, upsert=True)
    
    def get_from_db(self):
        if self.dp_id is None:
            raise ValueError('dp_id is required')
        collection = MongoUtils.get_collection(self.db_name, self.collection_name)
        data_dict = collection.find_one({'_id': self.dp_id})
        self.populate_from_dict(data_dict)
        return self


class SensorDataProvider(DataProvider):
    def __init__(self, dp_id, provider_id=None, source_id=None, sensor_type=None, sensor_value_type=None, data_type=None, data_format=None, raw_data_path=None, raw_data_file_name=None, pickle_path=None, pickle_file_name=None):
        super().__init__(dp_id, provider_id=provider_id, source_id=source_id, raw_data_path=raw_data_path, 
                         raw_data_file_name=raw_data_file_name, pickle_path=pickle_path, pickle_file_name=pickle_file_name)
        self.sensor_type = sensor_type
        self.sensor_value_type = sensor_value_type
        self.data_type = data_type
        self.data_format = data_format

    def to_dict(self):
        data_dict = super().to_dict()
        data_dict['sensor_type'] = self.sensor_type
        data_dict['sensor_value_type'] = self.sensor_value_type
        data_dict['data_type'] = self.data_type
        data_dict['data_format'] = self.data_format
        return data_dict
    
    def populate_from_dict(self, data_dict):
        super().populate_from_dict(data_dict)
        self.sensor_type = data_dict.get('sensor_type', self.sensor_type)
        self.sensor_value_type = data_dict.get('sensor_value_type', self.sensor_value_type)
        self.data_type = data_dict.get('data_type', self.data_type)
        self.data_format = data_dict.get('data_format', self.data_format)
        return self


class SourceDataProvider(SensorDataProvider):
    def __init__(self, dp_id, provider_id=None, source_id=None, sensor_type=None, sensor_value_type=None, data_type=None, data_format=None, raw_data_path=None, raw_data_file_name=None, pickle_path=None, pickle_file_name=None):
        super().__init__(dp_id, provider_id=provider_id, source_id=source_id, raw_data_path=raw_data_path, 
                         raw_data_file_name=raw_data_file_name, pickle_path=pickle_path, pickle_file_name=pickle_file_name)
        self.sensor_type = sensor_type
        self.sensor_value_type = sensor_value_type
        self.data_type = data_type
        self.data_format = data_format
class TimeSeriesDataProvider(SensorDataProvider):

    @classmethod
    def get_collection_name(cls):
        return 'time_series_providers'
    

    def __init__(self,dp_id, provider_id=None, source_id=None, sensor_type=None, sensor_value_type=None, start_date=None, end_date=None, 
                 frequency=None, data_type=None, data_format=None,
                   raw_data_path=None, raw_data_file_name=None, pickle_path=None, pickle_file_name=None):
        super().__init__(dp_id, provider_id=provider_id, source_id=source_id, 
                         sensor_type=sensor_type, sensor_value_type=sensor_value_type,
                         data_type=data_type, data_format=data_format,
                         raw_data_path=raw_data_path, raw_data_file_name=raw_data_file_name, pickle_path=pickle_path, pickle_file_name=pickle_file_name)
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
      
    def to_dict(self):
        data_dict = super().to_dict()
        data_dict['start_date'] = self.start_date
        data_dict['end_date'] = self.end_date
        data_dict['frequency'] = self.frequency
        return data_dict
        
    
    def populate_from_dict(self, data_dict):
        super().populate_from_dict(data_dict)
        self.start_date = data_dict.get('start_date', self.start_date)
        self.end_date = data_dict.get('end_date', self.end_date)
        self.frequency = data_dict.get('frequency', self.frequency)
        return self
    

class IMSTimeSeriesDataProvider(TimeSeriesDataProvider):
    sensor_dict = {
        'temperature': {
            'sensor_type': 'temperature',
            'sensor_value_type': 'celcius',
            'data_type': 'time_series',
            'data_format': 'float'
        },
        'humidity': {
            'sensor_type': 'humidity',
            'sensor_value_type': 'percent',
            'data_type': 'time_series',
            'data_format': 'float'
        },
        'wind_speed': {
            'sensor_type': 'wind_speed',
            'sensor_value_type': 'm/s',
            'data_type': 'time_series',
            'data_format': 'float'
        },
        'wind_direction': { 
            'sensor_type': 'wind_direction',
            'sensor_value_type': 'degree',
            'data_type': 'time_series',
            'data_format': 'float'
        },
        'rainfall': {
            'sensor_type': 'rainfall',
            'sensor_value_type': 'mm',
            'data_type': 'time_series',
            'data_format': 'float'
        },
        'grass_temperature': {
            'sensor_type': 'grass_temperature',
            'sensor_value_type': 'celcius',
            'data_type': 'time_series',
            'data_format': 'float'
        },
        'max_temperature': {
            'sensor_type': 'max_temperature',
            'sensor_value_type': 'celcius',
            'data_type': 'time_series',
            'data_format': 'float'
        },
        'min_temperature': {
            'sensor_type': 'min_temperature',
            'sensor_value_type': 'celcius',
            'data_type': 'time_series',
            'data_format': 'float'
        },
        'diffused_radiation': {
            'sensor_type': 'diffused_radiation',
            'sensor_value_type': 'w/m^2',
            'data_type': 'time_series',
            'data_format': 'float'
        }, 
        'global_radiation': {
            'sensor_type': 'global_radiation',
            'sensor_value_type': 'w/m^2',
            'data_type': 'time_series',
            'data_format': 'float'
        },
        'wind_chill': {
            'sensor_type': 'wind_chill',
            'sensor_value_type': 'celcius',
            'data_type': 'time_series',
            'data_format': 'float'
        }, 
        'gust_wind_speed': {
            'sensor_type': 'gust_wind_speed',
            'sensor_value_type': 'm/s',
            'data_type': 'time_series',
            'data_format': 'float'
        }, 
        'gust_wind_direction': {
            'sensor_type': 'gust_wind_direction',
            'sensor_value_type': 'degree',
            'data_type': 'time_series',
            'data_format': 'float'
        },
        'max_1_minute_wind_speed': {
            'sensor_type': 'max_1_minute_wind_speed',
            'sensor_value_type': 'm/s',
            'data_type': 'time_series',
            'data_format': 'float'
        },
        'max_10_minute_wind_speed': {
            'sensor_type': 'max_10_minute_wind_speed',
            'sensor_value_type': 'm/s',
            'data_type': 'time_series',
            'data_format': 'float'
        },
        'time_ending_max_10_minute_wind_speed': {
            'sensor_type': 'time_ending_max_10_minute_wind_speed',
            'sensor_value_type': 'hhmm',
            'data_type': 'time_series',
            'data_format': 'str'
        },
        'standard_deviation_wind_direction': {
            'sensor_type': 'standard_deviation_wind_direction',
            'sensor_value_type': 'degree',
            'data_type': 'time_series',
            'data_format': 'float'
        },
        'wet_temperature': {
            'sensor_type': 'wet_temperature',
            'sensor_value_type': 'celcius',
            'data_type': 'time_series',
            'data_format': 'float'
        }
    }


    def __init__(self, source_id, station_info, sensor_type, raw_data_file_name=None, pickle_file_name=None, raw_data_path=None, pickle_path=None, start_date=None, end_date=None, frequency=None):
        dp_id = 'ims' + source_id + sensor_type
        sensor_info = self.sensor_dict.get(sensor_type)
        if sensor_info is None:
            raise ValueError(f'sensor_type {sensor_type} not found')
        super().__init__(dp_id, provider_id='ims', source_id=source_id, 
                         sensor_type=sensor_info['sensor_type'], sensor_value_type=sensor_info['sensor_value_type'], data_type=sensor_info['data_type'], data_format=sensor_info['data_format'],
                           raw_data_path=raw_data_path or 'provider/raw/ims', raw_data_file_name=raw_data_file_name, pickle_path=pickle_path or 'provider/pickle/ims', pickle_file_name=pickle_file_name,
                           start_date=start_date, end_date=end_date, frequency=frequency)
        self.station_info = station_info
        self.station_name = station_info.get('name', None)
        self.station_id = station_info.get('_id', None)
        lat = station_info.get('lat', -5)
        lon = station_info.get('lon', -5)
        alt = station_info.get('alt', -500)
        self.set_location(lat=lat, lon=lon, alt=alt)
        
        
    
def add_all_ims_files(files=None):
    ims_data_dir = DirUtils.get_raw_data_dir(path='provider/raw/ims')
    station_di_path = DirUtils.get_raw_data_path('ims_stations.json', 'provider/raw/ims')
    with open(station_di_path, 'r') as f:
        station_di = json.load(f)
    station_di = {d['name']: d for d in station_di}
    pprint.pp(station_di.keys())

    #Traverse all files in the ims_data_dir
    plugin = IMSDataPlugin()
    for file in os.listdir(ims_data_dir):
        if files and file not in files:
            continue
        if file.endswith('.csv'):
            print(file)
            df = plugin.prepare_data(file_name=file)
            station_name = df['station'].iloc[0].replace(' ', '_').lower()
            pprint.pp(df.columns[1: ])
            df_sensor_types = list(df.columns[1: ])
            if 'station' in df_sensor_types:
                df_sensor_types.remove('station')
            station_info = station_di.get(station_name, None)
            if station_info is None:
                print(f'Station {station_name} not found')
                continue
                # raise ValueError(f'Station {station_name} not found')
            
            source = IMSSensorSource(source_id=f'ims_{station_info["name"]}', source_name=station_info['name'], source_type='ims_station', station_info=station_info, sensor_type_list=df_sensor_types)
            ndf = plugin.prepare_pickle(df=df, ndf=None)
            pickle_file_name = source.source_id
            plist_full, vlist_full = plugin.create_pickles(df=ndf, file_name=pickle_file_name, by_column=False)
            plist, vlist = plugin.create_pickles(df=ndf, file_name=pickle_file_name, by_column=True)
            source.save_to_db()

            
if __name__ == '__main__':
    add_all_ims_files(files=['ims_afek_78.csv'])
    ims_sensor_source = IMSSensorSource(source_id='ims_afeq_78', source_name='afeq', source_type='ims_station')
    ims_sensor_source.save_to_db()
