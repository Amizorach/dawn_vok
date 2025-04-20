from datetime import timedelta
import os
import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.providers.provider import DataProvider
from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.utils.dir_utils import DirUtils
from dawn_vok.vok.data_request.data_request_embedding import DataRequestEmbedding
from dawn_vok.vok.v_objects.vok_object import VOKObject

class RawDataProvider(VOKObject):
    @classmethod
    def get_db_name(cls):
        return 'raw_data'
    
    @classmethod
    def get_collection_name(cls):
        return 'raw_data_providers'
    
    def __init__(self, uid, provider_type=None, data_provider_id=None, file_name=None,  **kwargs):
        super().__init__(uid=uid, **kwargs)
        self.provider_type = provider_type
        self.data_provider_id = data_provider_id
        self.file_name= file_name
        self.pickle_dir = DirUtils.get_raw_data_path(file_name=file_name, path=f'provider/pickle/{data_provider_id}') if file_name is not None else None
        self.column_names = []
        self.base_df = None
        self.start_date = None
        self.end_date = None
        self.frequency = None
        self.agg = None

    def scan_file(self, agg='mean'):
        pi_path = self.pickle_dir
        if not os.path.exists(pi_path):
            print(f'File {pi_path} does not exist')
            return False
        self.base_df = pd.read_pickle(pi_path)
        if self.base_df is None:
            print(f'File {pi_path} is not a pickle file')
            return False
        self.column_names = self.base_df.columns.tolist()
        self.column_names = [s for s in self.column_names if s not in ['timestamp', 'date']]
        self.column_names = {s:s for s in self.column_names}
        self.start_date = self.base_df.index[0]
        self.end_date = self.base_df.index[-1]
        self.frequency = self.base_df.index.freq.freqstr
        self.agg = agg
        return True

   
    def to_dict(self):
        ret = super().to_dict()
        ret['data_provider_id'] = self.data_provider_id
        ret['file_name'] = self.file_name
        ret['provider_type'] = self.provider_type
        ret['column_names'] = self.column_names
     
        ret['frequency'] = self.frequency
        ret['agg'] = self.agg
        DictUtils.put_datetime(ret, 'start_date', self.start_date)
        DictUtils.put_datetime(ret, 'end_date', self.end_date)
        return ret
    
    def populate_from_dict(self, di):
        super().populate_from_dict(di)
        self.data_provider_id = di.get('data_provider_id', self.data_provider_id)
        self.file_name = di.get('file_name', self.file_name)
        self.provider_type = di.get('provider_type', self.provider_type)
        self.column_names = di.get('column_names', self.column_names)
        self.start_date = DictUtils.parse_datetime(di, 'start_date', self.start_date)
        self.end_date = DictUtils.parse_datetime(di, 'end_date', self.end_date)
        self.frequency = di.get('frequency', self.frequency)
        self.agg = di.get('agg', self.agg)
        if self.file_name is not None:
            self.pickle_dir = DirUtils.get_raw_data_path(file_name=self.file_name, path=f'provider/pickle/{self.data_provider_id}')
        return self
    
    def get_base_df(self):
        if self.base_df is None:
            try:
                self.base_df = pd.read_pickle(self.pickle_dir)
            except FileNotFoundError:
                 # Handle case where pickle file doesn't exist
                 # Log an error?
                 print(f"Error: Pickle file not found at {self.pickle_dir}")
                 return None # Or raise
            except Exception as e:
                 # Handle other potential unpickling errors
                 print(f"Error reading pickle file {self.pickle_dir}: {e}")
                 return None # Or raise  
        return self.base_df
    
    def get_data(self, column_names=None, start_date=None, end_date=None, frequency=None, agg=None):
        if not os.path.exists(self.pickle_dir):
            print(f'File {self.pickle_dir} does not exist')
            return None
        if agg is None:
            agg = self.agg
        if frequency is None:
            frequency = self.frequency

        # This is so you can see what its doing 
        # self.sensor_ids = {'temperature':['temperature', 'max_temperature']}
        if start_date is not None:
            start_date = DictUtils.parse_datetime_direct(start_date)
        if end_date is not None:
            end_date = DictUtils.parse_datetime_direct(end_date)
        if column_names is not None:
            if isinstance(column_names, str):
                column_names = [column_names]
            column_names = [c for c in column_names if c in self.column_names]
            if len(column_names) == 0:
                return None
        dfr = self.get_base_df()
        if start_date is not None:
            dfr = dfr[start_date:]
        if end_date is not None:
            dfr = dfr[:end_date]
        if dfr.empty:
            return None
        if column_names is not None:
            columns = [c for c in column_names if c in dfr.columns]
            if len(columns) == 0:
                return None
            dfs = dfr[columns]
        else:
            dfs = dfr
        #drop columns that are all nan
        dfs = dfs.dropna(axis=1, how='all')
        if dfs.empty:
            return None
        if frequency is not None:
            try:
                dfs = dfs.resample(f'{frequency}S').agg(agg)
            except Exception as e:
                 print(f"Error during resampling/aggregation for column '{column_names}': {e}")
                 return None # Or raise

        print(dfs)
        return dfs
    
class SourceDataProvider(RawDataProvider):
    def __init__(self, source_id, data_provider_id=None, file_name=None, obj_type='source_data_provider'):
        uid = f'{source_id}'
        super().__init__(uid=uid, 
                         obj_type=obj_type,
                         provider_type='source',
                         data_provider_id=data_provider_id,
                         file_name=file_name,
                         meta_data={'source_id': source_id})
        self.source_id = source_id
        self.sensor_ids = self.column_names

class DataProviderFactory:
    @classmethod
    def get_data_provider(cls, source_id, provider_type):
        if provider_type == 'source':
            return SourceDataProvider(source_id=source_id)
        else:
            raise ValueError(f'Provider type {provider_type} is not supported')

if __name__ == "__main__":

  
    file_list = [{'file':'ims_ariel_21.pkl', 'source_id':'ims_ariel_21'}, 
                 {'file':'ims_afeq.pkl', 'source_id':'ims_afeq_48'},
                 {'file':'ims_mizpe_ramon_379.pkl', 'source_id':'ims_mizpe_ramon_379'},
                 {'file':'ims_paran_207.pkl', 'source_id':'ims_paran_207'},
                 {'file':'ims_qarne_shomron_20.pkl', 'source_id':'ims_qarne_shomron_20'},
                 {'file':'ims_haifa_univirsity_42.pkl', 'source_id':'ims_haifa_univirsity_42'},
                 {'file':'ims_haifa_refineries_41.pkl', 'source_id':'ims_haifa_refineries_41'},
                 {'file':'ims_haifa_technion_43.pkl', 'source_id':'ims_haifa_technion_43'},
              
                
                 
                 
                 ]
    for file in file_list:
        sourcedp = SourceDataProvider(source_id=file['source_id'], data_provider_id='ims', file_name=file['file'])
        sourcedp.scan_file()
        print(sourcedp.to_dict())
        sourcedp.save_to_db()
    exit()