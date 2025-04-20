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

class RawDataRetriever:
    """
    This class is used to retrieve raw data from the data provider.
    """
    def __init__(self):
        self.data_providers = {}

        self.init_data_providers()

    def init_data_providers(self):
        self.data_providers = {}

        for dp in SourceDataProvider.get_all(populate=True):
            self.data_providers[dp.source_id] = dp

    def get_data(self, source_id, sensor_id, start_date, frequency=10*60, agg='mean'):
        if not source_id or not sensor_id or not start_date:
            raise ValueError('source_id, sensor_id and start_date are required')
        start_date = DictUtils.parse_datetime_direct(start_date)
        end_date = start_date + timedelta(seconds=frequency*143)
        dp = self.get_data_provider(source_id)
        if dp is None:
            print(f'Data provider for source_id {source_id} and sensor_id {sensor_id} not found')
            return None
        data = dp.get_data(sensor_id, start_date, end_date, frequency, agg)
        if data is None:
            print(f'Data for source_id {source_id} and sensor_id {sensor_id} not found')
            return None
        return data
    
    def get_data_provider(self, source_id):
        if source_id not in self.data_providers:
            print(f'Data provider for source_id {source_id} not found')
            return None
        return self.data_providers[source_id]

    def fake_init_data_providers(self, source_provider_list):
        for source_provider in source_provider_list:
            source_id = source_provider['source_id']
            file_name = source_provider['file_name']
            provider_id = source_provider['provider_id']
            dp = RawDataProvider(source_id, provider_id, file_name)
            dp.init_file()
            if dp.sensor_ids is None:
                print(f'Data provider for source_id {source_id} and file_name {file_name} not found')
                continue
            self.data_providers[source_id] = dp
            print(f'Data provider for source_id {source_id} and file_name {file_name} initialized')



class RawDataSampleCreator:
    def __init__(self, sensor_types, start_date, end_date, source_ids=None, frequency=10*60, agg='mean'):
        self.source_ids = source_ids
        self.sensor_types = sensor_types
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.agg = agg
        self.source_dfs = {}
        self.start_date = DictUtils.parse_datetime_direct(start_date)
        self.end_date = DictUtils.parse_datetime_direct(end_date)

    def load_providers(self):
        retriever = RawDataRetriever()
        if self.source_ids is None:
            self.source_ids = list(retriever.data_providers.keys())
        for source_id in self.source_ids:
            if source_id not in retriever.data_providers:
                print(f'Data provider for source_id {source_id} not found')
                continue
            self.source_dfs[source_id] = retriever.data_providers[source_id].get_data(column_names=self.sensor_types,
                                                                                       start_date=self.start_date, end_date=self.end_date,
                                                                                         frequency=self.frequency, agg=self.agg)
        
    def create_samples(self, sample_size=144, sample_resolution=6, max_samples=-1, max_samples_per_provider=-1, add_embeddings=False, shuffle=True, shuffle_providers=True):
        #create samples of 1000 rows each
        self.load_providers()
        df_samples = []
        embeddings = []
        full_embeddings = []
        samples=[]
        freq = 10*60
        if max_samples != -1 and max_samples_per_provider == -1:
            max_samples_per_provider = max_samples 
        if max_samples != -1 and max_samples_per_provider != -1 and max_samples_per_provider > max_samples:
            max_samples_per_provider = max_samples
        for source_id, df in self.source_dfs.items():
            provider_samples = []
            if df is None:
                print(f'Data provider {source_id} is not found')
                continue
            if self.end_date < df.index[0] or self.start_date > df.index[-1]:
                print(f'Data provider {source_id} is not in the time range {self.start_date} to {self.end_date}')
                continue

            # if max_samples_per_provider > 0 and len(provider_samples) >= max_samples_per_provider:
            #     print(f'Data provider {source_id} has {len(provider_samples)} samples, skipping')
            #     continue
            # df = dp.get_base_df()
            df_length = len(df)

            for i in tqdm(range(0, df_length, sample_resolution), desc='Creating samples', total=df_length//sample_resolution):
                if i+sample_size > df_length:
                    break
                sample = df.iloc[i:i+sample_size]
          
            # if the sample is all zeros, skip it
                flat = sample.values.ravel()

        # 1) skip if everything is zero
                if flat.sum() == 0:
                    continue
             
                mi = min(flat)
                ma = max(flat)
                if mi == ma:
                    continue
                # 2) skip if all entries are the same value
              
                start_time = sample.index[0]
                end_time = start_time + timedelta(seconds=freq*143)
                if add_embeddings:
                    for col in sample.columns:
                        
                        data_request_embedding = DataRequestEmbedding(source_id=source_id, sensor_type=col, start_time=start_time, end_time=end_time)
                        embedding_di = data_request_embedding.get_embedding()
                        if embedding_di is None:
                            continue
                        samp = sample[col].values
                        #normalize this between 0 and 1
                        if  min(samp) == max(samp):
                            print(f"Sample {i} is all the same value, skipping")
                            continue
                        samp = (samp - samp.min()) / (samp.max() - samp.min())
                        samples.append(samp)
                        full_embeddings.append(embedding_di)

                        embeddings.append({'sensor':col, 'start_time':sample.index[0]})
                provider_samples.append(sample)
        
            print(f'{source_id}: embeddings: {len(embeddings)}, samples: {len(samples)}, provider_samples: {len(provider_samples)}')
            if not provider_samples:
                continue

            if shuffle:
                mask = np.random.choice(len(provider_samples), size=max_samples, replace=False)
                provider_samples = [provider_samples[i] for i in mask]
                if add_embeddings:
                    full_embeddings = [full_embeddings[i] for i in mask]
                    embeddings = [embeddings[i] for i in mask]
               
            if max_samples_per_provider > 0 and len(provider_samples) >= max_samples_per_provider:
                provider_samples = provider_samples[:max_samples_per_provider] 
            samples.extend(provider_samples)
     
        if max_samples > 0 and len(samples) > max_samples:
            mask = np.random.choice(len(samples), size=max_samples, replace=False)
            if add_embeddings:
                full_embeddings = [full_embeddings[i] for i in mask]
                embeddings = [embeddings[i] for i in mask]
            samples = [samples[i] for i in mask]
        if shuffle_providers:
            samples = np.random.permutation(samples)
            if add_embeddings:
                full_embeddings = [full_embeddings[i] for i in mask]
                embeddings = [embeddings[i] for i in mask]
        
        return samples, embeddings, full_embeddings

    def get_original_size(self):
        return sum(df.memory_usage(deep=True).sum() for df in self.source_dfs.values())



class FakeRawDataRetriever:
    def __init__(self, file_names, allowed_sensor_types=None):
        self.file_names = file_names
        if not isinstance(self.file_names, list):
            self.file_names = [self.file_names]

        self.allowed_sensor_types = allowed_sensor_types or ['temperature', 'humidity', 'wind_speed', 'solar_radiation', 'rain_intensity', 'pressure', 'wind_gust', 'wind_chill', 'dew_point', 'visibility', 'uv_index', 'wind_speed_gust']

    def load_data(self):
        self.source_dfs = {}

        for file_name in self.file_names:
            if file_name.startswith('ims'):
                dir_path = DirUtils.get_raw_data_path(file_name=file_name, path='provider/pickle/ims')
            elif file_name.startswith('synoptic'):
                dir_path = DirUtils.get_raw_data_path(file_name=file_name, path='provider/pickle/synoptic')
            else:
                raise ValueError(f'File name {file_name} is not supported')
            df = pd.read_pickle(dir_path)
            if df is None:
                return None
            print(df)
            print(df.columns)
            al = df.columns.intersection(self.allowed_sensor_types)
            #drop all the rows that have nans until we reach this first row that is not all nan
            df = df[al]

            # 1) mark -5 as missing
            df.replace(-5, np.nan, inplace=True)
            df.dropna(axis=1, how='all', inplace=True)

        # 2) forward‐fill via linear interpolation
        #    limit_direction="forward" only fills NaNs *after* known values
            #    if you also want to fill leading NaNs, use limit_direction="both"
            df.interpolate(
                method='linear',
                axis=0,
                limit_direction='forward',
                limit=6,
                inplace=True
            )

        # 3) (optional) fill any leading NaNs by back‐filling:
        #    self.df.interpolate(..., limit_direction='both', inplace=True)

            # 4) fill remaining NaNs with the column mean
            df.fillna(0, inplace=True)
            still_na = df.isna().sum()
            if still_na.any():
                print("⚠️ Columns still containing NaNs after mean‐fill:")
                print(still_na[still_na > 0])
            # 5) final sanity check
            if df.isna().any().any():
                raise ValueError('There are still NaNs in the DataFrame')
            # if start_date is None:

        # if end_date is None:
        #     end_date = pd.Timestamp('2022-01-02')
            self.source_dfs[file_name] = df
        return self.source_dfs

    def create_samples(self, sample_size=1024, sample_resolution=6, max_samples=-1):
        #create samples of 1000 rows each
        self.load_data()
        df_samples = []
        embeddings = []
        full_embeddings = []
        samples=[]
        freq = 10*60
        for file_name, df in self.source_dfs.items():
            source_id = hex(hash(file_name))
            df_length = len(df)
            for i in tqdm(range(0, df_length, sample_resolution), desc='Creating samples', total=df_length//sample_resolution):
                if i+sample_size > df_length:
                    break
                sample = df.iloc[i:i+sample_size]
          
            # if the sample is all zeros, skip it
                flat = sample.values.ravel()

        # 1) skip if everything is zero
                if flat.sum() == 0:
                    continue

                # 2) skip if all entries are the same value
                if np.unique(flat).size < 24:
                    print(f"Sample {i} is all the same value, skipping")
                    continue
                if  min(flat) == max(flat):
                    print(f"Sample {i} is all the same value, skipping")
                    continue
                start_time = sample.index[0]
                end_time = start_time + timedelta(seconds=freq*143)
                for col in sample.columns:
                    
                    data_request_embedding = DataRequestEmbedding(source_id=source_id, sensor_type=col, start_time=start_time, end_time=end_time)
                    embedding_di = data_request_embedding.get_embedding()
                    if embedding_di is None:
                        continue
                    samp = sample[col].values
                    #normalize this between 0 and 1
                    if  min(samp) == max(samp):
                        print(f"Sample {i} is all the same value, skipping")
                        continue
                    samp = (samp - samp.min()) / (samp.max() - samp.min())
                    samples.append(samp)
                    full_embeddings.append(embedding_di)

                    embeddings.append({'sensor':col, 'start_time':sample.index[0]})
                df_samples.append(sample)
        
        print(len(embeddings), len(samples), len(df_samples))
        if max_samples > 0:
            mask = np.random.choice(len(samples), size=max_samples, replace=False)
            full_embeddings = [full_embeddings[i] for i in mask]
            embeddings = [embeddings[i] for i in mask]
            samples = [samples[i] for i in mask]
        return full_embeddings, embeddings, samples

    def get_original_size(self):
        return sum(df.memory_usage(deep=True).sum() for df in self.source_dfs.values())


if __name__ == "__main__":

    #create samples
    sampler = RawDataSampleCreator(source_ids=None,
                                    sensor_types=['temperature', 'humidity'], start_date='2024-01-04', end_date='2024-10-05', frequency=10*60, agg='mean')
    samples, embeddings, full_embeddings = sampler.create_samples(sample_size=144, sample_resolution=6, max_samples=1000, max_samples_per_provider=-1, 
                                                                  add_embeddings=False, shuffle=True, shuffle_providers=True)
    print(f'samples: {len(samples)}, embeddings: {len(embeddings)}, full_embeddings: {len(full_embeddings)}')
    exit()
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
    #save the data providers to the db
    retriever = RawDataRetriever()
    retriever.init_data_providers()
    # retriever.fake_init_data_providers([
    #     {'source_id': 'ims_ariel_21', 'provider_id': 'ims', 'file_name': 'ims_ariel_21.pkl'},
    #     {'source_id': 'ims_afeq_48', 'provider_id': 'ims', 'file_name': 'ims_afeq.pkl'}
    # ])
    print(retriever.data_providers)
    for source_id, dp in retriever.data_providers.items():
        print(source_id, dp.sensor_ids)

    data = retriever.get_data('ims_ariel_21', 'temperature', '2024-01-04', 10*60, 'mean')
    data = retriever.get_data('ims_afeq_48', 'humidity', '2024-01-04', 10*60, 'mean')
    # retriever = FakeRawDataRetriever(file_name='ims_ariel_21.pkl')
    # full_embeddings, embeddings, samples = retriever.create_samples()
    # full_embeddings = []
    # full_samples = []
    # for i in tqdm(range(len(embeddings)), desc='Creating full embeddings and samples', total=len(embeddings)):
    #     start_time = embeddings[i]['start_time']
    #     freq = 10*60
    #     end_time = start_time + timedelta(seconds=freq*144)
    #     data_request_embedding = DataRequestEmbedding(source_id='test', sensor_type=embeddings[i]['sensor'], start_time=start_time, end_time=end_time)
    #     full_embeddings.append(data_request_embedding.get_embedding())
    #     full_samples.append(samples[i])
    # print(len(full_embeddings), len(full_samples))
    # print(full_embeddings[1])
    # print(full_samples[1])
