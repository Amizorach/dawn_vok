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
from dawn_vok.raw_data.raw_data_providers.raw_data_provider import RawDataProvider, SourceDataProvider

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

    # def fake_init_data_providers(self, source_provider_list):
    #     for source_provider in source_provider_list:
    #         source_id = source_provider['source_id']
    #         file_name = source_provider['file_name']
    #         provider_id = source_provider['provider_id']
    #         dp = RawDataProvider(source_id, provider_id, file_name)
    #         dp.init_file()
    #         if dp.sensor_ids is None:
    #             print(f'Data provider for source_id {source_id} and file_name {file_name} not found')
    #             continue
    #         self.data_providers[source_id] = dp
    #         print(f'Data provider for source_id {source_id} and file_name {file_name} initialized')



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

    #save the data providers to the db
    retriever = RawDataRetriever()
    # retriever.init_data_providers()
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
