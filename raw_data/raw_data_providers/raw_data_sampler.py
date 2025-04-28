
from datetime import timedelta

import numpy as np
from tqdm import tqdm
from dawn_vok.raw_data.raw_data_providers.raw_data_retriever import RawDataRetriever
from dawn_vok.utils.dict_utils import DictUtils


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
              

                # if add_embeddings:
                #     print('will add embeddings')
                start_time = sample.index[0]
                end_time = start_time + timedelta(seconds=freq*143)
                for col in sample.columns:
                    
                    #data_request_embedding = DataRequestEmbedding(source_id=source_id, sensor_type=col, start_time=start_time, end_time=end_time)
                    # embedding_di = data_request_embedding.get_embedding()
                    # if embedding_di is None:
                    #     continue
                    samp = sample[col].values
                    #normalize this between 0 and 1
                    if  min(samp) == max(samp):
                        print(f"Sample {i} is all the same value, skipping")
                        continue
                    samp = (samp - samp.min()) / (samp.max() - samp.min())
                    samples.append(samp)
                    # full_embeddings.append(embedding_di)

                    embeddings.append({'sensor':col, 'start_time':sample.index[0]})
                    provider_samples.append(samp)
        
            print(f'{source_id}: embeddings: {len(embeddings)}, samples: {len(samples)}, provider_samples: {len(provider_samples)}')
            if not provider_samples:
                continue

            if shuffle:
                mask = np.random.choice(len(provider_samples), size=min(max_samples, len(provider_samples)), replace=False)
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


if __name__ == "__main__":

    #create samples
    sampler = RawDataSampleCreator(source_ids=None,
                                    sensor_types=['temperature', 'humidity'], start_date='2024-01-04', end_date='2024-10-05', frequency=10*60, agg='mean')
    samples, embeddings, full_embeddings = sampler.create_samples(sample_size=144, sample_resolution=1, max_samples=10000, max_samples_per_provider=-1, 
                                                                  add_embeddings=False, shuffle=True, shuffle_providers=True)
    print(f'samples: {len(samples)}, embeddings: {len(embeddings)}, full_embeddings: {len(full_embeddings)}')
    print(len(samples[0]))
    exit()
    