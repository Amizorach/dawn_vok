
from datetime import datetime, timedelta
from pprint import pprint

import torch
from dawn_vok.vok.embedding.cyclic_emb.frequency_embedding import FrequencyEmbedding
from dawn_vok.vok.embedding.cyclic_emb.timestamp_encoder import TimestampEncoder
from dawn_vok.utils.dict_utils import DictUtils
import numpy as np
from dawn_vok.vok.embedding.embedding_paramater.embedding_paramater_header import EMPHeaderBuilder
from dawn_vok.vok.embedding.embedding_paramater.v_embedding_paramater import VOKEmbeddingParamater
from dawn_vok.vok.embedding.embedding_paramater.v_embedding_paramater_db import VOKEmbeddingParamaterDB
from dawn_vok.vok.embedding.encoders.timerange_st_enc import TimeRangeSTEncoder
from dawn_vok.vok.embedding.static_emb.time_range_encoding import RichTimeRangeEncoding
from dawn_vok.vok.pipelines.meta_data.frequency_classefier.frequency_classefier_utils import TimeRangeClassifierUtils

class VOKDataEmbeddingBuilder:
    def __init__(self, start_time=datetime(2010, 1, 1), end_time=datetime(2028, 1, 10), data_dim = 144, latent_dim = 64):
        self.start_time = start_time
        self.end_time = end_time
        self.full_time_range = self.end_time.timestamp() - self.start_time.timestamp()
        self.sources = {}
        self.sensor_types = {}
        self.formulations = {}
        self.freqs = [10*60, 30*60, 60*60, 2*60*60, 6*60*60, 12*60*60, 24*60*60]
        
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.clu = TimeRangeClassifierUtils()
       
       
        self.avial_sources ={}
        self.avial_sensor_types ={}
        self.avial_formulations ={}
        self.embedding_di={}
        self.meta_data = []
        # self.retrieve_latents = {}
        # self.latent_cache = {}
        # self.load_data()
        self.latent_db = VOKEmbeddingParamaterDB()
        self.time_range_encoder = TimeRangeSTEncoder()
        self.latent_db.load_db()
        self.build_latent_cache()
        self.load_data()

    def load_data(self, allowed_source_ids=None, allowed_sensor_types=None, allowed_formulations=None, allowed_freq=None):

        emb_paramaters = VOKEmbeddingParamater.get_all()
        for emb_paramater in emb_paramaters:
            if emb_paramater.param_type == 'source':
                ret = emb_paramater.get_latents(dim_size=self.latent_dim-16).reshape(1, -1)
                sret = np.array(emb_paramater.static_latent_id).reshape(1, -1)
                ret = np.concatenate([sret, ret], axis=1)
                self.sources[emb_paramater.param_id]['latent'] = ret

            elif emb_paramater.param_type == 'sensor_type':
                self.sensor_types[emb_paramater.param_id]['latent'] = emb_paramater.get_latents(dim_size=self.latent_dim)
            elif emb_paramater.param_type == 'formulation':
                self.formulations[emb_paramater.param_id]['latent'] = emb_paramater.get_latents(dim_size=self.latent_dim)
        # for emb_paramater in emb_paramaters:
        #     if emb_paramater.param_type == 'source':
        #         if not allowed_source_ids or emb_paramater.param_id in allowed_source_ids:    
        #             self.sources[emb_paramater.param_id.replace('source_', '')] = emb_paramater
        #     elif emb_paramater.param_type == 'sensor_type':
        #         if not allowed_sensor_types or emb_paramater.param_id in allowed_sensor_types:
                    
        #             self.sensor_types[emb_paramater.param_id.replace('sensor_type_', '')] = emb_paramater
        #     elif emb_paramater.param_type == 'formulation':
        #         if not allowed_formulations or emb_paramater.param_id in allowed_formulations:
        #             self.formulations[emb_paramater.param_id.replace('formulation_', '')] = emb_paramater
        # if allowed_freq:
        #     self.allowed_freq = [f for f in self.allowed_freq if f in allowed_freq]

    def build_latent_cache(self):
        for meta_data in self.latent_db.meta_data_map.values():
            if meta_data['param_type'] == 'source':
                self.sources[meta_data['param_id']] = meta_data
                self.sources[meta_data['param_id']]['latent_id'] = self.latent_db.full_latents[meta_data['column_index']][1:]
            elif meta_data['param_type'] == 'sensor_type':
                self.sensor_types[meta_data['param_id']] = meta_data
                self.sensor_types[meta_data['param_id']]['latent_id'] = self.latent_db.full_latents[meta_data['column_index']][1:]
            elif meta_data['param_type'] == 'formulation':
                self.formulations[meta_data['param_id']] = meta_data
                self.formulations[meta_data['param_id']]['latent_id'] = self.latent_db.full_latents[meta_data['column_index']][1:]
        # print(len(self.sources))
        # print(len(self.sensor_types))
        # print(len(self.formulations))
        # print(self.formulations)
        # for source in self.sources.values():
        #     if source.static_latent_id:
        #         ret = source.get_latents(self.latent_dim-16).reshape(1, -1)
        #         sret = np.array(source.static_latent_id).reshape(1, -1)
        #         ret = np.concatenate([sret, ret], axis=1)
        #         latents[f'source_{source.param_id}'] = ret
        #     else:
        #         latents[f'source_{source.param_id}'] = source.get_latents(self.latent_dim)
        # for sensor_type in self.sensor_types.values():
        #     sensor_type_id = sensor_type.param_id.replace('sensor_type_', '')
        #     latents[f'sensor_type_{sensor_type_id}'] = sensor_type.get_latents(self.latent_dim)
        # for formulation in self.formulations.values():
        #     formulation_id = formulation.param_id.replace('formulation_', '')
        #     latents[f'formulation_{formulation_id}'] = formulation.get_latents(self.latent_dim)
        # self.latent_cache = latents
        # return self.latent_cache
     
    def get_random_start_time(self, freq):
        secs = np.random.randint(0, int(self.full_time_range-self.data_dim*freq))
        return self.start_time + timedelta(seconds=secs)
    
    def prepare_meta_data(self, source_id=None, sensor_type=None, formulation=None, start_time=None, freq=None):
        if not source_id:
            source_id = np.random.choice(list(self.sources.keys()))
        if not sensor_type:
            sensor_type = np.random.choice(list(self.sensor_types.keys()))
        if not formulation:
            formulation = np.random.choice(list(self.formulations.keys()))
        if not freq:
            freq = np.random.choice(self.freqs)
        start_time = DictUtils.parse_datetime_direct(start_time, default=None)
        if not start_time :
            #use start date and end date to create a random start time
            start_time = self.get_random_start_time(freq)
        if not source_id in self.sources \
            or not sensor_type in self.sensor_types \
            or not formulation in self.formulations \
            or start_time < self.start_time \
            or start_time > self.end_time-timedelta(seconds=int(self.data_dim*freq)):
            print('some of the meta data is not valid') 
            return None
       
        end_time = start_time + timedelta(seconds=int(freq*self.data_dim))  
        meta_data = {
            'source_id': str(source_id),
            'source_id_index': self.sources[source_id]['system_uid'],
            'db_column_index': self.sources[source_id]['column_index'],
            'sensor_type': str(sensor_type),
            'sensor_type_index': self.sensor_types[sensor_type]['system_uid'],
            'db_column_index': self.sensor_types[sensor_type]['column_index'],
            'formulation': str(formulation),
            'formulation_index': self.formulations[formulation]['system_uid'],
            'db_column_index': self.formulations[formulation]['column_index'],
            'start': start_time,
            'end': end_time,
            'freq': int(freq)
        }

        return meta_data
    

    # def get_source_latent(self, source_id=None, dim_size=64):
    #     if source_id in self.latent_cache:
    #         return self.latent_cache[source_id]
    #     source = self.sources.get(source_id, None)
    #     if not source:
    #         print(f'source_id {source_id} not found')
    #         raise ValueError(f'source_id {source_id} not found')
    #     if source.static_latent_id:
    #             ret = source.get_latents(dim_size-16).reshape(1, -1)
    #             sret = np.array(source.static_latent_id).reshape(1, -1)
    #             ret = np.concatenate([sret, ret], axis=1)
    #     return source.get_latents(dim_size)
    
    # def get_source_static_latent(self, source_id=None, dim_size=16):
    #     source = self.sources.get(source_id, None)
    #     if not source:
    #         print(f'source_id {source_id} not found')
    #         raise ValueError(f'source_id {source_id} not found')
    #     return source.static_latent_id
    
    # def get_sensor_type_static_latent(self, sensor_type=None, dim_size=16):
    #     sensor_type = self.sensor_types.get(sensor_type, None)
    #     if not sensor_type:
    #         print(f'sensor_type {sensor_type} not found')
    #         raise ValueError(f'sensor_type {sensor_type} not found')
    #     return sensor_type.get_latents(16)
    
    # def get_formulation_static_latent(self, formulation=None, dim_size=16):
    #     formulation = self.formulations.get(formulation, None)
    #     if not formulation:
    #         print(f'formulation {formulation} not found')
    #         raise ValueError(f'formulation {formulation} not found')
    #     return formulation.get_latents(16)
    
    # def get_sensor_type_latent(self, sensor_type=None, dim_size=64):
    #     if sensor_type in self.latent_cache:
    #         return self.latent_cache[sensor_type]
    #     sensor_type = self.sensor_types.get(sensor_type, None)
    #     if not sensor_type:
    #         print(f'sensor_type {sensor_type} not found')
    #         raise ValueError(f'sensor_type {sensor_type} not found')
    #     return sensor_type.get_latents(dim_size)
    
    # def get_formulation_latent(self, formulation=None, dim_size=64):
    #     if formulation in self.latent_cache:
    #         return self.latent_cache[formulation]
    #     formulation = self.allowed_formulations.get(formulation, None)
    #     if not formulation:
    #         print(f'formulation {formulation} not found')
    #         raise ValueError(f'formulation {formulation} not found')
    #     return formulation.get_latents(dim_size)
    
    def get_time_range_latent(self, start_time=None, end_time=None, frequency=10*60, dim_size=64):
        
        encoding = self.time_range_encoder.encode(start=start_time, freq=frequency, end=end_time)
        if dim_size:
            if encoding.shape[0] < dim_size:
                encoding = np.concatenate([encoding, np.zeros(dim_size-encoding.shape[0])])
            elif encoding.shape[0] > dim_size:
                raise ValueError(f'encoding shape {encoding.shape[0]} is greater than dim_size {dim_size}')
        return encoding.reshape(1, -1)
    

    
    # def get_emp_latent(self, source_id=None, sensor_type=None, formulation=None, dim_size=64):
    #     if source_id != None:
    #         if source_id in self.emp_latents:
    #             return self.emp_latents[source_id]
    #         source = self.sources.get(source_id, None)

    #         if not source:
    #             print(f'source_id {source_id} not found')
    #             raise ValueError(f'source_id {source_id} not found')
    #         if source.static_latent_id:
    #             ret = source.get_latents(dim_size-16).reshape(1, -1)
    #             sret = np.array(source.static_latent_id).reshape(1, -1)
    #             ret = np.concatenate([sret, ret], axis=1)
    #         else:
    #             ret = source.get_latents(dim_size)
    #         self.emp_latents[source_id] = ret
    #         return ret
            
    #     if sensor_type != None:
    #         if sensor_type in self.emp_latents:
    #             return self.emp_latents[sensor_type]
    #         sensor_type = self.sensor_types.get(sensor_type, None)
    #         if not sensor_type:
    #             print(f'sensor_type {sensor_type} not found')
    #             raise ValueError(f'sensor_type {sensor_type} not found')
    #         ret = sensor_type.get_latents(dim_size)
    #         self.emp_latents[sensor_type] = ret
    #         return ret
    #     if formulation != None:
    #         if formulation in self.emp_latents:
    #             return self.emp_latents[formulation]
    #         formulation = self.allowed_formulations.get(formulation, None)
    #         if not formulation:
    #             print(f'formulation {formulation} not found')
    #             raise ValueError(f'formulation {formulation} not found')
    #         ret = formulation.get_latents(dim_size)
    #         self.emp_latents[formulation] = ret
    #         return ret
    #     return None
            
    def encode(self, source_id=None, sensor_type=None, formulation=None, start_time=None, freq=None, meta_data=None):
        if not meta_data:
            meta_data = self.prepare_meta_data(source_id, sensor_type, formulation, start_time, freq)
        
        embedding_di = {}
        embedding_di['meta_data'] = meta_data
        # Encode components
        # st = TimestampEncoder().encode(meta_data['start_time'])
        # en = TimestampEncoder().encode(meta_data['end_time'])
        # self.embedding_di['time_range'] = np.concatenate([st, en]) # iwant this 1,64
        # self.embedding_di['time_range'] = self.embedding_di['time_range'].reshape(1, -1)
        # self.embedding_di['freq'] = FrequencyEmbedding().encode(meta_data['freq'], dim_size=64)
        
        # self.embedding_di['freq'] = self.embedding_di['freq'].reshape(1, -1)
        # source = self.allowed_source_ids.get(meta_data['source_id'], None)
        # if not source:
        #     raise ValueError(f'source_id {meta_data["source_id"]} not found')
        # self.embedding_di['source_id'] = self.get_source_latent(source_id=meta_data['source_id'])
        embedding_di['source_id'] = self.sources[meta_data['source_id']]['latent']
        embedding_di['sensor_type'] = self.sensor_types[meta_data['sensor_type']]['latent']
        embedding_di['formulation'] = self.formulations[meta_data['formulation']]['latent']
        embedding_di['time_range'] = self.get_time_range_latent(start_time=meta_data['start'], 
                                                                     end_time=meta_data['end'], 
                                                                     frequency=meta_data['freq'])
      
        return embedding_di
    
    @staticmethod
    def calc_time_enc(dt):
        max_dt = datetime(2027,1,1).timestamp() 
        min_dt = datetime(2010,1,1).timestamp()
        ret = (dt.timestamp() - min_dt)/(max_dt - min_dt)
        return ret
    @staticmethod
    def decode_time_enc(enc):
        max_dt = datetime(2027,1,1).timestamp() 
        min_dt = datetime(2010,1,1).timestamp()
        try:
            return datetime.fromtimestamp(enc*(max_dt - min_dt) + min_dt)
        except Exception as e:
            print(f"error: {e}")
            return None
    
    @staticmethod
    def encode_freq(freq):
        return freq/60/60/24/365
    @staticmethod
    def decode_freq(enc):
        return enc*60*60*24*365
    
    def get_vok_vdb_latent(self, source_id=None, sensor_type=None, formulation=None, start_time=None, freq=None, meta_data=None):
        
        if not meta_data:
            meta_data = self.prepare_meta_data(source_id, sensor_type, formulation, start_time, freq)
        src_latent = self.sources[meta_data['source_id']]['latent_id']
        sensor_type_latent = self.sensor_types[meta_data['sensor_type']]['latent_id']
        formulation_latent = self.formulations[meta_data['formulation']]['latent_id']

        time_range_latent = self.clu.get_ground_truth(meta_data)
        # print(f"time_range_latent: {time_range_latent}")
        # time_range_latent = np.zeros(16)
        # time_range_latent[0] = self.calc_time_enc(meta_data['start_time'])
        # time_range_latent[1] = self.encode_freq(meta_data['freq'])
        # print('src_latent',max(src_latent))
        # print('src_latent',min(src_latent))
        # print('sensor_type_latent',max(sensor_type_latent))
        # print('sensor_type_latent',min(sensor_type_latent))
        # print('formulation_latent',max(formulation_latent))
        # print('formulation_latent',min(formulation_latent))
        # print('time_range_latent',max(time_range_latent))
        # print('time_range_latent',min(time_range_latent))
        if (max(src_latent) > 5):
            print('src_latent',src_latent)

            exit()
        return np.concatenate([src_latent, sensor_type_latent, formulation_latent, time_range_latent['full']]).reshape(1, -1)
   
   
    def get_embedding(self, embedding_di=None):
        if not embedding_di:
            embedding_di = self.encode()
        meta_data = embedding_di.get('meta_data', None)
        if meta_data:
            del embedding_di['meta_data']
      
        latents = []
        global_index = 0
        for k, v in embedding_di.items():
            if v.shape[0] == 1:
                if isinstance(v, np.ndarray):
                    tensor = torch.from_numpy(v).float()
                else:
                    tensor = torch.tensor(v).float()
                latents.append(tensor)
                global_index += 1
            else:
                for i in range(v.shape[0]):
                    tensor = torch.from_numpy(v[i].reshape(1, -1)).float()
                    latents.append(tensor)
        # for i in range(len(latents)):
        #     print(latents[i].shape)
        ret = torch.stack(latents, dim=0)
        ret = ret.squeeze(1)
        return ret, meta_data
    


    def get_sample_train_data_set(self, sample_size=100):
        if not self.sources:
            self.load_data()
        data_set = []
        for i in range(sample_size):
            lats, meta_data = self.get_embedding()
            gt = self.get_vok_vdb_latent(meta_data=meta_data)
            data_set.append((lats, meta_data, gt))

        
       
        return data_set
if __name__ == '__main__':
    builder = VOKDataEmbeddingBuilder()
    
    # builder.load_data(allowed_sensor_types=['air_temperature', 'humidity', 'radiation'])
    # emb_di = builder.encode()
    # pprint(emb_di)
    # emb = builder.get_embedding(emb_di)
    data_set = builder.get_sample_train_data_set(sample_size=10)
    # builder.encode()
    # print(builder.get_embedding())
    # print(builder.retrieve_latents)
    # print(builder.get_vok_vdb_latent())