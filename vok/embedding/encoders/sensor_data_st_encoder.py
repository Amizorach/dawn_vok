from pprint import pprint
import numpy as np
from dawn_vok.vok.embedding.encoders.emp_st_enc import FormulationSTEncoder, SensorTypeSTEncoder, SourceSTEncoder
from dawn_vok.vok.embedding.encoders.timerange_st_enc import TimeRangeSTEncoder
from dawn_vok.vok.embedding.encoders.base.model_st_encoder import ModelSTEncoder
from dawn_vok.utils.dict_utils import DictUtils
from datetime import datetime, timedelta
import torch
from dawn_vok.vok.embedding.encoders.frequency_encoder import FrequencyEncoder

   
class SensorDataSTEncoder(ModelSTEncoder):
    @classmethod
    def get_config(cls):
        return {
            'info_map': {
                0: 'source_start',
                15: 'source_end',
                16: 'sensor_type_start',
                31: 'sensor_type_end',
                32: 'formulation_start',
                47: 'formulation_end',
                48: 'start_time_start',
                
            }
        }

    def __init__(self):
        uid = 'sensor_data_st_encoder'
        self.model = ()
        super().__init__(uid=uid)
        cfg = self.config
      
        self.tr_encoder = TimeRangeSTEncoder()
        self.freq_encoder = FrequencyEncoder()
        self.source_encoder = SourceSTEncoder()
        self.sensor_type_encoder = SensorTypeSTEncoder()
        self.formulation_encoder = FormulationSTEncoder()
        self.sources = {}
        self.sensor_types = {}
        self.formulations = {}
        self.latent_dim = 64
        # self.latent_db = VOKEmbeddingParamaterDB()
        # self.latent_db.load_db()
        # self.build_latent_cache()
        # self.load_data()
    
    def encode(self, source_id, sensor_type, formulation, start_time, freq, end_time=None, data_length=None):
        """
        start: datetime object
        freq: int, frequency in minutes
        """
        if not end_time and not data_length:
            raise ValueError("Either end or data_length must be provided")
        
        se = self.tr_encoder.encode(start=start_time, freq=freq, end=end_time, data_length=data_length)
        source_enc = self.source_encoder.encode(source_id)
        sensor_type_enc = self.sensor_type_encoder.encode(sensor_type)
        formulation_enc = self.formulation_encoder.encode(formulation)
        print(se.shape)
        print(source_enc.shape)
        print(sensor_type_enc.shape)
        print(formulation_enc.shape)
        fin = []
        fin.extend(source_enc)
        fin.extend(sensor_type_enc)
        fin.extend(formulation_enc)
        fin.extend(se)
        for f in fin:
            print(f.shape)
        return np.array(fin, dtype=np.float32)
        
    def decode(self, v):
        source = self.latent_db.search(torch.tensor(v[0:15], dtype=torch.float32))
        sensor_type = self.latent_db.search(torch.tensor(v[16:31], dtype=torch.float32))
        formulation = self.latent_db.search(torch.tensor(v[32:47], dtype=torch.float32))
        source_dec = self.source_encoder.decode(source)
        sensor_type_dec = self.sensor_type_encoder.decode(sensor_type)
        formulation_dec = self.formulation_encoder.decode(formulation)
        start_time = self.tr_encoder.decode(v[48:])

        di = {}
        di['source'] = source_dec
        di['sensor_type'] = sensor_type_dec
        di['formulation'] = formulation_dec
        di['start_time'] = start_time

        pprint(di)

        return di
    
    def get_ground_truth(self, source_id, sensor_type, formulation, start_time, freq, end_time=None, data_length=None):
        if not end_time and not data_length:
            raise ValueError("Either end or data_length must be provided")
        source_enc = self.source_encoder.get_ground_truth(source_id)
        sensor_type_enc = self.sensor_type_encoder.get_ground_truth(sensor_type)
        formulation_enc = self.formulation_encoder.get_ground_truth(formulation)
        samples = [
            {
                'start': start_time,
                'freq': freq,
                'end': end_time,
                'data_length': data_length
            }
        ]
        se = self.tr_encoder.get_ground_truth_batch(samples).squeeze(0)
        print(source_enc.shape)
        print(sensor_type_enc.shape)
        print(formulation_enc.shape)
        print(se.shape)
        
        return np.concatenate([source_enc, sensor_type_enc, formulation_enc, se])

    def samples_to_encodings(self, samples):
        raise NotImplementedError("This method should be implemented by the subclass")

    def decode_from_latent(self, latents):
        source = self.source_encoder.decode_from_latent(latents[:16])
        print('source', source)
        sensor_type = self.sensor_type_encoder.decode_from_latent(latents[16:32])
        print('sensor_type', sensor_type)
        formulation = self.formulation_encoder.decode_from_latent(latents[32:48])
        print('formulation', formulation)
        tr = self.tr_encoder.decode_from_latent(latents[48:])
        print('tr', tr)
        return {
            'source': source,
            'sensor_type': sensor_type,
            'formulation': formulation,
            'tr': tr
        }
if __name__ == "__main__":
    encoder = MetaDataSTEncoder()
    v = encoder.encode(source_id='ds_ims_besor_farm_58', 
                       sensor_type='relative_humidity', 
                       formulation='agg_mean',
                       start_time=datetime(2024,11,21,12,32), 
                       freq=10, 
                       end_time=datetime(2024,11,22,12,42), 
                       )
    print(v.shape)
    gt = encoder.get_ground_truth(source_id='ds_ims_besor_farm_58', 
                                  sensor_type='relative_humidity', 
                                  formulation='agg_mean',
                                  start_time=datetime(2024,11,21,12,32), 
                                  freq=10, 
                                  end_time=datetime(2024,11,22,12,42), 
                                  )
    print(gt.shape) 
    # e = encoder.decode_from_latent(gt)
    # print(e)
    # dt = encoder.decode(v)
    # print(dt)
