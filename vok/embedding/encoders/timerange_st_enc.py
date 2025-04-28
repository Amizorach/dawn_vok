import os
from pprint import pprint
import numpy as np
from dawn_vok.vok.embedding.encoders.base.model_st_encoder import ModelSTEncoder
from dawn_vok.utils.dict_utils import DictUtils
from datetime import datetime, timedelta
import torch
from dawn_vok.vok.embedding.encoders.frequency_encoder import FrequencyEncoder
from dawn_vok.vok.embedding.encoders.timestamp_st_enc import TimeStampSTEncoder
from dawn_vok.vok.pipelines.meta_data.time_range_classifier.time_range_classefier_model import TimeRangeClassifierModel
   

    
class TimeRangeSTEncoder(ModelSTEncoder):
    @classmethod
    def get_config(cls):
        ts_config = TimeStampSTEncoder.get_config()
        ts_dim = len(ts_config['info_map'])
        info_map = [''] * (ts_dim * 3)
        for k, v in ts_config['info_map'].items():
            info_map[k] = f'start_{v}'
            info_map[k + ts_dim] = f'middle_{v}'
            info_map[k + ts_dim * 2] = f'end_{v}'
        freq_config = FrequencyEncoder.get_config()
        freq_dim = len(freq_config['info_map'])
        info_map.extend([f'freq_{v}' for v in freq_config['info_map'].values()])
       
        return {
            'info_map': info_map
        }

    def __init__(self):
        uid = 'time_range_encoder'
        model = TimeRangeClassifierModel()
        super().__init__(uid=uid, model=model)
        cfg = self.config
        self.min_year = TimeStampSTEncoder.get_config()['min_year']
        self.max_year = TimeStampSTEncoder.get_config()['max_year']
        self.ts_encoder = TimeStampSTEncoder()
        self.freq_encoder = FrequencyEncoder()

    def encode(self, start, freq, end=None, data_length=None):
        """
        start: datetime object
        freq: int, frequency in minutes
        """
        if not end and not data_length:
            raise ValueError("Either end or data_length must be provided")
        start_dt = DictUtils.parse_datetime_direct(start)

        if not end:
            end_dt = start_dt + timedelta(minutes=freq * data_length)
        else:
            end_dt = DictUtils.parse_datetime_direct(end)
        if not data_length:
            data_length = (end_dt - start_dt).total_seconds() / (freq * 60)

        # parse start datetime
        # compute middle and end
        mid_dt = start_dt + timedelta(seconds=(end_dt - start_dt).total_seconds() / 2)
        # print(f"encode: Start: {start}, Frequency: {freq}, End: {end}, Data Length: {data_length}")

        st = self.ts_encoder.encode(start_dt)
        mt = self.ts_encoder.encode(mid_dt)
        et = self.ts_encoder.encode(end_dt)

        # frequency features
        fr = self.freq_encoder.encode(freq)

        # concatenate all parts
        # st, mt, et are arrays or tensors of same dim
        parts = [torch.tensor(st, dtype=torch.float32, device=fr.device),
                 torch.tensor(mt, dtype=torch.float32, device=fr.device),
                 torch.tensor(et, dtype=torch.float32, device=fr.device),
                 fr]
        return torch.cat(parts, dim=-1).unsqueeze(0)  # final feature vector
        
    def decode(self, v):
        # decodes the start date and frequency  
        # 1) pick predicted class indices for each head
        start_dt = self.ts_encoder.decode(v[:self.ts_encoder.dim_size])
        end_dt = self.ts_encoder.decode(v[self.ts_encoder.dim_size*2:self.ts_encoder.dim_size*3])
        freq = self.freq_encoder.decode(v[self.ts_encoder.dim_size*3:])
        data_length = (end_dt - start_dt).total_seconds() / (freq * 60)
        print(f"Start: {start_dt}, End: {end_dt}, Frequency: {freq}, Data Length: {data_length}")
        return start_dt, end_dt, freq, data_length
    
    def samples_to_encodings(self, samples):
        # prepare padded encodings
        x = []
        for i, f in enumerate(samples):
            start = DictUtils.parse_datetime(f, 'start', default=None)
            freq = DictUtils.parse_value(f, 'freq', default=None)
            end = DictUtils.parse_datetime(f, 'end', default=None)
            if start is None or freq is None or end is None:
                raise ValueError("start, freq, and end must be provided")
            encs = self.encode(start=start, freq=freq, end=end)
            # print(encs.shape)
            L = encs.shape[0]
            x.append(encs)
        x = torch.cat(x, dim=0)
        return x

    def decode_from_latent(self, latents):
        if isinstance(latents, np.ndarray):
            latents = torch.tensor(latents, dtype=torch.float32)
        ret_latents = self.model.decoder(latents)
        print('start', ret_latents['start'].shape)
        print('end', ret_latents['end'].shape)
        print('freq', ret_latents['freq'].shape)
        start_dt = self.ts_encoder.decode_from_latent(ret_latents['start'])
        end_dt = self.ts_encoder.decode_from_latent(ret_latents['end'])
        freq = self.freq_encoder.decode_from_latent(ret_latents['freq'])
        print('start_dt', start_dt)
        print('end_dt', end_dt)
        print('freq', freq)
        data_length = (end_dt[0] - start_dt[0]).total_seconds() / (freq * 60)

        print(f"Start: {start_dt}, End: {end_dt}, Frequency: {freq}, Data Length: {data_length}")
        # start_dt = self.freq_encoder.decode_from_latent(ret_latents['freq'][0])
        # print('start_dt', start_dt)
        # for i, lat in enumerate(ret_latents):
        #     print(f'latent {i}', lat)
        #     start_dt = self.ts_encoder.decode_from_latent(lat[:self.ts_encoder.dim_size])
        #     end_dt = self.ts_encoder.decode_from_latent(lat[self.ts_encoder.dim_size*2:self.ts_encoder.dim_size*3])
        #     freq = self.freq_encoder.decode_from_latent(lat[self.ts_encoder.dim_size*3:])
        #     data_length = (end_dt - start_dt).total_seconds() / (freq * 60)
        #     print(f"Start: {start_dt}, End: {end_dt}, Frequency: {freq}, Data Length: {data_length}")
        return ret_latents

if __name__ == "__main__":
    encoder = TimeRangeSTEncoder()
    # v = encoder.encode(datetime(2024,11,21,12,32), 10, end=datetime(2024,11,22,12,42))
    # print(v.shape)
    # dt = encoder.decode(v.squeeze(0))
    # print(dt)
    samples = [
        {
            'start': datetime(2024,11,21,12,32),
            'freq': 10,
            'end': datetime(2024,11,22,12,42)
        },
        # {
        #     'start': datetime(2019,7,11, 3, 0),
        #     'freq': 10,
        #     'end': datetime(2019,7,16, 8, 10)
        # }
    ]
    gt = encoder.get_ground_truth_batch(samples)
    print('gt', gt)
    e = encoder.decode_from_latent(gt)
    # print(e)
    # print(gt.shape)
