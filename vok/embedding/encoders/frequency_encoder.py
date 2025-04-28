from pprint import pprint
import numpy as np
from dawn_vok.utils.dict_utils import DictUtils
from datetime import datetime, timedelta
import torch

from dawn_vok.vok.embedding.encoders.base.model_st_encoder import ModelSTEncoder

class FrequencyEncoder(ModelSTEncoder):
    @classmethod
    def get_config(cls):
        return {
            'dim_size': 16,
            'freq_bins': [1, 10, 15, 30, 60, 120, 240, 480, 720, 1440,
                          1440*2, 1440*3, 1440*7, 1440*30, 1440*90,
                          1440*180, 1440*365],
            'num_harmonics': 7,
            'info_map': {
                0: 'log_freq',
                1: 'freq',
                2: 'sin_freq',
                3: 'cos_freq',
                4: 'sin_2_freq',
                5: 'cos_2_freq',
                6: 'sin_3_freq',
                7: 'cos_3_freq',
                8: 'sin_4_freq',
                9: 'cos_4_freq',
                10: 'sin_5_freq',
                11: 'cos_5_freq',
                12: 'sin_6_freq',
                13: 'cos_6_freq',
                14: 'sin_7_freq',
                15: 'cos_7_freq',
            }
        }
    def __init__(self):
        from dawn_vok.vok.pipelines.meta_data.frequency_classefier.frequency_classefier_model import FrequencyClassifierModel


        uid = 'frequency_encoder'
        model = FrequencyClassifierModel()
        super().__init__(uid=uid, model=model)
        self.freq_bins = self.config['freq_bins']
        self.num_harmonics = self.config['num_harmonics']
        P = torch.tensor(self.freq_bins, dtype=torch.float32)

        # normalized log2(period)
        logs = torch.log2(P)
        self.log_norm = (logs - logs.min()) / (logs.max() - logs.min())
        # normalized frequency (1/period)
        inv = 1.0 / P
        self.freq_norm = (inv - inv.min()) / (inv.max() - inv.min())

        # Fourier features up to num_harmonics
        omega = 2 * torch.pi * inv
        feats = [self.log_norm.unsqueeze(1), self.freq_norm.unsqueeze(1)]
        for k in range(1, self.num_harmonics + 1):
            feats.append(torch.sin(k * omega).unsqueeze(1))
            feats.append(torch.cos(k * omega).unsqueeze(1))

        # stack into [n_bins, dim]
        self.features = torch.cat(feats, dim=1)

    def encode(self, freq):
        """
        freq: scalar, list, or 1D tensor of periods (minutes)
        returns: torch.Tensor of shape [dim] or [N, dim]
        """
        # convert input to 1D tensor
        if not torch.is_tensor(freq):
            f = torch.tensor(freq, dtype=torch.float32).view(-1)
        else:
            f = freq.clone().detach().view(-1)

        # find nearest bin index per value
        P = torch.tensor(self.freq_bins, dtype=torch.float32)
        diffs = torch.abs(P.unsqueeze(0) - f.unsqueeze(1))  # [N, n_bins]
        idx = diffs.argmin(dim=1)

        # lookup features
        out = self.features[idx]
        return out.squeeze(0) if out.size(0) == 1 else out

    def decode(self, v):
        """
        v: tensor of shape [dim] or [N, dim]
        returns: period(s) in minutes corresponding to the nearest feature vector (full 16-D) bin
        """
        # prepare vectors
        print('v', v.shape)
        vecs = v.unsqueeze(0) if v.dim() == 1 else v  # [N, dim]
        # compute distance to each bin's feature vector
        # features: [n_bins, dim]
        # expand for broadcasting: vecs[:, None, :] - features[None, :, :]
        diffs = vecs.unsqueeze(1) - self.features.unsqueeze(0)  # [N, n_bins, dim]
        d2 = (diffs * diffs).sum(dim=2)  # [N, n_bins] squared distances
        idx = d2.argmin(dim=1)  # [N]
        print('idx', idx)
        periods = torch.tensor(self.freq_bins, dtype=torch.float32, device=vecs.device)
        out = periods[idx]
        return out.squeeze(0) if out.size(0) == 1 else out
    
    
    def decode_batch_logits(self, logits: dict):
        """
        logits: dict of tensors, each of shape [B, num_classes] for heads
        minute_bin_size: the bin size used when training (so we can invert the minute bin)
        Returns: list of datetime objects of length B
        """
        # print('logits', logits)
        freq_logits = logits['freq']              # shape [1, 17] in your example
        max_idx_tensor = torch.argmax(freq_logits, dim=1)  
        # print('max_idx_tensor:', max_idx_tensor)  # tensor([1])

        # if you want the Python int index:
        idx = max_idx_tensor.item()               
        # print('idx (int):', idx)                  # 1

        # now extract the logit at that index for each batch item:
        # since batch size B=1 here, we do:
        # value_tensor = freq_logits[0, max_idx_tensor]
        # print('value_tensor:', value_tensor)      # e.g. tensor([14.2260], grad_fn=…)

        # and to get a Python float:
        # value = value_tensor.item()
        out = self.config['freq_bins'][idx]
        # 1) pick predicted class indices for each head
        # preds = { head: torch.argmax(logits[head], dim=1).cpu().numpy()
        #           for head in ['freq']
        #           if head in logits }
        # #  print(f"Preds: {preds}")
        # B = next(iter(preds.values())).shape[0]
        # out = []
        # for i in range(B):
        #     # 2) build the “normalized feature” vector v of length len(info_map)
        #     v = torch.zeros(len(self.info_map), dtype=torch.float32)
        #     # year_norm lives at index where info_map==“year_norm” (should be 0)
        #     v[0] = (preds["freq"][i])
        #     # month_norm at index 1
        #     # 3) decode single vector
        #     dt = self.decode(v)
        #     out.append(dt)
        # print('out', out)
        return out
    
        
    def decode_from_latent(self, latents):
        # print('latents', latents.shape)
        lats = self.model.decoder(latents)
        # print('lats', lats.shape)
        
        ret = self.decode_batch_logits({'freq': lats})
        # print('ret', ret)
        return ret
  
if __name__ == "__main__":
    encoder = FrequencyEncoder()
    v = encoder.encode(10)
    print(v.shape)
    dt = encoder.decode(v)
    print(dt)
