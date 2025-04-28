from datetime import datetime
import os

import numpy as np
from dawn_vok.vok.embedding.encoders.frequency_encoder import FrequencyEncoder
from dawn_vok.vok.embedding.encoders.source_st_enc import TimeStampSTEncoder
from dawn_vok.vok.pipelines.meta_data.frequency_classefier.frequency_classefier_model import FrequencyClassifierModel
import torch

from dawn_vok.vok.pipelines.meta_data.time_classifier.time_classefier_model import TimeClassifierModel


class VClassifierUtils:
    def __init__(self, model, encoder, file_path=None):
        self.file_path = file_path
        self.model = model
        self.encoder = encoder
        if model is not None:
            if file_path:
                path = os.path.abspath(file_path)
                self.model.encoder.load_state_dict(torch.load(path))
            else:
                self.model.load_model_state_dict()

    def get_ground_truth(self, sample):
        if isinstance(sample, dict):
            sample = [sample]
        return self.get_ground_truth_batch(sample, device='cpu')

    def get_ground_truth_batch(self, samples: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
        B = samples.shape[0]
        # prepare padded encodings
        x = torch.zeros(B, 16, dtype=torch.float32, device=device)
        for i, f in enumerate(samples):
            encs = self.encoder.encode(f)        # => 1D tensor of length L ≤ 16
            L = encs.shape[0]
            x[i, :L] = encs

        # run through model in eval mode
        self.model.eval()
        with torch.no_grad():
            latents = self.model.encoder(x)                  # => [B, 16]
        return latents

class FrequencyClassifierUtils(VClassifierUtils):
    def __init__(self, file_path=None):
        model = FrequencyClassifierModel()
        encoder = FrequencyEncoder()
        super().__init__(model, encoder, file_path)
          
    #change this to get_ground_truth_batch  

class TimeStampClassifierUtils(VClassifierUtils):
    def __init__(self, file_path=None):
        model = TimeClassifierModel()
        encoder = TimeStampSTEncoder()
        super().__init__(model, encoder, file_path)

class TimeRangeClassifierUtils(VClassifierUtils):
    def __init__(self, file_path_ts=None, file_path_freq=None):
        model = TimeClassifierModel()
        encoder = TimeStampSTEncoder()
        super().__init__(model, encoder, file_path_ts)
        self.freq_utils = FrequencyClassifierUtils(file_path=file_path_freq)
       
    def get_ground_truth_batch(self, samples, device: str = 'cpu') -> torch.Tensor:
        start_list = []
        end_list = []
        freq_list = []
        for sample in samples:
            start_list.append(sample['start'])
            end_list.append(sample['end'])
            freq_list.append(sample['freq'])
        start_latents = super().get_ground_truth_batch(np.array(start_list), device)
        end_latents = super().get_ground_truth_batch(np.array(end_list), device)
        freq_latents = self.freq_utils.get_ground_truth_batch(np.array(freq_list), device)
        ret = {
            'start': start_latents,
            'end': end_latents,
            'freq': freq_latents,
            'full': torch.cat((start_latents[0], end_latents[0], freq_latents[0]), dim=0)
        }
        return ret


if __name__ == "__main__":
    # utils = FrequencyClassifierUtils(file_path=
    #     "/home/amiz/dawn/dawn_data/models/frequency_classifier/1_0_0/encoder.pt"
    # )
    # samples = torch.tensor([10, 20, 1440, 10, 10], dtype=torch.float32)
    # print(utils.get_ground_truth_batch(samples))

    utils = TimeRangeClassifierUtils()
    sample_list = [
        {
            "start": datetime(2021, 1, 1, 12, 0, 0),
            "end": datetime(2021, 1, 1, 12, 0, 1),
            "freq": 1
        },
        {
            "start": datetime(2021, 1, 1, 12, 0, 1),
            "end": datetime(2021, 1, 1, 12, 0, 2),
            "freq": 10
        },
        {
            "start": datetime(2021, 1, 1, 12, 0, 2),
            "end": datetime(2021, 3, 1, 12, 0, 3),
            "freq": 1440
        },
        {
            "start": datetime(2022, 11, 1, 12, 12, 3),
            "end": datetime(2022, 11, 11, 1, 12, 4),
            "freq": 1440
        }
    ]
    print(utils.get_ground_truth_batch(sample_list, device='cpu'))