from datetime import datetime, timedelta
import random

import torch
from torch.utils.data import Dataset
from dawn_vok.vok.embedding.encoders.frequency_encoder import FrequencyEncoder
from dawn_vok.vok.embedding.encoders.source_st_enc import TimeStampSTEncoder


class FrequencyClassifierDataset(Dataset):
    """
    Generates pairs of (feature_vector, timestamp).
    The feature vector `x` is engineered to contain information
    about year, month, day, hour, and minute derived from the timestamp `ts`.
    """
    def __init__(self,# Note: minute_bin_size is used by Trainer/Model for output head size
                 input_dim=16,       # Ensure this is >= 16 for the features below
                 length=1000):
        super().__init__() # Good practice to call super().__init__()
      
        # self.minute_bin = minute_bin_size # Keep if needed elsewhere
        self.input_dim = input_dim
        self.length = length
        self.freq_encoder = FrequencyEncoder()
       
        self.data = []
        self.prepare_data()

    def prepare_data(self):
        freq_bins = self.freq_encoder.freq_bins
        for i in range(self.length):
            bin_idx = random.randint(0, len(freq_bins)-1)
            period = freq_bins[bin_idx]
            enc = self.freq_encoder.encode(period)
            feat_len = enc.shape[0]
            x = torch.zeros(self.input_dim, dtype=torch.float32)
            if feat_len > self.input_dim:
                raise ValueError(f"Encoded feature-length {feat_len} exceeds input_dim {self.input_dim}")
            x[:feat_len] = enc
            y = torch.tensor([bin_idx], dtype=torch.float32)
            self.data.append((x, y))


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

if __name__ == "__main__":
    dataset = FrequencyClassifierDataset()
    print(len(dataset))
    print(dataset[0])
