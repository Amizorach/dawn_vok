from datetime import datetime, timedelta
import random

import torch
from torch.utils.data import Dataset
from dawn_vok.vok.embedding.encoders.source_st_enc import TimeStampSTEncoder


class TimeClassifierDataset(Dataset):
    """
    Generates pairs of (feature_vector, timestamp).
    The feature vector `x` is engineered to contain information
    about year, month, day, hour, and minute derived from the timestamp `ts`.
    """
    def __init__(self,
                 start_year=1990,
                 end_year=2029,
                 minute_bin_size=10, # Note: minute_bin_size is used by Trainer/Model for output head size
                 input_dim=16,       # Ensure this is >= 9 for the features below
                 length=1000):
        super().__init__() # Good practice to call super().__init__()
        self.start = datetime(start_year, 1, 1)
        self.end   = datetime(end_year, 12, 31, 23, 59)
        # self.minute_bin = minute_bin_size # Keep if needed elsewhere
        self.input_dim = input_dim
        self.length = length
        self.time_stamp_encoder = TimeStampSTEncoder()
        # Ensure input_dim is sufficient for the engineered features
        # Current features require at least 9 dimensions.
        required_dims = 16
        if self.input_dim < required_dims:
             print(f"Warning: input_dim ({self.input_dim}) is less than {required_dims}, "
                   f"which are needed for the default engineered features. "
                   f"Consider increasing input_dim.")
             # Or raise ValueError(f"input_dim must be at least {required_dims} for default features")
        self.data = []
        self.prepare_data()

    def prepare_data(self):
        for i in range(self.length):
            ts = self.start + timedelta(seconds=random.randint(0, int((self.end - self.start).total_seconds())))
            enc = self.time_stamp_encoder.encode(ts)
            feat_len = enc.shape[0]
            x = torch.zeros(self.input_dim, dtype=torch.float32)
            if feat_len > self.input_dim:
                raise ValueError(f"Encoded feature-length {feat_len} exceeds input_dim {self.input_dim}")
            x[:feat_len] = enc
            self.data.append((x, ts))


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]
