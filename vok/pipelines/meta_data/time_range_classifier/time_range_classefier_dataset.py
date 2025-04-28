from datetime import datetime, timedelta
import random

import torch
from torch.utils.data import Dataset
from dawn_vok.vok.embedding.encoders.timerange_st_enc import TimeRangeSTEncoder
from dawn_vok.vok.embedding.encoders.source_st_enc import TimeStampSTEncoder
from dawn_vok.vok.pipelines.meta_data.frequency_classefier.frequency_classefier_utils import TimeRangeClassifierUtils


class TimeRangeClassifierDataset(Dataset):
    """
    Generates pairs of (feature_vector, timestamp).
    The feature vector `x` is engineered to contain information
    about year, month, day, hour, and minute derived from the timestamp `ts`.
    """
    def __init__(self,
                 start_year=1990,
                 end_year=2029,
                 input_dim=64, 
                 freq_bins=[1, 10, 15, 30, 60, 120, 240, 480, 720, 1440, 1440*2, 1440*3, 1440*7, 1440*30, 1440*90, 1440*180, 1440*365],
                 data_length_bins=[10, 100, 144, 180, 300],
                 length=1000):
        super().__init__() # Good practice to call super().__init__()
        self.start = datetime(start_year, 1, 1)
        self.end   = datetime(end_year, 12, 31, 23, 59)
        self.freq_bins = freq_bins
        self.data_length_bins = data_length_bins
        # self.minute_bin = minute_bin_size # Keep if needed elsewhere
        self.input_dim = input_dim
        self.length = length
        self.time_stamp_encoder = TimeRangeSTEncoder()
        # Ensure input_dim is sufficient for the engineered features
        # Current features require at least 9 dimensions.
        required_dims = 64
        if self.input_dim < required_dims:
             print(f"Warning: input_dim ({self.input_dim}) is less than {required_dims}, "
                   f"which are needed for the default engineered features. "
                   f"Consider increasing input_dim.")
             # Or raise ValueError(f"input_dim must be at least {required_dims} for default features")
        self.data = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.prepare_data()

    def prepare_data(self):
        x_list = []
        y_list = []
        for i in range(self.length):
            ts = self.start + timedelta(seconds=random.randint(0, int((self.end - self.start).total_seconds())))
            freq = random.choice(self.freq_bins)
            data_length = random.choice(self.data_length_bins)
            end = ts + timedelta(minutes=freq * data_length)
            if random.random() < 0.5:
                end1 = None
            else:
                end1 = end
            enc = self.time_stamp_encoder.encode(start=ts, freq=freq, end=end1, data_length=data_length)
            feat_len = enc.shape[0]
            x = torch.zeros(self.input_dim, dtype=torch.float32)
            if feat_len > self.input_dim:
                raise ValueError(f"Encoded feature-length {feat_len} exceeds input_dim {self.input_dim}")
            x[:feat_len] = enc
            x_list.append(x)

            y_list.append({'start': ts, 'end': end, 'freq': freq})

        clu = TimeRangeClassifierUtils()
        latents = clu.get_ground_truth_batch(y_list, device='cpu')
        self.x_list = x_list
        self.latents = latents
        # print(f"X: {self.x_list[0].shape}")
        # print(f"Y: {self.latents.keys()}") 
        self.y_list = []
        for i in range(len(self.x_list)):
            self.y_list.append({'start': torch.tensor(self.latents['start'][i], dtype=torch.float32, device=self.device),
                                 'end': torch.tensor(self.latents['end'][i], dtype=torch.float32, device=self.device), 
                                 'freq': torch.tensor(self.latents['freq'][i], dtype=torch.float32, device=self.device)})
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x_list[idx], self.y_list[idx]


if __name__ == "__main__":
    dataset = TimeRangeClassifierDataset(length=3)
    print(len(dataset))
    print(dataset[0])
