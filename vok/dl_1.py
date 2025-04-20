import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_lengths=[36, 144, 288, 1008], window=2016, num_samples=10000):
        """
        Args:
            data: DataFrame with datetime index and normalized values.
            seq_lengths: List of possible sequence lengths to sample.
            window: Maximum lookback window for position encoding.
            num_samples: Total number of random sequences to generate.
        """
        # Convert the DataFrame to a numpy array of float32
        self.data = data.values.astype(np.float32).tolist() 
        self.seq_lengths = seq_lengths
        self.window = window
        self.total_length = len(data)
        self.num_samples = num_samples  # Use the provided number of samples
        

    def __len__(self):
        # Return exactly num_samples
        return self.num_samples
    
    def __getitem__(self, idx):
        # Randomly select a sequence length from the provided options
        seq_len = np.random.choice(self.seq_lengths)
        
        # Get a random starting index; ensure we don't go out of bounds
        start_idx = np.random.randint(0, self.total_length - seq_len - 1)
        
        # Extract the sequence from the data
        sequence = self.data[start_idx:start_idx + seq_len]
        
        # Create sinusoidal position encoding
        positions = np.arange(seq_len) / self.window
        position_enc = np.array([
            np.sin(2 * np.pi * positions),
            np.cos(2 * np.pi * positions)
        ]).T.astype(np.float32)
        
        return {
            'data': torch.tensor(sequence),
            'pos_enc': torch.tensor(position_enc),
            'seq_len': seq_len
        }


def collate_fn(batch):
    """Dynamic padding for variable-length sequences in a batch."""
    # Determine the maximum sequence length in the batch
    max_len = max(item['data'].shape[0] for item in batch)
    
    padded_data = []
    padded_pos = []
    masks = []
    seq_lengths = []
    for item in batch:
        seq_len = item['data'].shape[0]
        seq_lengths.append(seq_len)

        # Pad the data with zeros
        # Assume data shape is (sequence_length, feature_dim)
        padding_data = torch.zeros((max_len - seq_len, item['data'].shape[1]))
        padded_data.append(torch.cat([item['data'], padding_data], dim=0))
        
        # Pad the position encoding in the same way
        padding_pos = torch.zeros((max_len - seq_len, item['pos_enc'].shape[1]))
        padded_pos.append(torch.cat([item['pos_enc'], padding_pos], dim=0))
        
        # Create a mask: 0 for real data, 1 for padding
        mask = torch.cat([torch.zeros(seq_len), torch.ones(max_len - seq_len)])
        masks.append(mask)
    
    return {
        'data': torch.stack(padded_data),        # shape: (batch_size, max_len, feature_dim)
        'pos_enc': torch.stack(padded_pos),        # shape: (batch_size, max_len, pos_dim)
        'mask': torch.stack(masks).bool(),
        'seq_lengths': torch.tensor(seq_lengths)         # shape: (batch_size, max_len)
    }


# Usage example
if __name__ == "__main__":
    # Assuming df is your preprocessed DataFrame
    # Example data creation:
    # date_rng = pd.date_range(start='2024-01-01', end='2024-12-31', freq='10T')
    # data = np.random.randn(len(date_rng), 2)  # 2 features
    # df = pd.DataFrame(data, index=date_rng, columns=['feature1', 'feature2'])
    df = pd.read_pickle('notebooks/daily_data.pkl')
    df.fillna(0, inplace=True)
    print(df)

    df = df[['normalized']]
    dataset = TimeSeriesDataset(df)
    tr_dat, te_dat = train_test_split(dataset, test_size=0.2, random_state=42)
    dataloader = DataLoader(tr_dat, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # Test one batch
    batch = next(iter(dataloader))
    print(f"Batch shapes:")
    print(f"Data: {batch['data'].shape}")
    print(f"Position encoding: {batch['pos_enc'].shape}")
    print(f"Mask: {batch['mask'].shape}")
    print(batch['data'][0])