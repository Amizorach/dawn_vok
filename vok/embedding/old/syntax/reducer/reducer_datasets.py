import torch
from torch.utils.data import Dataset

class SyntaxEmbeddingReducerVectorDataset(Dataset):
    def __init__(self, data_list, pad_length=10, device=None):
        """
        Args:
            data_list (list): List of dictionaries with key 'data' that holds a tensor data (or list) of shape (x, 384).
            pad_length (int): Target number of rows for each sample, default is 10.
        """
        self.data_list = data_list
        self.pad_length = pad_length
        self.data = []
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        for d in self.data_list:
            if isinstance(d,dict):
                d = d['data']
            raw_x = torch.tensor(d, dtype=torch.float32)  # Shape: (x, 384)
            padded_x, mask = self.pad_and_create_mask(raw_x, pad_length)
            # Here we set the target as the padded_x as well (for reconstruction tasks)
            y = padded_x.clone()
            self.data.append((padded_x.to(self.device), y.to(self.device), mask.to(self.device)))
    def add_dirty_data(self, noise_level=0.1, num_samples=10):
        dirty_data = []
        for d in self.data:
            for i in range(num_samples):
                dd = d[0].clone()
                #add noise to the data
                dd += torch.randn_like(dd) * noise_level
                dirty_data.append((dd.to(self.device), d[1].to(self.device), d[2].to(self.device)))
        
        self.data.extend(dirty_data)
        return dirty_data
    def pad_and_create_mask(self, x, pad_length):
        """
        Pads or truncates the tensor x to have pad_length rows and creates a mask indicating valid rows.
        
        Args:
            x (torch.Tensor): Input tensor of shape (n, 384) where n can vary.
            pad_length (int): The fixed sequence length to output.
        
        Returns:
            tuple: padded_x of shape (pad_length, 384) and mask of shape (pad_length,)
        """
        seq_len = x.shape[0]
        valid_length = min(seq_len, pad_length)
        # Create a mask: 1 for valid positions, 0 for padded positions.
        mask = torch.zeros(pad_length, dtype=torch.float32)
        mask[:valid_length] = 1
        
        # If the input is shorter than pad_length, pad with zeros;
        # if it is longer, truncate to pad_length.
        if seq_len < pad_length:
            padded_x = torch.zeros((pad_length, x.shape[1]), dtype=x.dtype)
            padded_x[:seq_len, :] = x
        else:
            padded_x = x[:pad_length, :]
        
        return padded_x, mask
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Returns a tuple: (input, target, mask)
        return self.data[idx][0], self.data[idx][1], self.data[idx][2]
    


class SyntaxEmbeddingReducerInferenceDataset(Dataset):
    def __init__(self, data_list, pad_length=10, device=None):
        self.data_list = data_list
        self.pad_length = pad_length
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data = []
        for d in self.data_list:
            if isinstance(d,dict):
                d = d['data']
            raw_x = torch.tensor(d, dtype=torch.float32)  # Shape: (x, 384)
            padded_x = self.pad_data(raw_x, pad_length)
            self.data.append(padded_x.to(self.device))
            
    def pad_data(self, x, pad_length):
        seq_len = x.shape[0]
        valid_length = min(seq_len, pad_length)
        padded_x = torch.zeros((pad_length, x.shape[1]), dtype=x.dtype)
        padded_x[:valid_length, :] = x[:valid_length, :]
        return padded_x
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    