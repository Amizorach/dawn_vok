from torch.utils.data import Dataset
from dawn_vok.vok.v_objects.vok_object import VOKObject
import torch

class VOKDataset(VOKObject, Dataset):
    def __init__(self, dataset_id, name=None):
       Dataset.__init__(self)
       uid = f'dataset_{dataset_id}'
       VOKObject.__init__(self, uid=uid, name=name, obj_type='dataset')

    
    def pad_and_create_mask(self, x, pad_length):
        """
        Pads or truncates the tensor x to have pad_length rows and creates a mask indicating valid rows.
        
        Args:
            x (torch.Tensor): Input tensor of shape (n, 384) where n can vary.
            pad_length (int): The fixed sequence length to output.
        
        Returns:
            tuple: padded_x of shape (pad_length, 384) and mask of shape (pad_length,)
        """
        # if len(x.shape) == 0:
        #     return None, None
       
        # print('pad_and_create_mask', x.shape)

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
        # print('padded_x', padded_x.shape)
        return padded_x, mask
     
    def pad_data(self, x, pad_length):
        seq_len = x.shape[0]
        valid_length = min(seq_len, pad_length)
        padded_x = torch.zeros((pad_length, x.shape[1]), dtype=x.dtype)
        padded_x[:valid_length, :] = x[:valid_length, :]
        return padded_x
    
    def load_data(self):
        raise NotImplementedError("Subclasses must implement this method")