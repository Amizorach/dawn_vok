import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from torch.nn import functional as F

from vok.embedding.static_emb.static_emb_utils import StaticEmbeddingUtils
class TransformerEncoder(nn.Module):
    def __init__(self, 
                 input_encoding_dim=64, 
                 output_dim=256, 
                 ff_hidden_dim=512):
        super().__init__()
        self.input_encoding_dim = input_encoding_dim
        
        # Allowed latent sizes that we support.
        self.allowed_latent_sizes = [36, 72, 144, 288, 1008]
        
        # For each allowed latent size, create a fully-connected layer that takes the 
        # concatenated input (encoding + latent) and maps it to 256 dimensions.
        self.fc_map = nn.ModuleDict({
            str(s): nn.Linear(input_encoding_dim + s, 256) for s in self.allowed_latent_sizes
        })
        self.def_fc = nn.Linear(input_encoding_dim + 144, 256)
        # The rest of the layers remain the same.
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 512)
        self.relu2 = nn.ReLU()
        self.proj_out = nn.Linear(512, output_dim)
        
    def forward(self, encoding, latent):
        """
        encoding: (batch_size, input_encoding_dim)   # fixed size
        latent:   (batch_size, latent_dim)             # variable size
        
        This method adjusts the latent to one of the allowed sizes:
          - If latent size is smaller than an allowed size, pad it with zeros.
          - If latent size exceeds 1008, trim it to 1008.
        Then the adjusted latent is concatenated with the encoding and processed.
        """
        latent_size = latent.shape[1]
        target_size = None
        
        # Find the next allowed size (or use 1008 if latent is larger than all allowed sizes)
        for s in self.allowed_latent_sizes:
            if latent_size <= s:
                target_size = s
                break
        if target_size is None:
            target_size = 1008  # For latent sizes larger than 1008, we trim to 1008
        
        # Adjust the latent: pad if too short, trim if too long.
        if latent_size < target_size:
            pad_amount = target_size - latent_size
            latent = F.pad(latent, (0, pad_amount))  # pads zeros to the end of the last dimension
        elif latent_size > target_size:
            latent = latent[:, :target_size]
        
        # Concatenate the fixed-size encoding with the adjusted latent.
        x = torch.cat([encoding, latent], dim=-1)
        x = torch.clamp(x, -10, 10)

        # Select and apply the appropriate fully-connected layer based on target_size.
        fc_layer = self.fc_map[str(target_size)]
        x = fc_layer(x)
        # x = self.def_fc(x)  
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.proj_out(x)
        return x
class TransformerEncoder1(nn.Module):
    def __init__(self, 
                 input_encoding_dim=64, 
                 input_latent_dim=144, 
                 output_dim=256, 
                 num_layers=4, 
                 num_heads=4,
                 ff_hidden_dim=512):
        super().__init__()
        self.input_dim = input_encoding_dim + input_latent_dim  # 68 + 144 = 212
        self.proj_in = nn.Linear(self.input_dim, ff_hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ff_hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.proj_out = nn.Linear(ff_hidden_dim, output_dim)
        self.fc = nn.Linear(self.input_dim, 256)
        self.fc_1008 = nn.Linear(1008, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 512)
        self.relu2 = nn.ReLU()
        self.proj_out = nn.Linear(512, output_dim)
    def forward(self, encoding, latent):
        """
        encoding: (batch_size, 68)
        latent:   (batch_size, 144)
        Returns:  (batch_size, 256)
        """
        x = torch.cat([encoding, latent], dim=-1) 
        if torch.isnan(x).any():
            print(x)
            print(encoding)
            print(latent)
            exit()
        # (batch_size, 212)
        x = torch.clamp(x, -10, 10)

        # x = self.proj_in(x).unsqueeze(1)           # (batch_size, 1, ff_hidden_dim)
        # if torch.isnan(x).any():
        #     print(x)
        #     print(encoding)
        #     print(latent)
        #     exit()
        # x = self.transformer(x)                    # (batch_size, 1, ff_hidden_dim)
        # if torch.isnan(x).any():
        #     print(x)
        #     print(encoding)
        #     print(latent)
        #     exit()
        # x = x.squeeze(1)                           # (batch_size, ff_hidden_dim)
        # if torch.isnan(x).any():
        #     print(x)
        #     print(encoding)
        #     print(latent)
        #     exit()
        if x.shape[1] == 1008:
            x = self.fc_1008(x)
        else:
            x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.proj_out(x)
        return x

# ---------------------------
# LatentDecoder:
# Reconstructs the original 144D latent from the 256D encoded vector.
# ---------------------------
class LatentDecoder(nn.Module):
    def __init__(self, input_dim=256, output_dim=144, hidden_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        x: (batch_size, 256)
        returns: (batch_size, 144)
        """
        return self.model(x)

# ---------------------------
# TransformerAutoencoder:
# Wraps the encoder and decoder. Provides load_data() for creating batches.
# ---------------------------
class TransformerAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoder()
        self.decoder = LatentDecoder()
        # Stateless context encoder

    def forward(self, encoding, latent):
        z = self.encoder(encoding, latent)  # (batch, 256)
        recon = self.decoder(z)             # (batch, 144)
        return recon

    def load_data(self, ol):
        """
        ol: list of dicts.
        Each dict contains: 
            'sensor_type', 'source_id', 'start_ts', 'end_ts', 'frequency', 'data' (144D array)
        Returns: A TensorDataset of (encoding, latent) tensors.
        """
        context_vecs = []
        latent_vecs = []
        for item in ol:
            context = StaticEmbeddingUtils.encode_data_context(item['sensor_type'], item['source_id'], item['start_ts'], item['end_ts'], item['frequency'])
            context_vecs.append(context)
            latent_vecs.append(item['data'])
        context_tensor = torch.tensor(np.stack(context_vecs), dtype=torch.float32)
        latent_tensor = torch.tensor(np.stack(latent_vecs), dtype=torch.float32)
        return TensorDataset(context_tensor, latent_tensor)
