#!/usr/bin/env python3
"""
train_raw_data_request_model.py

Encodes raw data request embeddings into a latent space,
then compares the encoded latent to the ground-truth latent
using a FragmentedLatentLoss. Periodically plots the encoded vs.
ground truth for a sample.
"""

from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.plt_utils import PlotUtils
from dawn_vok.vok.data_request.data_request_embedding import DataRequestEmbedding
from dawn_vok.vok.loss.fragmented_latent_loss import FragmentedLatentLoss
import matplotlib.pyplot as plt
from tqdm import trange

# ─── Configuration ─────────────────────────────────────────────────────────────
MONGO_DB         = 'embeddings'
MONGO_COLLECTION = 'embedding_paramaters'  # watch your spelling!

INPUT_DIM   = 136  # must match DataRequestEmbedding.get_embedding() length
LATENT_DIM  = 32

NUM_SAMPLES   = 5000
BATCH_SIZE    = 64
LEARNING_RATE = 1e-4
EPOCHS        = 1500
PLOT_INTERVAL = 25  # epochs

# ─── Model Definitions ─────────────────────────────────────────────────────────
class RawDataRequestFCEncoderModel(nn.Module):
    """MLP encoder: INPUT_DIM → 128 → LATENT_DIM."""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class RawDataRequestFCDecoderModel(nn.Module):
    """MLP decoder: LATENT_DIM → 128 → INPUT_DIM (unused in this script)."""
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class RawDataRequestModel(nn.Module):
    """
    Autoencoder that returns (z, reconstruction), but we use z vs gt.
    """
    def __init__(self, fragments: list):
        super().__init__()
        self.encoder = RawDataRequestFCEncoderModel(INPUT_DIM, LATENT_DIM)
        self.decoder = RawDataRequestFCDecoderModel(LATENT_DIM, INPUT_DIM)
        self.fragments = fragments

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

# ─── Dataset Definition ────────────────────────────────────────────────────────
class FakeRawDataRequestDataset(Dataset):
    """
    Synthetic DataRequestEmbedding dataset for latent learning.
    Returns (embedding, ground_truth_latent).
    """
    def __init__(self, input_dim: int = INPUT_DIM, num_samples: int = NUM_SAMPLES):
        self.input_dim = input_dim
        self.num_samples = num_samples
        self.samples = []

    def prepare_dataset(self):
        cursor = MongoUtils.get_collection(MONGO_DB, MONGO_COLLECTION) \
                   .find({'param_type': 'sensor_type'})
        self.sensor_di = {
            doc['param_id']: doc for doc in cursor
            if 'param_id' in doc and 'latents' in doc and 'mean' in doc['latents']
        }
        if not self.sensor_di:
            raise RuntimeError("No valid sensor embeddings found.")
        self.sources = [0.1,0.3,0.5,0.7,0.9]
        self.agg_list = ['mean','median','max','min','sum','count']
        self.freqs = [10*60,30*60,60*60,2*60*60,6*60*60,12*60*60,24*60*60]
        self.min_time = datetime(2021,1,1)
        self.max_time = datetime(2024,1,1)
        self.time_range = int((self.max_time-self.min_time).total_seconds())

    def generate_samples(self):
        while len(self.samples) < self.num_samples:
            emb, gt = self._create_sample()
            assert emb.shape[-1]==self.input_dim
            assert gt.shape[-1]==LATENT_DIM
            self.samples.append((torch.tensor(emb, dtype=torch.float32),
                                  torch.tensor(gt, dtype=torch.float32)))

    def _create_sample(self):
        src = float(np.random.choice(self.sources))
        sensor = np.random.choice(list(self.sensor_di))
        agg = np.random.choice(self.agg_list)
        freq = float(np.random.choice(self.freqs))
        start = datetime.fromtimestamp(np.random.randint(0,self.time_range)+self.min_time.timestamp())
        end = start+timedelta(seconds=freq)
        print(f"start: {start}, end: {end}")
        dre = DataRequestEmbedding(
            source_id=src, sensor_type=sensor,
            start_time=start, end_time=end,
            agg=agg, freq=freq, data_dim=self.input_dim)
        emb = dre.get_embedding()
        gt = dre.get_gt()
        return emb, gt

    def __len__(self): return len(self.samples)
    def __getitem__(self,idx): return self.samples[idx]

# ─── Trainer ───────────────────────────────────────────────────────────────────
class RawDataRequestModelTrainer:
    def __init__(self):
        self.fragments = [{'start':0,'end':4,'loss':'mse','weight':0.3}, 
                          {'start':4,'end':5,'loss':'mse','weight':0.3},
                          {'start':5,'end':21,'loss':'cosine','weight':0.4}]
        self.model = RawDataRequestModel(self.fragments)
        self.criterion = FragmentedLatentLoss(self.fragments, LATENT_DIM)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion.to(self.device)
        ds = FakeRawDataRequestDataset()
        ds.prepare_dataset()
        ds.generate_samples()
        self.loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        self.loss_history = []
        self.embed_history = []
        plt.ion()
        self.fig, (self.ax_loss, self.ax_embed) = plt.subplots(1, 2, figsize=(12, 4))
    def train(self):
       
        self.model.train()
        for epoch in trange(1,EPOCHS+1,desc='Epochs'):
            total_loss=0
            for x,gt in self.loader:
                x,gt = x.to(self.device),gt.to(self.device)
                self.optim.zero_grad()
                z,_ = self.model(x)
                loss = self.criterion(z, gt)
                loss.backward()
                self.optim.step()
                total_loss += loss.item()

            if epoch%PLOT_INTERVAL==0 or epoch==1:
                avg=total_loss/len(self.loader)
                self.loss_history.append(avg)
                print(f"Epoch {epoch} avg loss {avg:.4f}")
                self.plot_samples(gt,z)

    def plot_samples(self, gt, z):
        """Plot the first latent: encoded vs ground truth."""
        self.ax_loss.clear()
        self.ax_loss.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker='o')
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Avg Loss")
        self.ax_loss.set_title("Training Loss")
        self.ax_loss.grid(True)

        self.ax_embed.clear()
        self.ax_embed.plot(range(21), gt.cpu().detach().numpy()[0][0:21], marker='x', linestyle='--', label='Ground-Truth')
        self.ax_embed.plot(range(21), z.cpu().detach().numpy()[0][0:21], marker='o', linestyle='-', label='Predicted (Encoded)')
        self.ax_embed.set_xlabel("Dimension Index (4-31)")
        # self.ax_embed.set_title(f"Sample {rand_index} Embedding vs GT")
        self.ax_embed.legend()
        self.ax_embed.grid(True)
        self.fig.canvas.draw()
        plt.pause(0.01)

if __name__=='__main__':
    trainer=RawDataRequestModelTrainer()
    trainer.train()
