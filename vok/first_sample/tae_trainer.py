import hashlib
import numpy as np
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from vok.first_sample.multi_res_sc import MultiResolutionSampleCreator
from vok.first_sample.trans_ae import TransformerAutoencoder

class TransformerAutoencoderTrainer:
    def __init__(self, model, lr=0.001, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        # self.load()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(12, 6))

    def train(self, dataloader, epochs=10, verbose=True):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for encoding, latent in dataloader:
                if torch.isnan(encoding).any() or torch.isnan(latent).any():
                    print(encoding.shape, latent.shape)
                    print(encoding)
                    print(latent)
                    exit()
                encoding = encoding.to(self.device)
                latent = latent.to(self.device)

                self.optimizer.zero_grad()
                reconstructed = self.model(encoding, latent)
                if torch.isnan(reconstructed).any():
                    print(reconstructed.shape)
                    print(reconstructed)
                    exit()
                loss = self.criterion(reconstructed, latent)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * encoding.size(0)

            avg_loss = epoch_loss / len(dataloader.dataset)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

            if (epoch + 1) % 1 == 0:
                self.visualize_reconstructions(dataloader)

    def visualize_reconstructions(self, dataloader):
        self.model.eval()
        encoding, latent = next(iter(dataloader))
        encoding = encoding.to(self.device)
        latent = latent.to(self.device)
        with torch.no_grad():
            recon = self.model(encoding, latent)

        self.fig.suptitle("Reconstruction Samples")
        # Clear and update each subplot for the first three samples.
        for i in range(3):
            self.axs[i].cla()
            self.axs[i].plot(latent[i].cpu().numpy(), label="Original", color="blue")
            self.axs[i].plot(recon[i].cpu().numpy(), label="Reconstructed", color="orange", linestyle="dashed")
            self.axs[i].legend()
            self.axs[i].set_title(f"Sample {i}")
        plt.pause(0.01)
        self.model.train()

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for encoding, latent in dataloader:
                encoding = encoding.to(self.device)
                latent = latent.to(self.device)
                reconstructed = self.model(encoding, latent)
                loss = self.criterion(reconstructed, latent)
                total_loss += loss.item() * encoding.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Validation Loss: {avg_loss:.6f}")
        return avg_loss

    def save(self, path="autoencoder.pt"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="my_autoencoder.pt"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

# ---------------------------
# Example usage:
# ---------------------------
if __name__ == "__main__":
    # Create a dummy raw_data list of dictionaries.
    df = pd.read_pickle('notebooks/daily_data.pkl')
    df.fillna(0, inplace=True)
    print(df)
    df.set_index('date', inplace=True)
    # Convert to list of sequences
    df = df[['normalized']]
    df.rename(columns={'normalized': 'value'}, inplace=True)
    print(df)
    # Define how many samples to generate for each resolution.
    # For example, use 10 samples at original resolution, 5 at 20-minute resolution, etc.
    sample_counts = {1: 10000, 2: 1000, 3: 1000, 6: 1000}

    # Create the multi-resolution sample creator.
    sample_creator = MultiResolutionSampleCreator(df, sensor_type="temp", source_id="sensor_01", sample_counts=sample_counts)
    
    # Generate samples.
    samples = sample_creator.create_samples()
    # Instantiate the autoencoder model.
    autoencoder = TransformerAutoencoder()
    # Load data into a TensorDataset.
    dataset = autoencoder.load_data(samples)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Instantiate and run the trainer.
    trainer = TransformerAutoencoderTrainer(autoencoder, lr=1e-3)
    trainer.train(dataloader, epochs=20)
    trainer.evaluate(dataloader)
    trainer.save("my_autoencoder.pt")