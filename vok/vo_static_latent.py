import json
from pprint import pprint
import matplotlib
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from vok.dl_1 import TimeSeriesDataset, collate_fn

matplotlib.use('TkAgg')

import torch
import torch.nn as nn

class FCAutoEncoder(nn.Module):
    def __init__(self, base_size=1008, latent_size=128, hidden_sizes=None):
        """
        Args:
            base_size (int): The common size to which each sample is expanded (and from which it is decoded).
            latent_size (int): The size of the latent representation.
            hidden_sizes (list of int): Sizes of the hidden layers in the shared encoder/decoder.
        """
        super(FCAutoEncoder, self).__init__()
        self.base_size = base_size
        self.latent_size = latent_size
        # Supported original sequence lengths.
        self.supported_sizes = [36, 72, 144, 288, 1008]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Input scalers: Map from an original sequence length (e.g. 144) to base_size (1008)
        self.input_scalers = nn.ModuleDict({
            str(size): nn.Linear(size, base_size)
            for size in self.supported_sizes
        })
        # Output scalers: Map from base_size (1008) back to the original sequence length.
        self.output_scalers = nn.ModuleDict({
            str(size): nn.Linear(base_size, size)
            for size in self.supported_sizes
        })
        
        # Shared encoder/decoder core operating at the base size.
        if hidden_sizes is None:
            hidden_sizes = [512, 256]
        
        # Build encoder: from base_size to latent_size.
        encoder_layers = []
        prev_size = base_size
        for size in hidden_sizes:
            encoder_layers.append(nn.Linear(prev_size, size))
            encoder_layers.append(nn.ReLU())
            prev_size = size
        encoder_layers.append(nn.Linear(prev_size, latent_size))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder: from latent_size back to base_size.
        decoder_layers = []
        prev_size = latent_size
        for size in reversed(hidden_sizes):
            decoder_layers.append(nn.Linear(prev_size, size))
            decoder_layers.append(nn.ReLU())
            prev_size = size
        decoder_layers.append(nn.Linear(prev_size, base_size))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.to(self.device)
    
    def forward(self, x, seq_lengths):
        """
        Args:
            x (Tensor): A batch of padded sequences of shape (batch_size, 1008).
            seq_lengths (Tensor or list): Effective lengths for each sample in the batch.
            
        Objective:
          - For each sample, extract the first L elements (its effective part).
          - Expand it (using a dedicated fully connected layer) from L -> 1008.
          - Process the common representation with the encoder/decoder.
          - Contract (using another dedicated layer) from 1008 -> L.
          - Pad the result back to 1008 (if desired).
        """
        batch_size = x.shape[0]
        expanded = []
        for i in range(batch_size):
            L = int(seq_lengths[i])
            if str(L) not in self.input_scalers:
                raise ValueError(f"Unsupported sequence length {L}. Supported: {self.supported_sizes}")
            # Extract effective data from the padded sample.
            x_eff = x[i, :L]
            # Expand from L to base_size.
            expanded_sample = self.input_scalers[str(L)](x_eff)
            expanded.append(expanded_sample)
        # Stack expanded samples: (batch_size, base_size)
        expanded = torch.stack(expanded, dim=0)
        
        # Shared encoding and decoding.
        latent = self.encoder(expanded)
        decoded = self.decoder(latent)
        
        outputs = []
        for i in range(batch_size):
            L = int(seq_lengths[i])
            if str(L) not in self.output_scalers:
                raise ValueError(f"Unsupported sequence length {L}. Supported: {self.supported_sizes}")
            # Shrink the decoded sample from base_size back to L.
            shrunk = self.output_scalers[str(L)](decoded[i])
            # Optionally, pad the shrunk output back to base_size.
            pad_len = self.base_size - L
            padded = torch.cat([shrunk, torch.zeros(pad_len, device=self.device)], dim=0)
            outputs.append(padded)
        # Final output: (batch_size, base_size)
        outputs = torch.stack(outputs, dim=0)
        return outputs



class StaticEncoderTrainer:
    def __init__(self, model, train_loader, test_loader, num_epochs=30, learning_rate=1e-3):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.train_losses = []
        self.test_losses = []
        
        # Initialize plot
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
    def train_autoencoder(self):     
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            total_train_samples = 0
            
            for batch in self.train_loader:
                print(batch)
                # Extract padded input and the effective sequence lengths.
                x = batch['data']       # shape: (batch_size, 1008)
                seq_lengths = batch['seq_lengths']  # shape: (batch_size,)
                x = x.to(self.model.device)
                seq_lengths = seq_lengths.to(self.model.device)
                
                self.optimizer.zero_grad()
                # Forward pass (note: the model expects seq_lengths)
                outputs = self.model(x, seq_lengths)  # shape: (batch_size, 1008)
                
                # Compute loss only on the effective parts.
                loss = 0.0
                batch_size = x.size(0)
                for i in range(batch_size):
                    L = int(seq_lengths[i].item())
                    # Only compute loss on the first L elements.
                    loss += self.criterion(outputs[i, :L], x[i, :L])
                loss = loss / batch_size
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * batch_size
                total_train_samples += batch_size
            
            train_loss = train_loss / total_train_samples
            self.train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            test_loss = 0.0
            total_test_samples = 0
            with torch.no_grad():
                for batch in self.test_loader:
                    x = batch['data']
                    seq_lengths = batch['seq_lengths']
                    x = x.to(self.model.device)
                    seq_lengths = seq_lengths.to(self.model.device)
                    
                    outputs = self.model(x, seq_lengths)
                    loss = 0.0
                    batch_size = x.size(0)
                    for i in range(batch_size):
                        L = int(seq_lengths[i].item())
                        loss += self.criterion(outputs[i, :L], x[i, :L])
                    loss = loss / batch_size
                    
                    test_loss += loss.item() * batch_size
                    total_test_samples += batch_size
            test_loss = test_loss / total_test_samples
            self.test_losses.append(test_loss)
            
            # Update plot
            self.ax1.clear()
            self.ax1.plot(self.train_losses, label='Training Loss')
            self.ax1.plot(self.test_losses, label='Validation Loss')
            self.ax1.set_title('Training Progress')
            self.ax1.set_xlabel('Epoch')
            self.ax1.set_ylabel('Loss')
            self.ax1.legend()
            
            plt.pause(0.001)
            print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {test_loss:.4f}")
            
        plt.ioff()
        plt.show()
        return self.train_losses, self.test_losses

    def _update_plots(self, epoch):
        # Loss plot
        self.ax1.clear()
        self.ax1.plot(self.train_losses, label='Training Loss')
        self.ax1.plot(self.test_losses, label='Validation Loss')
        self.ax1.set_title('Training Progress')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        
        # Reconstruction plot
        if epoch % 5 == 0 or epoch == self.num_epochs-1:
            self.ax2.clear()
            with torch.no_grad():
                sample_batch = next(iter(self.test_loader))
                inputs = sample_batch['data'][:5].to(self.model.device)
                outputs = self.model(inputs).cpu().numpy()
                inputs = inputs.cpu().numpy()
                
                for i in range(3):  # Show first 3 samples
                    seq_len = sum(~sample_batch['mask'][i])
                    self.ax2.plot(inputs[i,:seq_len,0], label='Original', alpha=0.6)
                    self.ax2.plot(outputs[i,:seq_len,0], label='Reconstructed', linestyle='--')
                
            self.ax2.set_title('Sample Reconstructions')
            self.ax2.legend()
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def load_data():
    df = pd.read_pickle('notebooks/daily_data.pkl')
    df.fillna(0, inplace=True)
    
    # Convert to list of sequences
    sequences = df[['normalized']]  # Assuming normalized column contains sequences
    
    # Split dataset
    train_data, test_data = train_test_split(sequences, test_size=0.2)
    
    train_dataset = TimeSeriesDataset(train_data)
    test_dataset = TimeSeriesDataset(test_data)
    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, 
                            batch_size=32, 
                            shuffle=True, 
                            collate_fn=collate_fn)
    
    test_loader = DataLoader(test_dataset,
                           batch_size=32,
                           collate_fn=collate_fn)
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = load_data()
    exit()
    ae = FCAutoEncoder(base_size=1008, latent_size=64)

    trainer = StaticEncoderTrainer(ae, train_loader, test_loader)
    trainer.train_autoencoder()
