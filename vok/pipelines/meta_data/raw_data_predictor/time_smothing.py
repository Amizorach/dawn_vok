import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from dawn_vok.vok.embedding.cyclic_emb.timestamp_encoder import TimestampEncoder
from dawn_vok.vok.embedding.static_emb.time_range_encoding import RichTimeRangeEncoding

class TimeBlockDataset(Dataset):
    def __init__(self, start_year=1990, end_year=2029, block_minutes=10):
        self.start_year = start_year
        self.end_year = end_year
        self.block_minutes = block_minutes
        total_minutes = (end_year - start_year + 1) * 365 * 24 * 60
        self.total_blocks = total_minutes // block_minutes
        self.norm_factor = self.total_blocks/10
        self.current_data = []
        self.update_data()
        self.meta_data = []

    def update_idx_data(self):
        #choose 10000 random index in the total_blocks
        random_index = np.random.randint(0, self.total_blocks, 10000)
        self.current_data = []
        for index in random_index:
            normalized_idx = index / self.norm_factor
            self.current_data.append(torch.tensor([normalized_idx], dtype=torch.float32))
    
    def update_data(self):
        random_index = np.random.randint(0, self.total_blocks, 1000)
        self.current_data = []
        self.meta_data = []
        for index in random_index:
            dt = datetime(self.start_year, 1, 1) + timedelta(minutes=int(index)*10)
            enc = self.encode_time(dt)
            self.current_data.append(enc)
            self.meta_data.append(dt)
        self.current_data = torch.stack(self.current_data, dim=0)
    def __len__(self):
        return len(self.current_data)

    def __getitem__(self, idx):
        return self.current_data[idx]
    
    def encode_time(self, dt):
        enc = RichTimeRangeEncoding(start=dt, end=dt, frequency=10*60).get_encoding()
        return torch.tensor(enc, dtype=torch.float32)
  # (input, target) pair

class TimeAutoencoder(nn.Module):
    def __init__(self, input_dim=32, latent_dim=16, output_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class TimeAutoencoderTrainer:
    def __init__(self, 
                 start_year=1990, 
                 end_year=2029, 
                 block_minutes=10,
                 latent_dim=32, 
                 batch_size=512, 
                 lr=1e-4, 
                 epochs=1000):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TimeAutoencoder(latent_dim).to(self.device)
        self.epochs = epochs
        self.block_minutes = block_minutes
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.dataset = TimeBlockDataset(start_year, end_year, block_minutes)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.random_datetimes = []

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            minute_loss = 0.0
            self.dataset.update_data()
            self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
            samples = []
            for i, x in enumerate(self.loader):
                x = x.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(x)
                loss = self.criterion(y_hat[:,0], x[:,0])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * x.size(0)
                if len(samples) < 5:
                    samples.append((y_hat, self.dataset.meta_data[i]))
            avg_loss = total_loss / len(self.dataset)

            # Report epoch loss
            print(f"Epoch {epoch+1}/{self.epochs} - Avg Loss: {avg_loss:.6f}")
            self.evaluate_samples(samples)
            # Evaluate a few samples
            # self.evaluate_samples()
    def decode_timestamp(self, normalized_timestamp):
        self.max_timestamp = datetime(2027, 1, 1).timestamp()
        self.min_timestamp = datetime(2020, 1, 1).timestamp()
        return normalized_timestamp * (self.max_timestamp - self.min_timestamp) + self.min_timestamp
    def evaluate_samples(self, samples):
        """
        Expects `samples` as a list of tuples: (model_output_tensor, original_datetime).
        Prints decoded datetime and error.
        """
        self.model.eval()
        for output_tensor, true_dt in samples:
            # Step 1: Decode normalized tensor to datetime
            predicted_datetimes = self.decode(output_tensor[:, 0])  # batch_size x 1

            for i, pred_dt in enumerate(predicted_datetimes):
                if i > 3:
                    break
                error_minutes = abs((true_dt - pred_dt).total_seconds()) / 60
                print(f"Sample {i+1}:")
                print(f"  Original datetime:  {true_dt}")
                print(f"  Predicted datetime: {pred_dt}")
                print(f"  Error: {error_minutes:.2f} minutes\n")

        # def evaluate_samples(self):
    #     self.model.eval()
    #     with torch.no_grad():
    #         # Use normalized input samples in [0, 1]
    #         sample_raw_blocks = torch.tensor([[0.0], [100000.0], [200000.0]], dtype=torch.float32)
    #         sample_blocks = sample_raw_blocks / self.dataset.total_blocks  # normalize to [0, 1]
    #         sample_blocks = sample_blocks.to(self.device)

    #         # Predict normalized, then decode
    #         reconstructed = self.model(sample_blocks)
    #         predicted_raw_blocks = reconstructed * self.dataset.total_blocks  # denormalize

    #         # Compute absolute error in blocks → convert to minutes
    #         error_blocks = (predicted_raw_blocks - sample_raw_blocks.to(self.device)).abs()
    #         error_minutes = error_blocks / self.dataset.norm_factor

    #     for i, err in enumerate(error_minutes.cpu().numpy()):
    #         print(f"Sample {i}: Error = {err[0]:.2f} minutes")

    def decode(self, normalized_output_tensor):
        """
        Converts normalized model output → block index → datetime
        """
        # Step 1: scale to block index
        block_indices = normalized_output_tensor * self.dataset.total_blocks

        # Step 2: convert to minutes
        total_minutes = block_indices / self.dataset.norm_factor

        # Step 3: convert to datetime
        start_datetime = datetime(self.dataset.start_year, 1, 1, 0, 0)
        datetimes = [start_datetime + timedelta(minutes=mins.item()) for mins in total_minutes]

        return datetimes
    # def evaluate_samples(self, num_samples=3):
    #     self.model.eval()
    #     self.dataset.update_data()
    #     self.loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
    #     with torch.no_grad():
    #         for i, x in enumerate(self.loader):
    #             x = x.to(self.device)
    #             reconstructed = self.model(x)
    #             meta_data = self.dataset.meta_data[i]
    #             print(reconstructed)
    #             predicted_blocks = reconstructed * self.dataset.total_blocks

    #             # 4. Convert back to predicted datetime
    #             predicted_minutes = predicted_blocks * self.block_minutes
    #             predicted_datetimes = [
    #                self.decode(predicted_minutes)
    #             ]

    #             # 5. Print results
    #             error_min = abs((predicted_minutes[i] - x[i]) /(self.block_minutes*10)).item()
    #             print(f"Sample {i}:")
    #             print(f"  Original datetime:   {meta_data}")
    #             print(f"  Predicted datetime:  {predicted_datetimes[i]}")
    #             print(f"  Error: {error_min:.2f} minutes\n")


# Example usage:
trainer = TimeAutoencoderTrainer()
trainer.train()
