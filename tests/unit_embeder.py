import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import uuid

# --- Unit Embedder Class ---
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import uuid

from unit_samples import UNIT_DEFINITIONS
# --- Unit Embedder Class ---
class UnitEmbedder:
    def __init__(
        self,
        unit_definitions,
        model_name="paraphrase-MiniLM-L3-v2",
        cache_dir="./unit_embedder_cache",
        # latent_dim=16
    ):
        """
        Initializes the UnitEmbedder class which handles encoding and decoding of multiple unit types into full latents.
        
        Args:
            unit_definitions (dict): Definitions of all the units.
            unit_types_dims (dict): Dictionary specifying latent dimensions for each unit type (e.g., "source", "sensor").
            model_name (str): Pre-trained model name from SentenceTransformers.
            cache_dir (str): Directory for storing cached data like latents, etc.
            latent_dim (int): Latent dimension size to reduce to.
        """
        self.unit_definitions = unit_definitions
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.model_latent_size = 384
        self.latent_dim = self.model_latent_size * 3 # Target latent dimension after network reduction
        self.unit_ids = list(unit_definitions.keys())

        # Build latent table for all units
        self.latent_table = self._build_latent_table()

    def _format_field(self, field, value):
        """Formats a text field with its value."""
        return f"{field}: {value}"

    def encode_unit(self, unit_id):
        """Encodes the unit into a full latent vector."""
        unit = self.unit_definitions[unit_id]
        # Encode unit type, parameter, and descriptors
        type_vec = self._encode_text("unit_type", unit["unit_type"])
        param_vec = self._encode_text("parameter", unit["parameter"])

        descs = unit.get("descriptors", {})
        if descs:
            desc_embeds = [
                self._encode_text(k, v) for k, v in descs.items()
            ]
            desc_vec = np.mean(desc_embeds, axis=0)  # Average descriptors
        else:
            desc_vec = np.zeros(self.unit_types_dims["descriptors"])

        # Concatenate the vectors (unit_type + parameter + descriptors)
        full_vec = np.concatenate([type_vec, param_vec, desc_vec])
        print('type_vec.shape', type_vec.shape)
        print('param_vec.shape', param_vec.shape)
        print('desc_vec.shape', desc_vec.shape)
        print('full_vec.shape', full_vec.shape)
        print('self.latent_dim', self.latent_dim)
        return full_vec  # Let the network handle dimensionality reduction

    def _encode_text(self, field, value):
        """Encodes a text field using the SentenceTransformer model."""
        raw = self.model.encode([self._format_field(field, value)])[0]
        return raw

    def _build_latent_table(self):
        """Build the latent table with reduced dimension vectors for all units."""
        latents = np.array([self.encode_unit(uid) for uid in self.unit_ids])
        return latents


# --- Retrieval Encoder ---
class EnhancedRetrievalEncoder(nn.Module):
    def __init__(self, input_dim=48, output_dim=48):
        super(EnhancedRetrievalEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),  # Increase layer size to capture more complexity
            nn.LeakyReLU(0.1),          # LeakyReLU for more powerful learning
            nn.Linear(512, 256),        # Intermediate projection layer
            nn.LeakyReLU(0.1),
            nn.Linear(256, output_dim)  # Output dimensions
        )

    def forward(self, x):
        return F.normalize(self.fc(x), p=2, dim=1)  # Normalize output to unit vector


# --- Retrieval Trainer ---
class RetrievalTrainer(nn.Module):
    def __init__(self, input_dim=1152, output_dim=64, latent_table=None, target_ids=None, device="cpu"):
        super(RetrievalTrainer, self).__init__()

        # Ensure the input_dim matches the latent vector size of 1152
        assert input_dim == 1152, f"Expected input_dim to be 1152, but got {input_dim}"

        self.device = device
        self.db_latents = torch.tensor(latent_table, dtype=torch.float32, device=device)
        self.target_ids = target_ids

        # Latent dimensionality reduction network
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),  # First layer to reduce input size
            nn.ReLU(),
            nn.Linear(512, 256),  # Intermediate layer to further reduce size
            nn.ReLU(),
            nn.Linear(256, output_dim)  # Output reduced dimension (64)
        )

        self.loss_fn = nn.TripletMarginLoss(margin=2.0)  # Using Triplet Loss to train retrieval

    def forward(self, x):
        """Forward pass with dimensionality reduction."""
        return F.normalize(self.fc(x), p=2, dim=1)  # Normalize the output to a unit vector

    def train_step(self, batch_queries, batch_target_ids, optimizer):
        """Training step for the retrieval model."""
        self.train()
        optimizer.zero_grad()

        query_tensor = torch.tensor(batch_queries, dtype=torch.float32, device=self.device)
        query_tensor = F.normalize(query_tensor, p=2, dim=1)  # Normalize queries before encoding

        # Step 1: Encode queries and get the predictions
        encoded_queries = self(query_tensor)

        # Step 2: Get the target latents for each query in the batch (positive samples)
        target_latents = torch.stack([
            self.db_latents[self.target_ids.index(tid)] for tid in batch_target_ids
        ])

        # Step 3: Get a batch of negative samples (random latents from the database)
        negative_samples_idx = torch.randint(0, len(self.db_latents), (len(batch_target_ids),))
        negative_latents = self.db_latents[negative_samples_idx]

        # Step 4: Compute Triplet Loss
        loss = self.loss_fn(encoded_queries, target_latents, negative_latents)

        # Step 5: Backpropagate and update weights
        loss.backward()
        optimizer.step()

        return loss.item()


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait before stopping if loss doesn't improve.
            min_delta (float): The minimum change to qualify as an improvement.
            verbose (bool): Whether to print the messages or not.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def should_stop(self, current_loss, current_epoch):
        """
        Checks whether the training should stop based on the current loss and epoch.

        Returns:
            bool: Whether to stop training early.
        """
        if self.best_loss is None:
            self.best_loss = current_loss
            self.best_epoch = current_epoch
            return False
        
        # Check if loss has improved
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_epoch = current_epoch
            self.epochs_without_improvement = 0  # reset patience counter
            return False
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered at epoch {current_epoch + 1} (no improvement in {self.patience} epochs)")
                return True
            return False



    # Initialize early stopping with patience

## Sample Usage
if __name__ == "__main__":

    # Step 1: Initialize embedder and encode units
    embedder = UnitEmbedder(UNIT_DEFINITIONS)
    latent_table = embedder.latent_table
    full_latents = [embedder.encode_unit(uid) for uid in embedder.unit_ids]

    # Step 2: Train the retrieval encoder
    encoder = RetrievalTrainer(input_dim=1152, output_dim=64, latent_table=latent_table, target_ids=embedder.unit_ids)
    optimizer = optim.Adam(encoder.parameters(), lr=1e-5)

    # Step 3: Run a few training steps
    for epoch in range(1000):
        total_loss = 0
        for i in range(0, len(full_latents), 16):
            batch_q = full_latents[i:i + 16]
            batch_ids = embedder.unit_ids[i:i + 16]
            loss = encoder.train_step(batch_q, batch_ids, optimizer)
            total_loss += loss
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
    
    # Step 4: Test retrieval
    encoder.eval()
    with torch.no_grad():
        query_tensor = torch.tensor(full_latents[:10], dtype=torch.float32)
        query_tensor = F.normalize(query_tensor, p=2, dim=1)
        encoded = encoder(query_tensor).cpu().numpy()

        for i, vec in enumerate(encoded):
            top_k = embedder.decode_latent(vec, return_top_k=5)
            print(f"Query {i} expected: {embedder.unit_ids[i]}")
            for rank, (uid, score) in enumerate(top_k):
                print(f"  Top {rank+1}: {uid} (score: {score:.4f})")
