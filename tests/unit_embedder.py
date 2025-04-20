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
import random

from dawn_vok.vok.model_utils.early_stopping import EarlyStopping
from dawn_vok.utils.dir_utils import DirUtils # Added import

# --- Unit Definitions (Combined from the second block) ---
sensor_types = [
    "temperature", "humidity", "radiation", "soil_moisture", "air_pressure",
    "wind_speed", "co2", "ph", "ec", "light_intensity", "rainfall", "wind_direction", "wind_speed", "solar_radiation", "soil_temperature", "soil_moisture", "soil_ph", "soil_ec", "soil_organic_matter", "soil_nitrogen", "soil_phosphorus", "soil_potassium"
]
manufacturers = ["Netafim", "Bosch", "Parrot", "AgriTech", "SensorCo", "WeatherCo", "PlantCo", "SoilCo", "WaterCo", "LightCo"]
locations = ["Greenhouse A", "Greenhouse B", "Field C", "Field D", "Orchard E", "Farm F", "Field G", "Field H", "Field I", "Field J"]
models = [f"M{100 + i}" for i in range(20)]  # e.g., M100–M119
value_types = ["float", "int", "str", "bool", "celcius", "fahrenheit", "kelvin", "pascal", "hPa", "kPa", "MPa", "GPa", "m/s", "km/h", "mph", "kph",
                "km/s", "m/s", "km/h", "mph", "kph", "km/s"]
UNIT_DEFINITIONS = {
    str(uuid.uuid4()): {
        "unit_type": "sensor",
        "parameter": random.choice(sensor_types),
         "value_type": random.choice(value_types),
        "descriptors": {
            # Ensure some units might have missing descriptors for testing
            k: v for k, v in {
               
                "manufacturer": random.choice(manufacturers),
                "location": random.choice(locations),
                "model": random.choice(models)
            }.items() if random.random() > 0.1 # ~10% chance to skip a descriptor
        }
    }
    for _ in range(10000)
}
# Add a unit with definitely no descriptors for robust testing
no_desc_id = str(uuid.uuid4())
UNIT_DEFINITIONS[no_desc_id] = {
        "unit_type": "sensor",
        "parameter": "special_parameter",
        "descriptors": {}
}
print(f"Total unit definitions: {len(UNIT_DEFINITIONS)}")

# --- Unit Embedder Class ---
class UnitEmbedder:
    def __init__(
        self,
        unit_definitions,
        model_name="paraphrase-MiniLM-L3-v2",
        cache_dir="./unit_embedder_cache",
    ):
        """
        Initializes the UnitEmbedder class which handles encoding units into full latents.
        """
        self.unit_definitions = unit_definitions
        self.model = SentenceTransformer(model_name)
        self.encode_map = {}

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # This is the dimension output by the SentenceTransformer model
        self.model_latent_size = self.model.get_sentence_embedding_dimension() # More robust way to get dim
        # The full latent dim is 3 * model_latent_size (type + param + avg_desc)
        self.full_latent_dim = self.model_latent_size * 3
        self.unit_ids = list(unit_definitions.keys())

        # Build latent table for all units
        print("Building latent table...")
        self.latent_table = self._build_latent_table()
        print(f"Latent table built with shape: {self.latent_table.shape}")

    def _format_field(self, field, value):
        """Formats a text field with its value."""
        return f"{field}: {value}"

    def encode_unit(self, unit_id):
        """Encodes the unit into a full latent vector."""
        if unit_id not in self.unit_definitions:
            raise ValueError(f"Unit ID {unit_id} not found in definitions.")
        unit = self.unit_definitions[unit_id]

        # Encode unit type, parameter

        type_vec = self._encode_text("unit_type", unit["unit_type"])
        param_vec = self._encode_text("parameter", unit["parameter"])

        descs = unit.get("descriptors", {})
        if descs:
            desc_embeds = [
                self._encode_text(k, v) for k, v in descs.items() if v # Handle potential None values
            ]
            value_type = self._encode_text("value_type", unit["value_type"])
            if desc_embeds: # Check if list is not empty after filtering Nones
                #  desc_embeds = np.array(desc_embeds)

                #  #the first desc is the value type
                #  #we want to make sure it's encoded strongly
                #  weights = np.array([1.0/desc_embeds[i].shape[0]*2 for i in range(len(desc_embeds))])
                #  weights[0] = 1.0 # Make the value type weight stronger
                #  desc_vec = np.sum(desc_embeds.T * weights, axis=1)
                 desc_vec = np.mean(desc_embeds, axis=0) 
                 desc_vec = desc_vec + value_type # Add the value type to the descriptor vector
            else:
                 # Handle case where descriptors exist but values are None/empty
                 desc_vec = np.zeros(self.model_latent_size)
        else:
            # Use zeros of the correct dimension if descriptors dict is empty
            desc_vec = np.zeros(self.model_latent_size) # FIX: Use model_latent_size

        # Ensure all vectors have the expected shape before concatenating
        assert type_vec.shape == (self.model_latent_size,), f"Type vec shape error: {type_vec.shape}"
        assert param_vec.shape == (self.model_latent_size,), f"Param vec shape error: {param_vec.shape}"
        assert desc_vec.shape == (self.model_latent_size,), f"Desc vec shape error: {desc_vec.shape}"

        # Concatenate the vectors (unit_type + parameter + descriptors)
        full_vec = np.concatenate([type_vec, param_vec, desc_vec])
        # print('type_vec.shape', type_vec.shape) # Keep for debug if needed
        # print('param_vec.shape', param_vec.shape)
        # print('desc_vec.shape', desc_vec.shape)
        # print('full_vec.shape', full_vec.shape)
        # print('self.full_latent_dim', self.full_latent_dim) # Compare with actual shape
        assert full_vec.shape == (self.full_latent_dim,), "Concatenated vector dimension mismatch"
        return full_vec

    def _encode_text(self, field, value):

        """Encodes a text field using the SentenceTransformer model."""
        text_to_encode = self._format_field(field, str(value)) # Ensure value is string
        # The model expects a list of sentences
        if text_to_encode not in self.encode_map:
            embedding = self.model.encode([text_to_encode])[0]
            self.encode_map[text_to_encode] = embedding
        else:
            embedding = self.encode_map[text_to_encode]
        return embedding

    def _build_latent_table(self):
        """Build the latent table with full dimension vectors for all units."""
        latents = []
        for uid in self.unit_ids:
            try:
                latents.append(self.encode_unit(uid))
            except Exception as e:
                print(f"Error encoding unit {uid}: {e}")
                # Handle error appropriately, e.g., skip unit or use a default vector
                # Using zeros as a placeholder here:
                latents.append(np.zeros(self.full_latent_dim))
        print(f"Latent table built with shape: {np.array(latents).shape}")
        return np.array(latents)


# --- Retrieval Trainer ---
class RetrievalTrainer(nn.Module):
    # FIX: input_dim must match embedder.full_latent_dim
    def __init__(self, input_dim, output_dim=64, latent_table=None, target_ids=None, device="cpu"):
        super(RetrievalTrainer, self).__init__()

        # Ensure the input_dim matches the latent vector size
        expected_input_dim = latent_table.shape[1] # Get dim from actual data
        assert input_dim == expected_input_dim, f"Expected input_dim {expected_input_dim}, but got {input_dim}"

        self.device = device
        self.db_latents_full = torch.tensor(latent_table, dtype=torch.float32) # Keep original full latents
        self.target_ids = target_ids
        self.output_dim = output_dim

        # Latent dimensionality reduction network
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            #nn.Dropout(0.2), # Optional: Add dropout for regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            #nn.Dropout(0.2), # Optional: Add dropout
            nn.Linear(256, output_dim) # Output reduced dimension
        )

        self.loss_fn = nn.TripletMarginLoss(margin=1.0) # Adjusted margin slightly

        # Store encoded database latents (reduced dimension) after training/loading
        self.db_latents_encoded = None
        self.to(device) # Move model to device

    def forward(self, x):
        """Forward pass with dimensionality reduction and normalization."""
        # Ensure input is on the correct device
        x = x.to(self.device)
        encoded = self.fc(x)
        return F.normalize(encoded, p=2, dim=1) # Normalize the output

    def encode_database(self):
        """Encodes the entire database using the trained network."""
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            # Encode in batches if the database is very large
            # For simplicity here, encode all at once if feasible
            full_db_tensor = self.db_latents_full.to(self.device)
            self.db_latents_encoded = self(full_db_tensor).cpu() # Encode and move back to CPU
        print(f"Database encoded to shape: {self.db_latents_encoded.shape}")


    def train_step(self, batch_queries_full, batch_target_ids, optimizer):
        """Training step for the retrieval model."""
        self.train() # Set model to training mode
        optimizer.zero_grad()

        # Convert full latents to tensor and move to device
        query_tensor_full = torch.tensor(np.array(batch_queries_full), dtype=torch.float32).to(self.device)
        # query_tensor_full = F.normalize(query_tensor_full, p=2, dim=1) # Normalize input? Maybe not necessary before network

        # Step 1: Encode queries (pass full latents through the network)
        encoded_queries = self(query_tensor_full) # Shape: (batch_size, output_dim)

        # Step 2: Get the target latents (positive samples) - these are the *full* latents
        # We need to find the corresponding full latent vector for the target ID
        target_indices = [self.target_ids.index(tid) for tid in batch_target_ids]
        target_latents_full = self.db_latents_full[target_indices].to(self.device)
        # Encode the targets using the *current* network state to get positive pairs in reduced space
        encoded_targets_positive = self(target_latents_full) # Shape: (batch_size, output_dim)


        # Step 3: Get negative samples
        # Simple random sampling: Pick random full latents from the DB *excluding* the current batch targets
        negative_indices = []
        possible_indices = list(range(len(self.target_ids)))
        for target_idx in target_indices:
            # Create a list of indices excluding the current target's index
            potential_negatives = [idx for idx in possible_indices if idx != target_idx]
            # Choose one random negative index
            negative_indices.append(random.choice(potential_negatives))

        negative_latents_full = self.db_latents_full[negative_indices].to(self.device)
        # Encode the negative samples using the *current* network state
        encoded_latents_negative = self(negative_latents_full) # Shape: (batch_size, output_dim)

        # Detach negative and positive samples from the graph for the loss calculation
        # The gradients should only flow back through the anchor (encoded_queries)
        encoded_targets_positive = encoded_targets_positive.detach()
        encoded_latents_negative = encoded_latents_negative.detach()

        # Step 4: Compute Triplet Loss
        loss = self.loss_fn(encoded_queries, encoded_targets_positive, encoded_latents_negative)

        # Step 5: Backpropagate and update weights
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.item()

    # FIX: Implement retrieval using the trained encoder
    def retrieve(self, query_vector_full, return_top_k=5):
        """
        Retrieves the top K most similar items from the database for a given full query vector.
        Assumes encode_database() has been called after training.
        """
        if self.db_latents_encoded is None:
            print("Warning: Database not encoded. Encoding now.")
            self.encode_database()
            if self.db_latents_encoded is None:
                 raise RuntimeError("Failed to encode database.")


        self.eval() # Set to evaluation mode
        with torch.no_grad():
            # Encode the single query vector (needs to be batch format [1, input_dim])
            query_tensor_full = torch.tensor(np.array([query_vector_full]), dtype=torch.float32).to(self.device)
            encoded_query = self(query_tensor_full).cpu() # Encode and move to CPU

            # Compute cosine similarity between the encoded query and all encoded DB items
            # Similarities shape: (1, num_db_items)
            similarities = cosine_similarity(encoded_query, self.db_latents_encoded)[0]

            # Get the indices of the top K most similar items
            # Argsort sorts in ascending order, so we take the end of the sorted list
            top_k_indices = np.argsort(similarities)[-return_top_k:][::-1] # Descending order

            # Get the corresponding unit IDs and scores
            results = [
                (self.target_ids[i], similarities[i]) for i in top_k_indices
            ]
            return results

# --- Early Stopping Class --- (Keep as is, but ensure it's used)
# class EarlyStopping:
#     def __init__(self, min_epochs=10, patience=10, min_delta=0.0001, verbose=True, checkpoint_path=None, checkpoint_file='best_retrieval_model.pt'):
#         """
#         Args:
#             patience (int): How many epochs to wait before stopping if loss doesn't improve.
#             min_delta (float): The minimum change to qualify as an improvement.
#             verbose (bool): Whether to print messages.
#             path (str): Path to save the best model.
#         """
#         self.min_epochs = min_epochs
#         self.patience = patience
#         self.min_delta = min_delta
#         self.verbose = verbose
#         self.path = os.path.join(checkpoint_path, checkpoint_file) if checkpoint_path and checkpoint_file else None
#         self.best_loss = None
#         self.best_epoch = 0
#         self.epochs_without_improvement = 0

#     def should_stop(self, current_loss, current_epoch, model):
#         """
#         Checks whether the training should stop based on the current loss and epoch.
#         Saves the model if the loss improves.

#         Returns:
#             bool: Whether to stop training early.
#         """
#         if self.best_loss is None:
#             self.best_loss = current_loss
#             self.best_epoch = current_epoch
#             self.save_checkpoint(current_loss, model)
#             return False

#         # Check if loss has improved
#         if current_loss < self.best_loss - self.min_delta:
#             if self.verbose:
#                 print(f"Loss improved ({self.best_loss:.6f} --> {current_loss:.6f}). Saving model...")
#             self.best_loss = current_loss
#             self.best_epoch = current_epoch
#             self.epochs_without_improvement = 0
#             if self.path:
#                 self.save_checkpoint(current_loss, model)
#             return False
#         else:
#             if current_epoch < self.min_epochs:
#                 return False
#             self.epochs_without_improvement += 1
#             if self.verbose:
#                 print(f"No improvement for {self.epochs_without_improvement} epochs. current loss: {current_loss:.6f}, best loss: {self.best_loss:.6f}")
#             if self.epochs_without_improvement >= self.patience:
#                 if self.verbose:
#                     print(f"Early stopping triggered at epoch {current_epoch + 1} (best loss {self.best_loss:.6f} at epoch {self.best_epoch + 1})")
#                 return True
#             return False

#     def save_checkpoint(self, loss, model):
#         """Saves model when validation loss decreases."""
#         torch.save(model.state_dict(), self.path)
#         if self.verbose:
#             print(f"Model saved to {self.path}")


## Sample Usage
if __name__ == "__main__":

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Step 1: Initialize embedder
    # This automatically builds the latent table (full vectors)
    embedder = UnitEmbedder(UNIT_DEFINITIONS)

    # The full latent table is needed for training queries and targets
    full_latent_table = embedder.latent_table
    unit_ids = embedder.unit_ids

    # Step 2: Initialize the retrieval trainer/encoder
    # The input_dim must match the shape of vectors in full_latent_table
    input_dimension = embedder.full_latent_dim
    output_dimension = 64 # Desired reduced dimension
    encoder = RetrievalTrainer(
        input_dim=input_dimension,
        output_dim=output_dimension,
        latent_table=full_latent_table, # Pass the full table
        target_ids=unit_ids,
        device=device
    )
    optimizer = optim.Adam(encoder.parameters(), lr=1e-4) # Slightly adjusted LR
    early_stopper = EarlyStopping(model_id='unit_embedder', version='1.0.0', patience=0, verbose=True, checkpoint_file='best_retrieval_model.pt') # Use early stopping

    # Step 3: Training loop
    num_epochs = 200 # Increased epochs, rely on early stopping
    batch_size = 256  # Adjusted batch size

    print("\n--- Starting Training ---")
    indices = list(range(len(unit_ids)))

    for epoch in range(num_epochs):
        random.shuffle(indices) # Shuffle data each epoch
        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            # Use indices to get corresponding full latents and IDs
            batch_q_full = full_latent_table[batch_indices]
            batch_ids = [unit_ids[idx] for idx in batch_indices]

            # Ensure batch has enough samples for TripletLoss (needs > 1)
            if len(batch_ids) > 1:
                loss = encoder.train_step(batch_q_full, batch_ids, optimizer)
                epoch_loss += loss
                num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_epoch_loss:.6f}")

        # Early stopping check
        if early_stopper.should_stop(avg_epoch_loss, epoch, encoder):
            break

    print("--- Training Finished ---")

    # Load the best model saved by EarlyStopping
    print(f"Loading best model from {early_stopper.path}")
    encoder.load_state_dict(torch.load(early_stopper.path, map_location=device))
    encoder.to(device) # Ensure model is on correct device after loading

    # Step 4: Encode the database with the *trained* encoder
    print("\n--- Encoding Database with Trained Model ---")
    encoder.encode_database()

    # Step 5: Test retrieval
    print("\n--- Testing Retrieval ---")
    encoder.eval() # Set to evaluation mode

    # Select a few samples to test retrieval
    num_test_samples = 5
    test_indices = random.sample(range(len(unit_ids)), num_test_samples)

    for i in test_indices:
        query_id = unit_ids[i]
        # Get the original *full* latent vector for the query
        query_full_vector = full_latent_table[i]

        print(f"\nQuerying for Unit ID: {query_id}")
        print(f"Unit Details: {UNIT_DEFINITIONS[query_id]}")

        # Use the retrieve method which handles encoding the query and finding neighbors
        top_k_results = encoder.retrieve(query_full_vector, return_top_k=5)

        print(f"  Top 5 Retrieved Results:")
        for rank, (retrieved_uid, score) in enumerate(top_k_results):
            print(f"    {rank + 1}: ID: {retrieved_uid} (Score: {score:.4f})")
            # Optional: Print details of retrieved unit for comparison
            print(f"       Details: {UNIT_DEFINITIONS[retrieved_uid]}")
        print("-" * 20)