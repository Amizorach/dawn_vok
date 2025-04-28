import random
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from dawn_vok.vok.embedding.encoders.source_st_enc import TimeStampSTEncoder
# ----------------------------
# 1) Dataset that returns (x, timestamp) - CORRECTED VERSION
# ----------------------------
class TimeClassifierDataset(Dataset):
    """
    Generates pairs of (feature_vector, timestamp).
    The feature vector `x` is engineered to contain information
    about year, month, day, hour, and minute derived from the timestamp `ts`.
    """
    def __init__(self,
                 start_year=1990,
                 end_year=2029,
                 minute_bin_size=10, # Note: minute_bin_size is used by Trainer/Model for output head size
                 input_dim=32,       # Ensure this is >= 9 for the features below
                 length=1000):
        super().__init__() # Good practice to call super().__init__()
        self.start = datetime(start_year, 1, 1)
        self.end   = datetime(end_year, 12, 31, 23, 59)
        # self.minute_bin = minute_bin_size # Keep if needed elsewhere
        self.input_dim = input_dim
        self.length = length
        self.time_stamp_encoder = TimeStampSTEncoder()
        # Ensure input_dim is sufficient for the engineered features
        # Current features require at least 9 dimensions.
        required_dims = 16
        if self.input_dim < required_dims:
             print(f"Warning: input_dim ({self.input_dim}) is less than {required_dims}, "
                   f"which are needed for the default engineered features. "
                   f"Consider increasing input_dim.")
             # Or raise ValueError(f"input_dim must be at least {required_dims} for default features")

        # Store range for normalization, handle edge case where start_year == end_year
        self.year_range = float(end_year - start_year)
        if self.year_range <= 0:
             self.year_range = 1.0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1) Sample a random datetime
        total_seconds = int((self.end - self.start).total_seconds())
        rand_sec = random.randint(0, total_seconds) if total_seconds > 0 else 0
        ts = self.start + timedelta(seconds=rand_sec)
        ts = datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute)

        # 2) Encode via whatever structured encoder you’ve attached
        enc: torch.Tensor = self.time_stamp_encoder.encode(ts)
        feat_len = enc.shape[0]

        # 3) Build full input vector, placing enc at front
        x = torch.zeros(self.input_dim, dtype=torch.float32)
        if feat_len > self.input_dim:
            raise ValueError(f"Encoded feature-length {feat_len} exceeds input_dim {self.input_dim}")
        x[:feat_len] = enc

        return x, ts


class TimeClassifierEncoder(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, latent_dim=144):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class TimeClassifierDecoder(nn.Module):
    def __init__(self, latent_dim=64, num_years=30, minute_bins=60):
        super().__init__()
        self.heads = nn.ModuleDict({
            "year":   nn.Linear(latent_dim, num_years), # Predict index relative to base_year
            "month":  nn.Linear(latent_dim, 12),             # Predict 0-11
            "day":    nn.Linear(latent_dim, 31),             # Predict 0-30
            "hour":   nn.Linear(latent_dim, 24),             # Predict 0-23
            "minute": nn.Linear(latent_dim, minute_bins),# Predict bin index 0 to N-1
        })

    def forward(self, x):
        return {k: head(x) for k, head in self.heads.items()}

# ----------------------------
# 2) Model (Unchanged from original)
# ----------------------------
class TimeClassifier(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, latent_dim=64, base_year=1990, max_year=2029, minute_bin_size=10):
        super().__init__()
        self.num_years = max_year - base_year + 1
        self.minute_bins = 60 // minute_bin_size # Calculate number of bins

        # Input validation for minute_bin_size
        if 60 % minute_bin_size != 0:
             raise ValueError("60 must be divisible by minute_bin_size")

        self.encoder = TimeClassifierEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = TimeClassifierDecoder(latent_dim, self.num_years, self.minute_bins)

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)
        # Return logits for each head

# ----------------------------
# 3) Trainer (Unchanged from original, uses corrected Dataset now)
# ----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from datetime import datetime, timedelta
import random
# Assume TimeClassifierDataset and TimeClassifier classes are defined as before
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from datetime import datetime, timedelta
import random
# Assume TimeClassifierDataset and TimeClassifier classes are defined as before

class TimeClassifierTrainer:
    def __init__(self, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.time_stamp_encoder = TimeStampSTEncoder()
        # Extract params with defaults
        self.start_year = kwargs.get("start_year", 1990)
        self.end_year = kwargs.get("end_year", 2029)
        self.minute_bin_size = kwargs.get("block_minutes", 10)
        self.input_dim = kwargs.get("input_dim", 32)

        self.model = TimeClassifier(
            input_dim=self.input_dim,
            hidden_dim=kwargs.get("hidden_dim", 64),
            base_year=self.start_year,
            max_year=self.end_year,
            minute_bin_size=self.minute_bin_size
        ).to(self.device)

        self.dataset = TimeClassifierDataset(
            start_year=self.start_year,
            end_year=self.end_year,
            minute_bin_size=self.minute_bin_size,
            input_dim=self.input_dim,
            length=kwargs.get("dataset_size", 1000)
        )
        self.loader = DataLoader(self.dataset,
                                 batch_size=kwargs.get("batch_size", 512),
                                 shuffle=True,
                                 collate_fn=self.time_collate_fn,
                                 num_workers=kwargs.get("num_workers", 0))
        self.opt = optim.Adam(self.model.parameters(), lr=kwargs.get("lr", 1e-3))
        self.epochs = kwargs.get("epochs", 50)

        self.loss_weights = {"year":0.3, "month":0.08, "day":0.05, "hour":0.05, "minute":0.01}
        self.heads_to_report_error = ["year", "month", "day", "hour", "minute"]
        print(f"Loss weights: {self.loss_weights}")
        print(f"Minute bin size: {self.minute_bin_size}")


    def time_collate_fn(self, batch):
        xs, ts_list = zip(*batch)
        xs = torch.stack(xs, dim=0)
        return xs, list(ts_list)

    def compute_datetime_loss(self, logits, timestamps):
        # (This function remains the same as before - calculates CE loss)
        batch_size = len(timestamps)
        total_loss = 0.0
        targets = { k: torch.empty(batch_size, dtype=torch.long, device=self.device) for k in self.loss_weights }

        for i, ts in enumerate(timestamps):
            if "year" in targets:   targets["year"][i]   = ts.year - self.start_year
            if "month" in targets:  targets["month"][i]  = ts.month - 1
            if "day" in targets:    targets["day"][i]    = ts.day - 1
            if "hour" in targets:   targets["hour"][i]   = ts.hour
            if "minute" in targets: targets["minute"][i] = ts.minute // self.minute_bin_size

        loss_components = {}
        for k, target_tensor in targets.items():
             if k not in logits: continue
             if k not in self.loss_weights: continue

             n_cls = logits[k].size(1)
             if target_tensor.max() >= n_cls or target_tensor.min() < 0:
                 raise ValueError(f"Head '{k}' target index out of bounds [0, {n_cls-1}]")

             head_loss = F.cross_entropy(logits[k], target_tensor)
             loss_components[k] = head_loss.item()
             total_loss += self.loss_weights[k] * head_loss

        return total_loss, loss_components


    # --- MODIFIED TRAIN METHOD ---
    def train(self):
        """Runs the training loop, calculates errors, and prints a sample comparison."""
        print(f"\nStarting training for {self.epochs} epochs...")
        last_batch_logits = None # Store logits from the last batch for sample printing
        last_batch_ts = None     # Store timestamps from the last batch

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_loss_components = {k: 0.0 for k in self.loss_weights}
            epoch_total_errors = {k: 0.0 for k in self.heads_to_report_error}
            num_samples = 0

            for batch_idx, (x_batch, ts_batch) in enumerate(self.loader):
                x_batch = x_batch.to(self.device)
                batch_size = x_batch.size(0)

                # Forward pass
                logits = self.model(x_batch)

                # Store the last batch's logits and timestamps for sample printing later
                if batch_idx == len(self.loader) - 1:
                     last_batch_logits = {k: v.detach() for k, v in logits.items()} # Detach to avoid holding graph
                     last_batch_ts = ts_batch

                # Compute standard cross-entropy loss
                loss, loss_comps = self.compute_datetime_loss(logits, ts_batch)

                # --- Calculate Error in Units (inside no_grad) ---
                with torch.no_grad():
                    actual_vals = {} # (Calculation as before)
                    if "year" in self.heads_to_report_error: actual_vals["year"] = torch.tensor([ts.year for ts in ts_batch], dtype=torch.float32, device=self.device)
                    if "month" in self.heads_to_report_error: actual_vals["month"] = torch.tensor([ts.month for ts in ts_batch], dtype=torch.float32, device=self.device)
                    if "day" in self.heads_to_report_error: actual_vals["day"] = torch.tensor([ts.day for ts in ts_batch], dtype=torch.float32, device=self.device)
                    if "hour" in self.heads_to_report_error: actual_vals["hour"] = torch.tensor([ts.hour for ts in ts_batch], dtype=torch.float32, device=self.device)
                    if "minute" in self.heads_to_report_error: actual_vals["minute"] = torch.tensor([ts.minute for ts in ts_batch], dtype=torch.float32, device=self.device)

                    for k in self.heads_to_report_error:
                        if k not in logits: continue
                        pred_indices = torch.argmax(logits[k], dim=1)
                        pred_values = None
                        if k == "year": pred_values = pred_indices.float() + self.start_year
                        elif k == "month": pred_values = pred_indices.float() + 1
                        elif k == "day": pred_values = pred_indices.float() + 1
                        elif k == "hour": pred_values = pred_indices.float()
                        elif k == "minute": pred_values = pred_indices.float() * self.minute_bin_size
                        if pred_values is not None:
                            batch_errors = torch.abs(pred_values - actual_vals[k])
                            batch_total_error = torch.sum(batch_errors)
                            epoch_total_errors[k] += batch_total_error.item()
                # --- End Error Calculation ---

                # Backward pass and optimization
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # Accumulate CE loss components and samples
                epoch_loss += loss.item() * batch_size
                for comp_k in epoch_loss_components:
                     if comp_k in loss_comps:
                         epoch_loss_components[comp_k] += loss_comps[comp_k] * batch_size
                num_samples += batch_size

            # --- Calculate Averages and Print Epoch Summary ---
            if num_samples > 0:
                avg_epoch_loss = epoch_loss / num_samples
                avg_comps = {k: v / num_samples for k, v in epoch_loss_components.items()}
                avg_errors = {k: v / num_samples for k, v in epoch_total_errors.items()}
            else: # Handle empty dataset case
                avg_epoch_loss = 0.0
                avg_comps = {k: 0.0 for k in self.loss_weights}
                avg_errors = {k: 0.0 for k in self.heads_to_report_error}

            # Print Epoch Loss / Error Summary
            print(f"Epoch {epoch+1}/{self.epochs}  Avg CE Loss: {avg_epoch_loss:.6f}")
            error_str = "  Avg Abs Errors: "
            unit_map = {"year": "yrs", "month": "mon", "day": "dys", "hour":"hrs", "minute": "min"}
            error_parts = []
            for k in self.heads_to_report_error:
                if k in avg_errors:
                    error_parts.append(f"{k.capitalize()}={avg_errors[k]:.2f} {unit_map.get(k, '')}")
            error_str += " | ".join(error_parts)
            print(error_str)

            # --- Print Sample Comparison ---
            if last_batch_logits is not None and last_batch_ts is not None:
                try: # Wrap in try-except in case something goes wrong with indexing/conversion
                    with torch.no_grad():
                        # Use the first sample from the last batch
                        sample_idx = 0
                        original_dt = last_batch_ts[sample_idx]

                        # Get predicted indices for this sample
                        pred_indices = {}
                        for k in self.heads_to_report_error:
                            if k in last_batch_logits:
                                pred_indices[k] = torch.argmax(last_batch_logits[k][sample_idx,:]).item()
                        pred_dt = self.time_stamp_encoder.decode_batch_logits(last_batch_logits, self.minute_bin_size)
                        # Convert indices to values
                        # pred_year = pred_indices.get("year", -1) + self.start_year
                        # pred_month = pred_indices.get("month", -1) + 1
                        # pred_day = pred_indices.get("day", -1) + 1
                        # pred_hour = pred_indices.get("hour", -1)
                        # pred_minute = pred_indices.get("minute", -1) * self.minute_bin_size
                        # pred_dt = datetime(pred_year, pred_month, pred_day, pred_hour, pred_minute)
                        print("  --- Sample Comparison (Last Batch, Idx 0) ---")
                        print(f"  Original DT : {original_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"  Predicted   : {pred_dt[0].strftime('%Y-%m-%d %H:%M:%S')}")
                        print("  ---------------------------------------------")
                except Exception as e:
                    print(f"  Error printing sample comparison: {e}") # Print error if sample display fails

            # Clear last batch info for next epoch
            last_batch_logits = None
            last_batch_ts = None
            # --- End Print Sample Comparison ---


# ----------------------------
# Example Usage
# ----------------------------
if __name__=="__main__":
     # Make sure TimeClassifierDataset and TimeClassifier are defined before this
     trainer = TimeClassifierTrainer(
         start_year=1990,
         end_year=2030,
         block_minutes=1,
         input_dim=32,
         hidden_dim=128,
         batch_size=256,
         lr=5e-4,
         epochs=200,
         dataset_size=50000,
         num_workers=0
     )
     trainer.train()