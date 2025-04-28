# --- Start of Code ---
from datetime import datetime, timedelta
import pprint

# External libraries
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F # Ensure F is imported for the loss function

# Your custom dawn_vok imports (ensure paths are correct)
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.dict_utils import DictUtils
# Assuming POCEncoderCosineModel is correctly defined in test_dr
from dawn_vok.vok.data_request.test_dr import POCEncoderCosineModel
# Remove unused imports if desired (SyntaxEmbedding, SensorTypeEmbeddingParamater)
# from dawn_vok.vok.embedding.base.discrete_embedding import SyntaxEmbedding
from dawn_vok.vok.embedding.cyclic_emb.frequency_embedding import FrequencyEmbedding
from dawn_vok.vok.embedding.cyclic_emb.timestamp_encoder import TimestampDecoder, TimestampEncoder
from dawn_vok.vok.embedding.embedding_paramater.v_embedding_paramater import VOKEmbeddingParamater
# from dawn_vok.vok.embedding.emb_param.embedding_paramater import SensorTypeEmbeddingParamater


    


class DataRequestEmbedding:
    def __init__(self, source_id=None, sensor_type=None, start_time=None, end_time=None, agg='mean', freq=10*60, data_dim = 144):
        self.source_id = source_id
        self.sensor_type = sensor_type
        self.start_time = start_time
        self.end_time = end_time
        self.agg = agg
        self.freq = freq
        self.data_dim = data_dim
        self.embedding_di={}

    def encode(self, sensor_emb=None, freq_emb=None, end_emb=None):
        if not self.sensor_type or not self.start_time:
            raise ValueError('sensor_type and start_time are required')
        # Ensure start_time is datetime
        self.start_time = DictUtils.parse_datetime_direct(self.start_time)
        self.end_time = self.start_time + timedelta(seconds=int(self.freq*self.data_dim)) # <-- Cast to int() here
        # Encode components
        self.embedding_di['start_time'] = TimestampEncoder().encode(self.start_time)
        self.embedding_di['end_time'] = TimestampEncoder().encode(self.end_time)
        self.embedding_di['freq'] = FrequencyEmbedding().encode(self.freq)
        # Get sensor embedding
        # if not sensor_emb:
            # Make sure MongoDB is accessible and configured
        st_di = MongoUtils.get_collection('embeddings', 'embedding_paramaters').find_one({'uid': f'sensor_type_{self.sensor_type}'})
        if not st_di:
            # print(f'sensor_type embedding not found for {self.sensor_type}')
            self.embedding_di = {}
            return None
            # raise ValueError(f'sensor_type embedding not found for {self.sensor_type}')
        # Assuming the embedding is stored under ['latents']['mean']
        self.embedding_di['sensor_type'] = st_di['latents']['lat_64']
        sensor_type_gt = st_di['latents']['lat_16']
        # else:
        #     self.embedding_di['sensor_type'] = sensor_emb
        # --- Create the ground truth (gt) vector ---
        gt = np.zeros(32, dtype=np.float32) # Use np.zeros for float type
        # Ensure components have expected lengths
        gt[0:4] = self.embedding_di['start_time'][0:4] # Assign slice directly
        gt[4] = np.array(float(self.embedding_di['freq'][1]), dtype=np.float32)
        gt[5:5+16] = sensor_type_gt
        # print(f"Final gt: {gt[0:5]}")

        # Assign float directly to element
       
            # print(f"Frequency embedding: {self.embedding_di['freq'][1]}")
            # print(f"Final gt: {gt[0:5]}")
            # <<< Optional: Add print right here to verify >>>
     
        # Assign slice directly
        self.embedding_di['gt'] = gt
        return self.embedding_di
    def decode(self):
        # Placeholder for potential decoding logic
        pass

    def get_embedding_dim(self):
        # This seems to return data_dim, not necessarily the embedding dim?
        return self.data_dim

    def get_embedding(self):
        # Ensure encode() is called to populate embedding_di
        if not self.embedding_di:
            self.encode()
        if not self.embedding_di:
            return None
        # Concatenate parts to form the input embedding 'x'
        # Ensure all parts are NumPy arrays for concatenation
        start_emb = np.array(self.embedding_di['start_time'])
        end_emb = np.array(self.embedding_di['end_time'])
        freq_emb = np.array(self.embedding_di['freq'])
        sensor_emb = np.array(self.embedding_di['sensor_type'])
        
        # Check shapes before concatenation if needed
        # print(start_emb.shape, end_emb.shape, freq_emb.shape, sensor_emb.shape)
        comb = np.concatenate([start_emb, end_emb, freq_emb, sensor_emb])
        return comb

    def get_gt(self):
        return self.embedding_di['gt']

# ================================================================
# Define CombinedLoss class (using the user's preferred name)
# ================================================================
class CombinedLoss(nn.Module):
    """
    Calculates a combined loss based on different metrics for different segments
    of the input vectors. The final loss is the mean over the batch of the
    equally weighted sum of the three segment losses.

    Loss = mean( (1/3 * MSE_part1) + (1/3 * MSE_part2) + (1/3 * CosineDist_part3) )

    Segments:
    - Part 1 (Index 0): Mean Squared Error
    - Part 2 (Indices 1-4): Mean Squared Error
    - Part 3 (Indices 5-end): Cosine Distance (1 - Cosine Similarity)
    """
    def __init__(self, eps=1e-8):
        super(CombinedLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if pred.shape != gt.shape:
            raise ValueError(f"Prediction shape {pred.shape} must match GT shape {gt.shape}")
        # Expecting 32 dim output from model and for GT
        if pred.ndim < 2 or pred.shape[1] != 32:
             raise ValueError(f"Inputs need >= 2 dims and vector length 32. Got {pred.shape}")

        # Part 1: MSE index 0
        loss_part1_batch = (pred[:, 0] - gt[:, 0])**2

        # Part 2: MSE indices 1-4
        squared_diff_part2 = (pred[:, 1:5] - gt[:, 1:5])**2
        loss_part2_batch = torch.mean(squared_diff_part2, dim=1)

        # Part 3: Cosine Distance indices 5-end (indices 5-31)
        slice_gt_part3 = gt[:, 5:]
        slice_pred_part3 = pred[:, 5:]
        # Use torch.nn.functional.cosine_similarity
        cosine_similarity_batch = F.cosine_similarity(
            slice_pred_part3, slice_gt_part3, dim=1, eps=self.eps
        )
        # Clamp similarity before distance calculation for robustness
        cosine_similarity_batch = torch.clamp(cosine_similarity_batch, -1.0 + self.eps, 1.0 - self.eps)
        loss_part3_batch = 1.0 - cosine_similarity_batch

        # Combine with equal weighting
        total_loss_batch = (1/3.0) * loss_part1_batch + \
                           (1/3.0) * loss_part2_batch + \
                           (1/3.0) * loss_part3_batch

        # Final mean loss over batch
        final_loss = torch.mean(total_loss_batch)
        return final_loss
# ================================================================


def train(model: nn.Module,
          dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          epochs: int = 10,
          plot_interval: int = 5):
    """
    Training loop with CombinedLoss:
      - uses CombinedLoss for training
      - interactive loss curve every `plot_interval` epochs
      - at each plot update, shows one sample's encoded vs ground-truth vector
    """
    model.to(device)
    # <<< CHANGE 1: Instantiate the CombinedLoss >>>
    criterion = CombinedLoss().to(device) # Move criterion to device

    # grab one fixed sample from the dataloader for visualization
    try:
        sample_x, sample_gt = next(iter(dataloader))
        sample_x  = sample_x[:20].to(device)
        sample_gt = sample_gt[:20].to(device)
        plot_samples = True
    except StopIteration:
        print("Warning: DataLoader is empty. Cannot grab sample for plotting.")
        plot_samples = False
        sample_x, sample_gt = None, None # Ensure they are defined

    # prepare interactive plots
    plt.ion()
    fig, (ax_loss, ax_embed) = plt.subplots(1, 2, figsize=(12, 4))
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train() # Set model to training mode at the start of epoch
        total_loss = 0.0
        batch_count = 0
        for x, gt in dataloader:
            x, gt = x.to(device), gt.to(device)
            optimizer.zero_grad()

            # <<< CHANGE 2: Assume model returns (_, prediction) >>>
            # Keep original model call signature if it's required by POCEncoderCosineModel.
            # Assume the second output is the prediction vector (shape N, 32).
            try:
                # Give the outputs meaningful names if possible.
                output1, pred = model(x, gt) # Assuming pred is the (N, 32) prediction
            except Exception as e:
                print(f"\nError during model forward pass: {e}")
                print(f"Input shape: {x.shape}, GT shape: {gt.shape}")
                # Optionally re-raise or handle differently
                raise e # Stop training if model call fails

            # <<< CHANGE 3: Calculate loss using CombinedLoss >>>
            # Use the prediction 'pred' (the model's second output) and ground truth 'gt'.
            loss = criterion(pred, gt)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            batch_count += x.size(0)

        # Avoid division by zero if dataset is empty
        if batch_count > 0:
           avg_loss = total_loss / batch_count
        else:
           avg_loss = 0.0

        loss_history.append(avg_loss)
        print(f"Epoch {epoch:03d} — avg loss: {avg_loss:.6f}") # Increased padding/precision

        # update plots every `plot_interval` epochs (or on last epoch)
        if plot_samples and (epoch % plot_interval == 0 or epoch == epochs):
            # 1) Update loss curve (No changes needed)
            ax_loss.clear()
            ax_loss.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Avg Loss")
            ax_loss.set_title("Training Loss")
            ax_loss.grid(True)

            # 2) Compute and plot the fixed sample's encoding vs GT
            model.eval() # Set model to evaluation mode for plotting
            with torch.no_grad():
                # <<< CHANGE 4: Keep model call, use 2nd output for plotting >>>
                # This call should also return (_, prediction)
                # The variable 'encoded_sample' now holds the prediction for the sample.
                try:
                    _, encoded_sample = model(sample_x, sample_gt)
                except Exception as e:
                     print(f"\nError during model forward pass for plotting: {e}")
                     continue # Skip plotting this epoch if model fails

            # Select a random index from the plotted samples
            num_plot_samples = encoded_sample.size(0)
            if num_plot_samples > 0:
                rand_index = np.random.randint(0, num_plot_samples)
                print(f"Plotting sample index: {rand_index} from plotted batch")
                # 'enc' is the prediction vector for the sample
                enc = encoded_sample[rand_index].cpu().numpy()   # shape (32,)
                gtv = sample_gt[rand_index].cpu().numpy()        # shape (32,)

                ax_embed.clear()
                # Plotting prediction ('encoded') vs GT
                # Plotting indices 4-31 (length 28)
                if len(enc) == 32: # Basic check for expected length
                    dims_to_plot = range(5) # Indices 0 to 27 for plotting
                    ax_embed.plot(dims_to_plot, enc[0:5], marker='o', linestyle='-', label='Predicted (Encoded)') # Update label
                    ax_embed.plot(dims_to_plot, gtv[0:5], marker='x', linestyle='--', label='Ground-Truth')
                    ax_embed.plot(range(5, 32), enc[5:32], marker='o', linestyle='-', label='Predicted (Encoded)') # Update label
                    ax_embed.plot(range(5, 32), gtv[5:32], marker='x', linestyle='--', label='Ground-Truth')
                    ax_embed.set_xlabel("Dimension Index (4-31)")
                    ax_embed.set_title(f"Sample {rand_index} Embedding vs GT")
                    ax_embed.legend()
                    ax_embed.grid(True)
                    print(f"Plotting sample {rand_index} from plotted batch")
                    print(f"Encoded sample: {TimestampDecoder().decode(enc[0])}")
                    print(f"Ground-Truth: {TimestampDecoder().decode(gtv[0])}")
                else:
                    print(f"Warning: Sample {rand_index} has unexpected length {len(enc)}. Skipping plot.")

            # redraw
            fig.canvas.draw()
            plt.pause(0.01) # Slightly longer pause for update

    plt.ioff()
    plt.show()

# --- Main execution block ---
if __name__ == '__main__':
    # --- Configuration ---
    MONGO_DB = 'embeddings' # Or get from config
    MONGO_COLLECTION = 'embedding_paramaters' # Or get from config
    NUM_SAMPLES_TO_GENERATE = 1000 # Number of data points
    BATCH_SIZE = 64 # Adjusted batch size
    LEARNING_RATE = 1e-4 # Potentially adjusted learning rate
    EPOCHS = 500 # Adjusted epochs
    PLOT_INTERVAL = 25 # Adjusted plot interval

    # --- Fetch Sensor Embeddings ---
    print(f"Connecting to MongoDB to fetch sensor embeddings from {MONGO_DB}/{MONGO_COLLECTION}...")
    try:
        sensor_embeddings_di = MongoUtils.get_collection(MONGO_DB, MONGO_COLLECTION).find({'param_type': 'sensor_type'})
        # Filter out entries without necessary keys right away
        sensor_embeddings_di = {
            di['param_id']: di
            for di in sensor_embeddings_di
            if 'param_id' in di and 'latents' in di and 'mean' in di['latents']
        }
        sensor_list = list(sensor_embeddings_di.keys())
        if not sensor_list:
             raise ValueError("No valid sensor type embeddings found in MongoDB collection.")
        print(f"Found {len(sensor_list)} sensor types.")
    except Exception as e:
        raise SystemExit(f"Error fetching sensor embeddings from MongoDB: {e}")


    # --- Data Generation Parameters ---
    sources = [0.1, 0.3, 0.5, 0.7, 0.9] # Example source IDs
    min_time = datetime(2021, 1, 1)
    max_time = datetime(2024, 1, 1) # Reduced time range for potentially denser data
    # Example frequencies in seconds

    frequency_list = [10*60.0, 30*60.0, 60*60.0, 2*60*60.0, 6*60*60.0, 12*60*60.0, 24*60*60.0]
    agg_list = ['mean', 'median', 'max', 'min', 'sum', 'count'] # Aggregation types (if used by model/loss?)
    data_dim = 144 # Related to end_time calculation

    # --- Generate Data ---
    embedding_list = [] # Input 'x' list
    gt_list = []        # Target 'gt' list
    print(f"Generating {NUM_SAMPLES_TO_GENERATE} data samples...")
    generated_count = 0
    skipped_count = 0
    time_range_seconds = int((max_time - min_time).total_seconds())

    while generated_count < NUM_SAMPLES_TO_GENERATE:
        # Randomly select parameters for each sample
        source = np.random.choice(sources)
        sensor = np.random.choice(sensor_list)
        agg = np.random.choice(agg_list)
        freq = np.random.choice(frequency_list)
        # Generate random start time within the range
        random_seconds = np.random.randint(0, time_range_seconds) 
        # print(f"Random seconds: {random_seconds}")
        start_time = min_time + timedelta(seconds=int(random_seconds)) # Cast numpy int to standard int
        # print(f"Generating data for sensor {sensor}, frequency {freq}, start time {start_time}, end time {max_time}, source {source}")

        try:
            # Create embedding object
            data_request_embedding = DataRequestEmbedding(
                source_id=source, sensor_type=sensor, start_time=start_time,
                agg=agg, freq=freq, data_dim=data_dim
            )
            # Get input 'x' (calls .encode() internally)
            input_embedding = data_request_embedding.get_embedding()
             # Get target 'gt' (created by .encode())
            target_gt = data_request_embedding.embedding_di['gt']
            # print(f"Input embedding shape: {input_embedding.shape}, Target GT shape: {target_gt.shape}")
            # print(f"Input embedding: {input_embedding[0:5]}")
            # print(f"Target GT: {target_gt[0:5]}")
            # Basic check for generated data validity (e.g., shape)
            if input_embedding is not None and target_gt is not None and target_gt.shape == (32,):
                 embedding_list.append(input_embedding)
                 gt_list.append(target_gt)
                 generated_count += 1
            else:
                 print(f"Warning: Invalid data generated for sensor {sensor}. Skipping.")
                 skipped_count += 1

        except ValueError as e:
            # Catch errors during encoding (e.g., sensor not found)
            print(f"Skipping sample due to encoding error: {e} (sensor: {sensor}, time: {start_time})")
            skipped_count += 1
            continue # Skip this sample if encode fails
        except Exception as e:
            print(f"Unexpected error during data generation: {e}. Skipping sample.")
            skipped_count +=1
            continue

        # Optional: Add a condition to prevent infinite loops if errors are persistent
        if skipped_count > NUM_SAMPLES_TO_GENERATE * 2: # Stop if too many errors
            print("Warning: Excessive errors during data generation. Stopping.")
            break

        # Progress indicator
        if (generated_count + skipped_count) % 100 == 0:
             print(f"  Processed {generated_count+skipped_count}, Generated {generated_count}, Skipped {skipped_count}...")

    if not embedding_list or not gt_list:
        raise SystemExit("No valid data generated. Check database connection, sensor types, and embedding logic.")
    embedding_list = np.array(embedding_list, dtype=np.float32)
    gt_list = np.array(gt_list, dtype=np.float32)
    print(f"\nGenerated data shapes: X={embedding_list.shape}, GT={gt_list.shape}")

    # --- Prepare PyTorch Dataset and DataLoader ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert numpy arrays to tensors
    X_tensor  = torch.from_numpy(embedding_list).float()
    GT_tensor = torch.from_numpy(gt_list).float()

    dataset = TensorDataset(X_tensor, GT_tensor)
    # Pin memory can speed up CPU-to-GPU transfers
    use_pin_memory = (device.type == 'cuda')
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=use_pin_memory)

    # --- Initialize Model and Optimizer ---
    # Ensure POCEncoderCosineModel is defined/imported correctly
    # It needs to accept x, gt and output (_, prediction) where prediction is (N, 32)
    model = POCEncoderCosineModel() # Add any necessary model arguments
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Start Training ---
    print("Starting training...")
    train(model, loader, optim, device, epochs=EPOCHS, plot_interval=PLOT_INTERVAL)
    print("Training finished.")