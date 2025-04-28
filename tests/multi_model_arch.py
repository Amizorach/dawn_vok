import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dawn_vok.raw_data.raw_data_providers.raw_data_sampler import RawDataSampleCreator
import torch.optim as optim


class SimpleMPLModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SimpleMPLModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class TranslatorModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(TranslatorModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class MultiModelArch1:
    def __init__(self, input_dim: int, latent_dim: int, output_dim: int, initial_models: int = 5):
        self.models = []
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.translator = TranslatorModel(self.latent_dim, 512, self.output_dim)
        for _ in range(initial_models):
            self.add_model()

    def add_model(self):
        model = SimpleMPLModel(self.input_dim, 512, self.latent_dim)
        self.models.append(model)

    def prepare_arch(self):
        for model in self.models:
            model.to(self.device)
        self.translator.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        ret = []
        for model in self.models:
            ret.append(model(x))
        
        
        dec = self.translator(enc)
        return dec
    
class MultiModelArch:
    def __init__(self, input_dim: int, latent_dim: int, output_dim: int, initial_models: int = 5):
        self.models = nn.ModuleList() # Use ModuleList to register models correctly
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.translator = TranslatorModel(self.latent_dim, 512, self.output_dim) # Assuming TranslatorModel takes latent_dim as input
        for _ in range(initial_models):
            self.add_model()
        self.prepare_arch() # Move models to device

    def add_model(self):
        model = SimpleMPLModel(self.input_dim, 512, self.latent_dim)
        self.models.append(model) # Appends to ModuleList

    def prepare_arch(self):
        # Move all models and the translator to the specified device
        for model in self.models:
            model.to(self.device)
        self.translator.to(self.device)

    def get_models_and_translator(self):
        # Helper to easily access components
        return self.models, self.translator

    # The forward method is no longer central to the training logic
    # def forward(self, x):
    #    # This method is less relevant in the new structure
    #    pass
class MultiModelDataset(Dataset):
    def __init__(self, samples, embeddings, full_embeddings):
        self.samples = samples
        self.embeddings = embeddings
        self.full_embeddings = full_embeddings

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    sampler = RawDataSampleCreator(source_ids=None,
                                    sensor_types=None, start_date='2024-01-04', end_date='2024-10-05', frequency=10*60, agg='mean')
    samples, embeddings, full_embeddings = sampler.create_samples(sample_size=144, sample_resolution=6, max_samples=10000, max_samples_per_provider=500, 
                                                                  add_embeddings=False, shuffle=True, shuffle_providers=True)
    print(f'samples: {len(samples)}, embeddings: {len(embeddings)}, full_embeddings: {len(full_embeddings)}')
    print(len(samples[0]))
    # model = SimpleMPLModel(1, 10, 1)
    # multi_model_arch = MultiModelArch(input_dim=144, latent_dim=256, output_dim=144)
    # multi_model_arch.add_model()
    # multi_model_arch.prepare_arch()
    # create a random batch of 144x144
    # batch = torch.randn(256, 144)
    samples = torch.tensor(samples, dtype=torch.float32)
    dataloader = DataLoader(samples, batch_size=256, shuffle=True)
    # x = torch.randn(1, 144)
    # print(multi_model_arch.forward(x))

    # dataset = MultiModelDataset(samples, embeddings, full_embeddings)
    # print(len(dataset))
    # print(dataset.__getitem__(0))
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # for batch in dataloader:
    #     print(batch)
    #     break
# --- Initialization ---
    input_dim = 144 # Example
    latent_dim = 144
    output_dim = 144 # Example
    num_models = 15

    architecture = MultiModelArch(input_dim, latent_dim, output_dim, num_models)
    models, translator = architecture.get_models_and_translator()
    device = architecture.device

    # --- Optimizers ---
    # Option A: Separate optimizers (recommended for clarity)
    model_optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]
    translator_optimizer = optim.Adam(translator.parameters(), lr=0.001)


    # Option B: Single optimizer (requires careful management if models have different learning needs)
    # all_params = list(translator.parameters())
    # for model in models:
    #     all_params.extend(list(model.parameters()))
    # optimizer = optim.Adam(all_params, lr=0.001)
    # rand_vector = torch.rand(144).to(device)
    # for model in models:
    #     print(f"model: {model}")
    #     print(f"model.forward(rand_vector): {model.forward(rand_vector)}")
    # --- Loss Function ---
    criterion = nn.MSELoss() # Or your specific loss function

    # --- Training Loop ---
    num_epochs = 100
    # dataloader = ... (Your DataLoader)
    full_use_count = [0] * len(models)

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch}/{num_epochs-1} ---")
        # Optional: Track losses per model for averaging later if needed
        epoch_model_losses = [[] for _ in range(len(models))]
        mod_losses = [[] for _ in range(len(models))]
        epoch_total_loss = 0.0
        batches_processed = 0
        for batch_idx, data in enumerate(dataloader): # Assuming data and target from dataloader
            data = data.to(device)
            target = data
            batch_size = data.size(0) # Get batch size for averaging later if needed

            # Zero gradients for all optimizers at the start of the batch
            translator_optimizer.zero_grad()
            for opt in model_optimizers:
                opt.zero_grad()

            batch_pathway_losses = [] # To store losses for each pathway in this batch
            #create a random boolean array
            random_model_mask = np.random.randint(0, 2, size=len(models))
            random_model_mask = random_model_mask.astype(bool)
            model_losses = [] 
            # Iterate through each model pathway
            for i, model in enumerate(models):
                rand = np.random.rand()
                # if rand < 0.5:
                #     print(f"skipping model {i}, rand: {rand}")
                #     continue
                # --- Forward Pass for Pathway i ---
                model.train()
                translator.train()

                latent_representation = model(data)
                output = latent_representation#translator(latent_representation)

                # --- Calculate Loss for Pathway i ---
                loss = criterion(output, target)
                loss_item = loss.item() # Get Python number for printing/logging
                batch_pathway_losses.append(loss_item )
                epoch_model_losses[i].append(loss_item ) # Store for epoch average
                epoch_total_loss += loss_item * batch_size # Accumulate total loss scaled by batch size
                model_losses.append(loss)
                # --- Print Individual Model Loss ---
                # print(f"  Epoch {epoch}, Batch {batch_idx}, Model {i}, Loss: {loss_item:.4f}")
                # --- Backward Pass for Pathway i ---
              

            # print(f"model_losses: {model_losses}")
            #get the indexes for the top 2 models
            top5= []
            for i, loss in enumerate(model_losses):
                if len(top5) < 5:
                    top5.append(i)
                elif loss < model_losses[top5[0]]:
                    top5[0] = i
                elif loss < model_losses[top5[1]]:
                    top5[1] = i
                elif loss < model_losses[top5[2]]:
                    top5[2] = i
                elif loss < model_losses[top5[3]]:
                    top5[3] = i
                elif loss < model_losses[top5[4]]:
                    top5[4] = i
            # print(f"top_2_indexes: {top2}")
            # top5.append(np.random.randint(0, len(models)))

            for i, loss in enumerate(model_losses):
                if i in top5 or epoch < 10:
                    full_use_count[i] += 1
                    loss.backward()
                    mod_losses[i].append(loss.item())
                # --- Optimizer Step for Model i ---
                    model_optimizers[i].step()

                    # --- Zero Gradients ONLY for Model i ---
                    model_optimizers[i].zero_grad()
            # --- Optimizer Step for Shared Translator ---
            translator_optimizer.step()

            # --- Zero Gradients for Translator ---
            translator_optimizer.zero_grad()

            batches_processed += 1
            total_samples_processed = batches_processed * batch_size # Approx. if last batch is smaller

            # --- Periodic Summary Print ---
            if batch_idx % 100 == 0: # Print progress every 100 batches
                avg_batch_loss = sum(batch_pathway_losses) / len(models)
                print(f"----> Epoch {epoch}, Batch {batch_idx}: Avg Loss across pathways in this batch: {avg_batch_loss:.4f}")

        # --- End of Epoch Summary ---
        print(f"\n--- Epoch {epoch} Summary ---")
        # Calculate average loss per model over the epoch
        for i in range(len(models)):
            avg_model_epoch_loss = sum(mod_losses[i]) / len(mod_losses[i]) if mod_losses[i] else 0
            print(f"  Model {i} Avg Epoch Loss: {avg_model_epoch_loss:.4f}")
        # Calculate overall average loss for the epoch
        # total_samples_in_epoch = len(dataloader.dataset) # More accurate way
        # avg_epoch_loss = epoch_total_loss / (total_samples_in_epoch * len(models)) # Avg loss per sample per pathway
        # Simplified average if dataset size isn't readily available:
        avg_epoch_loss_approx = sum(sum(m_loss) for m_loss in epoch_model_losses) / sum(len(m_loss) for m_loss in epoch_model_losses) if batches_processed > 0 else 0
        print(f"  Overall Avg Epoch Loss across all pathways: {avg_epoch_loss_approx:.4f}")
        print(f"--------------------------")
        print(f"full_use_count: {full_use_count}")
        
# --- Evaluation ---
# Similar loop structure might be needed if you evaluate per pathway,
# or you might define a separate evaluation function that averages outputs, etc.
# Remember to set models to eval mode: model.eval(), translator.eval()