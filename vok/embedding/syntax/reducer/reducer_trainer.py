import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

class SyntaxEmbeddingReducerTrainer:
    def __init__(self, generator_id, model, dataset, batch_size=8, lr=1e-3, epochs=100):
        self.generator_id = generator_id
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.mse_criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.cosine_similarity = nn.CosineSimilarity(dim=1)  # Assuming you want to compute cosine similarity along the feature dimension

    def compute_cosine_loss(self, reconstruction, target):
        # Compute the cosine similarity and return the negative of it as loss
        cosine_sim = self.cosine_similarity(reconstruction, target)
        loss = 1 - cosine_sim.mean()  # The loss will be high when cosine similarity is low, and vice versa
        return loss
    
    def compute_masked_loss(self, reconstruction, target, mask):
        """
        Computes the masked mean squared error loss.
        
        Args:
            reconstruction (torch.Tensor): Reconstructed output of shape [B, T, D].
            target (torch.Tensor): Ground truth tensor of shape [B, T, D].
            mask (torch.Tensor): Mask tensor of shape [B, T], where 1 indicates valid positions.
            
        Returns:
            torch.Tensor: The computed masked loss.
        """
        # Expand mask to match the dimensions of the reconstruction/target.
        # mask_expanded shape: [B, T, 1]
        mask_expanded = mask.unsqueeze(-1)
        
        # Compute the element-wise squared error.
        squared_error = (reconstruction - target) ** 2
        
        # Apply the mask: Only valid positions contribute to the loss.
        masked_squared_error = squared_error * mask_expanded
        
        # Sum the errors and then divide by the number of valid elements.
        # The total number of valid elements is the sum of mask values multiplied by the feature dimension.
        total_valid = mask_expanded.sum()
        
        # Avoid division by zero.
        if total_valid == 0:
            return torch.tensor(0.0, device=reconstruction.device)
        
        loss = masked_squared_error.sum() / total_valid
        return loss

    def train(self):
        self.model.train()
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 8))
        best_model = None
        plt.ion()
        loss_history = []

        for epoch in range(self.epochs):
            reconstructed =[]
            ground_truths = []
            epoch_loss = 0.0
            # Iterate over the training data one sample at a time
            for data, ground_truth, mask in self.dataloader:
                print(data.shape, ground_truth.shape, mask.shape)
                self.optimizer.zero_grad()

                reconstruction, encoded = self.model(data)
                reconstructed.append(reconstruction[0].detach().cpu().numpy())
                ground_truths.append(ground_truth[0].detach().cpu().numpy())
                loss = self.mse_criterion(reconstruction, ground_truth)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.detach().item()
            avg_loss = epoch_loss / len(self.dataloader)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch} loss: {avg_loss:.4f}")

            # Save model and update plot every 10 epochs.
            if epoch % 10 == 5:
                best_model = self.model.state_dict()
                self.save_model(f"autoencoder_{epoch}.pth")
                # Use the first training sample for plotting progress.
                idx = random.randint(0, len(reconstructed)-1)
                col_idx = random.randint(0, 5)
                self.plot_predictions(reconstructed[idx][col_idx], ground_truths[idx][col_idx], epoch, loss_history)
            if best_model:
                self.model.load_state_dict(best_model)
                self.save_model()
        self.fig.show()
        plt.show(block=True)
        plt.ioff()

    def inference(self, inference_dataset, batch_size=1):
        """
        Extracts latent representations using only the encoder of the model.

        Args:
            inference_dataset (Dataset): Your inference dataset.
            batch_size (int): Batch size for inference.
        
        Returns:
            torch.Tensor: Concatenated latent representations.
        """
        # Set the model to evaluation mode.
        self.model.eval()
        
        dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)
        latent_representations = []

        with torch.no_grad():
            for data in dataloader:
                # Assuming your model has an attribute 'encoder'
                encoded = self.model.encoder(data)
                latent_representations.append(encoded.cpu())
        
        # Concatenate latent representations along the batch dimension.
        latent_representations = torch.cat(latent_representations, dim=0)
        
        return latent_representations

    def plot_predictions(self, embedding, output, epoch, loss_history):
        """
        Creates a two-panel plot:
          - The first panel compares the original embedding with its reconstruction.
          - The second panel shows the training loss history.
        """

        # Plot the original and reconstructed embedding.
        emb_np = embedding
        out_np = output
        x_axis = range(len(emb_np))
        self.axs[0].clear()
        self.axs[0].plot(x_axis, emb_np, label='Original')
        self.axs[0].plot(x_axis, out_np, label='Reconstructed')
        self.axs[0].set_title(f'Embedding Reconstruction at Epoch {epoch}')
        self.axs[0].legend()

        # Plot the loss history.
        self.axs[1].clear()
        self.axs[1].plot(loss_history[-100:], label='Loss')
        self.axs[1].set_title('Training Loss History')
        self.axs[1].set_xlabel('Epoch')
        self.axs[1].set_ylabel('Loss')
        self.axs[1].legend()

        plt.tight_layout()
        plt.pause(0.1)

    def save_model(self, filename=None):
        filename = filename or f"autoencoder_{self.generator_id}.pth"
        """Saves the model state to a file."""
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename=None):
        filename = filename or f"autoencoder_{self.generator_id}.pth"
        try:
            md = torch.load(filename)
            print(f"Model loaded from {filename}")
            self.model.load_state_dict(md)
            print(f"Model loaded from {filename}")
        except Exception as e:
            print(f"Error loading model from {filename}: {e}")
