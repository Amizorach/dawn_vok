
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.vok.embedding.syntax_emb.syntax_emb import EmbeddedDiscreteValue


class SyntaxReductionAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SyntaxReductionAutoEncoder, self).__init__()
        # Encoder: two-layer MLP with ReLU activation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        # Decoder: two-layer MLP with ReLU activation
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class SyntaxEmbeddingReductionTrainer:
    def __init__(self, short_emb_size=32, generator_id='syntax_builder'):
        self.short_emb_size = short_emb_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 5))
        self.generator_id = generator_id
    def train(self, num_epochs=100, emb_size=None, orig_scheme_id='full_embedding', out_scheme_id=None):
       
        self.short_emb_size = emb_size or self.short_emb_size
        if out_scheme_id is None:
            out_scheme_id = f'{orig_scheme_id}_reduced_{self.short_emb_size}'
        # Retrieve embeddings from the external source.
        # Assumes each embedding is a dict with key 'embedding' mapping to a list of numbers.
        self.embeddings = EmbeddedDiscreteValue.get_embeddings(generator_id=self.generator_id, embedding_type='syntax')
        print(self.embeddings.keys())
        latents = []
        for emb_id, emb in self.embeddings.items():
            lat = emb.get('embedding', {}).get(orig_scheme_id, None)
            print(emb.get('embedding', {}).keys(), orig_scheme_id)
            if lat is None:
                print(f"No latent found for {emb_id}")
                continue
            latents.append(lat)
        print(latents)
        print(len(latents))
        self.latents = latents
        self.input_dim = len(latents[0])
        if self.input_dim == 0:
            raise ValueError("No embeddings found")
        
        self.autoencoder = SyntaxReductionAutoEncoder(
            input_dim=self.input_dim, 
            hidden_dim=self.short_emb_size, 
            output_dim=self.input_dim
        ).to(self.device)
        self.load_model()
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.create_train_data()

        # Set up interactive plotting mode
        plt.ion()
        loss_history = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            # Iterate over the training data one sample at a time
            for embedding in self.train_data:
                self.optimizer.zero_grad()
                output, _ = self.autoencoder(embedding)
                loss = self.loss_fn(output, embedding)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(self.train_data)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch} loss: {avg_loss:.4f}")

            # Save model and update plot every 10 epochs.
            if epoch % 10 == 0:
                best_model = self.autoencoder.state_dict()
                self.save_model(f"autoencoder_{epoch}.pth")
                # Use the first training sample for plotting progress.
                sample_embedding = self.train_data[0]
                sample_output, _ = self.autoencoder(sample_embedding)
                self.plot_predictions(sample_embedding, sample_output, epoch, loss_history)
        if best_model:
            self.autoencoder.load_state_dict(best_model)
            self.save_model()
        plt.ioff()
        # plt.show()

    def plot_predictions(self, embedding, output, epoch, loss_history):
        """
        Creates a two-panel plot:
          - The first panel compares the original embedding with its reconstruction.
          - The second panel shows the training loss history.
        """

        # Plot the original and reconstructed embedding.
        emb_np = embedding.cpu().detach().numpy()
        out_np = output.cpu().detach().numpy()
        x_axis = range(len(emb_np))
        self.axs[0].clear()
        self.axs[0].plot(x_axis, emb_np, label='Original')
        self.axs[0].plot(x_axis, out_np, label='Reconstructed')
        self.axs[0].set_title(f'Embedding Reconstruction at Epoch {epoch}')
        self.axs[0].legend()

        # Plot the loss history.
        self.axs[1].clear()
        self.axs[1].plot(loss_history, label='Loss')
        self.axs[1].set_title('Training Loss History')
        self.axs[1].set_xlabel('Epoch')
        self.axs[1].set_ylabel('Loss')
        self.axs[1].legend()

        plt.tight_layout()
        plt.pause(0.1)
        plt.show(block=False)

    def create_train_data(self):
        """Converts the list of embeddings into a stacked tensor for training."""
        train_list = []
        for lat in self.latents:
            # Ensure each 'embedding' is a list; defaults to empty list if not present.
            embedding_tensor = torch.tensor(lat, dtype=torch.float32).to(self.device)
            train_list.append(embedding_tensor)
        self.train_data = torch.stack(train_list)

    def save_model(self, filename=None):
        filename = filename or f"autoencoder_{self.short_emb_size}.pth"
        """Saves the model state to a file."""
        torch.save(self.autoencoder.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename=None):
        filename = filename or f"autoencoder_{self.short_emb_size}.pth"
        try:
            md = torch.load(filename)
            print(f"Model loaded from {filename}")
            self.autoencoder.load_state_dict(md)
            print(f"Model loaded from {filename}")
        except Exception as e:
            print(f"Error loading model from {filename}: {e}")


    def update_embeddings(self, emb_size=32, orig_scheme_id='full_latent', out_scheme_id=None):
        if out_scheme_id is None:
            out_scheme_id = f'{orig_scheme_id}_reduced_{self.short_emb_size}'
        """Trains the model and updates the embeddings with the new reduced embeddings under the field 'short_embedding'."""
        self.train(emb_size=emb_size, orig_scheme_id=orig_scheme_id, out_scheme_id=out_scheme_id)
        for emb in self.embeddings.values():
            lat = emb.get('embedding', {}).get(orig_scheme_id, None)
            if lat is None:
                print(f"No latent found for {emb.get('emb_id')}")
                continue
            input_tensor = torch.tensor(lat, dtype=torch.float32).to(self.device)
            # Extract the encoded (reduced) representation.
            _, encoded = self.autoencoder(input_tensor)
            emb['embedding'][out_scheme_id] = encoded.cpu().detach().numpy().tolist()
        MongoUtils.update_many(
            EmbeddedDiscreteValue.get_db_name(),
            EmbeddedDiscreteValue.get_collection_name(),
            list(self.embeddings.values())
        )

if __name__ == "__main__":
    trainer = SyntaxEmbeddingReductionTrainer(generator_id='syntax_single_embedding')
    trainer.update_embeddings(
        emb_size=32,
        orig_scheme_id='full_embedding',
        out_scheme_id='reduced_32'
    )
