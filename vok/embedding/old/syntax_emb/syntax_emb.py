from datetime import datetime
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from pymongo import UpdateOne
from sentence_transformers import SentenceTransformer
import torch

from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.dir_utils import DirUtils
from torch import nn


class EmbeddedDiscreteValue:
    @classmethod
    def get_collection_name(cls):
        return "embedded_discrete_values"
    @classmethod
    def get_db_name(cls):
        return "models"
    

    def __init__(self, emb_id=None, embedding=None, short_embedding=None,  generator_id=None):
        self.emb_id = emb_id
        self.embedding = embedding
        self.short_embedding = short_embedding
        self.embedding_type = "syntax"
        self.generator_id = generator_id
        self.updated_at = datetime.now()
        self.embedding_size = len(self.embedding) if self.embedding is not None else 0
        self.short_embedding_size = len(self.short_embedding) if self.short_embedding is not None else 0

    def populate_from_dict(self, di):
        self.emb_id = di.get('emb_id', self.emb_id)
        self.embedding = di.get('embedding', self.embedding)
        self.short_embedding = di.get('short_embedding', self.short_embedding)
        self.embedding_size = di.get('embedding_size', self.embedding_size)
        self.short_embedding_size = di.get('short_embedding_size', self.short_embedding_size)
        self.generator_id = di.get('generator_id', self.generator_id)
        self.updated_at = di.get('updated_at', self.updated_at)
        return self
    
    def to_dict(self):
        return {
            '_id': f'{self.generator_id}_{self.emb_id}',
            'generator_id': self.generator_id,

            'emb_id': self.emb_id,
            'embedding': self.embedding,
            'short_embedding': self.short_embedding,
            'embedding_size': len(self.embedding) if self.embedding is not None else 0, 
            'short_embedding_size': len(self.short_embedding) if self.short_embedding is not None else 0,
            'embedding_type': self.embedding_type,
        }
    def save_to_db(self):
        col = MongoUtils.get_collection(db_name=self.get_db_name(), collection_name=self.get_collection_name())
        col.update_one(
            {'_id': self.get_id(self.table_id, self.model_id, self.version_id)},
            {'$set': self.to_dict()},
            upsert=True
        )

    @classmethod
    def get_embeddings(cls, generator_id=None, emb_id=None, populate=False, as_dict=True,   embedding_type=None):
        col = MongoUtils.get_collection(db_name=cls.get_db_name(), collection_name=cls.get_collection_name())
        match = {}
        if emb_id:
            match = {'emb_id': emb_id}
        if generator_id:
            match['generator_id'] = generator_id
        if embedding_type:
            match['embedding_type'] = embedding_type
        embeddings = list(col.find(match))
        if not as_dict:
            if populate:    
                ret = []
                for emb in embeddings:
                    obj = cls()
                    obj.populate_from_dict(emb)
                    ret.append(obj)
                return ret
            else:
                return embeddings
        else:
            if populate:
                ret = {}
                for emb in embeddings:
                    obj = cls()
                    obj.populate_from_dict(emb)
                    ret[emb['emb_id']] = obj
                return ret
            else:
                return {emb['emb_id']: emb for emb in embeddings}
      
    @classmethod
    def add_embedding(cls, generator_id, emb_id, embedding, short_embedding):
        obj = cls(generator_id=generator_id, emb_id=emb_id, embedding=embedding, short_embedding=short_embedding)
        obj.save_to_db()
    
    @classmethod
    def add_embeddings(cls, generator_id, embeddings, short_embeddings):
        emb_list = []
        all_emb_ids = set(list(embeddings.keys()) + list(short_embeddings.keys()))
        for emb_id in all_emb_ids:
            obj = cls(generator_id=generator_id, emb_id=emb_id, embedding=embeddings.get(emb_id, None), short_embedding=short_embeddings.get(emb_id, None))
            emb_list.append(obj.to_dict())
        print(emb_list)
        col = MongoUtils.get_collection(db_name=cls.get_db_name(), collection_name=cls.get_collection_name())
        bulk_updates = []
    # Assuming emb_list is a list of dictionaries that include an "emb_id" key and other update fields.
        for emb in emb_list:
            # Construct an _id based on generator_id and a unique emb_id from each element.
            doc_id = f"{generator_id}_{emb['emb_id']}"
            # Create an update operation for each document.
            bulk_updates.append(
                UpdateOne({'_id': doc_id}, {'$set': emb}, upsert=True)
            )

        if bulk_updates:
            col.bulk_write(bulk_updates)
        exit()
        return emb_list
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
    def __init__(self, generator_id, short_emb_size=32):
        self.generator_id = generator_id
        self.short_emb_size = short_emb_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 5))

    def train(self, num_epochs=100):
        # Retrieve embeddings from the external source.
        # Assumes each embedding is a dict with key 'embedding' mapping to a list of numbers.
        self.embeddings = EmbeddedDiscreteValue.get_embeddings(generator_id=self.generator_id)
        self.input_dim = len(self.embeddings[list(self.embeddings.keys())[0]].get('embedding', []))
        if self.input_dim == 0:
            raise ValueError("No embeddings found")
            
        self.autoencoder = SyntaxReductionAutoEncoder(
            input_dim=self.input_dim, 
            hidden_dim=self.short_emb_size, 
            output_dim=self.input_dim
        ).to(self.device)
        
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
                self.save_model(f"autoencoder_{epoch}.pth")
                # Use the first training sample for plotting progress.
                sample_embedding = self.train_data[0]
                sample_output, _ = self.autoencoder(sample_embedding)
                self.plot_predictions(sample_embedding, sample_output, epoch, loss_history)

        plt.ioff()
        plt.show()

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
        for emb in self.embeddings.values():
            # Ensure each 'embedding' is a list; defaults to empty list if not present.
            embedding_tensor = torch.tensor(emb.get('embedding', []), dtype=torch.float32).to(self.device)
            train_list.append(embedding_tensor)
        self.train_data = torch.stack(train_list)

    def save_model(self, filename):
        """Saves the model state to a file."""
        torch.save(self.autoencoder.state_dict(), filename)
        print(f"Model saved to {filename}")

class SyntaxEmbeddingTable:
    @classmethod
    def get_collection_name(cls):
        # Collection storing the 'next_available_index' counter document
        return "embedding_managers" # Default, can be overridden

    @classmethod
    def get_db_name(cls):
        return "models" # Default, can be overridden
    @classmethod
    def get_id(cls, table_id, model_id, version_id):
        return f"{table_id}_{model_id}_{version_id}"
    
    @classmethod
    def get_table_dir(cls, table_id, model_id, version_id):
        return DirUtils.get_model_path(model_id, version_id)
    
    def __init__(
        self,
        table_id=None,
        model_id=None,
        version_id=None,
        vocab= None,
        model_name="paraphrase-MiniLM-L3-v2",
    ):
        """
        Initializes the UnitEmbedder class which handles encoding units into full latents.
        """
        self.vocab = vocab or vocab_dict
    
        self.model = SentenceTransformer(model_name)
        self.encode_map = {}
        self.model_id = model_id
        self.table_id = table_id
        self.version_id = version_id
        self.model_latent_size = self.model.get_sentence_embedding_dimension() # More robust way to get dim

        # self.file_path = os.path.join(cache_dir, f"{self.get_id(self.table_id, self.model_id, self.version_id)}.pkl")

    def build_table(self):
        if not self.vocab:
            raise ValueError("Vocab is empty")
        for field, values in self.vocab.items():
            for value in values:
                self.encode_map[f'{field}_{value}'] = self._encode_text(field, value)
        print(f"Table built with {len(self.encode_map)} entries")
    
   
    def _format_field(self, field, value):
        """Formats a text field with its value."""
        return f"{field}: {value}"
    
    def _encode_text(self, field, value):

        """Encodes a text field using the SentenceTransformer model."""
        text_to_encode = self._format_field(field, str(value)) # Ensure value is string
        # The model expects a list of sentences
        if f'{field}_{value}' not in self.encode_map:
            embedding = self.model.encode([text_to_encode])[0]
            self.encode_map[f'{field}_{value}'] = embedding
        else:
            embedding = self.encode_map[f'{field}_{value}']
        return embedding
    
    def save_key_map(self):
        generator_id = self.get_id(self.table_id, self.model_id, self.version_id)
        embeddings = {k: v.tolist() for k, v in self.encode_map.items()}
        EmbeddedDiscreteValue.add_embeddings(generator_id, embeddings, {})

    def to_dict(self):
        list_map = {k: v.tolist() for k, v in self.encode_map.items()}
        return {
            '_id': self.get_id(self.table_id, self.model_id, self.version_id),
            'table_id': self.table_id,
            'model_id': self.model_id,
            'version_id': self.version_id,
            'vocab': self.vocab,
            'encode_map': list_map
        }
    def populate_from_dict(self, di):
        self.model_id = di.get('model_id', self.model_id)
        self.table_id = di.get('table_id', self.table_id)
        self.version_id = di.get('version_id', self.version_id)
        self.vocab = di.get('vocab', {})    
        self.encode_map = {k: np.array(v) for k, v in di.get('encode_map', {}).items()}
        print(f"Encode map loaded with {len(self.encode_map)} entries")

    def save_to_db(self):
        self.save_key_map()
        col = MongoUtils.get_collection(db_name=self.get_db_name(), collection_name=self.get_collection_name())
        col.update_one(
            {'_id': self.get_id(self.table_id, self.model_id, self.version_id)},
            {'$set': self.to_dict()},
            upsert=True
        )
    
    # def save_to_file(self, fn = None):
    #     fn = fn or f"{self.get_id(self.table_id, self.model_id, self.version_id)}.pkl"
    #     path = os.path.join(self.get_table_dir(self.table_id, self.model_id, self.version_id), fn)
    #     with open(path, 'wb') as f:
    #         pickle.dump(self, f)

    @classmethod
    def load_from_db(cls, table_id, model_id, version_id):
        col = MongoUtils.get_collection(db_name=cls.get_db_name(), collection_name=cls.get_collection_name())
        di = col.find_one({'_id': cls.get_id(table_id, model_id, version_id)})
        if di:
            obj = cls()
            obj.populate_from_dict(di)
            return obj
        else:
            return None



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

  
if __name__ == "__main__":
    # tbl = SyntaxEmbeddingTable(table_id="unit_definitions1", model_id="syntax_emb", version_id="v1")
    # tbl.build_table()
    # tbl.save_to_db()
    trainer = SyntaxEmbeddingReductionTrainer(generator_id="unit_definitions1_syntax_emb_v1")
    trainer.train(num_epochs=1000)