import json
import os
import torch
import torch.nn as nn
import torch
import torch.nn as nn

from dawn_vok.utils.dir_utils import DirUtils
from dawn_vok.utils.id_utils import IDUtils
from dawn_vok.vok.v_objects.vok_object import VOKObject
from dawn_vok.vok.vmodel.models.basic_models.encoder_decoder import EncoderDecoderModel

class SyntaxEmbeddingReducerTransformerEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, sequence_length=10, num_layers=2, num_heads=4, dim_feedforward=128, dropout=0.1):
        """
        Args:
            input_dim (int): Number of features per token (e.g., 384).
            latent_dim (int): Size of the latent embedding (e.g., 128).
            sequence_length (int): Number of tokens per sample.
            num_layers (int): Number of layers in the Transformer encoder.
            num_heads (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feed-forward network in the Transformer.
            dropout (float): Dropout probability.
        """
        super(SyntaxEmbeddingReducerTransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

        # Project each token from input_dim to latent_dim.
        self.input_proj = nn.Linear(input_dim, latent_dim)
        
        # Create the transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, input_dim]
        
        Returns:
            torch.Tensor: Aggregated latent representation of shape [batch_size, latent_dim]
        """
        # Project tokens.
        x = self.input_proj(x)  # [batch, sequence_length, latent_dim]
        
        # Process with the transformer encoder.
        encoded_seq = self.transformer_encoder(x)  # [batch, sequence_length, latent_dim]
        
        # Aggregate the token representations (mean pooling).
        encoded = torch.mean(encoded_seq, dim=1)  # [batch, latent_dim]
        return encoded
    
class SyntaxEmbeddingReducerTransformerDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, sequence_length=10):
        """
        Args:
            input_dim (int): Number of features per token.
            latent_dim (int): Size of the latent embedding.
            sequence_length (int): The number of tokens to reconstruct.
        """
        super(SyntaxEmbeddingReducerTransformerDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        
        # Decoder network: expands the latent vector to reconstruct the full sequence.
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, sequence_length * output_dim),
            nn.ReLU(),
            nn.Linear(sequence_length * output_dim, sequence_length * output_dim)
        )

    def forward(self, encoded):
        """
        Args:
            encoded (torch.Tensor): Aggregated latent representation of shape [batch_size, latent_dim]
        
        Returns:
            torch.Tensor: Reconstructed tensor of shape [batch_size, sequence_length, input_dim]
        """
        reconstruction_flat = self.decoder(encoded)
        batch_size = encoded.shape[0]  # [batch, sequence_length * input_dim]
        reconstruction = reconstruction_flat.view(batch_size, self.sequence_length, self.output_dim)  # reshape
        return reconstruction



class SyntaxReducerModel(EncoderDecoderModel):
    def __init__(self, version='1.0.0', model_config=None, latent_db=None, latent_dim=None):
        """
        Args:
            input_dim (int): Number of features per token.
            latent_dim (int): Size of the latent embedding.
            latent_db (optional): A vector database or similar object expected to have a 
                                  find_closest(embedding, num_results) method.
            sequence_length (int): Number of tokens per sample.
            num_layers (int): Number of layers in the Transformer encoder.
            num_heads (int): Number of attention heads.
            dim_feedforward (int): Dimension for the transformer feedforward network.
            dropout (float): Dropout probability.
        """
        model_id = 'syntax_reducer'
        self.latent_dim = latent_dim
        name = 'Syntax Reducer'
        super().__init__(model_id=model_id, version=version, name=name, model_config=model_config, latent_db=latent_db)
       
    def load_encoder(self, encoder_config):
        return SyntaxEmbeddingReducerTransformerEncoder(**encoder_config)
    
    def load_decoder(self, decoder_config):
        return SyntaxEmbeddingReducerTransformerDecoder(**decoder_config)
    
    def get_base_config(self):
        input_dim = 384
        latent_dim = self.latent_dim or 64
        output_dim = 384
        sequence_length = 10
        return {
            'config_id': 'syntax_reducer_default',
            'encoder_config': {'input_dim': input_dim, 
                               'latent_dim': latent_dim,
                               'sequence_length': sequence_length,
                               'num_layers': 2, 'num_heads': 4, 'dim_feedforward': 128, 'dropout': 0.1 },
            'decoder_config': {'input_dim': latent_dim,
                               'output_dim': output_dim,
                               'sequence_length': sequence_length},
        }
  
   
            
if __name__ == '__main__':
    model = SyntaxReducerModel()
    print(model.model_config)
    print(model.encoder)
    print(model.decoder)
    model.save_to_db()
