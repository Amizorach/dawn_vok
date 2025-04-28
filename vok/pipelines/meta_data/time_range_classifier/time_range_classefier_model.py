import json
import os
import torch
import torch.nn as nn
import torch
import torch.nn as nn

from dawn_vok.utils.dir_utils import DirUtils
from dawn_vok.utils.id_utils import IDUtils
from dawn_vok.vok.pipelines.meta_data.time_classifier.time_classefier_model import TimeClassifierDecoder
from dawn_vok.vok.v_objects.vok_object import VOKObject
from dawn_vok.vok.vmodel.models.basic_models.encoder_decoder import EncoderDecoderModel

class TimeRangeClassifierEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Args:
            input_dim (int): input dimension
            hidden_dim (int): hidden dimension
            latent_dim (int): latent dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Project each token from input_dim to latent_dim.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, input_dim]
        
        Returns:
            torch.Tensor: Aggregated latent representation of shape [batch_size, latent_dim]
        """
        # Project tokens.
        return self.encoder(x)
    

class TimeRangeClassifierDecoder(nn.Module):
    def __init__(self, latent_dim):
        """
        Args:
            input_dim (int): Number of features per token.
            latent_dim (int): Size of the latent embedding.
            sequence_length (int): The number of tokens to reconstruct.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.heads = nn.ModuleDict({
            "start":  nn.Linear(latent_dim, 16), 
            "end":  nn.Linear(latent_dim, 16),        
            "freq":  nn.Linear(latent_dim, 16),      
        })

    def forward(self, x):
        return {k: head(x) for k, head in self.heads.items()}



class TimeRangeClassifierModel(EncoderDecoderModel):
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
        model_id = 'time_range_classifier'
        self.latent_dim = latent_dim
        name = 'Time Range Classifier'
        super().__init__(model_id=model_id, version=version, name=name, model_config=model_config, latent_db=latent_db)
       
    def load_encoder(self, encoder_config):
        return TimeRangeClassifierEncoder(**encoder_config)
    
    def load_decoder(self, decoder_config):
        return TimeRangeClassifierDecoder(**decoder_config)
    
    def get_base_config(self):
        input_dim = 64  
        hidden_dim =128
        latent_dim = self.latent_dim or 32
        minute_bins = 60
        # output_dim = 16
        return {
            'config_id': 'time_range_classifier_default',
            'encoder_config': {'input_dim': input_dim, 
                               'hidden_dim': hidden_dim,
                               'latent_dim': latent_dim,
                                },
            'decoder_config': {'latent_dim': latent_dim,
                            },
        }
    
  
    
            
if __name__ == '__main__':
    model = TimeRangeClassifierModel()
    print(model.model_config)
    print(model.encoder)
    print(model.decoder)
    model.save_to_db()
