import torch
import torch.nn as nn
import torch
import torch.nn as nn

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
    def __init__(self, input_dim, latent_dim, sequence_length=10):
        """
        Args:
            input_dim (int): Number of features per token.
            latent_dim (int): Size of the latent embedding.
            sequence_length (int): The number of tokens to reconstruct.
        """
        super(SyntaxEmbeddingReducerTransformerDecoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        
        # Decoder network: expands the latent vector to reconstruct the full sequence.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, sequence_length * input_dim),
            nn.ReLU(),
            nn.Linear(sequence_length * input_dim, sequence_length * input_dim)
        )

    def forward(self, encoded):
        """
        Args:
            encoded (torch.Tensor): Aggregated latent representation of shape [batch_size, latent_dim]
        
        Returns:
            torch.Tensor: Reconstructed tensor of shape [batch_size, sequence_length, input_dim]
        """
        reconstruction_flat = self.decoder(encoded)  # [batch, sequence_length * input_dim]
        reconstruction = reconstruction_flat.view(-1, self.sequence_length, self.input_dim)  # reshape
        return reconstruction

   
class SyntaxEmbeddingReducerModel(nn.Module):
    def __init__(self, input_dim, latent_dim, latent_db=None, sequence_length=10, 
                 num_layers=2, num_heads=4, dim_feedforward=128, dropout=0.1):
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
        super(SyntaxEmbeddingReducerModel, self).__init__()
        self.encoder = SyntaxEmbeddingReducerTransformerEncoder(
            input_dim, latent_dim, sequence_length, num_layers, num_heads, dim_feedforward, dropout
        )
        self.decoder = SyntaxEmbeddingReducerTransformerDecoder(input_dim, latent_dim, sequence_length)
        self.latent_db = latent_db  # Expecting an object with a find_closest method.

    def forward(self, x, run_encoder=True, run_decoder=True, infer_latent=False):
        """
        Args:
            x (torch.Tensor): Input tensor. If run_encoder is False, x is assumed to be
                              an already encoded representation.
            run_encoder (bool): Whether to run the encoder on x.
            run_decoder (bool): Whether to run the decoder on the encoded representation.
            infer_latent (bool): If True and decoder is not run, use the latent_db to retrieve data.
        
        Returns:
            Depending on the flags:
              - If run_decoder is True: returns (decoded, encoded)
              - If infer_latent is True and decoder is not run: returns (encoded, retrieved)
              - Otherwise: returns (None, encoded)
        """
        if run_encoder:
            encoded = self.encoder(x)
        else:
            encoded = x
        
        decoded = None
        if run_decoder:
            decoded = self.decoder(encoded)
        elif infer_latent:
            if self.latent_db is not None:
                # Use the vector database's find_closest function to retrieve the nearest match.
                # This calls find_closest(encoded, 1) and returns the closest match for each encoded vector.
                retrieved = self.latent_db.find_closest(encoded, 1)
                return encoded, retrieved
        
        return decoded, encoded
        