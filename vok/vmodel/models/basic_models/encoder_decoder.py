          
from dawn_vok.vok.vmodel.models.v_model import VOKModel


class EncoderDecoderModel(VOKModel):
    def __init__(self, model_id=None, version=None, name=None, model_config=None, latent_db=None, model_file_name=None, encoder_file_name=None, decoder_file_name=None):
        super().__init__(model_id=model_id, version=version, name=name, model_config=model_config, latent_db=latent_db, model_file_name=model_file_name)
        self.model_config = self.get_base_config()
        if model_config is not None:
            self.model_config.update(model_config)
        self.encoder = self.load_encoder(self.model_config.get('encoder_config', {}))
        self.decoder = self.load_decoder(self.model_config.get('decoder_config', {}))
        self.latent_db = latent_db  # Expecting an object with a find_closest method.
        self.encoder_file_name = encoder_file_name or 'encoder.pt'
        self.decoder_file_name = decoder_file_name or 'decoder.pt'

    def load_encoder(self, encoder_config):
        raise NotImplementedError("Subclasses must implement this method")
    
    def load_decoder(self, decoder_config):
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_model_structure(self):
        return {
            'encoder': self.get_module_structure(self.encoder),
            'decoder': self.get_module_structure(self.decoder)
        }
    
    
    def load_model_state_dict(self, file_name=None, device='cpu'):
        ret = True
        if self.encoder is not None:
            ret = ret and self.load_layer_state_dict(self.encoder, self.encoder_file_name, device)
        if self.decoder is not None:
            ret = ret and self.load_layer_state_dict(self.decoder, self.decoder_file_name, device)
        return ret
        
    def save_model_state_dict(self, file_name=None):
        ret = True
        if self.encoder is not None:
            ret = ret and self.save_layer_state_dict(self.encoder, self.encoder_file_name)
        if self.decoder is not None:
            ret = ret and self.save_layer_state_dict(self.decoder, self.decoder_file_name)
        return ret

    def eval(self):
        if self.encoder is not None:
            self.encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def train(self):
        if self.encoder is not None:
            self.encoder.train()
        if self.decoder is not None:
            self.decoder.train()

    def to(self, device):
        if self.encoder is not None:
            self.encoder.to(device)
        if self.decoder is not None:
            self.decoder.to(device)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, encoding):
        return self.decoder(encoding)
    
    def infer_latent(self, encoding):
        if self.latent_db is None:
            raise ValueError("latent_db is not set")
        return self.latent_db.find_closest(encoding, 1)
    
    def forward(self, x, run_encoder=True, run_decoder=True, infer_latent=False):
        """
        Args:
            x (torch.Tensor): Input tensor. If run_encoder is False, x is assumed to be
                              an already encoded representation. Shape depends on run_encoder.
                              If run_encoder=True: [batch_size, sequence_length, input_dim]
                              If run_encoder=False: [batch_size, latent_dim] (encoder output)
            run_encoder (bool): Whether to run the encoder on x.
            run_decoder (bool): Whether to run the decoder on the encoded representation.
            infer_latent (bool): If True and decoder is not run, use the latent_db to retrieve data.
        
        Returns:
            Depending on the flags:
              - If run_decoder is True: returns (decoded, encoded)
              - If infer_latent is True and decoder is not run: returns (retrieved, encoded) 
                Note: Changed order to (retrieved, encoded) for consistency (output, latent)
              - Otherwise: returns (None, encoded)
        """
        encoded = None
        if run_encoder:
            # if x.shape[-1] != self.model_config['encoder_config']['input_dim'] or \
            #    x.shape[1] != self.model_config['encoder_config']['sequence_length']:
            #      raise ValueError(f"Input shape mismatch for encoder. Expected [*, {self.model_config['encoder_config']['sequence_length']}, {self.model_config['encoder_config']['input_dim']}], got {x.shape}")
            encoded = self.encode(x)
        else:
            # If not running encoder, input 'x' is assumed to be the encoding
            # if x.shape[-1] != self.model_config['encoder_config']['latent_dim']:
            #      raise ValueError(f"Input shape mismatch for pre-encoded input. Expected [*, {self.model_config['encoder_config']['latent_dim']}], got {x.shape}")
            encoded = x
        
        decoded = None
        retrieved = None
        
        if run_decoder:
            decoded = self.decode(encoded)
            return decoded, encoded
        elif infer_latent:
            if self.latent_db is not None:
                # Use the vector database's find_closest function.
                retrieved = self.infer_latent(encoded) 
                return retrieved, encoded # Return retrieved data and the encoding used
            else:
                # Raise error or return None? Let's return None, None and warn.
                print("Warning: infer_latent=True but latent_db is not set. Returning (None, encoded).")
                return None, encoded
        else:
            # Neither decoding nor inferring, just return encoding
            return None, encoded
        
  