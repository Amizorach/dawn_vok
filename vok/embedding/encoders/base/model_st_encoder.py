import os
from pprint import pprint
import numpy as np
from dawn_vok.vok.embedding.encoders.base.v_structured_encoder import VOKStructuredEncoder
from dawn_vok.utils.dict_utils import DictUtils
from datetime import datetime, timedelta
import torch

   

class ModelSTEncoder(VOKStructuredEncoder):
    def __init__(self, uid, model, file_path=None):
        self.file_path = file_path
        self.model = model
        if model is not None:
            if file_path:
                path = os.path.abspath(file_path)
                self.model.encoder.load_state_dict(torch.load(path))
            else:
                self.model.load_model_state_dict()
        super().__init__(uid=uid)

    def get_ground_truth(self, sample):
        if isinstance(sample, dict):

            sample = np.array([sample])
        return self.get_ground_truth_batch(sample, device='cpu')


    def get_ground_truth_batch(self, samples: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
       
        # prepare padded encodings
        x = self.samples_to_encodings(samples)
        # run through model in eval mode
        self.model.eval()
        with torch.no_grad():
            latents = self.model.encoder(x)    
            # print(latents.shape)
                          # => [B, 16]
        return latents

    def samples_to_encodings(self, samples):
        raise NotImplementedError("This method should be implemented by the subclass")

    def decode_from_latent(self, latents):
        if isinstance(latents, np.ndarray):
            latents = torch.tensor(latents, dtype=torch.float32)
        ret_latents = self.model.decoder(latents)
        print('ret_latents', ret_latents)
        for i, lat in enumerate(ret_latents):
            print(f'latent {i}', lat)

        return ret_latents
    