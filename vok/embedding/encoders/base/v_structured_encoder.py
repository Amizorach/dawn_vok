import numpy as np
from dawn_vok.vok.v_objects.vok_object import VOKObject
import torch

class VOKStructuredEncoder(VOKObject):
    @classmethod
    def get_db_name(cls):
        return 'embeddings'
    
    @classmethod
    def get_collection_name(cls):
        return 'structured_encoders'
    
    def __init__(self, uid=None, info_map=None, config=None):
        super().__init__(uid=uid, obj_type="structured_encoder", name="structured_encoder")
        self.config = config or self.get_config()
        self.info_map = info_map or self.config.get("info_map", {})
    
    def to_dict(self):
        ret = super().to_dict()
        ret["info_map"] = self.info_map
        return ret
    
    def populate_from_dict(self, d):
        super().populate_from_dict(d)
        self.info_map = d["info_map"]

    def encode(self, dt):
        raise NotImplementedError("Subclasses must implement this method")
    
    def decode(self, v):
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_ground_truth(self, sample):
        if isinstance(sample, dict):
            sample = np.array([sample])
        return self.get_ground_truth_batch(sample, device='cpu')

    def get_ground_truth_batch(self, samples, device=None):
        raise NotImplementedError("This method should be implemented by the subclass")

    def samples_to_encodings(self, samples):
        raise NotImplementedError("This method should be implemented by the subclass")

    def decode_from_latent(self, latents):
        raise NotImplementedError("This method should be implemented by the subclass")
    