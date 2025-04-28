from pprint import pprint
from dawn_vok.vok.embedding.embedding_paramater.v_embedding_paramater import VOKEmbeddingParamater
from dawn_vok.vok.embedding.embedding_paramater.v_embedding_paramater_db import VOKEmbeddingParamaterDB
from dawn_vok.vok.embedding.encoders.base.latent_db_st_encoder import LatentDBSTEncoder
from dawn_vok.vok.embedding.encoders.base.v_structured_encoder import VOKStructuredEncoder
from dawn_vok.utils.dict_utils import DictUtils
from datetime import datetime
import torch



class SourceSTEncoder(LatentDBSTEncoder):

    @classmethod
    def get_config(cls):
        return {
            "min_year": 1990,
            "max_year": 2029,
            "info_map": {
                0: "source_id",
                1: "latent",
            }
        }
    
    def __init__(self):
        uid = "source_st_encoder"  
        super().__init__(uid=uid, latent_db=VOKEmbeddingParamaterDB())

    def prepare_data(self):
        emb_paramaters = VOKEmbeddingParamater.get_by_param_type(param_type='source')
        for emb_paramater in emb_paramaters:
            self.emb_paramaters[emb_paramater.param_id] = emb_paramater
            self.gt[emb_paramater.param_id] = torch.tensor(emb_paramater.static_latent_id, dtype=torch.float32)
            self.latents[emb_paramater.param_id] = torch.tensor(emb_paramater.get_latents(self.token_dim), dtype=torch.float32)
    def encode(self, source_id):
        if source_id not in self.emb_paramaters:
            raise ValueError(f"Source ID {source_id} not found")
        
        return self.latents[source_id]

    def get_ground_truth(self, source_id):
        if source_id not in self.emb_paramaters:
            raise ValueError(f"Source ID {source_id} not found")
        return self.gt[source_id]
    
    def decode(self, v):
        return self.latent_db.search(v, top_k=1)
    
    def decode_from_latent(self, latents):
        return self.latent_db.search(latents, top_k=1)
    

if __name__ == "__main__":
    encoder = SourceSTEncoder()
    meta = encoder.emb_paramaters['ds_ims_avne_etan_2']
    system_uid = meta.system_uid
    enc = encoder.encode('ds_ims_avne_etan_2')
    print(enc)
    print(encoder.get_ground_truth('ds_ims_avne_etan_2'))
    v = encoder.get_ground_truth('ds_ims_avne_etan_2')
    print(encoder.decode(v))
    print(system_uid)
    pprint(meta.to_dict().keys())
    # print(encoder.latents)
    # encoder.encode(1)
    # encoder.decode(encoder.encode(1))
    # encoder.decode_batch_logits(encoder.encode(1), 1)
