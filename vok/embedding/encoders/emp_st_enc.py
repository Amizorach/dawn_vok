from pprint import pprint

import numpy as np
from dawn_vok.vok.embedding.embedding_paramater.v_embedding_paramater import VOKEmbeddingParamater
from dawn_vok.vok.embedding.embedding_paramater.v_embedding_paramater_db import VOKEmbeddingParamaterDB
from dawn_vok.vok.embedding.encoders.base.latent_db_st_encoder import LatentDBSTEncoder
from datetime import datetime
import torch


class EMPSTEncoder(LatentDBSTEncoder):
    @classmethod
    def get_config(cls):
        return {
            "info_map": {
                0: "param_id",
                1: "latent",
            }
        }
    def __init__(self, param_type, uid=None, token_dim=64, dim_size=16):
        uid = uid or f"{param_type}_st_encoder"
        latent_db = VOKEmbeddingParamaterDB()
        latent_db.load_latents()
        self.param_type = param_type

        super().__init__(uid=uid, latent_db=latent_db, token_dim=token_dim, dim_size=dim_size)


    def prepare_data(self):
        emb_paramaters = VOKEmbeddingParamater.get_by_param_type(param_type=self.param_type)
        for emb_paramater in emb_paramaters:
            self.emb_paramaters[emb_paramater.param_id] = emb_paramater
            self.gt[emb_paramater.param_id] = torch.tensor(emb_paramater.static_latent_id, dtype=torch.float32)
            self.latents[emb_paramater.param_id] = torch.tensor(emb_paramater.get_latents(self.token_dim), dtype=torch.float32)

    def encode(self, param_id):
        if param_id not in self.emb_paramaters:
            raise ValueError(f"{self.param_type} {param_id} not found")
        
        return self.latents[param_id]

    def get_ground_truth(self, param_id):
        if param_id not in self.emb_paramaters:
            raise ValueError(f"{self.param_type} {param_id} not found")
        return self.gt[param_id]
    
    def decode(self, v):
        return self.latent_db.search(v, top_k=1)
    
    def decode_from_latent(self, latents):
        if isinstance(latents, np.ndarray):
            latents = torch.tensor(latents, dtype=torch.float32)
        ret = self.latent_db.search(latents, top_k=1)
        src = self.latent_db.meta_data_map[ret[-1]]
        return src

class SourceSTEncoder(EMPSTEncoder):
    def __init__(self):
        super().__init__(param_type='source')

class SensorTypeSTEncoder(EMPSTEncoder):
    def __init__(self):
        super().__init__(param_type='sensor_type')

class FormulationSTEncoder(EMPSTEncoder):
    def __init__(self):
        super().__init__(param_type='formulation')

if __name__ == "__main__":
    source = 'ds_ims_avne_etan_2'
    st = 'air_temperature'
    formulation = 'agg_min'
    encoder = SourceSTEncoder()
    # meta = encoder.emb_paramaters[source]
    # system_uid = meta.system_uid
    enc = encoder.encode(source)
    print(enc)
    print(encoder.get_ground_truth(source))
    v = encoder.get_ground_truth(source)
    print(encoder.decode(v))
    # print(system_uid)
    # pprint(meta.to_dict().keys())

    encoder = SensorTypeSTEncoder()
    meta = encoder.emb_paramaters[st]
    system_uid = meta.system_uid
    enc = encoder.encode(st)
    # print(enc)
    # print(encoder.get_ground_truth(st))
    v = encoder.get_ground_truth(st)
    print(encoder.decode(v))
    # print(system_uid)
    # pprint(meta.to_dict().keys())

    encoder = FormulationSTEncoder()
    meta = encoder.emb_paramaters[formulation]
    system_uid = meta.system_uid
    enc = encoder.encode(formulation)
    print(enc)
    print(encoder.get_ground_truth(formulation))
    v = encoder.get_ground_truth(formulation)
    print(encoder.decode(v))
    # print(system_uid)
    # pprint(meta.to_dict().keys())
    # print(encoder.latents)
    # encoder.encode(1)
    # encoder.decode(encoder.encode(1))
    # encoder.decode_batch_logits(encoder.encode(1), 1)
