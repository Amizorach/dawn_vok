from dawn_vok.vok.embedding.embedding_paramater.v_embedding_paramater_db import VOKEmbeddingParamaterDB
from dawn_vok.vok.embedding.encoders.base.v_structured_encoder import VOKStructuredEncoder


class LatentDBSTEncoder(VOKStructuredEncoder):
    def __init__(self, uid=None, dim_size=16, latent_db=None, token_dim=64):
        uid = uid or "latent_db_st_encoder"
        super().__init__(uid=uid)
        self.dim_size = dim_size
        self.token_dim = token_dim
        self.latent_db = latent_db or VOKEmbeddingParamaterDB()
        self.latent_db.load_latents()
        self.latents = {}
        self.emb_paramaters = {}
        self.gt = {}
        self.prepare_data()

    def prepare_data(self):
        raise NotImplementedError("prepare_data must be implemented by subclass")
