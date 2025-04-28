from pprint import pprint
from dawn_vok.vok.embedding.db.vector_db.vok_searchable_db import VOKSearchableDB
from dawn_vok.vok.embedding.embedding_paramater.v_embedding_paramater import VOKEmbeddingParamater
import torch
class VOKEmbeddingParamaterDB:
    def __init__(self, latent_dim=16, meta_dim=1, device=None):
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.db = VOKSearchableDB(latent_dim=latent_dim, meta_dim=meta_dim, device=self.device)
        self.latent_dim = latent_dim
        self.meta_dim = meta_dim
        self.device = device
        self.latents=[]
        self.meta_data=[]
        self.full_latents=[]
        self.meta_data_map ={}

    def formulate_emb(self, emb_paramater, column_index):
        ret = {
            'column_index': column_index,
            'param_id': emb_paramater.param_id,
            'param_type': emb_paramater.param_type,
            'uid': emb_paramater.uid,
            'system_uid': emb_paramater.system_uid,
        }
        return ret
    
    def load_db(self):
        emb_paramaters = VOKEmbeddingParamater.get_all()
        self.meta_data = []
        self.latents = []
        for column_index, emb_paramater in enumerate(emb_paramaters):
            self.meta_data_map[emb_paramater.system_uid] = self.formulate_emb(emb_paramater, column_index)
            self.meta_data.append(emb_paramater.system_uid)
            self.latents.append(torch.tensor(emb_paramater.static_latent_id, dtype=torch.float32))
          
        
        for i in range(len(self.meta_data)):
            meta_data = torch.tensor([i/len(self.meta_data)], dtype=torch.float32)
            lat = self.latents[i]
            if lat.ndim == 2:
                lat = lat.squeeze(0)
            self.full_latents.append(torch.cat([meta_data, lat], dim=0))
        self.db.add_latents(torch.stack(self.full_latents, dim=0))
        self.db.finalize(sort_by_meta_pos=None)
        print(f'loaded {len(self.meta_data_map)}')

    def load_latents(self):
        emb_paramaters = VOKEmbeddingParamater.get_all()
        self.meta_data = []
        self.latents = []

        for column_index, emb_paramater in enumerate(emb_paramaters):
            self.meta_data_map[emb_paramater.system_uid] = self.formulate_emb(emb_paramater, column_index)
            self.meta_data.append(emb_paramater.system_uid)
            if emb_paramater.param_type == 'source':
                self.latents.append(torch.tensor(emb_paramater.static_latent_id, dtype=torch.float32))
            elif emb_paramater.param_type == 'sensor_type':
                self.latents.append(torch.tensor(
                    emb_paramater.get_latent(latent_id='sensor_type', latent_key=None, dim_size=16), dtype=torch.float32))
            elif emb_paramater.param_type == 'formulation':
                self.latents.append(torch.tensor(emb_paramater.get_latents(16), dtype=torch.float32))
        
        for i in range(len(self.meta_data)):
            meta_data = torch.tensor([i/len(self.meta_data)], dtype=torch.float32)
            lat = self.latents[i]
            if lat.ndim == 2:
                lat = lat.squeeze(0)
            self.full_latents.append(torch.cat([meta_data, lat], dim=0))
        self.db.add_latents(torch.stack(self.full_latents, dim=0))
        self.db.finalize(sort_by_meta_pos=None)
    
    def search(self, query, top_k=1):
        ret = self.db.search(query, top_k=top_k, return_indices=True)
        ind = int(ret[-1][0])
        return ret, self.meta_data[ind]
    
if __name__ == '__main__':
    db = VOKEmbeddingParamaterDB()
    db.load_db()
    for i in range(len(db.meta_data)):
        lat_10 = db.full_latents[i][1:]
        lat_10 = lat_10 + torch.randn(lat_10.shape) * 0.1
        ret, md = db.search(lat_10, top_k=1)
        if md != db.meta_data_map[db.meta_data[i]]['system_uid']:  
            print(f'index {i} is wrong')
            exit()
    print(f'checked {len(db.meta_data)}')
      
    pprint(db.meta_data_map)