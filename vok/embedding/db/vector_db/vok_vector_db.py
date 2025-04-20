import time
import numpy as np
import torch

from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.utils.id_utils import IDUtils
from dawn_vok.vok.embedding.db.vector_db.vok_searchable_db import VOKSearchableDB
from dawn_vok.vok.v_objects.vok_object import VOKObject

class VOKVectorDB(VOKObject):
    @classmethod
    def get_db_name(cls):
        return "vok_vector_db"
    
    @classmethod
    def get_collection_name(cls):
        return "vok_vector_db"
    
    def __init__(self, uid, vdb_id, latent_dim=32, index_type="metadata", vdb_name=None, backend="matrix", storage_type="mongo", max_capacity=-1, seed=42, device="cpu"):
        super().__init__(uid=uid, system_uid=vdb_id, obj_type='vok_vector_db', meta_data={'vdb_id': vdb_id, 'vdb_name': vdb_name})
        self.vdb_id = vdb_id
        self.latent_dim = latent_dim
        self.index_type = index_type
        self.backend = backend
        self.storage_type = storage_type
        self.vdb_name = vdb_name
        self.max_capacity = max_capacity
        self.current_total_count = 0
        self.seed = seed
        self.device = None
        self.supports_on_demand_vector_generation = False
        self.supports_vector_search = False
        self.supports_id_search = True
        self.supports_meta_search = False
        self.search_resources_allocated = False
        self.supports_index_creation = False
        self.latent_db = None

    def to_dict(self):
        ret = super().to_dict()
        ret['latent_dim'] = self.latent_dim
        ret['index_type'] = self.index_type
        ret['backend'] = self.backend
        ret['max_capacity'] = self.max_capacity
        ret['current_total_count'] = self.current_total_count
        ret['seed'] = self.seed

    def populate_from_dict(self, d):
        super().populate_from_dict(d)
        self.latent_dim = DictUtils.get_int(d, 'latent_dim', self.latent_dim)   
        self.index_type = DictUtils.get_str(d, 'index_type', self.index_type)
        self.backend = DictUtils.get_str(d, 'backend', self.backend)
        self.max_capacity = DictUtils.get_int(d, 'max_capacity', self.max_capacity)
        self.current_total_count = DictUtils.get_int(d, 'current_total_count', self.current_total_count)
        self.seed = DictUtils.get_int(d, 'seed', self.seed)
        return self
     
    def save_to_db(self):
        col = MongoUtils.get_collection(self.get_db_name(), self.get_collection_name())
        col.update_one({'_id': self.vdb_id}, {'$set': self.to_dict()}, upsert=True)
    
    @classmethod
    def load_from_db(cls, vdb_id, populate=True):
        col = MongoUtils.get_collection(cls.get_db_name(), cls.get_collection_name())
        d = col.find_one({'_id': vdb_id})
        if populate:
            return cls(vdb_id, **d).populate_from_dict(d)
        return d
    
    def load_latent_db(self):
        pass

    def add_latents(self, latents):
        pass

    def get_latents_by_ids(self, latent_ids):
        pass
    
    def get_latent_by_id(self, latent_id):
        pass

    def get_latent_by_metadata(self, metadata):
        pass

    def get_closest_latent(self, query_latent, k=1):
        pass

    def prepare_for_search(self):
        pass

    def release_search_resources(self):
        pass
    
    def move_to_device(self, device):
        pass
    



class VDBBackend:
    def __init__(self, dre_id, dre_name=None):
        super().__init__(dre_id, dre_name)
        self.dre_id = dre_id
        self.dre_name = dre_name

class VDBStorage(VOKObject):
    def __init__(self, vdb_id, storage_type="mongo", vdb_name=None):
        uid = f'vdb_storage_{vdb_id}'
        system_uid = IDUtils.get_system_unique_id({'name': vdb_name or uid, 'uid': uid, 'type': 'vdb_storage'})
        super().__init__(uid=uid, system_uid=system_uid, obj_type='vdb_storage', meta_data={'vdb_id': vdb_id, 'vdb_name': vdb_name})
        self.vdb_id = vdb_id
        self.vdb_name = vdb_name
        self.storage_type = storage_type

    def load_latent_db(self):
        raise NotImplementedError("Subclasses must implement this method")

    def add_latents(self, latents):
        raise NotImplementedError("Subclasses must implement this method")

    def describe(self):
        ret = ""
        ret += f"VDBStorage(vdb_id={self.vdb_id}, vdb_name={self.vdb_name}, storage_type={self.storage_type})"
        return ret

class VDBMongoStorage(VDBStorage):
    def __init__(self, vdb_id, vdb_name=None, db_name=None, collection_name=None, 
                 match_query=None, index_field=None, latent_scheme=None):
        super().__init__(vdb_id, storage_type="mongo", vdb_name=vdb_name)
        self.vdb_id = vdb_id
        self.vdb_name = vdb_name
        self.mongodb_db_name = db_name
        self.mongodb_collection_name = collection_name
        self.mongodb_collection= MongoUtils.get_collection(self.mongodb_db_name, self.mongodb_collection_name)
        self.match_query = match_query
        self.latent_scheme = latent_scheme or 'lat_16'
        self.index_field = index_field or '_id'
        self.latent_db = {}
        self.latent_db_keys = []    
   
    def describe(self):
        ret = super().describe()
        ret += f"\n\tdb_name={self.mongodb_db_name}, collection_name={self.mongodb_collection_name}"
        ret += f"\n\tmatch_query={self.match_query}, "
        ret += f"\n\tlatent_count={len(self.latent_db)}, latent_size={self.latent_dim}"
        return ret
    
    def load_latent_db(self):
        latents = self.mongodb_collection.find(self.match_query)
        self.latent_db = {str(latent[self.index_field]): DictUtils.parse_value(latent, f'latents.{self.latent_scheme}', None) for latent in latents}
        self.comb_db = []
        for k, latent in self.latent_db.items():
            lat = [float(k)] + latent
            self.comb_db.append(lat)
            print(lat)
        self.latent_dim = len(self.comb_db[0][1:])
        self.comb_db.sort(key=lambda x: x[0])
        print(self.comb_db[0])
        self.comb_db = torch.tensor(self.comb_db, dtype=torch.float32)
        print(f"Loaded {len(self.latent_db)} latents from {self.mongodb_db_name}.{self.mongodb_collection_name}")
        return self.latent_db
    
    def get_latent_by_id(self, latent_id):
        print(self.latent_db.keys())
        return DictUtils.parse_value(self.latent_db[latent_id], 'latents.lat_16', None)
    
    def get_latent_by_metadata(self, metadata):
        system_uid = metadata.get('system_uid')
        if not system_uid:
            raise ValueError("System uid is required")
        for index_item in self.index:
            if index_item[0] == system_uid:
                return self.latent_db[index_item[1]]
        return None
    
    def get_latent_by_column_id(self, column_id):
        if column_id < 0 or column_id >= len(self.index):
            raise ValueError("Column id is out of range")
        return DictUtils.parse_value(self.latent_db[str(self.index[column_id][0])], 'latents.lat_16', None)
    
    def add_latents(self, latents):
        for latent in latents:
            if not self.index_field in latent:
                raise ValueError(f"Latent {latent} does not have an index field")
            self.latent_db[latent[self.index_field]] =  torch.tensor(latent[1:], dtype=torch.float32)
        # MongoUtils.update_many(self.db_name, self.collection_name, self.latent_db)


class VOKRawDataEmbeddingDB(VOKVectorDB):
    def __init__(self):
        uid = f'vok_raw_data_embedding_db'
        vdb_id = IDUtils.get_system_unique_id({'name': uid, 'uid': uid, 'type': 'vdb'})
        super().__init__(uid=uid, vdb_id=vdb_id,  
                         vdb_name=uid,
                         latent_dim=16, index_type="metadata", 
                         backend="vok_searchable_db", 
                         storage_type="mongo", max_capacity=-1, seed=42, device="cpu")
        self.backend = VOKSearchableDB(latent_dim=self.latent_dim, meta_dim=1, device=self.device)
        self.backend_type = "vok_searchable_db"
        self.storage = VDBMongoStorage(vdb_id=self.vdb_id, 
                                       vdb_name=self.vdb_name,
                                       db_name="embeddings", collection_name="embedding_paramaters", index_field='system_uid')

    def load_latent_db(self):
        self.storage.load_latent_db()
        print(self.storage.describe())
        self.backend.add_latents(self.storage.comb_db)
        self.backend.finalize()
        print(self.backend.db_tensor.shape)
if __name__ == "__main__":
    # vdb_id = IDUtils.get_system_unique_id({'name': 'test_vdb', 'type': 'vdb'})
    # vdb = VDBMongoStorage(vdb_id=vdb_id, db_name="embeddings", collection_name="embedding_paramaters", index_field='system_uid')
    # vdb.load_latent_db()
    # for latent in vdb.latent_db.values():
    #     print(latent['system_uid'])
    #     print(latent['latents']['lat_16'])
    # print(vdb.get_latent_by_column_id(0))
    # # print(vdb.get_latent_by_metadata({'system_uid': '-7075778220887397562'}))
    # print(vdb.get_latent_by_id('-7075778220887397562'))
    # print(len(vdb.latent_db.values()[0]))
    vdb = VOKRawDataEmbeddingDB()
    vdb.load_latent_db()