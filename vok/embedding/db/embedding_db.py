 mport time


import numpy as np
import torch

from dawn_vok.db.mongo_utils import MongoUtils

class VOKVectorDB:
    def __init__(self, vdb_id, latent_dim=32, index_type="metadata", vdb_name=None, backend="mongo", max_capacity=-1, seed=42, device="cpu"):
        self.vdb_id = vdb_id
        self.latent_dim = latent_dim
        self.index_type = index_type
        self.backend = backend
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

    def load_latent_db(self):
        pass

    def add_latents(self, latents):
        pass

    def get_latents_by_ids(self, latent_ids):
        pass
    
    def get_latent_by_id(self, latent_id):
        pass

    def get_latent_by_metadata(self, metadata):
        pass+

    def get_closest_latent(self, query_latent, k=1):
        pass

    def prepare_for_search(self):
        pass

    def release_search_resources(self):
        pass
    
    def move_to_device(self, device):
        pass
    
    def to_dict(self):
        pass

    def populate_from_dict(self, d):
        pass
    
    def save_to_db(self):
        pass
    
    @classmethod
    def load_from_db(cls, vdb_id):
        pass

class EmbeddingDB:
    @classmethod
    def get_collection_name(cls):
        return "embedding_tables"
    
    @classmethod
    def get_db_name(cls):
        return "models"
    
    @classmethod
    def load_from_db(cls, table_id):
        collection = MongoUtils.get_collection(cls.get_db_name(), cls.get_collection_name())
        d = collection.find_one({"table_id": table_id})
        if d:
            return cls(table_id).populate_from_dict(d)
        return None
    
    def __init__(self, table_id, generator_id=None, latent_dim=32, db_type="dict", db_name=None):
        self.generator_id = generator_id
        self.table_id = table_id
        self.latent_dim = latent_dim
        self.table_db_type = db_type
        self.table_db_name = db_name

    def to_dict(self):
        ret = {
            "_id": self.table_id,
            "table_id": self.table_id,
            "generator_id": self.generator_id,
            "latent_dim": self.latent_dim,
            "table_db_type": self.table_db_type,
            "table_db_name": self.table_db_name
        }
        return ret
    
    def populate_from_dict(self, d):
        self.table_id = d.get("table_id", self.table_id)
        self.generator_id = d.get("generator_id", self.generator_id)
        self.latent_dim = d.get("latent_dim", self.latent_dim)
        self.table_db_type = d.get("table_db_type", self.table_db_type)
        self.table_db_name = d.get("table_db_name", self.table_db_name)

    def save_to_db(self):
        collection = MongoUtils.get_collection(self.get_db_name(), self.get_collection_name())
        collection.update_one(
            {"_id": self.table_id},
            {"$set": self.to_dict()},
            upsert=True
        )
    
    # def load_from_db(self):
    #     collection = MongoUtils.get_collection(self.get_db_name(), self.get_collection_name())
    #     d = collection.find_one({"table_id": self.table_id})
    #     if d:
    #         self.populate_from_dict(d) 
    #     return self
    
    

    def get_or_assign_vectors(self, column_ids_input):
        pass

    def prepare_for_search(self):
        pass

    def find_closest(self, query_vector, k=1):
        pass

    def clear_search_data(self):
        pass



