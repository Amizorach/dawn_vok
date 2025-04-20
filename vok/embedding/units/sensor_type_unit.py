
import pprint

import numpy as np
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.utils.id_utils import IDUtils
from dawn_vok.vok.dobjects.dobj.d_sensor_type import SensorType
from dawn_vok.vok.embedding.base.discrete_embedding import EmbeddedDiscreteValue


class DEmbedding:
    def __init__(self, key, embedding_scheme_id, latent=None):
        self.key = key
        self.embedding_scheme_id = embedding_scheme_id
        self.latent = latent
      
    def to_dict(self):
        return {
            '_id': self.get_id(),
            'key': self.key,
            'embedding_scheme_id': self.embedding_scheme_id,
            'latent': self.latent
        }
    
    def populate_from_dict(self, d):
        self.key = DictUtils.get_value(d, 'key', self.key)
        self.embedding_scheme_id = DictUtils.get_value(d, 'embedding_scheme_id', self.embedding_scheme_id)
        self.latent = DictUtils.get_value(d, 'latent', self.latent)
        self._id = self.get_id()

        return self
    
    def get_id(self):
        raise NotImplementedError("get_id must be implemented")
class EmbeddingUnit:
    @classmethod
    def get_db_name(cls):
        return 'embeddings'
    
    @classmethod
    def get_collection_name(cls):
        return 'embedding_units'
    
   
    def __init__(self, emb_id, system_uid, unit_type, emb_providers=None, default_emb_scheme=None):
        self.emb_id = emb_id
        self.system_uid = system_uid
        self.unit_type = unit_type
        self.emb_providers = emb_providers
        self.default_emb_scheme = default_emb_scheme
        self.embeddings ={}
    
    def __str__(self):
        return f"{self.system_uid}"
   
    def get_id(self):
        return f'{self.unit_type}_{self.emb_id}'
   
    def to_dict(self):
        return {
            'emb_id': self.emb_id,
            'system_uid': self.system_uid,
            'unit_type': self.unit_type,
            'emb_providers': self.emb_providers,
            'embeddings': self.embeddings,
            'default_emb_scheme': self.default_emb_scheme
        }
    
    def populate_from_dict(self, d):
        self.emb_id = DictUtils.get_value(d, 'emb_id', self.emb_id)
        self.system_uid = DictUtils.get_value(d, 'system_uid', self.system_uid)
        self.unit_type = DictUtils.get_value(d, 'unit_type', self.unit_type)
        self.emb_providers = DictUtils.get_value(d, 'emb_providers', self.emb_providers)
        self.embeddings = DictUtils.get_value(d, 'embeddings', self.embeddings)
        self.default_emb_scheme = DictUtils.get_value(d, 'default_emb_scheme', self.default_emb_scheme)
        return self
    
    def update_embeddings(self):
        pass

    def get_embedding(self, provider_scheme_di=None):
        if provider_scheme_di is None:
            if self.default_emb_scheme is None:
                raise ValueError(f"No default embedding scheme set for {self.system_uid}")
            provider_scheme_di = self.default_emb_scheme
        emb = np.array([])
        for k, v in provider_scheme_di.items():
            if k not in self.embeddings:
                raise ValueError(f"Embedding for provider scheme {k} not found")
            if v not in self.embeddings[k]:
                raise ValueError(f"Embedding for provider scheme {k} not found")
            emb = np.concatenate((emb, self.embeddings[k][v]))
        return emb
    
    def save_to_db(self):
        col = MongoUtils.get_collection(self.get_db_name(), self.get_collection_name())
        col.update_one({'_id': self.get_id()}, {'$set': self.to_dict()}, upsert=True)

class SensorTypeEmbeddingUnit(EmbeddingUnit):
    def __init__(self, sensor_info, system_uid=None):
        default_emb_scheme = {
            'syntax': ['lat_64']
        }
        if not sensor_info:
            raise ValueError(f"Sensor type {sensor_type} not found")
        super().__init__(emb_id=sensor_info.sensor_type, system_uid=system_uid or IDUtils.get_system_unique_id({'sensor_type':sensor_info.sensor_type}),
                         unit_type='sensor_type', emb_providers={'syntax': {'generator_id': 'unit_syntax', 'embedding_scheme_ids': None}},
                         default_emb_scheme=default_emb_scheme)
        self.sensor_type = sensor_info.sensor_type
        self.sensor_info = sensor_info
        self.sensor_type_uid = sensor_info.obj_id
       

    def to_dict(self):
        d = super().to_dict()
        d['sensor_type'] = self.sensor_type
        d['sensor_type_uid'] = self.sensor_type_uid
        # d['sensor_info'] = self.sensor_info.to_dict()
        return d
    
    def populate_from_dict(self, d):
        super().populate_from_dict(d)
        self.sensor_type = d['sensor_type']
        self.sensor_type_uid = d['sensor_type_uid']
        # self.sensor_info = d['sensor_info']
        return self
    
    def get_vocab(self):
        return self.sensor_info.get_vocab()
    
    def load_embeddings(self, generator_id, embedding_type='syntax'):
        # self.embeddings = self.sensor_info.get_embeddings(embedding_scheme_id)
        # col = MongoUtils.get_collection(EmbeddedDiscreteValue.get_db_name(), EmbeddedDiscreteValue.get_collection_name())
        emb = EmbeddedDiscreteValue.get_embeddings(generator_id=generator_id, emb_id=None, embedding_type=embedding_type, embedding_scheme_id=None, populate=True)
        # if emb:
        #     pprint.pprint(emb.keys())
        #     print(self.sensor_info.to_dict())
        #     for k, v in emb.items():
        #         print(k, v.emb_id, self.sensor_info.obj_id)
        #         if v.emb_id == self.sensor_info.obj_id:
        #             latent = v.get_embedding_scheme_latent(embedding_scheme_id)
        #             self.embeddings = latent
        #             print(k, latent)
        #             exit()
        # else:
        #     print(f"Embedding for {self.sensor_info.system_uid} not found")
            # raise ValueError(f"Embedding for {self.sensor_info.system_uid} not found")
        return emb

    def __str__(self):
        return f"{self.sensor_type} ({self.sensor_info.unit})"
    
    def update_embeddings(self):
        print(self.sensor_type_uid)
        for k, v in self.emb_providers.items():
            if k == 'syntax':
                emb = self.load_embeddings(v['generator_id'], k)
                if k not in self.embeddings:
                    self.embeddings[k] = {}
                pprint.pprint(self.to_dict())
                print(self.emb_id, emb.keys())
                pprint.pprint(emb)
                if self.sensor_type_uid in emb:
                    self.embeddings[k]= emb[self.sensor_type_uid].embedding
                    pprint.pprint(self.to_dict())
            else:
                raise ValueError(f"Embedding provider {k} not found")
class ProviderTypeEmbeddingUnit(EmbeddingUnit):
    def __init__(self, provider_info, system_uid=None):
        super().__init__(emb_id=provider_info.uid, system_uid=system_uid or IDUtils.get_system_unique_id({'provider_type':provider_info.uid}),
                         unit_type='provider_type', emb_providers={'syntax': {'generator_id': 'unit_syntax', 'embedding_scheme_ids': None}},
                         default_emb_scheme=default_emb_scheme)
        self.provider_type = provider_info.provider_type
        self.provider_info = provider_info
        self.provider_type_uid = provider_info.obj_id
        
if __name__ == "__main__":
    sensor_types = SensorType.get_all()
    for sensor_type in sensor_types:
        
        sensor_type_unit = SensorTypeEmbeddingUnit(sensor_type)
        sensor_type_unit.save_to_db()
        sensor_type_unit.update_embeddings()
        sensor_type_unit.save_to_db()

        print(sensor_type_unit)

