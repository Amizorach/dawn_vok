from datetime import datetime

import numpy as np
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.utils.id_utils import IDUtils
from dawn_vok.vok.embedding.syntax.syntax_reducer import SyntaxEmbeddingReductionTrainer
from dawn_vok.vok.embedding.syntax_emb.syntax_emb import EmbeddedDiscreteValue


class DObject:
    def __init__(self, system_uid, obj_type=None, uid=None, syntax_directives=None, norm_range=None, has_data=False):
        self.obj_type = obj_type
        self.uid = uid
        self.system_uid = system_uid
        self.syntax_directives = syntax_directives
        self.updated_at = datetime.now()
        self.norm_range = norm_range
        self.has_data = has_data

    @classmethod
    def get_db_name(cls):
        return 'meta_data'
    
    @classmethod
    def get_collection_name(cls):
        raise NotImplementedError("get_collection_name must be implemented by subclasses")
   
    def to_dict(self):
        ret = {
            '_id': self.system_uid,
            'obj_type': self.obj_type,
            'uid': self.uid,
            'system_uid': self.system_uid,
            'syntax_directives': self.syntax_directives,
            'norm_range': self.norm_range
        }
        DictUtils.put_datetime(ret, 'updated_at', self.updated_at)
        return ret
    
    def populate_from_dict(self, d):
        self.obj_type = d.get('obj_type', self.obj_type)
        self.uid = d.get('uid', self.uid)
        self.system_uid = d.get('system_uid', self.system_uid)
        self.syntax_directives = d.get('syntax_directives', self.syntax_directives)
        self.norm_range = d.get('norm_range', self.norm_range)
        self.has_data = d.get('has_data', self.has_data)
        return self
    
    def save_to_db(self):
        collection = MongoUtils.get_collection(self.get_db_name(), self.get_collection_name())
        collection.update_one({'system_uid': self.system_uid}, {'$set': self.to_dict()}, upsert=True)

    def load_from_db(self):
        collection = MongoUtils.get_collection(self.get_db_name(), self.get_collection_name())
        d = collection.find_one({'system_uid': self.system_uid})
        if d:
            self.populate_from_dict(d)
    
    @classmethod
    def get_by_system_uid(cls, system_uid):
        collection = MongoUtils.get_collection(cls.get_db_name(), cls.get_collection_name())
        d = collection.find_one({'system_uid': system_uid})
        if d:
            obj = cls(system_uid)
            obj.populate_from_dict(d)
            return obj
        return None

    
    @classmethod
    def get_all(cls):
        collection = MongoUtils.get_collection(cls.get_db_name(), cls.get_collection_name())
        return [cls(d['system_uid']).populate_from_dict(d) for d in collection.find()]
   
        
    def get_vocab(self):
        ret = []
       
    
       
        if self.syntax_directives:
            ret.extend(self.syntax_directives)

        return ret

    @classmethod
    def gather_full_vocab(cls):
        objects = cls.get_all()
        vocab = []
        for o in objects:
            vocab.extend(o.get_vocab())
        vocab = list(set(vocab))
        return vocab
    
    def get_embeddings(self, embedding_scheme_id):
        vocab = self.get_vocab()
        ev = {e.replace(' ', '_').lower():None for e in vocab}
        col = MongoUtils.get_collection(EmbeddedDiscreteValue.get_db_name(), EmbeddedDiscreteValue.get_collection_name())
        embeddings = col.find({'emb_id': {'$in': list(ev.keys())}})
        
        for e in embeddings:
            ev[e['emb_id']] = e.get('embedding', {}).get(embedding_scheme_id, None)
        count = 0

        for k, v in ev.items():
            if v is None:
                print(f"Embedding for {k} not found")
            else:
                print(f"{k}: {len(v)}")
                if mean_amb is None:
                    mean_amb = v
                else:
                    mean_amb = np.add(mean_amb, v)
                count += 1
        mean_amb = np.divide(mean_amb, count)
        print(f"Mean embedding: {mean_amb}")
        return mean_amb
 
    
if __name__ == '__main__':
    # SensorType.create_all_sensor_types()
    # vocab = SensorType.gather_full_vocab()
    # builder = SyntaxDBBuilder()
    # builder.build_syntax_db(vocab)
    # builder.save_to_db()
    trainer = SyntaxEmbeddingReductionTrainer()
    trainer.update_embeddings(emb_size=32, orig_scheme_id='full_embedding', out_scheme_id='reduced_32')
    trainer.save_model()
    trainer = SyntaxEmbeddingReductionTrainer(short_emb_size=16)
    trainer.update_embeddings(emb_size=16, orig_scheme_id='full_embedding', out_scheme_id='reduced_16')
    trainer.save_model()
    # "air_temperature": {
    #     "sensor_type": "temperature",
    #     "value_type": "float",
    #     "sensor_value_unit": "Celsius",
    #     "sensor_class": "environmental",
    #     "range_expected": [-50.0, 60.0],
    #     "precision": 0.1,
    #     "physical_quantity": "air_heat_content",
    #     "application_context": ["greenhouse", "outdoor", "weather_station"],
    #     "syntax_directives": [
    #         "Air temperature measures the thermal energy of the surrounding environment.",
    #         "This sensor is critical for evaluating plant growth conditions and climate control.",
    #         "It outputs values in degrees Celsius, typically ranging from -50 to 60."
    #     ]
    # },
    