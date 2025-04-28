import hashlib
from datetime import datetime
import random
from matplotlib import pyplot as plt
import numpy as np
import pprint
from pymongo import UpdateOne
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.utils.id_utils import IDUtils
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dawn_vok.vok.embedding.encoders.syntax_formater import SyntaxFormater
from dawn_vok.vok.v_objects.vobjects.formulations.v_agg import VOKFormulation
from dawn_vok.vok.v_objects.vobjects.measurments.v_measurment_unit import MeasurementUnit
from dawn_vok.vok.v_objects.vobjects.sensors.v_base_sensor_type import VOKBaseSensorType
from dawn_vok.vok.v_objects.vobjects.sensors.v_sensor_type import VOKSensorType
from dawn_vok.vok.v_objects.vobjects.source.v_source import VOKSource
from dawn_vok.vok.v_objects.vok_object import VOKObject

class DiscreteEmbedding(VOKObject):
    @classmethod
    def get_collection_name(cls):
        return "discrete_embeddings"
    
    @classmethod
    def get_db_name(cls):
        return "embeddings"
    
    # @classmethod
    # def create_embedding_scheme(cls, emb_size, di={}):
    #     ret = {'emb_size': emb_size}
    #     for k, v in di.items():
    #         ret[k] = v
    #     return ret
    def update_system_uid(self):
        self.system_uid = IDUtils.get_system_unique_id({
            'obj_type': 'discrete_embedding',
            'generator_id': self.generator_id,
            'embedding_type': self.embedding_type,
            'emb_id': self.emb_id,
            }, ret_type='int')
   
    def __init__(self, 
                 uid=None,
                 system_uid=None,
                 generator_id=None, emb_id=None, latent_schemes=None,
                 data=None, meta_data=None, embedding_type='syntax'):
        self.generator_id = generator_id
        self.emb_id = emb_id
        self.embedding_type = embedding_type
        self.syntax_directives_type = 'single_line'
        self.latent_schemes = latent_schemes or {}
        self.syntax_directives = []
        if data is None:
            data = []
        if isinstance(data, str):
            self.syntax_directives = [data]
        elif isinstance(data, list):
            self.syntax_directives = data
        else:
            print(data)
            raise ValueError('data must be a string or list')
        
        if not uid:
            uid = IDUtils.get_id([generator_id, emb_id])
        super().__init__(uid=uid, obj_type='discrete_embedding', syntax_directives=self.syntax_directives)
      
     
    def populate_from_dict(self, di):
        super().populate_from_dict(di)
        self.generator_id = di.get('generator_id', self.generator_id)
        self.emb_id = di.get('emb_id', self.emb_id)
        self.embedding_type = di.get('embedding_type', self.embedding_type)
        self.latent_schemes = di.get('latent_schemes', self.latent_schemes)
        self.update_system_uid()
        return self
    
  
    def to_dict(self):
        ret = super().to_dict()
        ret['generator_id'] = self.generator_id
        ret['emb_id'] = self.emb_id
        ret['latent_count'] = len(self.latent_schemes) if self.latent_schemes is not None else 0
        ret['embedding_type'] = self.embedding_type
        ret['latent_schemes'] = self.latent_schemes
        DictUtils.put_datetime(ret, 'updated_at', self.updated_at)

        return ret
   
    def save_to_db(self):
        
        col = MongoUtils.get_collection(db_name=self.get_db_name(), collection_name=self.get_collection_name())
        col.update_one(
            {'system_uid': self.system_uid},
            {'$set': self.to_dict()},
            upsert=True
        )
   
    def get_latent_scheme(self, latent_scheme_id):
        return self.latent_schemes.get(latent_scheme_id, None)
    
    def set_latent_scheme(self, latent_scheme_id, latent, save=True):
        self.latent_schemes[latent_scheme_id] = latent
        if save:
            self.save_to_db()
        return True
    
    @classmethod
    def save_multi_embeddings_to_db(cls, embeddings):
        MongoUtils.update_many(db_name=cls.get_db_name(), collection_name=cls.get_collection_name(), data=embeddings)

    @classmethod
    def get_embedding_managers(cls, generator_id=None,populate=False):
        col = MongoUtils.get_collection(db_name=cls.get_db_name(), collection_name=cls.get_collection_name())
        match = {}
        if generator_id:
            match['generator_id'] = generator_id
        
        embedding_managers = list(col.find(match))
        if populate:    
            ret = []
            for emb in embedding_managers:
                obj = cls(uid=emb['uid'])
                obj.populate_from_dict(emb)
                ret.append(obj)
            return ret
        else:
            return embedding_managers


  



    # @classmethod
    # def add_embeddings(cls, generator_id, embeddings, latent_schemes=None):
    #     emb_list = []
    #     all_emb_ids = set(list(embeddings.keys()) + list(latent_schemes.keys()))
    #     for uid in all_emb_ids:
    #         obj = cls(generator_id=generator_id, uid=uid, latent_schemes=latent_schemes.get(uid, None))
    #         emb_list.append(obj.to_dict())
    #     print(emb_list)
      
    #     return emb_list
    


    # se.save_to_db()
    # print(se.to_dict())
    # print(se.gather_vocab_for_update())

class SyntaxEmbedding(DiscreteEmbedding):

    @classmethod
    def create_embedding(cls, emb_str, uid, generator_id='syntax_single_embedding'):
        if not emb_str:
            raise ValueError('emb_str is required')
     
        return cls(generator_id=generator_id, emb_id=uid, data=emb_str)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_type = 'syntax'
        self.latent_vocab = {}
        self.hash_key = None

    def get_hash_key(self, dt):
        if isinstance(dt, str):
            self.hash_key = hashlib.md5(dt.encode()).hexdigest()
        elif isinstance(dt, list):
            self.hash_key = hashlib.md5(' '.join(dt).encode()).hexdigest()
        return self.hash_key
    
    def get_id(self):
        return f'syntax_embedding_{self.uid}'
    def to_dict(self):
        ret = super().to_dict()
        ret['latent_vocab'] = self.latent_vocab
        ret['hash_key'] = self.hash_key
        ret = DictUtils.np_to_list(ret)
        return ret

    def populate_from_dict(self, di):
        super().populate_from_dict(di)
        self.syntax_directives = di.get('data', self.syntax_directives)
        self.latent_vocab = di.get('latent_vocab', self.latent_vocab)
        self.hash_key = di.get('hash_key', self.hash_key)
        return self

    def update_vocab(self, save=False):
        self.latent_vocab = {}

        if not self.syntax_directives:
            self.syntax_directives = []
            return self.latent_vocab
        self.syntax_directives = list(set(self.syntax_directives))
        for dt in self.syntax_directives:
            hash_key = self.get_hash_key(dt)
            self.latent_vocab[hash_key] = {'raw': dt, 'hash': hash_key, 'latent': None}
        if save:
            self.save_to_db()
        return self.latent_vocab
    
    def update_embedding(self, embedding):
        self.embedding = embedding
        self.save_to_db()

    def get_vocab(self):
        if not self.latent_vocab:
            self.update_vocab()
        return self.latent_vocab
    
    def update_data(self, data, save=False):
        self.syntax_directives = data
        self.update_vocab(save=False)
        if save:
            self.save_to_db()
        return self.syntax_directives

    @classmethod
    def get_full_vocab_for_formatter(cls, generator_id=None, manger_ids=None):
        se = SyntaxEmbedding.get_embedding_managers(generator_id=generator_id, populate=True)
        if manger_ids:
            se = [s for s in se if s.get_id() in manger_ids]
        full_vocab = {}
        for s in se:
            for key, value in s.get_vocab().items():
                full_vocab[key] = value['raw']
        # full_vocab = list(set(full_vocab))

        return full_vocab, se
    

    @classmethod 
    def update_missing_embeddings(cls, generator_id=None, save=True):
        vocab, se = cls.get_full_vocab_for_formatter(generator_id=generator_id)
        print(se)
        mng_di = {mng.hash_key: mng for mng in se}
        for key, value in vocab.items():
            print(key, value)
            if key not in mng_di:
                emb = cls.create_embedding([value], uid=key, generator_id=generator_id)

                mng_di[key] = emb
        if save:
            for mng in mng_di.values():
                try:
                    mng.save_to_db()
                except Exception as e:
                    print(e)
        
    @classmethod
    def update_syntax_embedding(cls, generator_id=None, manger_ids=None, save=False):
        vocab, se = cls.get_full_vocab_for_formatter(generator_id=generator_id, manger_ids=manger_ids)
        vocab = list(set(vocab))
        builder = SyntaxFormater()
        pprint.pprint(vocab)
        builder.format_syntax(vocab)
        print(builder.encode_map.keys())
        for mng in se:
            base_latent =[]
            for key, value in mng.get_vocab().items():
                print(key)
                if key in builder.encode_map:
                    mng.latent_vocab[key]['latent'] = builder.encode_map[key]
                    base_latent.append(builder.encode_map[key].tolist())
            mng.set_latent_scheme('base', base_latent, save=False)
            mng.set_latent_scheme('mean', np.mean(base_latent, axis=0).tolist(), save=False)
        if save:
            for mng in se:
                try:
                    mng.save_to_db()
                except Exception as e:
                    print(e)
        return se
    
    
if __name__ == '__main__':
    MongoUtils.get_collection(db_name='embeddings', collection_name='discrete_embeddings').delete_many({})
    emb_dict = {}

    # bst = VOKSource.gather_full_vocab()
    # for s in bst:
    #     emb = SyntaxEmbedding.create_embedding(s, uid=s, generator_id='syntax_single_embedding')
    #     if emb.system_uid not in emb_dict:
    #         emb.save_to_db()
    #         emb_dict[emb.system_uid] = True
    source_list = VOKSource.get_all_by_obj_type('source')
    for s in source_list:
        v = s.get_vocab()
        emb = SyntaxEmbedding.create_embedding(v, uid = s.source_id, generator_id='syntax_single_embedding')
        if emb.system_uid not in emb_dict:
            emb.save_to_db()
            emb_dict[emb.system_uid] = True

    # meu = MeasurementUnit.gather_full_vocab()
    # for me in meu:
    #     emb = SyntaxEmbedding.create_embedding(me, uid = me, generator_id='syntax_single_embedding')
    #     if emb.system_uid not in emb_dict:
    #         emb.save_to_db()
    #         emb_dict[emb.system_uid] = True

    meu_list = MeasurementUnit.get_all_by_obj_type('measurement_unit')
    full_vocab = {}
    for mu in meu_list:
        v = mu.get_vocab()
        emb = SyntaxEmbedding.create_embedding(v, uid = mu.get_id(), generator_id='syntax_single_embedding')
        if emb.system_uid not in emb_dict:  
            emb.save_to_db()
            emb_dict[emb.system_uid] = True
    # bst = VOKBaseSensorType.gather_full_vocab() 
    # for v in bst:
        
    #     emb = SyntaxEmbedding.create_embedding(v, uid = v, generator_id='syntax_single_embedding')
    #     if emb.system_uid not in emb_dict:  
    #         emb.save_to_db()
    #         emb_dict[emb.system_uid] = True
    base_sensor_type_list = VOKBaseSensorType.get_all_by_obj_type('base_sensor_type')
    for st in base_sensor_type_list:
        v = st.get_vocab()
        emb = SyntaxEmbedding.create_embedding(v, uid = st.get_id(), generator_id='syntax_single_embedding')
        emb.save_to_db()
        emb_dict[emb.system_uid] = True
    # bst = VOKSensorType.gather_full_vocab()
    # for v in bst:
    #     emb = SyntaxEmbedding.create_embedding(v, uid = v, generator_id='syntax_single_embedding')
    #     if emb.system_uid not in emb_dict:  
    #         emb.save_to_db()
    #         emb_dict[emb.system_uid] = True

    sensor_type_list = VOKSensorType.get_all_by_obj_type('sensor_type')
    for s in sensor_type_list:
        v = s.get_vocab()
        emb = SyntaxEmbedding.create_embedding(v, uid = s.get_id(), generator_id='syntax_single_embedding')
        if emb.system_uid not in emb_dict:  
            emb.save_to_db()
            emb_dict[emb.system_uid] = True
    
    formulation_list = VOKFormulation.get_all_by_obj_type('formulation')
    for s in formulation_list:
        v = s.get_vocab()
        emb = SyntaxEmbedding.create_embedding(v, uid = s.get_id(), generator_id='syntax_single_embedding')
        if emb.system_uid not in emb_dict:  
            emb.save_to_db()
            emb_dict[emb.system_uid] = True
    se = SyntaxEmbedding.update_syntax_embedding(generator_id='syntax_single_embedding', save=True)
    
    se = {s.emb_id: s for s in se}
    for s in formulation_list:
        if s.get_id() not in se:
            continue
        
        s.latent_schemes = se[s.get_id()].latent_schemes
   

    exit()
    SyntaxEmbedding.update_syntax_embedding(generator_id='syntax_multi_embedding', save=True)

    exit()
    SyntaxEmbedding.update_missing_embeddings(generator_id='syntax_single_embedding', save=True)
    SyntaxEmbedding.update_syntax_embedding(generator_id='syntax_single_embedding', save=True)

