
import numpy as np
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.utils.id_utils import IDUtils
from dawn_vok.vok.v_objects.vok_object import VOKObject


class VOKEmbeddingParamater(VOKObject):
    @classmethod
    def get_collection_name(cls):
        return 'embedding_paramaters'

    @classmethod
    def get_db_name(cls):
        return 'embeddings'
    
    def __init__(self, uid=None, param_type=None, param_id=None, latents=None, static_latent_id=None):
        self.param_type = param_type
        self.param_id = param_id
        self.uid = uid or IDUtils.get_id([self.param_type, self.param_id])
        self.latents = latents or {}
        self.static_latent_id = static_latent_id or []
        super().__init__(uid=self.uid, obj_type='embedding_paramater')
        # _latents = {
        #     'measurement_unit': torch.tensor([0]*64),
        #     'base_sensor_type': torch.tensor([0]*64),
        #     'sensor_type': torch.tensor([0]*64),
        #     'sensor_info': torch.tensor([0]*64),
        # }
    def to_dict(self):
        ret = super().to_dict()
        ret['param_type'] = self.param_type
        ret['param_id'] = self.param_id
        ret['latents'] =self.latents
        ret['static_latent_id'] = self.static_latent_id
        return ret

    def populate_from_dict(self, di):
        super().populate_from_dict(di)
        self.param_type = di.get('param_type', self.param_type)
        self.param_id = di.get('param_id', self.param_id)
        self.latents = di.get('latents', self.latents)
        self.static_latent_id = di.get('static_latent_id', self.static_latent_id)
        return self

    def get_latent(self, latent_id, latent_key, dim_size =64):
        latent_key = latent_key or f'lat_{dim_size}'
        # print('get_latent', latent_id)
        lat = DictUtils.parse_value(self.latents, f'{latent_id}.{latent_key}', default=None)
      
        return lat

    def update_latent(self, latent_id, latent_key, latent_value):
        if (latent_key == 'static_latent_id'):
            self.static_latent_id = latent_value
        else:
            DictUtils.put_value(self.latents, f'{latent_id}.{latent_key}', value=latent_value)
        # print('update_latent', self.latents.keys())
        return self
    
    def get_latents(self, dim_size =64):
        ret = []
        latent_key = f'lat_{dim_size}'
        for v in self.latents.values():
            if not v:
                ret.append(np.array([0]*dim_size))
            else:
                ret.append(v.get(latent_key, np.array([0]*dim_size)))
        return np.array(ret)
    
    def get_latent_keys(self):
        return list(self.latents.keys())
    
    def get_latent_key_dim(self, latent_key):
        if not latent_key in self.latents:
            raise ValueError(f'latent_key {latent_key} not found')
        return self.latents[latent_key].shape[0]
        
    @classmethod
    def get_by_param_type(cls, param_type, populate=True):
        col = MongoUtils.get_collection(db_name=cls.get_db_name(), collection_name=cls.get_collection_name())
        ret = col.find({'param_type': param_type})
        if populate:
            ret = [cls().populate_from_dict(obj) for obj in ret]
        return list(ret)
