import pprint
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.utils.id_utils import IDUtils
from dawn_vok.vok.dobjects.dobj.d_sensor_type import SensorType
from dawn_vok.vok.embedding.base.discrete_embedding import SyntaxEmbedding
from dawn_vok.vok.v_objects.vok_object import VOKObject



    
import torch
import torch.nn as nn

class HeaderBuilder(nn.Module):
    def __init__(
        self,
        max_num_unit_types=50,
        max_seq_len=24,
        max_unit_len=6,
        unit_type_dim=8,
        position_dim=4,
    ):
        super().__init__()
        assert unit_type_dim + 2 * position_dim <= 16, "Header overflow risk"

        self.unit_type_embed = nn.Embedding(max_num_unit_types, unit_type_dim)
        self.global_pos_embed = nn.Embedding(max_seq_len, position_dim)
        self.unit_pos_embed = nn.Embedding(max_unit_len, position_dim)

    def populate_from_dict(self, di):
        self.unit_type_embed = nn.Embedding(di['max_num_unit_types'], di['unit_type_dim'])
        self.global_pos_embed = nn.Embedding(di['max_seq_len'], di['position_dim'])
        self.unit_pos_embed = nn.Embedding(di['max_unit_len'], di['position_dim'])
        self.create_base_embeddings(di)
        return self
    
      
        

    def create_base_embedding(self, di, dim_size=32):
        unit_type_id = di.get('unit_type_id', None)
        global_position = di.get('global_position', None)
        unit_position = di.get('unit_position', None)
        unit_length = di.get('unit_length', None)
        relative_indices = di.get('relative_indices', None)
        is_first = di.get('is_first', None)
        is_last = di.get('is_last', None)
        is_group_start = unit_position == 0
        has_data = di.get('has_data', None)

        # Embeddings
        unit_type_vec = self.unit_type_embed(unit_type_id)        # (B, N, unit_type_dim)
        global_pos_vec = self.global_pos_embed(global_position)   # (B, N, position_dim)
        unit_pos_vec = self.unit_pos_embed(unit_position)         # (B, N, position_dim)
        
        # Scalar metadata   
        scalar_meta = torch.stack([
            unit_length,
            relative_indices,
            is_first,
            is_last,
            is_group_start,
            has_data
        ], dim=-1)  # (B, N, 6)

        # Pad remaining space up to 10 scalar dimensions

        # Full 32D header
        header = torch.cat([
            unit_type_vec,
            global_pos_vec,
            unit_pos_vec,
            scalar_meta,
        ], dim=-1)  # (B, N, 32)
        if header.shape[-1] < self.dim_size:
            pad = torch.zeros(B, N, self.dim_size - header.shape[-1], device=header.device)
            header = torch.cat([header, pad], dim=-1)
        return header

    def forward(
        self,
        unit_type_ids, global_positions, unit_positions,
        unit_lengths, relative_indices,
        is_first, is_last, is_group_start, has_data
    ):
        return self.reconstruct(
            unit_type_ids, global_positions, unit_positions,
            unit_lengths, relative_indices,
            is_first, is_last, is_group_start, has_data
        )

class EmbeddingParamater(VOKObject):
    @classmethod
    def get_collection_name(cls):
        return 'embedding_paramaters'

    @classmethod
    def get_db_name(cls):
        return 'embeddings'
    
    def __init__(self, uid=None, param_type=None, param_id=None, embedding_manager_config=None):
        self.param_type = param_type
        self.param_id = param_id
        self.uid = uid or IDUtils.get_id([self.param_type, self.param_id])
        self.embedding_manager_config = embedding_manager_config or {}
        self.latents = {}
        super().__init__(uid=self.uid, obj_type='embedding_paramater')

    def to_dict(self):
        ret = super().to_dict()
        ret['param_type'] = self.param_type
        ret['param_id'] = self.param_id
        ret['embedding_manager_config'] = self.embedding_manager_config
        ret['latents'] = DictUtils.np_to_list(self.latents)
        return ret

    def populate_from_dict(self, di):
        super().populate_from_dict(di)
        self.param_type = di.get('param_type', self.param_type)
        self.param_id = di.get('param_id', self.param_id)
        self.embedding_manager_config = di.get('embedding_manager_config', self.embedding_manager_config)
        self.latents = di.get('latents', self.latents)
        return self

    def add_embedding_manager_config(self, embedding_type, embedding_manager_config):
        self.embedding_manager_config[embedding_type] = embedding_manager_config
        return self
    
    def get_embedding_manager_config(self, embedding_type):
        return self.embedding_manager_config.get(embedding_type, None)

    def get_embedding_manager_configs(self):
        return list(self.embedding_manager_config.values())

    def load_or_create_embedding_manager(self, embedding_type, mng_di=None):
        mng_di = mng_di or {}
        if embedding_type  == 'syntax':
            se = SyntaxEmbedding.create_embedding('12123')
            se.update_data(['fsfsfsfs', 'sadsdsad'], save=True)
            se.save_to_db()
            print(se.to_dict())
            SyntaxEmbedding.update_missing_embeddings(generator_id='syntax_single_embedding', save=True)
            SyntaxEmbedding.update_syntax_embedding(generator_id='syntax_single_embedding', save=True)
            self.embedding_manager_config[embedding_type] = se.to_dict()
        return self

class SensorTypeEmbeddingParamater(EmbeddingParamater):
    def __init__(self, sensor_type, sensor_info=None):
        self.sensor_type = sensor_type
        self.sensor_info = sensor_info
        embedding_managers = {'syntax': None}
        uid = f'sensor_type_{sensor_type}'
        super().__init__(uid=uid, param_type='sensor_type', param_id=sensor_type)

    def to_dict(self):
        ret = super().to_dict()
        ret['sensor_info'] = self.sensor_info.to_dict()

        return ret
    

    def populate_from_dict(self, di):
        super().populate_from_dict(di)
        self.sensor_type = di.get('sensor_type', self.sensor_type)
        if not self.sensor_info or not self.sensor_info.sensor_type == self.sensor_type:
            self.sensor_info = SensorType.get_by_uid(f'{self.sensor_type}')
            if not self.sensor_info:
                raise ValueError(f"Sensor type {self.sensor_type} not found")
            
        return self
    
    def prepare_embedding_managers(self):
        # for embedding_type, mng in self.embedding_manager_config.items():
        se = self.load_or_create_embedding_manager(embedding_type='syntax')
        pprint.pprint(se.latent_schemes)
        self.embedding_manager_config['syntax'] = se.emb_id
        pprint.pprint(se.to_dict().keys())
        pprint.pprint(se.to_dict())
        pprint.pprint(se.latent_schemes)
        self.latents = se.latent_schemes
        return self
    

    def load_or_create_embedding_manager(self, embedding_type, mng_di=None):
        mng_di = mng_di or SyntaxEmbedding.get_all()
        if mng_di is None:
            raise ValueError('mng_di is None')
        if isinstance(mng_di, list):
            mng_di = {mng.emb_id: mng for mng in mng_di}
        pprint.pprint(mng_di)
        if embedding_type  == 'syntax':
            directives = self.sensor_info.get_vocab()
            uid = f'syntax_single_embedding_{self.uid}'
            print(uid)
            se = mng_di.get(uid, None) or SyntaxEmbedding.create_embedding(directives, uid=uid)
            print(se.to_dict())
            vocab = self.sensor_info.get_vocab()

            se.update_data(vocab, save=True)
            se.save_to_db()
           
            self.embedding_manager_config[embedding_type] = se.emb_id
            return se
        return None
    
    @classmethod
    def update_all_sensor_types(cls):   
        # MongoUtils.get_collection(db_name='embeddings', collection_name='embedding_paramaters').delete_many({})
        sensor_types = SensorType.get_all()
        sens = {}
        for sensor_info in sensor_types:
            print(sensor_info.sensor_type)
            if sens.get(sensor_info.sensor_type, None) is not None:
                continue
            sens[sensor_info.sensor_type] = sensor_info
            ep = SensorTypeEmbeddingParamater(sensor_info.sensor_type, sensor_info)
            ep.save_to_db()
            # ep = EmbeddingParamater.get_by_uid(ep.uid)
            ep.prepare_embedding_managers()
            ep.save_to_db()
        SyntaxEmbedding.update_missing_embeddings(generator_id='syntax_single_embedding', save=True)
        SyntaxEmbedding.update_syntax_embedding(generator_id='syntax_single_embedding', save=True)

if __name__ == '__main__':
    SensorTypeEmbeddingParamater.update_all_sensor_types()