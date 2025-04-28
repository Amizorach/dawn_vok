

from pprint import pprint
from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.vok.embedding.db.vector_db.vok_searchable_db import VOKSearchableDB
from dawn_vok.vok.embedding.embedding_paramater.v_embedding_paramater import VOKEmbeddingParamater
from dawn_vok.vok.embedding.embedding_paramater.v_embedding_paramater_db import VOKEmbeddingParamaterDB
from dawn_vok.vok.pipelines.meta_data.raw_data_request.raw_data_retriever_dataset import RawDataRetrieverDataset
from dawn_vok.vok.vmodel.models.v_model import VOKModel
from dawn_vok.vok.vmodel.pipeline.v_pipeline_node import VOKPipelineNode
from dawn_vok.vok.vmodel.trainer.v_trainer import VOKTrainer
from dawn_vok.vok.pipelines.emp.syntax_reducer.emp_syntax_reducer_datasets import EMPSyntaxReducerDataset, EMPSyntaxReducerInferenceDataset

import torch
import torch.nn as nn
class RawDataRetrieverModel(VOKModel):
    def __init__(self, version='1.0.0', model_config=None, latent_db=None, token_dim=64, seq_len=8, latent_dim=96,
                  num_heads=4, num_layers=2, dropout=0.1, model_file_name=None):
        super().__init__(
            model_id='raw_data_retriever',
            name='Raw Data Retriever',
            version=version,
            model_config=model_config,
            latent_db=latent_db,
            model_file_name=model_file_name,
        )
        self.token_dim = token_dim
        self.seq_len = seq_len
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Pooling token to latent
        self.fc = nn.Linear(token_dim, latent_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, token_dim]
        # Add positional embedding
        # x = torch.cat([x, self.pos_embedding], dim=1)
        # # Transformer expects [seq_len, batch_size, token_dim]
        x = x.permute(1, 0, 2)
        # Encode

        x = self.transformer_encoder(x)
        # x: [seq_len, batch_size, token_dim]
        # Take first token (or mean)
        pooled = x[0]  # or x.mean(dim=0)
        # Project to latent
        latent = self.fc(pooled)
        # latent: [batch_size, latent_dim]
        return latent
        
    def get_base_config(self):
        token_dim = 64
        latent_dim = 96
        sequence_length = 8
        return {
            'config_id': 'raw_data_retriever_default',
            'config': {'token_dim': token_dim, 
                               'latent_dim': latent_dim,
                               'sequence_length': sequence_length,
                               'num_layers': 2, 'num_heads': 4, 'dim_feedforward': 128, 'dropout': 0.1 },
        }
     
    

class RawDataRetrieverTrainer(VOKTrainer):
    def __init__(self, node, device, load_model=True, latent_db=None, model_file_name=None):
        super().__init__(node, device, load_model, filename=model_file_name)
        self.criterion = nn.MSELoss()
        self.latent_db = latent_db
        self.model_file_name = model_file_name

    def run_batch(self, data_batch):
        # unpack the whole batch
        x, y = data_batch

        # if you’re using a device field, move tensors there:
        x = x.to(self.device)
        y = y.to(self.device)

        # run your pipeline once
        st_latent = self.pipeline_node.model(x)
        batch_ret = {'preds': st_latent, 'y': y}
        # make sure 'y' (and mask, if needed) are in the returned dict
        # batch_ret['y'] = y
        # batch_ret['mask'] = mask

        return batch_ret
    
    def compute_loss(self, batch_ret):
        preds = batch_ret['preds']
        y = batch_ret['y']
        # print(f"preds: {preds.shape}")
        # print(f"y: {y.shape}")
        loss = self.criterion(preds, y.squeeze(1))
        return loss
    
    def run_batch_inference(self, data_batch):
        # unpack the whole batch
        x, y = data_batch

        # if you’re using a device field, move tensors there:
        x = x.to(self.device)

        # run your pipeline once
        st_latent = self.pipeline_node.model(x)
        batch_ret = {'preds': st_latent, 'y': y}
        # make sure 'y' (and mask, if needed) are in the returned dict
        # batch_ret['y'] = y
        # batch_ret['mask'] = mask

        return batch_ret
    
    def finalize_inference(self, latents, dataset):
        from dawn_vok.vok.embedding.data_emb.v_data_embedding import VOKDataEmbeddingBuilder
        batch_ret = []
        # pprint(dataset.meta_data[0])
        # print(latents[0][:17])
        # print(dataset.gt[0].shape)
        # print(dataset.gt[0].squeeze(0)[:17])
        meta_data_map = {}
        for v in self.latent_db.meta_data_map.values():
            meta_data_map[v.system_uid] = {
                'param_id': v.param_id,
                'param_type': v.param_type,
                
            }
        # for i, md in enumerate(dataset.meta_data):
        #     meta_data_map[md['source_id_index']] = md
        #     meta_data_map[md['sensor_type_index']] = md['sensor_type']
        #     meta_data_map[md['formulation_index']] = md['formulation']

        pprint(meta_data_map.keys())
       
        self.latent_db.db.move_to('cpu')
        for i, latent in enumerate(latents):
            ret = {
                # 'latent': latent,
                'meta_data': dataset.meta_data[i],
            }
            ret['closest_source'], source_id_index = self.latent_db.search(latent[0:16], top_k=1)
           
            # pprint(self.dataset.meta_data[ind])
            ret['closets_orig_source'], orig_index = self.latent_db.search(dataset.gt[i].squeeze(0)[0:16], top_k=1)
            
            ret['closest_sensor_type'], sensor_type_index = self.latent_db.search(latent[16:32], top_k=1,)
            ret['closest_formulation'], formulation_index = self.latent_db.search(latent[32:48], top_k=1)
            mo = {}
            mo['encoded_timestamp'] = latent[48]
            mo['encoded_frequency'] = latent[49]
            # print(source_id_index, sensor_type_index, formulation_index)
            mo['source_id'] =  meta_data_map.get(source_id_index, None)
            mo['orig_source_id'] =  meta_data_map.get(orig_index, None)
            mo['sensor_type'] =  meta_data_map.get(sensor_type_index, None)
            mo['formulation'] =  meta_data_map.get(formulation_index, None)
            mo = DictUtils.torch_to_list(mo)
            ret['meta_data_output'] = mo
            ret['frequency'] = VOKDataEmbeddingBuilder.decode_freq(mo['encoded_frequency'])
            ret['start_time'] = VOKDataEmbeddingBuilder.decode_time_enc(mo['encoded_timestamp'])
            # ret['s_latent'] = dataset.gt[i].squeeze(0)
            # ret['latent'] = latent
            ret = DictUtils.torch_to_list(ret)
            batch_ret.append(ret)
        
            # su = float(dataset.gt[i].squeeze(0)[0])
            # print(su)
            # for i in range(len(self.latent_db.db.db_tensor)):
            #     su1 = float(self.latent_db.db.db_tensor[i][0])
            #     print(su, su1)
            #     if su1 == su:
            #         print(i)
            #         exit()
        # for i, md in enumerate(dataset.meta_data):
        #         print(i, md['source_id'], )
        # for i, md in enumerate(self.latent_db.meta_data):
        #     print(i, md)
        return batch_ret
    
class RawDataRetrieverPLN(VOKPipelineNode):
    def __init__(self, node_id, name=None, model_info=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = RawDataRetrieverModel(model_file_name='raw_data_retriever_model.pth')
        
        super().__init__(node_id, name, model_info, model=model)
        self.latent_db =  VOKEmbeddingParamaterDB()
        self.latent_db.load_latents()
        self.load_trainer()
        
    def load_trainer(self):
        if self.trainer is None:
            self.trainer = RawDataRetrieverTrainer(self, 
                             device=self.device, 
                             load_model=True,
                             latent_db=self.latent_db, 
                             model_file_name=self.model.model_file_name)
        return self.trainer
    
    def train_model(self, dataset=None):
        self.load_trainer()
        if dataset is None:
            dataset = RawDataRetrieverDataset()
            dataset.load_data(sample_size=100)
        self.trainer.train(dataset, epochs=100)

    def get_latents(self, dataset=None, key=None):
        self.load_trainer()
        if dataset is None:
            dataset = RawDataRetrieverDataset()
            dataset.load_data(sample_size=100)

        latent_di = self.trainer.inference(dataset, as_dict=True, key=key)
        return latent_di

if __name__ == "__main__":
    pln = RawDataRetrieverPLN(node_id=f'test_retriever', name=f'RawDataRetrieverPLN')

    dataset = RawDataRetrieverDataset()

    dataset.load_data(sample_size=10000)
    pln.train_model(dataset=dataset)
    latents = pln.get_latents(dataset=dataset, key='preds')
        # print(latents)
    pprint(latents[0])