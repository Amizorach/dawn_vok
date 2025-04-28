

from dawn_vok.vok.pipelines.meta_data.raw_data_request.raw_data_retriever_dataset import RawDataRetrieverDataset
from dawn_vok.vok.vmodel.models.v_model import VOKModel
from dawn_vok.vok.vmodel.pipeline.v_pipeline_node import VOKPipelineNode
from dawn_vok.vok.vmodel.trainer.v_trainer import VOKTrainer
from dawn_vok.vok.pipelines.emp.syntax_reducer.emp_syntax_reducer_datasets import EMPSyntaxReducerDataset, EMPSyntaxReducerInferenceDataset

import torch
import torch.nn as nn
class RawDataRetrieverModel(VOKModel):
    def __init__(self, version='1.0.0', model_config=None, latent_db=None, token_dim=64, seq_len=8, latent_dim=64,
                  num_heads=4, num_layers=2, dropout=0.1):
        super().__init__(
            model_id='raw_data_retriever',
            name='Raw Data Retriever',
            version=version,
            model_config=model_config,
            latent_db=latent_db
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
        x = x + self.pos_embedding
        # Transformer expects [seq_len, batch_size, token_dim]
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
        latent_dim = 64
        sequence_length = 8
        return {
            'config_id': 'raw_data_retriever_default',
            'config': {'token_dim': token_dim, 
                               'latent_dim': latent_dim,
                               'sequence_length': sequence_length,
                               'num_layers': 2, 'num_heads': 4, 'dim_feedforward': 128, 'dropout': 0.1 },
        }
     

class RawDataRetrieverTrainer(VOKTrainer):
    def __init__(self, node, device, load_model=True):
        super().__init__(node, device, load_model)
        self.criterion = nn.MSELoss()

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
        loss = self.criterion(preds, y.squeeze(1))
        return loss
    
class RawDataRetrieverPLN(VOKPipelineNode):
    def __init__(self, node_id, name=None, model_info=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = RawDataRetrieverModel()
        
        super().__init__(node_id, name, model_info, model=model)
        self.load_trainer()

    def load_trainer(self):
        if self.trainer is None:
            self.trainer = RawDataRetrieverTrainer(self, 
                             device=self.device, 
                             load_model=True)
        return self.trainer
    
    def train_model(self, dataset=None):
        self.load_trainer()
        dataset = dataset or RawDataRetrieverDataset()
        dataset.load_data(sample_size=10000)
        print('train_model', len(dataset.data))
        self.trainer.train(dataset, epochs=1000)

    # def get_latents(self, dataset=None, key=None):
    #     key = key or f'lat_{self.model.latent_dim}'
    #     self.load_trainer()
    #     if dataset is None:
    #         dataset = EMPSyntaxReducerInferenceDataset()
    #     latent_di = self.trainer.inference(dataset, as_dict=True)
    #     dataset.update_latents(latent_di, key=key)
    #     return latent_di


if __name__ == "__main__":
    pln = RawDataRetrieverPLN(node_id=f'test_retriever', name=f'RawDataRetrieverPLN')
    pln.train_model()
        # print(latents.shape)