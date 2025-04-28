

from dawn_vok.vok.pipelines.emp.syntax_reducer.emp_syntax_reducer_model import SyntaxReducerModel
from dawn_vok.vok.vmodel.pipeline.v_pipeline_node import VOKPipelineNode
from dawn_vok.vok.vmodel.trainer.v_trainer import VOKTrainer
from dawn_vok.vok.pipelines.emp.syntax_reducer.emp_syntax_reducer_datasets import EMPSyntaxReducerDataset, EMPSyntaxReducerInferenceDataset

import torch
class EMPSytaxReducerPLN(VOKPipelineNode):
    def __init__(self, node_id, latent_dim, name=None, model_info=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SyntaxReducerModel(latent_dim=latent_dim, version=f'1.0.{latent_dim}')

        super().__init__(node_id, name, model_info, model=model)
        self.load_trainer()

    def load_trainer(self):
        if self.trainer is None:
            self.trainer = VOKTrainer(self, 
                             device=self.device, 
                             load_model=True)
        return self.trainer
    
    def train_model(self, dataset=None):
        self.load_trainer()
        dataset = dataset or EMPSyntaxReducerDataset(
            
            latent_key='base')
        print('train_model', len(dataset.full_latent_list))
        self.trainer.train(dataset, epochs=100, log_interval=30)

    def get_latents(self, dataset=None, key=None):
        key = key or f'lat_{self.model.latent_dim}'
        self.load_trainer()
        if dataset is None:
            dataset = EMPSyntaxReducerInferenceDataset()
        latent_di = self.trainer.inference(dataset, as_dict=True)
        dataset.update_latents(latent_di, key=key)
        return latent_di


if __name__ == "__main__":
    for i in [16, 32, 48, 64, 112, 128, 256]:
        pln = EMPSytaxReducerPLN(node_id=f'se_lat_{i}', name=f'EMPSytaxReducerPLN_{i}', latent_dim=i)
        # pln.train_model()
        latents = pln.get_latents()
        # print(latents.shape)