

from email.headerregistry import DateHeader
import random
from dawn_vok.vok.embedding.encoders.timerange_st_enc import TimeRangeSTEncoder
from dawn_vok.vok.embedding.encoders.source_st_enc import TimeStampSTEncoder
from dawn_vok.vok.pipelines.meta_data.raw_data_request.raw_data_retriever_dataset import RawDataRetrieverDataset
from dawn_vok.vok.pipelines.meta_data.time_classifier.time_classefier_dataset import TimeClassifierDataset
from dawn_vok.vok.pipelines.meta_data.time_classifier.time_classefier_model import TimeClassifierModel
from dawn_vok.vok.pipelines.meta_data.time_range_classifier.time_range_classefier_dataset import TimeRangeClassifierDataset
from dawn_vok.vok.pipelines.meta_data.time_range_classifier.time_range_classefier_model import TimeRangeClassifierModel
from dawn_vok.vok.vmodel.models.basic_models.encoder_decoder import EncoderDecoderModel
from dawn_vok.vok.vmodel.models.v_model import VOKModel
from dawn_vok.vok.vmodel.pipeline.v_pipeline_node import VOKPipelineNode
from dawn_vok.vok.vmodel.trainer.v_trainer import VOKTrainer
from dawn_vok.vok.pipelines.emp.syntax_reducer.emp_syntax_reducer_datasets import EMPSyntaxReducerDataset, EMPSyntaxReducerInferenceDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class TimeRangeClassifierTrainer(VOKTrainer):
    def __init__(self, node, device, load_model=True, lr=None, **kwargs):
        super().__init__(node, device, load_model, lr=lr)
        self.criterion = nn.MSELoss()
  
        self.input_dim = kwargs.get("input_dim", 16)
        self.loss_weights = kwargs.get("loss_weights", {'start': 1, 'end': 1, 'freq': 1})
    def run_batch(self, data_batch):
        # unpack the whole batch
        x, y = data_batch

        # if you’re using a device field, move tensors there:
        x = x.to(self.device)
        # y = y.to(self.device)
        logits, latents = self.pipeline_node.model(x)
        # print(f"Logits: {logits.keys()}")
        # print(f"Logits['start']: {logits['start'].shape}")
        # print(f"Logits['end']: {logits['end'].shape}")
        # print(f"Logits['freq']: {logits['freq'].shape}")
        # print(f"Latents: {latents.shape}")
        # print(f"y: {y}")
        # run your pipeline once
        batch_ret = {'preds': logits, 'y': y, 'latents': latents}
        # make sure 'y' (and mask, if needed) are in the returned dict
        # batch_ret['y'] = y
        # batch_ret['mask'] = mask
       
        return batch_ret
    
    def time_collate_fn(self, batch):
        xs, ts_list = zip(*batch)
        xs = torch.stack(xs, dim=0)
        return xs, list(ts_list)

    def compute_loss(self, batch_ret):
        preds = batch_ret['preds']
        y = batch_ret['y']
        batch_size = len(y)
        total_loss = 0.0
        # targets = { k: torch.empty(batch_size, dtype=torch.long, device=self.device) for k in self.loss_weights }
        # for i in range(batch_size   ):
        #     if "start" in targets: targets["start"][i]  = y['start'][i]
        #     if "end" in targets:  targets["end"][i]  = y['end'][i]
        #     if "freq" in targets: targets["freq"][i] = y['freq'][i]

        loss_components = {}
        for k, target in y.items():
            #  preds = preds[0]
            if k not in preds: 
                 print(f"Head '{k}' not in preds")
                 print(f"Preds: {preds.keys()}")
                 continue
            if k not in self.loss_weights: continue
        #  print(f"Target tensor: {target_tensor}")
        #  n_cls = preds[k].size(1)
        #  if target.max() >= n_cls or target.min() < 0:
            #      raise ValueError(f"Head '{k}' target index out of bounds [0, {n_cls-1}]")
            # print(f"preds[k]: {preds[k].shape}")
            # print(f"target: {target.shape}")
            head_loss = self.criterion(preds[k], target)
            loss_components[k] = head_loss.item()
            total_loss += self.loss_weights[k] * head_loss
        # print(f"Total loss: {total_loss}")
        # print(f"Loss components: {loss_components}")
        # print(f"Preds: {preds}")
        # print(f"Targets: {targets}")
        return total_loss
    
   

   
        
class TimeRangeClassifierPLN(VOKPipelineNode):
    def __init__(self, node_id, name=None, model_info=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = TimeRangeClassifierModel()
        
        super().__init__(node_id, name, model_info, model=model)
        self.load_trainer()

    def load_trainer(self):
        if self.trainer is None:
            self.trainer = TimeRangeClassifierTrainer(self, 
                             lr=0.0001,
                             device=self.device, 
                             load_model=True)
        return self.trainer
    
    def train_model(self, dataset=None):
        self.load_trainer()
        dataset = dataset or TimeRangeClassifierDataset(length=50000)
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
    pln = TimeRangeClassifierPLN(node_id=f'test_time_range_classifier', name=f'TimeRangeClassifierPLN')
    pln.train_model()
        # print(latents.shape)