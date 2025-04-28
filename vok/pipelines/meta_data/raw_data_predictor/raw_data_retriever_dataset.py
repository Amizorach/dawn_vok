import torch
from datetime import datetime
from dawn_vok.vok.embedding.data_emb.v_data_embedding import VOKDataEmbeddingBuilder
from dawn_vok.vok.v_dataset.v_dataset import VOKDataset


class RawDataRetrieverDataset(VOKDataset):
    def __init__(self, dataset_id=None, name=None, sample_size=100):
        super().__init__(dataset_id=dataset_id, name=name)
        self.sample_size = sample_size
        self.meta_data = []
        self.latents = []
        self.gt = []
        self.data =[]
        self.emb = VOKDataEmbeddingBuilder()

    def load_data(self, sample_size=100):
        data_set = self.emb.get_sample_train_data_set(sample_size=sample_size)
        print('data_set', len(data_set))
        for lats, meta_data, gt in data_set:
    # convert or clone appropriately
            if isinstance(lats, torch.Tensor):
                tensor_lats = lats.detach().clone().float()
            else:
                tensor_lats = torch.as_tensor(lats, dtype=torch.float32)
            self.latents.append(tensor_lats)

            self.meta_data.append(meta_data)

            if isinstance(gt, torch.Tensor):
                tensor_gt = gt.detach().clone().float()
            else:
                tensor_gt = torch.as_tensor(gt, dtype=torch.float32)
            self.gt.append(tensor_gt)
        self.latents = torch.stack(self.latents, dim=0)
        self.gt = torch.stack(self.gt, dim=0)

        self.data = torch.cat([self.latents, self.gt], dim=1)
        for d in self.latents:
            #make sure data is float and no nan or inf
            d = d.float()
            if torch.isnan(d).any() or torch.isinf(d).any():
                raise ValueError('nan or inf in data')
            
           
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.gt[idx]
    
    def get_meta_data(self, idx):
        return self.meta_data[idx]
    

if __name__ == '__main__':
    dataset = RawDataRetrieverDataset()
    dataset.load_data(sample_size=100)
  