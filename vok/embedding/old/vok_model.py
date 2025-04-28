import os
import torch
import torch.nn as nn
import torch.optim as optim

from dawn_vok.utils.dir_utils import DirUtils
from dawn_vok.vok.model_utils.models.vok_criterion import VCriterion
from dawn_vok.vok.model_utils.models.vok_optimizer import VOptimizer
from dawn_vok.vok.v_objects.vok_object import VOKObject



class VModel(VOKObject):
    @classmethod
    def get_db_name(cls):
        return 'models'
    
    @classmethod
    def get_collection_name(cls):
        return 'vok_models'
    
    def __init__(self, model_id, version, name, model, criterion=None, optimizer=None):
        uid = f'vok_model_{model_id}_{version}_{name}'
        super().__init__(uid=uid, obj_type='model',  meta_data={'model_id': model_id, 'version': version, 'name': name})
        self.model_id = model_id
        self.version = version
        self.name = name
        self.model = model
        self.criterion = criterion or VCriterion(self)
        self.optimizer = optimizer or VOptimizer(self)

    def get_criterion(self):
        return self.criterion
    
    def get_optimizer(self):
        return self.optimizer
    
  
    def to(self, device):
        self.device = device
        self.model.to(device)
        # self.criterion.criterion.to(device)
        # self.optimizer.to(device)

    
    def to_dict(self):
        return {
            'model_id': self.model_id,
            'version': self.version,
            'name': self.name,
          
        }
    
    def populate_from_dict(self, d):
        self.model_id = d.get('model_id', self.model_id)
        self.version = d.get('version', self.version)
        self.name = d.get('name', self.name)
    
    def forward(self, x):
        self.model.forward(x)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
        
    def save_model(self, filename, path=None):
        if path is None:
            path = DirUtils.get_model_path(model_id=self.model_id, version=self.version, path=self.name)
        pth = os.path.join(path, filename)
        torch.save(self.model.state_dict(), pth)

    def load_model(self, filename, path=None):
        if path is None:
            path = DirUtils.get_model_path(model_id=self.model_id, version=self.version, path=self.name)
        pth = os.path.join(path, filename)
        try:
            state_dict = torch.load(pth, map_location=self.device)
            self.model.load_state_dict(state_dict)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False





class SensorTypeLatentModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_types, latent_dim=16):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)
        # Embedding table for latents
        self.codes = nn.Embedding(num_types, latent_dim)

    def forward(self, input_data):
        x = input_data[0]
        type_idx = input_data[1]
        z_pred = self.encoder(x)
        z_code = self.codes(type_idx)
        return z_pred, z_code

    def expand_codes(self, new_codes: torch.Tensor):
        """
        Expand the embedding table by appending new_codes (shape: [k, latent_dim]).
        """
        old_weights = self.codes.weight.data
        updated = torch.cat([old_weights, new_codes], dim=0)
        num_new = updated.size(0)
        self.codes = nn.Embedding(num_new, updated.size(1)).to(self.codes.weight.device)
        self.codes.weight.data.copy_(updated)


class VSensorTypeLatentModel(VModel):
    def __init__(self, input_dim, hidden_dims, num_types, latent_dim=16):
        model = SensorTypeLatentModel(input_dim, hidden_dims, num_types, latent_dim)
        super().__init__(model_id='sensor_type_latent', version='1.0', name='sensor_type_latent', model=model)



if __name__ == '__main__':
    model = VSensorTypeLatentModel(input_dim=10, hidden_dims=[100, 100], num_types=10, latent_dim=16)
    print(model)
