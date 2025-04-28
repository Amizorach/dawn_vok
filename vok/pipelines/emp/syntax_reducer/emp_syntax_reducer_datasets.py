from pprint import pprint
import numpy as np
import torch
from torch.utils.data import Dataset
from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.vok.embedding.embedding_paramater.v_embedding_paramater import VOKEmbeddingParamater
from dawn_vok.vok.v_dataset.v_dataset import VOKDataset


class EMPSyntaxReducerDataset(VOKDataset):
    def __init__(self, latent_key='base', pad_length=10, device=None, dataset_id=None, name=None):
        """
        Args:
            data_list (list): List of dictionaries with key 'data' that holds a tensor data (or list) of shape (x, 384).
            pad_length (int): Target number of rows for each sample, default is 10.
        """
        dataset_id = dataset_id or 'emp_syntax_reducer_dataset'
        name = name or 'EMPSyntaxReducerDataset'
        super().__init__(dataset_id=dataset_id, name=name)
        self.pad_length = pad_length
        self.full_latent_list = []
        self.data = []
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_paramaters = None
        self.latent_key = latent_key
        self.latents_di = {}
        self.meta_data = []

        self.load_data()
        # self.add_dirty_data(noise_level=0.3, num_samples=10)
        self.prepare_data()

    def prepare_data(self):
        self.data = []
        for d in self.full_latent_list:
            
            raw_x = torch.tensor(d, dtype=torch.float32)  # Shape: (x, 384)
            padded_x, mask = self.pad_and_create_mask(raw_x, self.pad_length)
            if padded_x is None:
                continue
            # Here we set the target as the padded_x as well (for reconstruction tasks)
            y = padded_x.clone()
            self.data.append((padded_x.to(self.device), y.to(self.device), mask.to(self.device)))
        
        # for d in self.data:
        #     print(d[0].shape)
        # print('self.data', len(self.data))
        # for i in range(len(self.data)):
        #     print(self.meta_data[i]['latent_map'], self.data[i][0].shape)

    def add_dirty_data(self, noise_level=0.1, num_samples=10):
        dirty_data = []
        for d in self.data:
            for i in range(num_samples):
                dd = d[0].clone()
                #add noise to the data
                dd += torch.randn_like(dd) * noise_level
                dirty_data.append((dd.to(self.device), d[1].to(self.device), d[2].to(self.device)))
        
        self.data.extend(dirty_data)
        return dirty_data
   
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # print('self.data[idx][0].shape', self.data[idx][0].shape)
        # Returns a tuple: (input, target, mask)
        return self.data[idx][0], self.data[idx][1], self.data[idx][2]
    

    def load_data(self):
        if not self.embedding_paramaters:
            ep = VOKEmbeddingParamater.get_all()
            self.embedding_paramaters = ep
        self.full_latent_list = []
        self.latents_di = {}
        self.meta_data = []
        for emp in self.embedding_paramaters:
            for latent_id, latent_keys in emp.latents.items():
                if not latent_keys or not isinstance(latent_keys, dict):
                    continue
                if not self.latent_key or self.latent_key not in latent_keys:
                    continue

                # for latent_key, latent_value in latent_keys.items():
                #     if not latent_value or not isinstance(latent_value, list):
                #         continue
                #     if latent_key == 'static_latent_id':
                #         continue
                latent_value = emp.get_latent(latent_id, self.latent_key)
                if latent_value:
                    self.latents_di[f'{emp.uid}_{latent_id}'] = latent_value
                    if isinstance(latent_value, list):
                        latent_value = np.array(latent_value)

                    self.full_latent_list.append(latent_value)
                    self.meta_data.append({
                        'latent_map': f'{emp.uid}_{latent_id}',
                        'latent_key': self.latent_key,
                        'latent_id': latent_id,
                        'uid': emp.uid
                    })
        return self.full_latent_list
    
    def get_static_latent_ids(self, as_dict=False):
        self.static_latent_ids = {}
        for emp in self.embedding_paramaters:
            if emp.static_latent_id:
                self.static_latent_ids[emp.uid] = emp.static_latent_id or np.zeros(16)
        if as_dict:
            return self.static_latent_ids
        else:
            return list(self.static_latent_ids.values())
    
    def update_static_latent_ids(self, static_latent_ids_di, override=True):
        for emp in self.embedding_paramaters:
            needs_update = False
            if emp.uid in static_latent_ids_di:
                if override or not emp.static_latent_id:
                    emp.static_latent_id = static_latent_ids_di[emp.uid]
                    needs_update = True
            elif not emp.static_latent_id:
                emp.static_latent_id = np.zeros(16)
                needs_update = True
            if needs_update:
                emp.save_to_db()
    
    def get_latents(self, as_dict=False):
        if as_dict:
            return self.latents_di
        else:
            return self.full_latent_list
        
        
    def update_latents(self, latents_di, key, override=True):
        if  latents_di is None or not key:
            print('No latents or key provided')
            return
        for i, emp in enumerate(self.embedding_paramaters):
            needs_update = False
            
            emp_lats = latents_di.get(emp.uid, {})
            for latent_id, lat in emp.latents.items():
                if lat is None:
                    continue
                print(lat.keys())
                print('latent_id', latent_id)
                tag = f'{emp.uid}_{latent_id}'
                lat = latents_di.get(tag, None)
                if lat is None:
                    continue
                if override or not emp.get_latent(latent_id, key):
                    emp.update_latent(latent_id, key, lat.tolist())
                    if emp.param_type == 'sensor_type' and latent_id == 'sensor_type' and key == 'lat_16':

                        emp.static_latent_id = lat.tolist()
                        pprint(emp.static_latent_id)
                    elif emp.param_type == 'formulation' and latent_id == 'formulation' and key == 'lat_16':
                        emp.static_latent_id = lat.tolist()
                        pprint(emp.static_latent_id)
                    needs_update = True
            if needs_update:
                print('saving emp', emp.uid)
                emp.save_to_db()
    def get_meta_data(self):
        return self.meta_data
    
    def get_meta_data_index(self, index):
        return self.meta_data[index]
    
    def get_meta_data_index_by_latent_map(self, latent_map):
        for i, md in enumerate(self.meta_data):
            if md['latent_map'] == latent_map:
                return i
        return None
    
class EMPSyntaxReducerTrainDataset(EMPSyntaxReducerDataset):
    def __init__(self, latent_key='base', pad_length=10, device=None):
        super().__init__(latent_key, pad_length, device)
        self.add_dirty_data(noise_level=0.3, num_samples=10)
            
class EMPSyntaxReducerInferenceDataset(EMPSyntaxReducerDataset):
    def __init__(self, pad_length=10, device=None):
        super().__init__(dataset_id='emp_syntax_reducer_inference_dataset', 
                         name='EMPSyntaxReducerInferenceDataset', pad_length=pad_length, device=device)
    

    def __getitem__(self, idx):
        return self.data[idx][0], self.meta_data[idx]
    
    