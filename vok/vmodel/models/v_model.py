import torch
import torch.nn as nn
import os
import json
from dawn_vok.utils.dir_utils import DirUtils
from dawn_vok.vok.v_objects.vok_object import VOKObject

class VOKModel(nn.Module, VOKObject):
    @classmethod
    def get_collection_name(cls):
        return 'vok_models'
    
    @classmethod
    def get_db_name(cls):
        return 'models'
  
    
    def __init__(self, model_id=None, version=None, name=None, model_config=None, latent_db=None, model_path=None, model_file_name=None, config_file_name=None):
        uid = f'model_{model_id}_{version}'
        nn.Module.__init__(self)    
        VOKObject.__init__(self, uid=uid, obj_type='model')
        self.model_id = model_id
        self.version = version
        self.name = name
        self.model_path = model_path
        self.model_file_name = model_file_name or 'model.pt'
        self.config_file_name = config_file_name or 'model_config.json'
        self.model_config = self.get_base_config()
        if model_config is not None:
            self.model_config.update(model_config)
        self.directory_path = DirUtils.get_model_path(model_id=self.model_id, version=self.version, path=model_path)

    def to_dict(self):
        ret = super().to_dict()
        ret['model_config'] = self.model_config
        ret['model_id'] = self.model_id
        ret['version'] = self.version
        ret['config_file_name'] = self.config_file_name
        ret['model_path'] = self.model_path
        ret['model_file_name'] = self.model_file_name
        ret['structure'] = self.get_module_structure()
        return ret
    
    def get_module_structure(self, module=None):
        """
        Recursively traverses a PyTorch module and builds a dictionary 
        representing its structure, including layer types and key parameters.
        """
        structure = {}
        if module is None:
            module = self
        module_type_name = type(module).__name__
        structure['type'] = module_type_name
        
        # --- Try to extract common identifying parameters ---
        params = {}
        # List common parameters found in various layers
        common_param_names = [
            'in_features', 'out_features', 'in_channels', 'out_channels', 
            'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 
            'num_features', 'eps', 'momentum', # BatchNorm/LayerNorm/InstanceNorm
            'p', # Dropout
            'hidden_size', 'num_layers', 'batch_first', 'bidirectional' # RNNs/Transformers (subset)
            # Add others as needed, e.g., dim, num_heads for attention
        ] 
        
        for p_name in common_param_names:
            if hasattr(module, p_name):
                value = getattr(module, p_name)
                # Avoid trying to serialize large tensors like weights/full bias vectors.
                # Capture boolean bias flag or essential configuration tuples/ints.
                if isinstance(value, torch.Tensor):
                    # Special case: if bias is a tensor, just record its presence (usually True)
                    if p_name == 'bias' and value is not None:
                        params[p_name] = True
                    # else: could record shape like f"Tensor{tuple(value.shape)}"
                elif value is not None: 
                    # Make sure boolean False is captured (e.g., bias=False)
                    if isinstance(value, bool) or value is not False:
                        params[p_name] = value
                elif p_name == 'bias' and value is None: # Explicitly record bias=None if attr exists
                    params[p_name] = None


        if params: # Only add 'params' key if we found any
            structure['params'] = params
                
        # --- Recursively handle children ---
        children_list = [] # Use a list for ordered modules like Sequential
        children_dict = {} # Use a dict for named modules
        has_named_children = False

        for name, child in module.named_children():
            has_named_children = True
            child_struct = self.get_module_structure(child)
            # If the module looks like a container with numeric names (like Sequential), use a list
            if name.isdigit(): 
                children_list.append(child_struct)
            else:
                children_dict[name] = child_struct

        # Decide how to store children based on naming convention found
        if children_dict: # If we found any named children (non-numeric names)
            structure['children'] = children_dict
            # If there were also numeric ones (mixed case, unlikely), add them too?
            # if children_list: 
            #    structure['children']['_sequential_children'] = children_list
        elif children_list: # Only numeric names found (like Sequential)
            structure['children'] = children_list
        # If no children, the 'children' key is omitted.
                
        return structure
    
    def populate_from_dict(self, d):
        super().populate_from_dict(d)
        self.model_id = d.get('model_id', self.model_id)
        self.version = d.get('version', self.version)
        self.model_path = d.get('model_path', self.model_path)
        self.model_config = d.get('model_config', self.model_config)
        self.config_file_name = d.get('config_file_name', self.config_file_name)
        self.directory_path = DirUtils.get_model_path(model_id=self.model_id, version=self.version, path=self.model_path)
        self.model_file_name = d.get('model_file_name', self.model_file_name)
        self.structure = d.get('structure', self.structure)
        return self
    
    def get_config_path(self):
        config_path = os.path.join(self.directory_path, self.config_file_name)
        return config_path
    
    
    def load_model_config(self):
        config_path = self.get_config_path()
        model_config = DirUtils.load_json(config_path)
        return model_config
    
    def save_model_config(self):
        config_path = self.get_config_path()
        model_config = self.model_config
        try:
            json.dump(model_config, open(config_path, 'w'))
        except Exception as e:
            print(f"Error saving model config to {config_path}: {e}")
    
    def load_layer_state_dict(self, layer, file_name, device='cpu'):
        if not layer or not file_name:
            return False
        model_path = os.path.join(self.directory_path, file_name)
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                layer.load_state_dict(state_dict)
                print(f"Model weights loaded from {model_path}")
                return True
            except Exception as e:
                print(f"Error loading model weights from {model_path}: {e}. Model weights not loaded.")
                return False
        else:
            print(f"Warning: Model weights file not found at {model_path}. Model weights not loaded.")
            return False
    def load_model_state_dict(self, file_name=None, device='cpu'):
        return self.load_layer_state_dict(self, file_name or self.model_file_name, device)
        
    def save_model_state_dict(self, file_name=None):

        
        return self.save_layer_state_dict(self, file_name or self.model_file_name)

    def save_layer_state_dict(self, layer, file_name):
        if not layer or not file_name:
            return False
        model_path = os.path.join(self.directory_path, file_name)
        os.makedirs(self.directory_path, exist_ok=True)
        torch.save(layer.state_dict(), model_path)
        print(f"Model weights saved to {model_path}")
        return True

    def save_model(self, to_file=True, file_name=None, db=False):
        if to_file:
            self.save_model_config()
            self.save_model_state_dict(file_name)
        if db:
            self.save_to_db()

  
  