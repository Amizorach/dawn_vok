
import torch
from dawn_vok.vok.v_objects.vok_object import VOKObject
from dawn_vok.vok.vmodel.creterion.vok_criterion import VCriterion
from dawn_vok.vok.vmodel.models.vok_optimizer import VOptimizer



class NodeInfo:
    def __init__(self, model_id, version, name, criterion=None, optimizer=None, input_dim=None, output_dim=None, model_config=None):
        self.model_id = model_id
        self.version = version
        self.name = name
        self.criterion = criterion
        self.optimizer = optimizer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_config = model_config

    def to_dict(self):
        return {
            'model_id': self.model_id,
            'version': self.version,
            'name': self.name,
            # 'criterion': self.criterion.to_dict(),
            # 'optimizer': self.optimizer.to_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'model_config': self.model_config,
        }
    
    def populate_from_dict(self, d):
        self.model_id = d.get('model_id', self.model_id)
        self.version = d.get('version', self.version)
        self.name = d.get('name', self.name)
        # self.criterion = d.get('criterion', )
        # self.optimizer = d.get('optimizer', self.optimizer.to_dict())
        self.input_dim = d.get('input_dim', self.input_dim)
        self.output_dim = d.get('output_dim', self.output_dim)
        self.model_config = d.get('model_config', self.model_config)
        return self
        
class VOKPipelineNode(VOKObject):
    # A pipeline node is a node in a pipeline. It is a model that is used to transform data.
    # It can be a model, a function, or a class.
    # It should allow self training and self saving.
    # IT can 'know' how to manage different models

    @classmethod
    def get_db_name(cls):
        return 'models'
    
    @classmethod
    def get_collection_name(cls):
        return 'pipeline_nodes'
    
    def __init__(self, node_id, name=None, node_info=None, model=None):
        uid = f'pipeline_node_{node_id}'
        super().__init__(uid=uid, obj_type='pipeline_node',  name=name)
        self.node_id = node_id
        self.name = name
        self.node_info = node_info or {}
        self.model = model
        self.trainer = None
        self.optimizer = None
        self.criterion = None
    def to_dict(self):
        return {
            'node_id': self.node_id,
            'name': self.name,
            'node_info': self.node_info,
        }
    
    def populate_from_dict(self, d):
        self.node_id = d.get('node_id', self.node_id)
        self.name = d.get('name', self.name)
        self.node_info = d.get('node_info', self.node_info)
        return self
    

    def get_model_class(self, model_id):
        raise NotImplementedError("Subclasses must implement get_model_class")
  
    def prepare_model(self, model_id=None):
        if self.model:
            return self.model
        if model_id is None:
            if len(self.model_info) == 1:
                model_id = list(self.model_info.keys())[0]
            else:
                raise ValueError("model_id is required for nodes that have multiple models")
            
        mod_info = self.model_info.get(model_id, None)
        if mod_info is None:
            raise ValueError(f"model_id {model_id} not found in model_info")
        
        model_class = self.get_model_class(model_id)
        if model_class is None:
            raise ValueError(f"model_class for model_id {model_id} not found")
        
        self.model = model_class(mod_info)
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()

  
    def get_criterion(self):
        if self.model is None:
            raise ValueError("model is not set")
        if self.criterion is None:
            self.criterion = VCriterion(self.model)
        return self.criterion
       # return self.criterion
    
    def get_optimizer(self):
        if self.model is None:
            raise ValueError("model is not set")
        if self.optimizer is None:
            self.optimizer = VOptimizer(self.model)
        return self.optimizer
        #return self.optimizer
    
  
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
        
    def save_model(self, filename=None, to_file=True, db=True):
        self.model.save_model(to_file=to_file, file_name=filename, db=db)

    def load_model(self, filename=None):
        self.model.load_model_state_dict(file_name=filename)


    
    
    
   
    
    
    
    