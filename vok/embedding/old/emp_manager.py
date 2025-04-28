  
from dawn_vok.vok.model_utils.models.vok_model import VModel


class EMPReducer:
    def __init__(self, latent_id, model_id, model_version='1.0.0'):
        self.model_id = model_id
        self.model_version = model_version
        self.latent_id = latent_id
        self.vocab = []


    def reduce(self, vocab):
        self.vocab.extend(vocab)
        return self.vocab
    
    

class EMPManager:
    def __init__(self):
        self.emp_builders = {}
        self.reducers_model_ids = {}

    def add_emp_builder(self, builder):
        self.emp_builders[builder.param_type] = builder

    def encode(self, info_dict):
        pass

    def decode(self, latent_dict):
        pass

class EMPSyntaxVModel(VModel):
    def __init__(self, model_id, model_version='1.0.0'):
        self.model_id = model_id
        self.model_version = model_version
        self.latent_model_info = {}

        super().__init__(model_id, model_version)

    
    def load_model(self, model_id, model_version):
        pass

    def save_model(self, model_id, model_version):
        pass

    def get_model(self, model_id, model_version):
        pass
    
    
