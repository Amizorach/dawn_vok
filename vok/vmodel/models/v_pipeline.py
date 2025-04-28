from dawn_vok.vok.v_objects.vok_object import VOKObject
from dawn_vok.vok.vmodel.pipeline.v_pipeline_node import VOKPipelineNode

class VOKPipeline(VOKObject):
    @classmethod
    def get_collection_name(cls):
        return 'pipelines'
    
    @classmethod
    def get_db_name(cls):
        return 'models'
    
    def __init__(self, pipeline_id, nodes=None):
        uid = f'pipeline_{pipeline_id}'
        super().__init__(uid=uid, obj_type='pipeline')
        self.nodes = nodes

    def to_dict(self):
        ret = super().to_dict()
        ret['nodes'] = [node.uid for node in self.nodes]
        return ret
    
    def populate_from_dict(self, d):
        super().populate_from_dict(d)
        self.nodes = [VOKPipelineNode(node_id=node_id) for node_id in d['nodes']]
        return self

class EmpSytaxPipeline(VOKPipeline):
    def __init__(self, model_id=None, version=None, name=None, model_config=None, latent_db=None):
        super().__init__(model_id=model_id, version=version, name=name, model_config=model_config, latent_db=latent_db)
        self.load_nodes()

    def load_nodes(self):
        self.node = VOKPipelineNode(node_id=self.node_id)   


