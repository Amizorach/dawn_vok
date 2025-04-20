

from dawn_vok.vok.v_objects.vobjects.formulations.v_formulation_config import FormulationConfig
from dawn_vok.vok.v_objects.vok_object import VOKObject


class VOKFormulation(VOKObject):
    @classmethod
    def get_collection_name(cls):
        return 'formulation'
    
    @classmethod
    def get_db_name(cls):
        return 'meta_data'
    
    def __init__(self, uid: str, formulation_type='agg', formulation_id='mean',
                 system_uid: str = None, meta_data: dict = None, name: str = None, syntax_directives: list = None):
        super().__init__(uid, obj_type='formulation', system_uid=system_uid, meta_data=meta_data, name=name, syntax_directives=syntax_directives)
        self.formulation_type = formulation_type
        self.formulation_id = formulation_id
        

    def to_dict(self):
        data = super().to_dict()
        data['formulation_type'] = self.formulation_type
        data['formulation_id'] = self.formulation_id
        return data
    
    def populate_from_dict(self, di: dict):
        super().populate_from_dict(di)
        self.formulation_type = di.get('formulation_type', self.formulation_type)
        self.formulation_id = di.get('formulation_id', self.formulation_id)
        return self

    @classmethod
    def create_all_formulations(cls):
        for formulation in FormulationConfig.aggregation_functions_info:
            uid = f'agg_{formulation["key"]}'
            formulation = cls(uid=uid, formulation_type='agg', formulation_id=formulation['key'], name=formulation['name'], syntax_directives=formulation['syntax_directives'])
            formulation.save_to_db()

if __name__ == '__main__':
    VOKFormulation.create_all_formulations()
