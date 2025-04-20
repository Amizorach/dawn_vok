

from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.vok.dobjects.dobj.config.sensor_type_conf import SensorTypeConfig
from dawn_vok.vok.v_objects.vobjects.sensors.sensor_config import SensorConfig
from dawn_vok.vok.v_objects.vok_object import VOKObject


class VOKBaseSensorType(VOKObject):
    @classmethod
    def get_db_name(cls) -> str:
        return 'meta_data'
    
    @classmethod
    def get_collection_name(cls) -> str:
        return 'sensor_type'
    
    @classmethod
    def get_all(cls, populate: bool = True):
        collection = MongoUtils.get_collection(cls.get_db_name(), cls.get_collection_name())
        li = collection.find({'obj_type': 'base_sensor_type'})
        if populate:
            return [cls(d['uid']).populate_from_dict(d) for d in li]
        return li


    
    def __init__(self, sensor_type: str, di: dict | None = None) -> None:
        if di is None:
            di = {}
        self.sensor_type: str = sensor_type
        self.physical_quantity: str | None = None
        self.allowed_units: list[str] = []
        uid = f'base_sensor_type_{sensor_type}'
        super().__init__(obj_type='base_sensor_type', uid=uid)
        if di:
            self.populate_from_dict(di)
        
    def get_vocab(self) -> list[str]:
        vocab = super().get_vocab()
        vocab.append(f"the base sensor type is {self.sensor_type}")
        return vocab
    
    def update_syntax_directives(self):
        self.syntax_directives = self.get_vocab()
        
    def to_dict(self) -> dict:
        data = super().to_dict()
        data['sensor_type'] = self.sensor_type
        data['physical_quantity'] = self.physical_quantity
        data['allowed_units'] = self.allowed_units
        return data
    
    def populate_from_dict(self, d: dict):
        super().populate_from_dict(d)
        self.sensor_type = d.get('sensor_type', self.sensor_type)
        self.physical_quantity = d.get('physical_quantity', self.physical_quantity)
        self.allowed_units = d.get('allowed_units', self.allowed_units)
        return self
    
    @classmethod
    def create_all_base_sensor_types(cls):
        base_sensor_types = []
        for st in SensorConfig.base_sensor_type_info:
            uid = st['uid']
            bst = VOKBaseSensorType(uid, st)
            bst.update_syntax_directives()
            base_sensor_types.append(bst)
        MongoUtils.update_many(cls.get_db_name(), cls.get_collection_name(), base_sensor_types)
        return base_sensor_types
    

if __name__ == '__main__':
    VOKBaseSensorType.create_all_base_sensor_types()
        