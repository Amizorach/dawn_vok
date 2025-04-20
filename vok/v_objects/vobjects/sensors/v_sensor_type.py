
import pprint

import numpy as np
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.id_utils import IDUtils
from dawn_vok.vok.dobjects.d_object import DObject
from dawn_vok.vok.dobjects.dobj.config.sensor_type_conf import SensorTypeConfig
from dawn_vok.vok.v_objects.vobjects.sensors.sensor_config import SensorConfig
from dawn_vok.vok.v_objects.vok_object import VOKObject
# from dawn_vok.vok.embedding.syntax.syntax_db_builder import SyntaxDBBuilder

class VOKSensorType(VOKObject):
    @classmethod
    def get_db_name(cls):
        return 'meta_data'
    
    @classmethod
    def get_collection_name(cls):
        return 'sensor_type'
    
    def __init__(self, uid, di={}):
        self.base_sensor_type = di.get('base_sensor_type', None)
        self.sensor_type = di.get('sensor_type', None)
        self.unit = di.get('unit', None)
        self.physical_quantity = di.get('physical_quantity', None)
        self.value_type = di.get('value_type', None)
        provider_type = di.get('provider_type', None)   
        system_uid = IDUtils.get_sensor_type_id(provider_type)
        super().__init__(obj_type='sensor_type', uid=uid, system_uid=system_uid)
        self.populate_from_dict(di)

    
      
        
    def get_vocab(self):
        ret = super().get_vocab()   
        if self.value_type:
            ret.append("the value type of the sensor is " + self.value_type)
        if self.base_sensor_type:
            ret.append("the base sensor type of the sensor is " + self.base_sensor_type)
        if self.sensor_type:
            ret.append("the sensor type of the sensor is " + self.sensor_type)
        if self.unit:
            ret.append("the unit of the sensor is " + self.unit)
        if self.physical_quantity:
            ret.append("the physical quantity of the sensor is " + self.physical_quantity)
        return ret

    def update_syntax_directives(self):
        self.syntax_directives = self.get_vocab()
        
    def to_dict(self):
        ret = super().to_dict()
        ret['base_sensor_type'] = self.base_sensor_type
        ret['sensor_type'] = self.sensor_type
        ret['unit'] = self.unit
        ret['physical_quantity'] = self.physical_quantity
        ret['value_type'] = self.value_type
        return ret
    
    def populate_from_dict(self, d):    
        super().populate_from_dict(d)
        self.value_type = d.get('value_type', self.value_type)
        self.base_sensor_type = d.get('base_sensor_type', self.base_sensor_type)
        self.sensor_type = d.get('sensor_type', self.sensor_type)
        self.unit = d.get('unit', self.unit)
        self.physical_quantity = d.get('physical_quantity', self.physical_quantity)
        return self
    
    
    @classmethod
    def create_all_sensor_types(cls):
        sensor_types = []
        for st in SensorTypeConfig.type_info:
            uid = st['uid']
            st = SensorType(uid, st)
            st.update_syntax_directives()
            sensor_types.append(st)


        MongoUtils.update_many(cls.get_db_name(), cls.get_collection_name(), sensor_types)
        return sensor_types
    
    @classmethod
    def gather_full_vocab(cls):
        sensor_types = cls.get_all()
        print(sensor_types)
        vocab = []
        for st in sensor_types:
            vocab.extend(st.get_vocab())
        vocab = list(set(vocab))
        return vocab
    


if __name__ == "__main__":
    SensorType.create_all_sensor_types()

  
