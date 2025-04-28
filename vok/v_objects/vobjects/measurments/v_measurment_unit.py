
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.vok.v_objects.vobjects.measurments.v_mes_config import MeasurementConfig
from dawn_vok.vok.v_objects.vok_object import VOKObject


class MeasurementUnit(VOKObject):
    @classmethod
    def get_db_name(cls):
        return 'meta_data'
    
    @classmethod
    def get_collection_name(cls):
        return 'measurement_unit'
    
    def __init__(self, unit, obj_type='measurement_unit', uid=None):
        if uid is None:
            uid = f'm_unit_{unit}'
        super().__init__(obj_type=obj_type, uid=uid)
        self.unit = unit
        self.symbol = None
    
    def get_vocab(self):
        vocab = self.syntax_directives
        vocab.append(f"the unit of the sensor is {self.unit}")
        return vocab
    
    def update_syntax_directives(self):
        self.syntax_directives = self.get_vocab()
        
    def to_dict(self):
        ret = super().to_dict()
        ret['unit'] = self.unit
        ret['symbol'] = self.symbol
        return ret
    
    def populate_from_dict(self, d):
        super().populate_from_dict(d)
        self.unit = d.get('unit', self.unit)
        self.symbol = d.get('symbol', self.symbol)
        self.syntax_directives = d.get('syntax_directives',self.syntax_directives)
        return self


    @classmethod
    def create_all_measurement_units(cls):
        measurement_units = []
        for key, value in MeasurementConfig.measurment_unit_info.items():
            me = MeasurementUnit(unit=key)
            me.populate_from_dict(value)
            me.update_syntax_directives()
            measurement_units.append(me)
        MongoUtils.update_many(cls.get_db_name(), cls.get_collection_name(), measurement_units)
        return measurement_units
    
    
if __name__ == '__main__':
    MeasurementUnit.create_all_measurement_units()