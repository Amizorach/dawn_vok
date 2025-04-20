from db.mongo_utils import MongoUtils

class SensorMD(MetaData):
    

    def __init__(self, sensor_id=None, sensor_name=None, sensor_type=None, sensor_unit=None, sensor_scale=None, provider_id=None):
        super().__init__(db_name='meta_data', collection_name='sensor_md', md_id=sensor_id, md_type='sensor', name=sensor_name)
        self.sensor_type = sensor_type
        self.sensor_unit = sensor_unit
        self.sensor_scale = sensor_scale
        self.provider_id = provider_id
        self.source_id = source_id

    def populate_from_dict(self, data):
        super().populate_from_dict(data)
        self.sensor_type = data.get('sensor_type', self.sensor_type)
        self.sensor_unit = data.get('sensor_unit', self.sensor_unit)
        self.sensor_scale = data.get('sensor_scale', self.sensor_scale)
        self.provider_id = data.get('provider_id', self.provider_id)
        self.source_id = data.get('source_id', self.source_id)
        return self
    
    def to_dict(self):
        ret = super().to_dict()
        ret['sensor_type'] = self.sensor_type
        ret['sensor_unit'] = self.sensor_unit
        ret['sensor_scale'] = self.sensor_scale
        ret['provider_id'] = self.provider_id
        ret['source_id'] = self.source_id
        return ret
    


if __name__ == '__main__':
    sensor_md = SensorMD(sensor_id='1', sensor_name='air_temperature', sensor_type='temperature', sensor_unit='C', sensor_scale=[-10,50], provider_id='1')
    sensor_md.save_to_db()
    print(list(sensor_md.get_all_sensors()))

