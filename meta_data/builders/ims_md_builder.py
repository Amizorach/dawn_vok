from dawn_vok.meta_data.sensor.sensor_md import SensorMD
from dawn_vok.meta_data.provider.provider_md import ProviderMD
from dawn_vok.meta_data.source.source_md import SourceMD

class IMSMDBuilder:
    @staticmethod
    def build_provider_md():
        provider_md = ProviderMD(provider_id='ims', provider_name='IMS', provider_type='env_data', provider_url='https://ims.com', provider_api_key='ims_api_key')
        provider_md.save_to_db()

        return provider_md
    
    @staticmethod
    def build_source_md(source_id, source_name):
        source_md = SourceMD(source_id=source_id, source_name=source_name, source_type='ims_station', provider_id='ims')
        source_md.save_to_db()
        return source_md
    
    @staticmethod
    def build_sensor_md(sensor_id, sensor_name, source_id):
        sensor_md = SensorMD(sensor_id=sensor_id, sensor_name=sensor_name, sensor_type='temperature', source_id=source_id)
        sensor_md.save_to_db()
        return sensor_md
  

if __name__ == '__main__':
    IMSMDBuilder.build_provider_md()
    # IMSMDBuilder.build_source_md('ims_station_1', 'IMS Station 1')
    # IMSMDBuilder.build_sensor_md('temperature_1', 'Temperature 1', 'ims_station_1')
