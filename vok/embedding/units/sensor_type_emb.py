
import numpy as np
from dawn_vok.vok.embedding.syntax_emb.syntax_emb import EmbeddedDiscreteValue


sensor_type_to_embedding = {
    'air_temperature': {
        'system_uid': 'air_temperature',
        'main_params': ['sensor_type_temperature', 'sensor_sub_type_air_temperature'],
        'extra_params': ['value_type_units_celcius', 'value_type_float', 'placement_ground'],
    },


    'humidity': {
        'system_uid': 'humidity',
        'main_params': ['sensor_humidity'],
        'extra_params': ['sensor_sub_type_humidity', 'value_type_units_percent', 'value_type_float'],
    },
    
}

class SensorTypeEmbedding:
    def __init__(self):
        self.sensor_type_to_embedding = sensor_type_to_embedding
        self.syntax_emb = EmbeddedDiscreteValue.get_embeddings(generator_id='syntax_builder')
        self.emb_cache = {}

    def get_embedding(self, sensor_type):
        if sensor_type in self.emb_cache:
            return self.emb_cache[sensor_type]
        if sensor_type not in self.sensor_type_to_embedding:
            raise ValueError(f"Sensor type {sensor_type} not found")
        emb_dict = self.sensor_type_to_embedding[sensor_type]
        main_emb = []
        for main_param in emb_dict.get('main_params', []):
            syntax_emb = self.syntax_emb.get(main_param, None)
            if syntax_emb is not None:
                emb = syntax_emb.get('embedding', {}).get('reduced_32', None)
                if emb is not None:
                    main_emb.append(emb)
        if len(main_emb) > 0:
            main_emb = np.mean(main_emb, axis=0)
        else:
            main_emb = np.zeros(32)
        emb_dict['main_params'] = main_emb
        extra_emb = []
        for extra_param in emb_dict.get('extra_params', []):
            syntax_emb = self.syntax_emb.get(extra_param, None)
            if syntax_emb is not None:
                emb = syntax_emb.get('embedding', {}).get('reduced_32', None)
                if emb is not None:
                    extra_emb.append(emb)
        if len(extra_emb) > 0:
            extra_emb = np.mean(extra_emb, axis=0)
        else:
            extra_emb = np.zeros(32)
        emb_dict['extra_params'] = extra_emb
        self.emb_cache[sensor_type] = np.concatenate([main_emb, extra_emb])
        return self.emb_cache[sensor_type]
      

if __name__ == '__main__':
    sensor_type_emb = SensorTypeEmbedding()
    print(sensor_type_emb.get_embedding('air_temperature'))
    print(sensor_type_emb.get_embedding('humidity'))