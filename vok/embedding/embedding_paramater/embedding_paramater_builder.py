
import pprint

import numpy as np
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.utils.id_utils import IDUtils
from dawn_vok.vok.embedding.base.discrete_embedding import DiscreteEmbedding
from dawn_vok.vok.embedding.embedding_paramater.v_embedding_paramater import VOKEmbeddingParamater
from dawn_vok.vok.v_objects.vobjects.formulations.v_agg import VOKFormulation
from dawn_vok.vok.v_objects.vobjects.measurments.v_measurment_unit import MeasurementUnit
from dawn_vok.vok.v_objects.vobjects.sensors.v_base_sensor_type import VOKBaseSensorType
from dawn_vok.vok.v_objects.vobjects.sensors.v_sensor_type import VOKSensorType
from dawn_vok.vok.v_objects.vobjects.source.v_source import VOKSource
from dawn_vok.vok.v_objects.vok_object import VOKObject


class EmbeddingParamaterBuilder:
    def __init__(self, param_type):
        self.param_type = param_type
        self.embeddings = {}


    def build(self):
        if self.param_type == 'sensor_type':
            return self.create_sensor_type_embedding_paramaters()
        elif self.param_type == 'source':
            return self.build_source_embedding_paramaters()
        elif self.param_type == 'formulation':
            return self.build_formulation_embedding_paramaters()
        print('Invalid param type ', self.param_type)
        return None
    
    def load_embeddings(self):
        if not self.embeddings:
            embeddings = DiscreteEmbedding.get_all()
            self.embeddings = {e.emb_id: e for e in embeddings}
        return self.embeddings
    
    def create_sensor_type_embedding_paramaters(self):
        latents = {
            'measurement_unit': None,
            'base_sensor_type': None,
            'sensor_type': None,
            'sensor_info': None,
        }
        emp_list = []
        mes = MeasurementUnit.get_all()
        base_sensor_types = VOKBaseSensorType.get_all()
        sensor_types = VOKSensorType.get_all()
        self.load_embeddings()
        mes = {m.uid: m for m in mes}
        base_sensor_types = {b.sensor_type: b for b in base_sensor_types}
        sensor_types = {s.sensor_type: s for s in sensor_types}
        for st, sensor_type in sensor_types.items():
            bst = base_sensor_types.get(sensor_type.base_sensor_type, None)
            if bst:
                lat = self.embeddings.get(bst.get_id(), None)
                if lat:
                    latents['base_sensor_type'] = lat.latent_schemes
            mest = mes.get(sensor_type.unit, None)
            if mest:
                lat = self.embeddings.get(mest.get_id(), None)
                if lat:
                    latents['measurement_unit'] = lat.latent_schemes
           
            st_lat = self.embeddings.get(sensor_type.get_id(), None)
            if st_lat:
                latents['sensor_type'] = st_lat.latent_schemes
            emp = VOKEmbeddingParamater(param_type='sensor_type', uid=sensor_type.get_id(),
                                        param_id=sensor_type.uid, latents=latents)
            emp_list.append(emp)
        return emp_list
    @classmethod
    def save_to_db(cls, emp_list):
        MongoUtils.update_many(db_name=VOKEmbeddingParamater.get_db_name(), collection_name=VOKEmbeddingParamater.get_collection_name(), data=emp_list)



    
    
class SourceEMPBuilder(EmbeddingParamaterBuilder):
    def __init__(self):
        super().__init__('source')
        self.latents = {
            'source': None,
        }
        self.emp_list = []

    def build(self):
        emp_list = []
        self.load_embeddings()
        sources = VOKSource.get_all()
        for source in sources:
            lat = self.embeddings.get(source.source_id, None)
            if lat:
                self.latents['source'] = lat.latent_schemes
                slat = DictUtils.parse_value(source.static_latent_ids, 'lat_16', np.zeros(16))
                self.static_latent_id = slat
                emp = VOKEmbeddingParamater(param_type='source', uid=source.get_id(), param_id=source.uid, 
                                            latents=self.latents.copy(), static_latent_id=self.static_latent_id)
                emp_list.append(emp)
            else:
                print("no latent for ", source.get_id(), source.uid)
                print(source.to_dict())
                print(list(self.embeddings.keys()))
                exit()
            print(emp.latents.get('source', {}).get('mean', [])[0:10])
        self.emp_list = emp_list

        return self.emp_list
    
    def load_emps(self):
        if not self.emp_list:
            self.emp_list = list(MongoUtils.get_collection(db_name=VOKEmbeddingParamater.get_db_name(), 
                                                      collection_name=VOKEmbeddingParamater.get_collection_name()).find({'param_type': 'source'}))
        return self.emp_list
    
    def get_vocab(self):
        # self.load_emps()
        print(len(list(self.emp_list)))
        vocab = []
        for emp in self.emp_list:
            vocab.append(emp.latents.get('source', {}).get('mean', []))
        pprint.pp(len(vocab))
        return vocab
    
class FormulationEMPBuilder(EmbeddingParamaterBuilder):
    def __init__(self):
        super().__init__('formulation')
        self.latents = {
            'formulation': None,
        }
    
    def build(self):
        emp_list = []
        print('Building formulation embedding paramaters')
        self.load_embeddings()
        formulations = VOKFormulation.get_all()
        print(len(formulations))
        for formulation in formulations:
            lat = self.embeddings.get(formulation.get_id(), None)
            if lat:
                self.latents['formulation'] = lat.latent_schemes
                
            emp = VOKEmbeddingParamater(param_type='formulation', uid=formulation.get_id(), param_id=formulation.uid,
                                         latents=self.latents.copy())
            emp_list.append(emp)

            pprint.pp(emp.latents.get('formulation', {}).get('mean', [])[0:10])
        self.emp_list = emp_list
        for emp in self.emp_list:
            print(emp.latents.get('formulation', {}).get('mean', [])[0:10])
        return self.emp_list     

class SensorTypeEMPBuilder(EmbeddingParamaterBuilder):
    def __init__(self):
        super().__init__('sensor_type')
        self.latents = {
            'measurement_unit': None,
            'base_sensor_type': None,
            'sensor_type': None,
            'sensor_info': None,
        }
    def build(self):
        
        emp_list = []
        mes = MeasurementUnit.get_all()
        base_sensor_types = VOKBaseSensorType.get_all()
        sensor_types = VOKSensorType.get_all()
        self.load_embeddings()
        mes = {m.uid: m for m in mes}
        base_sensor_types = {b.sensor_type: b for b in base_sensor_types}
        sensor_types = {s.sensor_type: s for s in sensor_types}
        for st, sensor_type in sensor_types.items():
            bst = base_sensor_types.get(sensor_type.base_sensor_type, None)
            if bst:
                lat = self.embeddings.get(bst.get_id(), None)
                if lat:
                    self.latents['base_sensor_type'] = lat.latent_schemes
            ut = f'm_unit_{sensor_type.unit}'.replace(' ', '_').replace('-', '_').replace('/', '_').lower()
            mest = mes.get(ut, None)
            if mest:
                lat = self.embeddings.get(mest.get_id(), None)
                if lat:
                    self.latents['measurement_unit'] = lat.latent_schemes
            else:
                print('No measurement unit found for ', sensor_type.unit)
              
            st_lat = self.embeddings.get(sensor_type.get_id(), None)
            if st_lat:
                self.latents['sensor_type'] = st_lat.latent_schemes
            emp = VOKEmbeddingParamater(param_type='sensor_type', uid=sensor_type.get_id(), param_id=sensor_type.uid, 
                                        latents=self.latents.copy())
            emp_list.append(emp)
        self.emp_list = emp_list
        return self.emp_list
    
   
 
if __name__ == '__main__':
    # MongoUtils.get_collection(db_name='embeddings', collection_name='embedding_paramaters').delete_many({})
    fl = []
    eb = SensorTypeEMPBuilder()
    eb2 = FormulationEMPBuilder()
    eb2.build()
    pprint.pp(eb2.emp_list)
    for emp in eb2.emp_list:
        print(emp.latents.get('formulation', {}).get('mean', [])[0:10])
    MongoUtils.update_many(db_name='embeddings', collection_name='embedding_paramaters', data=eb2.emp_list)
    
    # fl.extend(eb.build())
    eb1 = SourceEMPBuilder()
    eb1.build()
    for emp in eb1.emp_list:
        print(emp.latents.get('source', {}).get('mean', [])[0:10])
    MongoUtils.update_many(db_name='embeddings', collection_name='embedding_paramaters', data=eb1.emp_list)
    # SourceEMPBuilder.save_to_db(eb1.emp_list)
    
    fl.extend(eb2.build())
    eb3 = SensorTypeEMPBuilder()
    eb3.build()
    for emp in eb3.emp_list:
        print(emp.latents.get('sensor_type', {}).get('mean', [])[0:10])
    MongoUtils.update_many(db_name='embeddings', collection_name='embedding_paramaters', data=eb3.emp_list)
    exit(1)
    fl.extend(eb3.build())
    MongoUtils.update_many(db_name='embeddings', collection_name='embedding_paramaters', data=fl)
