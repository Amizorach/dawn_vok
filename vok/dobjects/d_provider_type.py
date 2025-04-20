
import pprint

import numpy as np
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.id_utils import IDUtils
from dawn_vok.vok.dobjects.d_object import DObject
from dawn_vok.vok.dobjects.dobj.config.provider_type_conf import ProviderTypeConfig
from dawn_vok.vok.embedding.syntax.syntax_db_builder import SyntaxDBBuilder
from dawn_vok.vok.embedding.syntax.syntax_reducer import SyntaxEmbeddingReductionTrainer
from dawn_vok.vok.embedding.syntax_emb.syntax_emb import EmbeddedDiscreteValue


class ProviderType(DObject):
    @classmethod
    def get_db_name(cls):
        return 'meta_data'
    
    @classmethod
    def get_collection_name(cls):
        return 'provider_type'
    # {
                            # "_id": "provider_noaa_nws_api",      # Updated
                            # "uid": "noaa_nws_api",              # New
                            # "name": "NOAA / National Weather Service API (api.weather.gov)",
                            # "url": "https://www.weather.gov/",
                            # "api_url": "https://www.weather.gov/documentation/services-web-api",
                            # "meta_url": "https://weather-gov.github.io/api/general-faqs",
                            # "syntax_directives": [
                            #     "This public API delivers official forecasts and recent weather observations primarily for locations within the United States.",
                            #     "It requires application identification via User-Agent for accessing standardized US weather service data streams.",
                            #     "Access involves querying by geographic coordinates which are then mapped to a specific forecast grid and associated reporting stations.",
                            #     "The service provides real-time alerts, hourly forecasts, and current conditions directly from government meteorological sources.",
                            #     "While offering recent instrumental readings, extensive historical climate records necessitate querying separate archival systems from the same agency."
                            # ],
    def __init__(self, uid, di={}):
        
        system_uid = IDUtils.get_provider_type_id(uid)
        super().__init__(obj_type='provider_type', uid=uid, system_uid=system_uid)
        self.name = di.get('name', None)
        self.url = di.get('url', None)
        self.api_url = di.get('api_url', None)
        self.meta_url = di.get('meta_url', None)
        self.populate_from_dict(di)
      
        
    def get_vocab(self):
        ret = super().get_vocab()   
        if self.name:
            ret.append("the name of the provider is " + self.name)
        # if self.url:
        #     ret.append("the url of the provider is " + self.url)
        # if self.api_url:
        #     ret.append("the api url of the provider is " + self.api_url)
        # if self.meta_url:
        #     ret.append("the meta url of the provider is " + self.meta_url)
      
        return ret

    def to_dict(self):
        ret = super().to_dict()
        ret['_id'] = self.system_uid
        ret['name'] = self.name
        ret['url'] = self.url
        ret['api_url'] = self.api_url
        ret['meta_url'] = self.meta_url
        return ret
    
    def populate_from_dict(self, d):    
        super().populate_from_dict(d)
        self.name = d.get('name', self.name)
        self.url = d.get('url', self.url)
        self.api_url = d.get('api_url', self.api_url)
        self.meta_url = d.get('meta_url', self.meta_url)
        self.system_uid = IDUtils.get_provider_type_id(self.uid)
        return self
    
    
    @classmethod
    def create_all_provider_types(cls):
        provider_types = ProviderTypeConfig.type_info
        providers = []
        for pt in provider_types:
            providers.append(ProviderType(pt['uid'], pt).to_dict())
        MongoUtils.update_many(cls.get_db_name(), cls.get_collection_name(), providers)
        return providers

# Using a fixed timestamp for consistency, replace with datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z" for current time

    
  
    
if __name__ == '__main__':
    ProviderType.create_all_provider_types()
    vocab = ProviderType.gather_full_vocab()
    pprint.pprint(vocab)
    builder = SyntaxDBBuilder()
    builder.build_syntax_db(vocab)
    builder.save_to_db()
    # trainer = SyntaxEmbeddingReductionTrainer()
    # trainer.update_embeddings(emb_size=32, orig_scheme_id='full_embedding', out_scheme_id='reduced_32')
    # trainer.save_model()
    # trainer = SyntaxEmbeddingReductionTrainer(short_emb_size=16)
    # trainer.update_embeddings(emb_size=16, orig_scheme_id='full_embedding', out_scheme_id='reduced_16')
    # trainer.save_model()
    # "air_temperature": {
    #     "sensor_type": "temperature",
    #     "value_type": "float",
    #     "sensor_value_unit": "Celsius",
    #     "sensor_class": "environmental",
    #     "range_expected": [-50.0, 60.0],
    #     "precision": 0.1,
    #     "physical_quantity": "air_heat_content",
    #     "application_context": ["greenhouse", "outdoor", "weather_station"],
    #     "syntax_directives": [
    #         "Air temperature measures the thermal energy of the surrounding environment.",
    #         "This sensor is critical for evaluating plant growth conditions and climate control.",
    #         "It outputs values in degrees Celsius, typically ranging from -50 to 60."
    #     ]
    # },
    