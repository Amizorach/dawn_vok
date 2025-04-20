import pprint
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.utils.id_utils import IDUtils
from dawn_vok.vok.embedding.system_uid.system_uid_emb import SystemUIDEmbedding
from dawn_vok.vok.v_objects.vobjects.location.v_location import VOKLocation
from dawn_vok.vok.v_objects.vok_object import VOKObject


class VOKSource(VOKObject):
    @classmethod
    def get_db_name(cls):
        return 'meta_data'
    
    @classmethod
    def get_collection_name(cls):
        return 'source'
    
    @classmethod
    def get_by_source_id(cls, source_id):
        return MongoUtils.get_collection(db_name=cls.get_db_name(), collection_name=cls.get_collection_name()).find_one({'source_id': source_id})
    
    def __init__(self,  source_id=None, name=None, provider_id=None, source_type='weather_station', location=None):
        uid = IDUtils.get_id(['ds', source_id])
        super().__init__(uid=uid, obj_type='source', name=name, syntax_directives=[])
        self.provider_id = provider_id
        self.source_id = source_id
        self.source_type = source_type
        self.static_latent_ids = {}
        self.location = None
        if location:
            self.parse_location(location)
        
    def parse_location(self, location):

        if not location:
            return
        loc = None
        try:
            if isinstance(location, dict):
                lat = location.get('lat', None)
                lon = location.get('lon', None)
                if lat and lon:
                    loc = [(lat, lon)]
            elif isinstance(location, list):
                loc = location
            elif isinstance(location, str):
                split_loc = location.split(',')
                loc = [(float(split_loc[0]), float(split_loc[1]))]
        except Exception as e:
            print(f"Error parsing location: {e}")
            return None
        vloc = VOKLocation.get_or_create_by_lat_lon(loc)
        vloc = list(vloc.values())[0]
        self.location = vloc

    def get_static_latent_id(self, dim=16):
        d = 'lat_' + str(dim)   
        if d not in self.static_latent_ids:
            self.static_latent_ids[d] = SystemUIDEmbedding.generate_embedding(self.system_uid, dim)
        return self.static_latent_ids[d]
        
    def to_dict(self):
        ret = super().to_dict()
        ret['provider_id'] = self.provider_id
        ret['source_id'] = self.source_id
        ret['source_type'] = self.source_type   
        ret['static_latent_ids'] = self.static_latent_ids
        if self.location:
            ret['location'] = self.location.to_dict() if not isinstance(self.location, dict) else self.location
        ret = DictUtils.np_to_list(ret)
        return ret
    
    def populate_from_dict(self, di):
        self.provider_id = di.get('provider_id', self.provider_id)
        self.source_id = di.get('source_id', self.source_id)
        self.source_type = di.get('source_type', self.source_type)
        self.static_latent_ids = di.get('static_latent_ids', self.static_latent_ids)
        location = di.get('location', self.location)
        if location:
            self.location = VOKLocation(uid=location['uid'], di=location)
        return self
    
    def get_source_type_syntax(self):
        if self.source_type == 'weather_station':
            return f"Data comes form a Weather Station, Weather station is a device that measures weather conditions such as temperature, humidity, wind speed, and precipitation."
        elif self.source_type == 'weather_forecast':
            return f"Data comes form a Weather Forecast, Weather forecast is a prediction of weather conditions for a specific period of time."
        elif self.source_type == 'weather_alert':
            return f"Data comes form a Weather Alert, Weather alert is a warning of weather conditions that are expected to occur."
        else:
            return f"the source type is {self.source_type}"
        
    def get_vocab(self):
        ret = super().get_vocab()
        ret.append(f"the source id is {self.source_id}")
        ret.append(f"the source name is {self.name}")
        ret.append(f"the provider id is {self.provider_id or 'unknown'}")
        ret.append(self.get_source_type_syntax())

        return ret
    
    def update_syntax(self):
        self.syntax_directives = self.get_vocab()


if __name__ == '__main__':
    # MongoUtils.get_collection(db_name='meta_data', collection_name='source').delete_many({})
    sources = MongoUtils.get_collection(db_name='meta_data', collection_name='source_md').find({})
    for s in sources:
        loc = s.get('location', None)
        lt = loc['lat']
        ln = loc['lon']
        loc['lat'] = ln
        loc['lon'] = lt
        # s = VOKSource(source_id=s['source_id'], name=s['source_name'], provider_id=s['provider_id'], location=loc)
        # s.parse_location(loc)
        # s.get_static_latent_id()
        # s.update_syntax()
        # s.save_to_db()
        s = VOKSource.get_by_source_id(s['source_id'])
        # if s:
        #     s.parse_location(loc)
        #     s.get_static_latent_id()
        #     s.update_syntax()
        #     s.save_to_db()
        pprint.pp(s)
    exit()
