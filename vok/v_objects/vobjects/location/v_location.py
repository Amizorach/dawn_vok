


import pprint

import requests
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.providers.google_earth.ge_location import GELocationUtils
from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.utils.id_utils import IDUtils
from dawn_vok.vok.v_objects.vok_object import VOKObject


class VOKLocation(VOKObject):
    @classmethod
    def get_db_name(cls):
        return 'meta_data'
    
    @classmethod
    def get_collection_name(cls):
        return 'location'

    @classmethod
    def get_by_lat_lon(cls, lat, lon):
        ret = cls.get_or_create_by_lat_lon([(lat, lon)])
        if ret:
            return list(ret.values())[0]
        return None

    @classmethod
    def get_or_create_by_lat_lon(cls, latlon_list):
        need_create = []
        locs = {}
        
        for lat, lon in latlon_list:
            loc = cls.get_by_id(IDUtils.get_id(['loc', lat, lon]))
            if not loc:
                need_create.append((lat, lon))
            else:
                locs[loc.uid] = loc
        if need_create:
            data = GELocationUtils.batch_process_locations(need_create)
            pprint.pp(data)
            for loc in data:
                uid = IDUtils.get_id(['loc', loc['lat'], loc['lon']])
                pprint.pp(loc)
                loc = VOKLocation(uid=uid, di=loc)
                loc.get_textual_context()
                loc.save_to_db()
                locs[loc.uid] = loc
        return locs


    def __init__(self, uid, lat=None, lon=None, di={}):
        super().__init__(obj_type='location', uid=uid)
        self.lat = lat
        self.lon = lon
        self.elevation = None
        self.slope = None
        self.aspect = None
        self.land_cover = None
        self.land_cover_label = None
        self.soil = None
        self.place_name = None
        self.osm_type = None
        self.address_type = None
        self.osm_category = None
        self.address = None
        self.populate_from_dict(di)


    def populate_from_dict(self, di):
        super().populate_from_dict(di)
        self.lat = di.get('lat', self.lat)
        self.lon = di.get('lon', self.lon)
        self.elevation = di.get('elevation', self.elevation)
        self.slope = di.get('slope', self.slope)
        self.aspect = di.get('aspect', self.aspect)
        self.land_cover = di.get('land_cover', self.land_cover)
        self.land_cover_label = di.get('land_cover_label', self.land_cover_label)
        self.soil = di.get('soil', self.soil)
        self.place_name = di.get('place_name', self.place_name)
        self.osm_type = di.get('osm_type', self.osm_type)
        self.address_type = di.get('address_type', self.address_type)
        self.osm_category = di.get('osm_category', self.osm_category)
        self.address = di.get('address', self.address)
        return self
    

    def to_dict(self):
        ret = super().to_dict()
        ret['lat'] = self.lat
        ret['lon'] = self.lon
        ret['elevation'] = self.elevation
        ret['slope'] = self.slope
        ret['aspect'] = self.aspect
        ret['land_cover'] = self.land_cover
        ret['land_cover_label'] = self.land_cover_label
        ret['soil'] = self.soil
        ret['place_name'] = self.place_name
        ret['osm_type'] = self.osm_type
        ret['address_type'] = self.address_type
        ret['osm_category'] = self.osm_category
        ret['address'] = self.address
        return ret
    
    def get_textual_context(self):
        """
        Reverse geocodes a lat/lon point using OpenStreetMap's Nominatim API.
        Returns a dictionary with place name, address, and classification.
        """
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            "format": "json",
            "lat": self.lat,
            "lon": self.lon,
            "zoom": 10,
            "addressdetails": 1,
            "accept-language": "en",
            "extratags": 0,
            "namedetails": 0,
            "normalizecity": 0,
        }
        headers = {"User-Agent": "VOK/1.0"}

        try:
            response = requests.get(url, params=params, headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            pprint.pp(data)
            if 'lat' in data:
                del data['lat']
            if 'lon' in data:
                del data['lon']
            self.populate_from_dict(data)
            # return {
            #     "place_name": data.get("display_name"),
            #     "osm_type": data.get("type"),
            #     "address_type": data.get("addresstype"),
            #     "osm_category": data.get("category"),
            #     "address": data.get("address", {})
            # }
        except Exception as e:
            return {"error": str(e)}
    def get_vocab(self):
        ret =[]
        if self.lat:
            ret.append("the latitude of the location is " + str(self.lat))
        if self.lon:
            ret.append("the longitude of the location is " + str(self.lon))
        return ret

def create_ims_locations():
    ims_stations = MongoUtils.get_collection('meta_data', 'source_md').find({'provider_id': 'ims'})
    points = []
    locations = []
    for station in ims_stations:
        lon = DictUtils.parse_value(station, 'location.lat')
        lat = DictUtils.parse_value(station, 'location.lon')
        if not lat or not lon:
            continue
       
        points.append((lat, lon))
    pprint.pp(points)
    data = GELocationUtils.batch_process_locations(points)
    pprint.pp(data)
    for loc in data:
        uid = IDUtils.get_id(['loc', loc['lat'], loc['lon']])
        pprint.pp(loc)
        loc = VOKLocation(uid=uid, di=loc)
        loc.get_textual_context()
        loc.save_to_db()
        locations.append(loc)


if __name__ == "__main__":
    # MongoUtils.get_collection('meta_data', 'location').delete_many({})
    # loc = VOKLocation(uid='loc_1', lon=35.1123, lat=32.8466)
    # loc.get_textual_context()
    # exit()
    create_ims_locations()
    exit()
    lat = 36.73582
    lon = -119.04663
    lat2 = 36.73582
    lon2 = -119.04663
    lat3 = 36.73582
    lon3 = -119.04663
    points = [(lat, lon), (lat2, lon2), (lat3, lon3)]
    data = GELocationUtils.batch_process_locations(points)
    loc = VOKLocation(uid='loc_1', di=data[0])
    print(loc.to_dict())

