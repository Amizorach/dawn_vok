from datetime import datetime, timezone
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.utils.id_utils import IDUtils


class VOKObject:
    @classmethod
    def get_db_name(cls):
        raise NotImplementedError("get_db_name must be implemented by subclasses")
    
    @classmethod
    def get_collection_name(cls):
        raise NotImplementedError("get_collection_name must be implemented by subclasses")
   
    def __init__(self, uid: str, obj_type: str = None, system_uid: str = None, meta_data: dict = None, name: str = None, syntax_directives: list = None):
        self.uid = uid
        self.obj_type = obj_type
        self.system_uid = system_uid
        self.meta_data = meta_data or {}
        self.name = name

        self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.update_system_uid()
        self.syntax_directives = syntax_directives or []
        self.latent_schemes = {}

    def to_dict(self) -> dict:
        # Use existing updated_at (set on init or save)
        ret = {
            '_id': self.get_id(),
            'obj_type': self.obj_type,
            'uid': self.uid,
            'system_uid': self.system_uid,
            'meta_data': self.meta_data,
            'name': self.name,
            'syntax_directives': self.syntax_directives,
            'latent_schemes': self.latent_schemes
        }
        # serialize timestamp
        DictUtils.put_datetime(ret, 'updated_at', self.updated_at)
        return ret
    
    def get_vocab(self):
        #make sure there are no duplicate lines in the syntax_directives
        return list(set(self.syntax_directives))
    def update_syntax_directives(self):
        self.syntax_directives = self.get_vocab()

    @classmethod
    def gather_full_vocab(cls):
        objects = cls.get_all()
       
        vocab = []
        for obj in objects:
            vocab.extend(obj.get_vocab())
        vocab = list(set(vocab))
        return vocab
    
    def populate_from_dict(self, d: dict) -> 'VOKObject':
        self.obj_type = d.get('obj_type', self.obj_type)
        self.uid = d.get('uid', self.uid)
        self.system_uid = d.get('system_uid', self.system_uid)
        self.meta_data = d.get('meta_data', self.meta_data)
        self.name = d.get('name', self.name)
        self.updated_at = DictUtils.parse_datetime(di=d, path='updated_at', default=self.updated_at)
        self.syntax_directives = d.get('syntax_directives', self.syntax_directives)
        self.latent_schemes = d.get('latent_schemes', self.latent_schemes)
        return self
    
    def update_system_uid(self) -> None:
        if not self.uid or not self.obj_type:
            return
        self.system_uid = IDUtils.get_system_unique_id({
            'obj_type': self.obj_type,
            'uid': self.uid
        })
   
    def get_id(self) -> str:
        return IDUtils.get_id([
            self.obj_type,
            self.uid
        ])
    
    def save_to_db(self, use_system_uid: bool = False) -> None:
        # update timestamp once
        self.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        d = self.to_dict()
        collection = MongoUtils.get_collection(
            self.get_db_name(), self.get_collection_name()
        )
        filter_field = 'system_uid' if use_system_uid else 'uid'
        collection.update_one(
            {filter_field: getattr(self, filter_field)},
            {'$set': d},
            upsert=True
        )

    def load_from_db(self, use_system_uid: bool = False) -> 'VOKObject':
        collection = MongoUtils.get_collection(
            self.get_db_name(), self.get_collection_name()
        )
        filter_field = 'system_uid' if use_system_uid else 'uid'
        d = collection.find_one({filter_field: getattr(self, filter_field)})
        if d:
            self.populate_from_dict(d)
        return self
    
    def set_latent_scheme(self, scheme_id: str, scheme: list, save: bool = True):
        self.latent_schemes[scheme_id] = scheme
        if save:
            self.save_to_db()
        return self
    
    def get_latent_scheme(self, scheme_id: str):
        return self.latent_schemes.get(scheme_id, None)
    
    @classmethod
    def get_by_system_uid(cls, system_uid: str, populate: bool = True):
        collection = MongoUtils.get_collection(cls.get_db_name(), cls.get_collection_name())
        d = collection.find_one({'system_uid': system_uid})
        if not d:
            return None
        if populate:
            obj = cls(d['uid'], d.get('obj_type'), system_uid=system_uid)
            return obj.populate_from_dict(d)
        return d

    @classmethod
    def get_by_uid(cls, uid: str, populate: bool = True):
        collection = MongoUtils.get_collection(cls.get_db_name(), cls.get_collection_name())
        d = collection.find_one({'uid': uid})
        if not d:
            return None
        if populate:
            obj = cls(uid, d.get('obj_type'))
            return obj.populate_from_dict(d)
        return d
    
    @classmethod
    def get_by_id(cls, doc_id: str, populate: bool = True):
        collection = MongoUtils.get_collection(cls.get_db_name(), cls.get_collection_name())
        d = collection.find_one({'_id': doc_id})
        if not d:
            return None
        if populate:
            obj = cls(d['uid'], d.get('obj_type'))
            return obj.populate_from_dict(d)
        return d

    @classmethod
    def get_all(cls, populate: bool = True):
        collection = MongoUtils.get_collection(cls.get_db_name(), cls.get_collection_name())
        li = collection.find()  
        if not li:
            return []
        if populate:
            return [cls(d['uid']).populate_from_dict(d) for d in li]
        return li
    
    @classmethod
    def get_all_by_obj_type(cls, obj_type: str, populate: bool = True):
        collection = MongoUtils.get_collection(cls.get_db_name(), cls.get_collection_name())
        li = collection.find({'obj_type': obj_type})
        if populate:
            return [cls(d['uid']).populate_from_dict(d) for d in li]
        return li

    @classmethod
    def save_many(cls, objects: list) -> None:
        docs = [o.to_dict() for o in objects]
        MongoUtils.update_many(
            cls.get_db_name(), cls.get_collection_name(), docs, index_field='uid'
        )
    
    @classmethod
    def delete_many(cls, objects: list) -> None:
        collection = MongoUtils.get_collection(cls.get_db_name(), cls.get_collection_name())
        uids = [o.uid for o in objects]
        collection.delete_many({'uid': {'$in': uids}})
