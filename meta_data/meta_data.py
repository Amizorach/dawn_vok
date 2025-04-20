from db.mongo_utils import MongoUtils

class MetaData:

    @staticmethod
    def get_collection(db_name, collection_name):
        return MongoUtils.get_collection(db_name, collection_name)

    @staticmethod
    def get_all_mds(db_name, collection_name):
        collection = MetaData.get_collection(db_name, collection_name)
        return list(collection.find())
    
    @staticmethod
    def get_md_by_id(db_name, collection_name, md_id):
        collection = MetaData.get_collection(db_name, collection_name)
        return collection.find_one({'md_id': md_id})
    
    @staticmethod
    def get_md_by_type(db_name, collection_name, md_type):
        collection = MetaData.get_collection(db_name, collection_name)
        return list(collection.find({'md_type': md_type}))
    
    
    

    def __init__(self, db_name, collection_name, md_id=None, md_type=None, name=None, data=None):
        self._id = md_id
        self.md_id = md_id
        self.md_type = md_type
        self.name = name
        self.db_name = db_name
        self.collection_name = collection_name
        self.data = data or {}

    def get_index_list(self):
        return  [
            ([('md_id', 1)], {'unique': True}),
            ([('md_type', 1)], {}),  # non-unique
            [('md_id', 1), ('md_type', 1)]  # default (non-unique) form still supported
        ]


    def to_dict(self):
        return {
            '_id': self._id,
            'md_id': self.md_id,
            'md_type': self.md_type,
            'name': self.name,
            'data': self.data
        }
    
    def populate_from_dict(self, data):
        self.md_id = data.get('md_id', self.md_id)
        self.md_type = data.get('md_type', self.md_type)
        self.name = data.get('name', self.name)
        self.data = data.get('data', self.data)
        
    def save_to_db(self):
        assert self.db_name is not None, "db_name is required"
        assert self.collection_name is not None, "collection_name is required"
        collection = self.get_collection(self.db_name, self.collection_name)
        collection.update_one({'_id': self._id}, {'$set': self.to_dict()}, upsert=True)

    def delete_from_db(self):
        assert self.db_name is not None, "db_name is required"
        assert self.collection_name is not None, "collection_name is required"
        collection = self.get_collection(self.db_name, self.collection_name)
        collection.delete_one({'_id': self._id})

