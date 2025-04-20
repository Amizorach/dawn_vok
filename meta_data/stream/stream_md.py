from datetime import datetime
from meta_data.meta_data import MetaData


class StreamMD(MetaData):

    def __init__(self, stream_id, stream_name,  stream_type, stream_created_at=None, stream_updated_at=None):
        self.stream_id = stream_id
        self.stream_type = stream_type
       
        self.stream_created_at = stream_created_at or datetime.utcnow()
        self.stream_updated_at = stream_updated_at or datetime.utcnow()
        super().__init__(db_name='meta_data', collection_name='stream_md', md_id=stream_id, md_type='stream', name=stream_name) 

    def to_dict(self):
        return {
            "stream_id": self.stream_id,
            "stream_type": self.stream_type,
            "stream_created_at": self.stream_created_at,
            "stream_updated_at": self.stream_updated_at
        }
    
    def populate_from_dict(self, data):
        self.stream_id = data.get("stream_id", self.stream_id)
        self.stream_type = data.get("stream_type", self.stream_type)
        self.stream_created_at = data.get("stream_created_at", self.stream_created_at)
        self.stream_updated_at = data.get("stream_updated_at", self.stream_updated_at)
        return self

