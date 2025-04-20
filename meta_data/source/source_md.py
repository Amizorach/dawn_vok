from meta_data.meta_data import MetaData

class SourceMD(MetaData):
    def __init__(self, source_id, source_name, source_type, provider_id):
        super().__init__(db_name='meta_data', collection_name='source_md', md_id=source_id, md_type='source', name=source_name)
        self.source_type = source_type
        self.provider_id = provider_id

    
    def to_dict(self):
        ret = super().to_dict()
        ret['source_type'] = self.source_type
        ret['provider_id'] = self.provider_id
        return ret

    
    def populate_from_dict(self, data):
        super().populate_from_dict(data)
        self.source_type = data.get('source_type', self.source_type)
        self.provider_id = data.get('provider_id', self.provider_id)
        return self



if __name__ == '__main__':
    source_md = SourceMD(source_id='1', source_name='source_1', source_type='temperature', provider_id='1')
    source_md.save_to_db()
    print(list(source_md.get_all_sources()))
