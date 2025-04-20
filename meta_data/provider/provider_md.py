from meta_data.meta_data import MetaData
from db.mongo_utils import MongoUtils

class ProviderMD(MetaData)  :
   
    def get_index_list(self):
        index_list = super().get_index_list()
        return index_list
      

    def __init__(self, provider_id=None, provider_name=None, provider_type=None, provider_url=None, provider_api_key=None, provider_api_secret=None, provider_api_token=None, provider_api_token_expiration=None):
        super().__init__(db_name='meta_data', collection_name='provider_md', md_id=provider_id, md_type='provider', name=provider_name)
        self.provider_type = provider_type
        self.provider_url = provider_url
        self.provider_api_key = provider_api_key
        self.provider_api_secret = provider_api_secret
        self.provider_api_token = provider_api_token
        self.provider_api_token_expiration = provider_api_token_expiration
        
   
    def to_dict(self):
        ret = super().to_dict()
        ret['provider_type'] = self.provider_type
        ret['provider_url'] = self.provider_url
        ret['provider_api_key'] = self.provider_api_key
        ret['provider_api_secret'] = self.provider_api_secret
        ret['provider_api_token'] = self.provider_api_token
        ret['provider_api_token_expiration'] = self.provider_api_token_expiration
        return ret

    def populate_from_dict(self, data):
        super().populate_from_dict(data)
        self.provider_type = data.get('provider_type', self.provider_type)
        self.provider_url = data.get('provider_url', self.provider_url)
        self.provider_api_key = data.get('provider_api_key', self.provider_api_key)
        self.provider_api_secret = data.get('provider_api_secret', self.provider_api_secret)
        self.provider_api_token = data.get('provider_api_token', self.provider_api_token)
        self.provider_api_token_expiration = data.get('provider_api_token_expiration', self.provider_api_token_expiration)  
        return self


if __name__ == '__main__':
    provider_md = ProviderMD(provider_id='1', provider_name='provider_1')
    provider_md.save_to_db()
    print(list(provider_md.get_all_providers()))