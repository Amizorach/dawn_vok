
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.vok.embedding.syntax_emb.syntax_emb import EmbeddedDiscreteValue


class SyntaxEmbedding(EmbeddedDiscreteValue):
    def __init__(self, generator_id, emb_id, data=None, meta_data=None, cache=None):
        if generator_id is None or emb_id is None:
            raise ValueError("emb_id is required")
        super().__init__(generator_id=generator_id, emb_id=emb_id, data=data, meta_data=meta_data, embedding_type='syntax')
        self.data = self.format_data(self.data)
        self.cache = cache
        
    def format_data(self, data):
        if data is None:
            return None
        if isinstance(data, dict):
            ret = data
        elif isinstance(data, str):
            ret = {data.replace(' ', '_').lower(): data}
        elif isinstance(data, list):
            ret = {d.replace(' ', '_').lower(): d for d in data}
        else:
            raise ValueError("data is not a dictionary or string")
        fr = {}
        for k, v in ret.items():
            fr[k] = {'raw': v, 'tokenized': []}
        
        return fr
   
    def to_dict(self):
        ret = super().to_dict()
        ret['data'] = DictUtils.np_to_list(self.data)
        return ret
    def get_vocab(self):
        return self.data
    
    def gather_vocab_for_update(self):
        vocab = {}
        for k, v in self.data.items():
            if v.get('tokenized', None) is None or len(v['tokenized']) == 0:
                if k in self.cache:
                    self.data[k]['tokenized'] = self.cache[k]
                else:
                    vocab[k] = v['raw']
           
        return vocab
    
    def update_vocab(self, full_vocab, scheme_id='tokenized'):
        mean_emb = []
        for k, v in self.data.items():
            if k in full_vocab:
                v[scheme_id] = full_vocab[k]
                mean_emb.append(full_vocab[k])
            elif v.get(scheme_id, None) is not None:
                mean_emb.append(v[scheme_id])
        self.embedding['mean'] = np.mean(mean_emb, axis=0).tolist()
        self.save_to_db()
        return True
    
    def get_embedding_scheme_latent(self, embedding_scheme_id):
        return self.embedding.get(embedding_scheme_id, None)
    
    def set_embedding_scheme_latent(self, embedding_scheme_id, latent, save=True):
        self.embedding[embedding_scheme_id] = latent
        if save:
            self.save_to_db()
        return True
    
    @classmethod
    def update_syntax_embedding_scheme(cls, generator_id=None, save=True):
        col = MongoUtils.get_collection(db_name=cls.get_db_name(), collection_name=cls.get_collection_name())
        match = { 'embedding_type': 'syntax'}
        if generator_id:
            match['generator_id'] = generator_id
        se = col.find(match)
        full_vocab = {}
        for s in se:
            vocab = s.get_vocab()
            for k, v in vocab.items():
                full_vocab[k] = v
           
        return True
    
