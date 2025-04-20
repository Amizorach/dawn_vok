import numpy as np
from sentence_transformers import SentenceTransformer
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.vok.embedding.base.discrete_embedding import EmbeddedDiscreteValue
from dawn_vok.vok.embedding.syntax.syntax_reducer import SyntaxEmbeddingReductionTrainer
from dawn_vok.vok.embedding.syntax.vocab import DVocab


class SyntaxDBBuilder():
    def __init__(self, generator_id=None, db_name=None, seed=42, device="cpu"):
        self.table_id = 'syntax_builder'
        self.generator_id = generator_id
        self.db_name = db_name
        self.seed = seed
        self.device = device
        self.model_name = "paraphrase-MiniLM-L3-v2"
        self.model = SentenceTransformer(self.model_name)
        self.encode_map = {}

        # This is the dimension output by the SentenceTransformer model
        self.model_latent_size = self.model.get_sentence_embedding_dimension() # More robust way to get dim
        self.latent_dim = self.model_latent_size
        # The full latent dim is 3 * model_latent_size (type + param + avg_desc)
        # self.db = EmbeddingDBDict(self.table_id, self.generator_id, self.db_name, self.latent_dim, self.seed, self.device)

        # self.unit_ids = list(unit_definitions.keys())

    def build_syntax_db(self, vocab=None):
        vocab = vocab or DVocab.get_syntax_vocab()
        self.model = SentenceTransformer(self.model_name)
        self.encode_map = {}
        self.build_table(vocab)

    
    def build_table(self, vocab=None):
        vocab = vocab or DVocab.get_vocab()

        if not vocab:
            raise ValueError("Vocab is empty")
        if isinstance(vocab, dict):
            for field, values in vocab.items():
                for value in values:
                    self.encode_map[f'{field}_{value}'] = self._encode_text(field, value)
        else:
            for value in vocab:
                key = value.replace(' ', '_').lower()
                self.encode_map[key] = self._encode_text(value=value)
        print(f"Table built with {len(self.encode_map)} entries")
    
   
    def _format_field(self, field, value):
        """Formats a text field with its value."""
        return f"{field}: {value}"
    
    def _encode_text(self, field=None, value=None):
        if value is None:
            return None
        if field is None:
            key = value.replace(' ', '_').lower()
            text_to_encode = str(value)
        else:
            text_to_encode = self._format_field(field, str(value)) # Ensure value is string
       
        # The model expects a list of sentences
        if key not in self.encode_map:
            embedding = self.model.encode([text_to_encode])[0]
            self.encode_map[key] = embedding
        else:
            embedding = self.encode_map[key]
        return embedding
   
    def get_id(self, table_id, model_id, version_id):
        return f"{table_id}_{model_id}_{version_id}"
    
    def save_key_map(self):
        generator_id = 'syntax_builder'
        embeddings = {k: v.tolist() for k, v in self.encode_map.items()}
        embedding_scheme_id = 'full_embedding'
        EmbeddedDiscreteValue.add_embedding_scheme_latents(generator_id=generator_id, embedding_scheme_id=embedding_scheme_id, latents=embeddings)
            

    def to_dict(self):
        list_map = {k: v.tolist() for k, v in self.encode_map.items()}
        return {
            '_id': self.get_id(self.table_id, self.model_id, self.version_id),
            'generator_id': self.generator_id,
            'table_id': self.table_id,
            'model_id': self.model_id,
            'version_id': self.version_id,
            'vocab': self.vocab,
            'encode_map': list_map,
        }
    def populate_from_dict(self, di):
        self.model_id = di.get('model_id', self.model_id)
        self.table_id = di.get('table_id', self.table_id)
        self.version_id = di.get('version_id', self.version_id)
        self.vocab = di.get('vocab', {})    
        self.encode_map = {k: np.array(v) for k, v in di.get('encode_map', {}).items()}
        print(f"Encode map loaded with {len(self.encode_map)} entries")

    def save_to_db(self):
        self.save_key_map()
        # col = MongoUtils.get_collection(db_name=self.get_db_name(), collection_name=self.get_collection_name())
        # col.update_one(
        #     {'_id': self.get_id(self.table_id, self.model_id, self.version_id)},
        #     {'$set': self.to_dict()},
        #     upsert=True
        # )
    
    # def save_to_file(self, fn = None):
    #     fn = fn or f"{self.get_id(self.table_id, self.model_id, self.version_id)}.pkl"
    #     path = os.path.join(self.get_table_dir(self.table_id, self.model_id, self.version_id), fn)
    #     with open(path, 'wb') as f:
    #         pickle.dump(self, f)

    @classmethod
    def load_from_db(cls, table_id, model_id, version_id):
        col = MongoUtils.get_collection(db_name=cls.get_db_name(), collection_name=cls.get_collection_name())
        di = col.find_one({'_id': cls.get_id(table_id, model_id, version_id)})
        if di:
            obj = cls()
            obj.populate_from_dict(di)
            return obj
        else:
            return None
        
if __name__ == "__main__":
    # col = MongoUtils.get_collection(db_name=EmbeddedDiscreteValue.get_db_name(), collection_name=EmbeddedDiscreteValue.get_collection_name())
    # if col != None:
    #     col.delete_many({})
    builder = SyntaxDBBuilder()
    builder.build_syntax_db()
    builder.save_to_db()
    trainer = SyntaxEmbeddingReductionTrainer()
    trainer.update_embeddings(emb_size=32, orig_scheme_id='full_embedding', out_scheme_id='reduced_32')
    trainer.save_model()
    trainer = SyntaxEmbeddingReductionTrainer(short_emb_size=16)
    trainer.update_embeddings(emb_size=16, orig_scheme_id='full_embedding', out_scheme_id='reduced_16')
    trainer.save_model()
