import numpy as np
from sentence_transformers import SentenceTransformer



class SyntaxFormater:
    def __init__(self, device="cpu"):
        self.table_id = 'syntax_builder'
        self.model_name = "paraphrase-MiniLM-L3-v2"
        self.model = SentenceTransformer(self.model_name)
        self.encode_map = {}

        # This is the dimension output by the SentenceTransformer model
        self.model_latent_size = self.model.get_sentence_embedding_dimension() # More robust way to get dim
        # The full latent dim is 3 * model_latent_size (type + param + avg_desc)
        # self.db = EmbeddingDBDict(self.table_id, self.generator_id, self.db_name, self.latent_dim, self.seed, self.device)

        # self.unit_ids = list(unit_definitions.keys())

    def format_syntax(self, vocab):
        if not vocab:
            raise ValueError("Vocab is empty")
        if not isinstance(vocab, dict) and not isinstance(vocab, list): 
            raise ValueError("Vocab is not a dictionary or list")
        if isinstance(vocab, list):
            vocab = {v.replace(' ', '_').lower(): v for v in vocab}
        for key, value in vocab.items():
            if key not in self.encode_map:
                self.encode_map[key] = self.encode_text(key, value)
        return self.encode_map
    
    def encode_text(self, key, value):
        embedding = self.model.encode([value])[0]
        return embedding

    
        

 