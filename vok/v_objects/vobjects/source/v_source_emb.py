class VOKSourceEmbedding:
    def __init__(self, source_id, source):
        self.source_id = source_id
        self.source = source
        self.embedding = None
        self.embedding_type = None
        
    def get_embedding(self):
        return self.source.get_static_latent_id()
    
    def get_embedding_type(self):
        return self.source.get_embedding_type()
    

        
        
