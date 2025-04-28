import torch
from dawn_vok.vok.embedding.emb_param.embedding_paramater import EmbeddingParamater
from dawn_vok.vok.embedding.syntax.syntax_embedding import SyntaxEmbedding
from dawn_vok.vok.embedding.syntax.reducer.reducer_datasets import SyntaxEmbeddingReducerVectorDataset, SyntaxEmbeddingReducerInferenceDataset
from dawn_vok.vok.embedding.syntax.reducer.reducer_models import SyntaxEmbeddingReducerModel
from dawn_vok.vok.embedding.syntax.reducer.reducer_trainer import SyntaxEmbeddingReducerTrainer
import pprint


class EMPSyntaxReducer:
    def __init__(self, input_dim=384, latent_dim=64):
        self.embedding = self.load_embedding()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dataset = None
        self.model = None
        self.trainer = None
        self.model_file_path = 'model.pth'
        self.embedding_paramaters = None
        self.full_latent_list = None
    def load_embedding(self):
        ep = EmbeddingParamater.get_all()
        self.embedding_paramaters = ep

        full_latent_list = []
        for ep1 in ep:
            full_latent_list.append(ep1.latents['base'])
        self.full_latent_list = full_latent_list
        return full_latent_list

    def prepare_for_train(self):
        self.load_embedding()
        self.model = SyntaxEmbeddingReducerModel(self.input_dim, self.latent_dim)
        self.dataset = SyntaxEmbeddingReducerVectorDataset(self.full_latent_list)
        self.dataset.add_dirty_data(noise_level=0.2, num_samples=10)

    def train(self):
        self.trainer = SyntaxEmbeddingReducerTrainer('param_syntax', self.model, self.dataset, batch_size=256, epochs=300)
        self.trainer.train()

    def prepare_for_inference(self):
        self.load_embedding()
        self.model = SyntaxEmbeddingReducerModel(self.input_dim, self.latent_dim)
        self.load_model()
        self.model.eval()
        self.dataset = SyntaxEmbeddingReducerInferenceDataset(self.full_latent_list)
        self.trainer = SyntaxEmbeddingReducerTrainer('param_syntax', self.model, self.dataset, batch_size=256, epochs=200)
   
    def update_embedding_paramater(self):
        self.prepare_for_inference()
        encoded = self.trainer.inference(self.dataset)
        lat_id = f'lat_{self.latent_dim}'
        for i, se in enumerate(self.embedding_paramaters):
            se.latents[lat_id] = encoded[i].tolist()
            se.save_to_db()
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_file_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_file_path))

if __name__ == "__main__":
    se_reducer = EMPSyntaxReducer(latent_dim=128)
    se_reducer.prepare_for_train()
    se_reducer.train()
    se_reducer.save_model()
    se_reducer.update_embedding_paramater()
    # data_list = []
    # for d in dataset:
    # #     data_list.append(d[0])
    # # print(data_list[0].shape)
    # # scheme_id = f'lat_{latent_dim}'
    # # inf_dataset = SyntaxEmbeddingReducerInferenceDataset(query_list)
    # # encoded = trainer.inference(inf_dataset)
    # # for i, se in enumerate(se_reducer.embedding):
    #     pprint.pp(se.embedding.keys())
    #     se.embedding[scheme_id]= encoded[i].tolist()
    #     se.save_to_db()
    # print(encoded.shape)
    # print(reconstructed[0].shape)
    # print(encoded[0].shape)
    # print(dataset[0][0].shape)
    # print(encoded[0])
    exit()
    sensors_types = SensorType.create_all_sensor_types()
    print(sensors_types)
    cache = {}
    se_emb = []
    for st in sensors_types:
        dt = st.get_vocab()
        pprint.pp(dt)
        se = SyntaxEmbedding(generator_id='unit_syntax', emb_id=st.system_uid, data=st.get_vocab(), cache=cache)
        se.save_to_db()
        se_emb.append(se)

    full_vocab = {}
    for se in se_emb:
        vocab = se.gather_vocab_for_update()
        for k, v in vocab.items():
            full_vocab[k] = v
    pprint.pp(full_vocab)
    builder = SyntaxFormater()
    builder.format_syntax(full_vocab)
    pprint.pp(builder.encode_map.keys())
    for se in se_emb:
        se.update_vocab(builder.encode_map)
        se.save_to_db()