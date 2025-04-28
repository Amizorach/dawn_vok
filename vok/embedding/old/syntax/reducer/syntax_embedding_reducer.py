from dawn_vok.vok.embedding.syntax.syntax_embedding import SyntaxEmbedding
from dawn_vok.vok.embedding.syntax.reducer.reducer_datasets import SyntaxEmbeddingReducerVectorDataset, SyntaxEmbeddingReducerInferenceDataset
from dawn_vok.vok.embedding.syntax.reducer.reducer_models import SyntaxEmbeddingReducerModel
from dawn_vok.vok.embedding.syntax.reducer.reducer_trainer import SyntaxEmbeddingReducerTrainer
import pprint

class SyntaxEmbeddingReducer:
    def __init__(self, generator_id=None):
        self.generator_id = generator_id
        self.embedding = self.load_embedding()
        
    def load_embedding(self):
        se = SyntaxEmbedding.get_embeddings(generator_id=self.generator_id, embedding_type='syntax', populate=True, as_dict=False)
        return se


if __name__ == "__main__":
    se_reducer = SyntaxEmbeddingReducer(generator_id='unit_syntax')
    latent_dim = 64
    base_query = {}
    for se in se_reducer.embedding:
        base_query[se.emb_id] ={}
        base_query[se.emb_id]['data'] = []
        data = se.data
        for lat in data.values():
            base_query[se.emb_id]['data'].append(lat['tokenized'])
    input_dim = 384
    query_list = list(base_query.values())
    
    dataset = SyntaxEmbeddingReducerVectorDataset(query_list)
    dataset.add_dirty_data(noise_level=0.1, num_samples=10)
    print(dataset[0])
    model = SyntaxEmbeddingReducerModel(input_dim, latent_dim)

    trainer = SyntaxEmbeddingReducerTrainer(generator_id='unit_syntax', model=model, dataset=dataset, batch_size=24, epochs=200)
    trainer.load_model()
    trainer.train()
    data_list = []
    for d in dataset:
        data_list.append(d[0])
    print(data_list[0].shape)
    scheme_id = f'lat_{latent_dim}'
    # inf_dataset = SyntaxEmbeddingReducerInferenceDataset(query_list)
    # encoded = trainer.inference(inf_dataset)
    # for i, se in enumerate(se_reducer.embedding):
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