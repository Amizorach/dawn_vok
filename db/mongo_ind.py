

class MongoIndexes:
    raw_data_collections = {
        'time_series_providers': [
            
            ([('provider_id', 1)], {}),
            ([('source_id', 1)], {}),

        ], 
        'raw_data_providers': [
            ([('uid', 1)], {'unique': True}),
            ([('system_uid', 1)], {'unique': True}),
            ([('data_provider_id', 1)], {}),
            ([('file_name', 1)], {}),
        ]
    }

    meta_data_collections = {
        'sensor_type': [
            ([('system_uid', 1)], {'unique': True}),
            ([('base_sensor_type', 1)], {}),
            ([('unit', 1)], {}),
        ],
        'sensor_md': [
            ([('sensor_id', 1)], {'unique': True}),
        ],
        'source_md': [
            ([('source_id', 1)], {'unique': True}),
            ([('source_name', 1)], {}),
            ([('source_type', 1)], {}),
            ([('provider_id', 1)], {})
        ],
        'provider_md': [
            ([('provider_id', 1)], {'unique': True}),
        ],
        'provider_type': [
            ([('system_uid', 1)], {'unique': True}),
            ([('uid', 1)], {}),
        ],
        'measurement_unit': [
            ([('uid', 1)], {'unique': True}),
            ([('system_uid', 1)], {'unique': True}),
            ([('unit', 1)], {'unique': True}),
        ],
        'location': [
            ([('uid', 1)], {'unique': True}),
            ([('system_uid', 1)], {'unique': True}),
            ([('lat', 1), ('lon', 1)], {'unique': True}),
        ],
        'source': [
            ([('uid', 1)], {'unique': True}),
            ([('system_uid', 1)], {'unique': True}),
            ([('source_id', 1)], {'unique': True}),
        ],
        'formulation': [
            ([('uid', 1)], {'unique': True}),
            ([('system_uid', 1)], {'unique': True}),
            ([('formulation_type', 1)], {}),
            ([('formulation_id', 1)], {}),
        ],
    }

    embedding_collections = {
        'discrete_embeddings': [
            ([('system_uid', 1)], {'unique': True}),
            ([('generator_id', 1)], {}),
            ([('emb_id', 1)], {}),
            ([('embedding_type', 1)], {}),
        ],
        'embedding_units': [
            ([('system_uid', 1)], {'unique': True}),
            ([('unit_type', 1)], {}),
            ([('emb_id', 1)], {}),
        ],
         'embedding_paramaters': [
            ([('uid', 1)], {'unique': True}),
                       ([('system_uid', 1)], {'unique': True}),
 
            ([('param_type', 1)], {}),
        ],
    }
    models_collections = {
        'embedding_managers': [
            ([('generator_id', 1)], {'unique': True}),
        ],
        'embedded_discrete_values': [
            ([('generator_id', 1), ('emb_id', 1)], {'unique': True}),
        ],
        'vok_models': [
            ([('uid', 1)], {'unique': True}),
            ([('system_uid', 1)], {'unique': True}),
            ([('model_id', 1)], {}),
            ([('name', 1)], {}),
            ([('model_id', 1), ('version', 1), ('name', 1)], {'unique': True}),
        ]
    }

    total_collections = {
        'raw_data': raw_data_collections,
        'meta_data': meta_data_collections,
        'models': models_collections,
        'embeddings': embedding_collections
    }