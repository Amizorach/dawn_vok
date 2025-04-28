import math
import torch
import torch.nn as nn

paramater_blue_print = {
    'sensor_type': {
       'size': 3,
       'id_vec':[0.5, -0.1, 0.3, 0.2, -0.7, 0.4],
    'latents':[ 
        'measurment_param',
        'base_sensor_type',
        'sensor_type',
        # 'sensor_type_id'
        # 'sensor_finger_print',
    ]
    },
    'source_id': {
        'id_vec':[-0.8, 0.2, 0.7, 0.1, -0.2, 0.4],

        'size': 1,
        'latents':[
            'source_id',
            #source_finger_print

        ]
    },
    'location': {
        'id_vec':[-0.2, -0.6, 0.1, 0.3, -0.5, 0.2],
        'size': 1,
        'latents':[
           'location'
        ]
    },
    'timerange': {
        'id_vec':[0.9, -0.9, 0.2, 0.1, -0.2, 0.4],
        'size': 2,
        'latents':[
           'start_time',
           'end_time'
        ]
    },
    'formulation': {
        'id_vec':[-0.3, 0.0, -0.5, 0.2, -0.7, 0.4],
        'size': 1,
        'latents':[
           'formulation'
        ]
    },
    'astrological_data': {
        'id_vec':[0.7, 0.1, 0.6, 0.2, -0.7, 0.4],
        'size': 1,
        'latents':[
           'astrological_data'
        ]
    }
}
class EMPHeaderBuilder:
    
    @classmethod
    def _cyclic_tensor(cls, val: float) -> torch.Tensor:
        return torch.tensor([val/100.0, math.cos(val), math.sin(val)], dtype=torch.float32)

    @classmethod
    def create_header(cls, data_id, data_loc, glb_start_loc, param_id, latent_id, has_info=True):
        """Creates the header from the input dictionary."""

        if not data_id or not param_id  or not latent_id:
           raise ValueError("data_id, param_id, and latent_id are required")
        pb_kl = list(paramater_blue_print.keys())
        param = paramater_blue_print.get(param_id)
        param_loc = pb_kl.index(param_id)+1
        latent_loc = param.get('latents').index(latent_id)+1

        data_global_loc =  0
        for i in range(param_loc-1):
            data_global_loc += paramater_blue_print.get(pb_kl[i]).get('size')
        data_global_loc += latent_loc
        param_vec = param.get('id_vec')
        param_size = param.get('size')
        # latent_size = param.get('latents').get(latent_loc).get('size')
        print(f'param_vec: {param_vec}, param_loc: {param_loc},  param_size: {param_size}, data_global_loc: {data_global_loc}, latent_loc: {latent_loc}')
        
        data_id_t = torch.tensor([data_id], dtype=torch.float32)
        print(f'data_id_t: {data_id_t}')
        # global_loc_t = torch.tensor(cls.to_cyclic_index(glb_start_loc+data_global_loc), dtype=torch.float32)
        # data_loc_t = torch.tensor([data_loc], dtype=torch.float32)
        # data_global_loc_t = torch.tensor(cls.to_cyclic_index(data_global_loc), dtype=torch.float32)
        # param_loc_t = torch.tensor(cls.to_cyclic_index(param_loc), dtype=torch.float32)
        param_vec_t = torch.tensor(param_vec, dtype=torch.float32)
        param_size_t = torch.tensor([param_size], dtype=torch.float32)
        # latent_loc_t = torch.tensor(cls.to_cyclic_index(latent_loc), dtype=torch.float32)
        latent_info_t = torch.tensor([1 if has_info else 0], dtype=torch.float32)
        latent_is_first_t = torch.tensor([1 if latent_loc == 1 else 0], dtype=torch.float32)
        latent_is_last_t = torch.tensor([1 if latent_loc == len(param.get('latents')) else 0], dtype=torch.float32)
        latent = torch.cat([data_id_t, cls._cyclic_tensor(glb_start_loc+data_global_loc), cls._cyclic_tensor(data_loc),
                             cls._cyclic_tensor(data_global_loc), 
                            cls._cyclic_tensor(param_loc),
                             param_vec_t, param_size_t, cls._cyclic_tensor(latent_loc),
                             latent_info_t, latent_is_first_t, latent_is_last_t], dim=-1)
        print(f'latent shape: {latent.shape}')
        pad = torch.zeros(*latent.shape[:-1], 32-latent.shape[-1], device=latent.device)
        latent = torch.cat([latent, pad], dim=-1)
        print(f'latent: {latent}')
        print(f'latent.shape: {latent.shape}')
       

        return True

if __name__ == '__main__':
    print(EMPHeaderBuilder.create_header(data_id=0.34543, data_loc=6, glb_start_loc=33, param_id='timerange', latent_id='start_time'))
    print(EMPHeaderBuilder.create_header(data_id=0.34543, data_loc=6, glb_start_loc=33, param_id='timerange', latent_id='end_time'))