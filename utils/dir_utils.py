import os

from dawn_vok.utils.dict_utils import DictUtils
from dawn_vok.utils.id_utils import IDUtils


class DirUtils:
    base_path = os.path.dirname('/home/amiz/dawn/dawn_data/')
    raw_data_path = os.path.join(base_path, 'raw_data')
    model_path = os.path.join(base_path, 'models')
    @classmethod
    def create_dir(cls, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    
    @classmethod
    def get_raw_data_dir(cls, path=None):
        if path is None:
            return cls.create_dir(cls.raw_data_path)
        else:
            return cls.create_dir(os.path.join(cls.raw_data_path, path))
    
    @classmethod
    def get_raw_data_path(cls, file_name, path=None):
        return os.path.join(cls.get_raw_data_dir(path=path), file_name)
    
    @classmethod
    def get_model_path(cls, model_id, version, path=None):
        if model_id is None:
            raise ValueError("model_id is required")
        if version is None:
            print("version is None, using latest")
            version = 'latest'
        version = IDUtils.clean_str(version)
        model_id = IDUtils.clean_str(model_id)
        dir = os.path.join(cls.model_path, model_id, version)
        if path is not None:
            dir = os.path.join(dir, path)
        cls.create_dir(dir)
        return dir
    
    @classmethod
    def get_base_dir(cls):
        return cls.base_path
    
    @classmethod
    def get_base_model_path(cls):
        return os.path.join(cls.model_path)
    

    
    @classmethod
    def get_checkpoints_dir(cls, model_id, version, checkpoint_file):
        return os.path.join(cls.get_model_path(model_id, version, 'checkpoints'), checkpoint_file)
    
    @classmethod
    def get_cache_dir(cls, model_id, version, path=None):
        dir = os.path.join(cls.base_path, 'cache')
        cls.create_dir(dir)
        return dir
    
    
if __name__ == '__main__':
    print(DirUtils.get_raw_data_dir(path='providers/synoptic'))
    print(DirUtils.get_base_dir())
        