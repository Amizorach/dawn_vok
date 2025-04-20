import pandas as pd
from raw_data.plugins.data_plugin import DataPlugin

class IMSDataPlugin(DataPlugin):
    def __init__(self):
        super().__init__('ims')
        self.data_dir = 'providers/raw/ims'
        self.pickle_dir = 'providers/pickle/ims'
        
    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        return df
    
    
