

import os
import numpy as np
import pandas as pd
from meta_data.stream.stream_md import StreamMD
from utils.dir_utils import DirUtils

class RawDataStream(StreamMD):
    def __init__(self, stream_id, stream_name, stream_type, pickle_file_path=None, columns=None, datetime_column='datetime', stream_created_at=None, stream_updated_at=None):
        super().__init__(stream_id=stream_id, stream_name=stream_name, stream_type=stream_type, stream_created_at=stream_created_at, stream_updated_at=stream_updated_at)
        self.pickle_file_path = pickle_file_path or f"{stream_id}.pkl"
        self.columns = columns
        self.data = None
        self.datetime_column = datetime_column

    def update_data(self, df):
        self.data = df
        self.data[self.datetime_column] = pd.to_datetime(self.data[self.datetime_column])
        self.data.set_index(self.datetime_column, inplace=True)
        self.data.sort_index(inplace=True)
        self.columns = self.data.columns
        self.start_date = self.data.index.min()
        self.end_date = self.data.index.max()
        # self.frequency = self.data.index.freq
        self.pickle_file_path = f"{self.stream_id}.pkl"#DirUtils.get_raw_data_path(f"{self.stream_id}.pkl")
        self.data.to_pickle(self.pickle_file_path)
        self.save_to_db()

    def load_data(self):
        if self.data is not None:
            return
        fp = DirUtils.get_raw_data_path(self.pickle_file_path)
        if not os.path.exists(fp):
            raise ValueError(f"Pickle file does not exist for stream {self.stream_id}")
        self.data = pd.read_pickle(fp)
        if self.data == None:
            raise ValueError(f"Data is None for stream {self.stream_id}")
        if self.columns is not None:
            self.data = self.data[self.columns]
        self.columns = self.data.columns
        self.data[self.datetime_column] = pd.to_datetime(self.data[self.datetime_column])
        self.data.set_index(self.datetime_column, inplace=True)
        self.data.sort_index(inplace=True)


    def manipulate(self, maniplator_fn):
        
        if maniplator_fn is None:
            return self.data
        elif maniplator_fn == 'diff':
            return self.data.diff()
        elif maniplator_fn == 'pct_change':
            return self.data.pct_change()
        elif maniplator_fn == 'log':
            return self.data.apply(lambda x: np.log(x))
        elif maniplator_fn == 'cumsum':
            return self.data.cumsum()
        elif maniplator_fn == 'cumprod':
            return self.data.cumprod()
        elif maniplator_fn == 'cummax':
            return self.data.cummax()
        return maniplator_fn(self.data)
    
    def get_data(self, start, end, columns=None, freq=None, agg_fn=None, maniplator_fn=None):
        if self.data is None:
            self.load_data()
        if freq is not None:
            agg_fn = agg_fn or 'mean'   
            self.data = self.data.resample(freq).agg(agg_fn)
        if maniplator_fn is not None:
            self.data = self.manipulate(maniplator_fn)
        if columns is not None:
            self.data = self.data[columns]
        if start is not None:
            self.data = self.data[start:]
        if end is not None:
            self.data = self.data[:end]
        return self.data

     
if __name__ == '__main__':
    stream = RawDataStream(stream_id='test', stream_name='test', stream_type='test', pickle_file_path='test.pkl')
    stream.update_data(pd.DataFrame({'datetime': ['2020-01-01', '2020-01-02', '2020-01-03'], 'value': [1, 2, 3]}))
    print(stream.get_data(start='2020-01-01', end='2020-01-03'))
