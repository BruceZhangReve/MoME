import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from data.m4 import M4Dataset, M4Meta
from utils.tools import StandardScaler

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

class Dataset_MTS(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None, features='S',
                  data_split = [0.7, 0.1, 0.2], scale=True, scale_statistic=None,target='OT'):
        # size [seq_len, label_len, pred_len]
        # info
        self.in_len = size[0]
        self.out_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.target=target
        
        self.scale = scale
    
        #self.inverse = inverse
        
        self.root_path = root_path
        self.features=features
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if (self.data_split[0] > 1):
            train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
        else:
            train_num = int(len(df_raw)*self.data_split[0]); 
            test_num = int(len(df_raw)*self.data_split[2])
            val_num = len(df_raw) - train_num - test_num; 
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            self.all_cols = cols_data.tolist()
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            self.all_cols = [self.target]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean = self.scale_statistic['mean'], std = self.scale_statistic['std'])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + self.out_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.in_len- self.out_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
 
    
class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask
    
    
    

    
    
class Dataset_stock(Dataset):
    def __init__(self, root_path, data_path='stock.csv',flag='train', scale=True, out_len=7, in_len=21):
        self.out_len = out_len
        self.in_len = in_len
        self.scale = scale
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = self.in_len
        self.pred_len = self.out_len
        self.label_len = self.out_len
        self.__read_data__()
        
    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        columns = df_raw.columns
        num_col = len(columns)
        train_num = int(num_col * 0.7)
        val_num = int(num_col * 0.15)
        test_num = num_col - train_num - val_num
        train_col = columns[:train_num]
        val_col = columns[train_num: train_num+val_num]
        test_col = columns[train_num+val_num:train_num+val_num+test_num]
        if self.flag == "train":
            df_data = df_raw[train_col]
        elif self.flag == "val":
            df_data = df_raw[val_col]
        else:
            df_data = df_raw[test_col]
            
        if self.scale:
            self.scaler = StandardScaler()
            # train_data = df_raw[train_col]
            self.scaler.fit(df_data.values)
            self.data = self.scaler.transform(df_data.values)   #[seq_len, num_stocks]
        else:
            self.data = df_data.values
        
        
        self.full_len = self.data.shape[0]
        self.num_ts_per_stock = int(self.full_len / (self.in_len+self.out_len))
        self.num_stock = self.data.shape[1]
        self.data = self.data.reshape(-1,1)  #[seq_len*num_stocks, 1]
            
            
    def __getitem__(self, index):
        
        num_stock = int(index / self.num_ts_per_stock)
        remain = index - num_stock * self.num_ts_per_stock 
        s_begin = num_stock * self.full_len + remain * (self.in_len+self.out_len)
        s_end = s_begin + self.in_len
        r_begin = s_end
        r_end = r_begin + self.out_len
        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        
        return seq_x, seq_y
    
    
    def __len__(self):
        return int(self.full_len/(self.in_len+self.out_len)) * self.num_stock
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# This is for the new benchmark
class Dataset_Telecom(Dataset):
    """Telecom multivariate time-series loader with categorical support."""
    def __init__(self,
                 root_path: str,
                 flag='train', #: str = 'train',
                 size=(96,48),#: tuple[int, int] = (96, 48),      # (in_len, out_len)
                 features: str = 'S',                   # 'S' | 'M' | 'MS'
                 data_split = (0.7, 0.1, 0.2),#: tuple[float, float, float] = (0.7, 0.1, 0.2),
                 scale: bool = True,
                 scale_statistic = None,#: dict | None = None,
                 target: str = 'TX_Bytes',
                 categorical_cols=None,#: list[str] | None = None):
                 ):

        assert flag in ('train', 'val', 'test')
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.in_len, self.out_len = size
        self.features = features
        self.data_split = data_split
        self.scale = scale
        self.scale_statistic = scale_statistic
        self.target = target
        self.cat_cols: list[str] = list(categorical_cols) if categorical_cols else []

        self._read_data(root_path)

    def _read_data(self, path: str):

        raw_objs = pd.read_json(path, lines=True)
        full_df = pd.concat(
            [pd.DataFrame(o.dataframe) for o in raw_objs.itertuples()],
            ignore_index=True
        )

        if self.features == 'S':
            full_df = full_df[[self.target]]
            if self.target not in self.cat_cols and full_df[self.target].dtype == object:
                self.cat_cols.append(self.target)
        else:
            full_df = full_df.drop(columns=['timestamp'], errors='ignore')

        self.cat_encoders = {}
        for col in self.cat_cols:
            if col not in full_df.columns:
                raise KeyError(f"Categorical column '{col}' not found")
            full_df[col] = full_df[col].fillna('UNKNOWN').astype(str)
            le = LabelEncoder().fit(full_df[col])
            full_df[col] = le.transform(full_df[col])
            self.cat_encoders[col] = le

        self.num_cols = [c for c in full_df.columns if c not in self.cat_cols]
        self.all_cols = self.num_cols + self.cat_cols

        full_df = full_df[self.all_cols]

        n = len(full_df)
        n_train = int(n * self.data_split[0])
        n_test  = int(n * self.data_split[2])
        n_val   = n - n_train - n_test
        borders = [0, n_train - self.in_len,
                   n_train + n_val - self.in_len, n]
        b1, b2 = borders[self.set_type], borders[self.set_type + 1]


        if self.scale and self.num_cols:
            self.scaler = StandardScaler()
            if self.scale_statistic is None:
                self.scaler.fit(full_df[self.num_cols].iloc[:n_train])
            else:
                self.scaler.mean_  = np.asarray(self.scale_statistic['mean'])
                self.scaler.scale_ = np.asarray(self.scale_statistic['std'])
                self.scaler.n_features_in_ = len(self.scaler.mean_)
            full_df[self.num_cols] = (
                self.scaler.transform(full_df[self.num_cols])
                .astype(np.float32) 
            )
        else:
            self.scaler = None

        self.data_x = full_df.iloc[b1:b2].to_numpy(dtype=np.float32)
        self.data_y = self.data_x  


    def __len__(self):
        return len(self.data_x) - self.in_len - self.out_len + 1

    def __getitem__(self, idx):
        s, e = idx, idx + self.in_len
        rs, re = e, e + self.out_len
        return (torch.from_numpy(self.data_x[s:e]),
                torch.from_numpy(self.data_y[rs:re]))

    def inverse_transform(self, arr: np.ndarray):
        """Inverse-standardise numeric part only."""
        if self.scaler is None or not self.num_cols:
            return arr
        arr = arr.copy()
        arr[..., :len(self.num_cols)] = self.scaler.inverse_transform(
            arr[..., :len(self.num_cols)])
        return arr

    def get_scale_statistic(self):
        if self.scaler is None:
            return None
        return {'mean': self.scaler.mean_.copy(),
                'std':  self.scaler.scale_.copy()}