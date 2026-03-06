import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings

warnings.filterwarnings('ignore')


def _preprocess_watershed_dataframe(file_path):
    """Load watershed TSV-like files and convert to common [date, OT, label] schema."""
    df_raw = pd.read_csv(file_path, sep='\t')

    # Some files may be read as a single tab-delimited column; split them explicitly.
    if df_raw.shape[1] == 1:
        expanded = df_raw.iloc[:, 0].astype(str).str.split('\t', expand=True)
        if expanded.shape[1] >= 3:
            expanded = expanded.iloc[:, -3:]
            expanded.columns = ['index', 'datetime', 'value']
            df_raw = expanded

    # Remove index-like helper columns exported by pandas.
    df_raw = df_raw.drop(columns=[c for c in df_raw.columns if str(c).lower().startswith('unnamed')], errors='ignore')

    if 'datetime' in df_raw.columns:
        date_col = 'datetime'
    elif 'date' in df_raw.columns:
        date_col = 'date'
    else:
        raise ValueError(f'Cannot find datetime column in watershed file: {file_path}')

    value_col = None
    for candidate in ['value', 'rainfall', 'streamflow', 'flow', 'ot']:
        if candidate in [c.lower() for c in df_raw.columns]:
            for col in df_raw.columns:
                if col.lower() == candidate:
                    value_col = col
                    break
            break

    if value_col is None:
        numeric_candidates = [c for c in df_raw.columns if c != date_col]
        if not numeric_candidates:
            raise ValueError(f'Cannot find value column in watershed file: {file_path}')
        value_col = numeric_candidates[-1]

    df = pd.DataFrame()
    df['date'] = pd.to_datetime(df_raw[date_col], errors='coerce')
    df['OT'] = pd.to_numeric(df_raw[value_col], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    # DAN_loader applies log-based normalization; apply log1p before scaling here.
    df['OT'] = np.log1p(df['OT'].clip(lower=0))

    ot_series = pd.Series(df['OT'].values, index=df['date'])
    ot_series = ot_series.interpolate(method='time').ffill().bfill().fillna(0.0)
    df['OT'] = ot_series.values.astype(np.float32)
    return df


class Dataset_MTS_ross_v1(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None, features='M',
                 data_split=[0.7, 0.1, 0.2], scale=True, scale_statistic=None, target='OT', timeenc=0, freq='h',
                 cycle=None, dataset_name=None, start_point=None, train_point=None, test_start=None, test_end=None,
                 train_seed=None, train_volume=None, val_seed=None, val_size=None, test_stride=16):
        # size [seq_len, label_len, pred_len]
        # info
        self.in_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.scale = scale
        # self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.dataset_name = dataset_name
        self.start_point = start_point
        self.train_point = train_point
        self.test_start = test_start
        self.test_end = test_end
        self.train_seed = train_seed
        self.train_volume = train_volume
        self.val_seed = val_seed
        self.val_size = val_size
        self.test_stride = test_stride
        self.sample_indices = None
        self.__read_data__()

    def __read_data__(self):
        file_path = os.path.join(self.root_path, self.data_path)
        # if self.data_path.startswith('watershed/'):
        #     df_raw = _preprocess_watershed_dataframe(file_path)
        # else:
        df_raw = pd.read_csv(file_path)

        is_ross_no_rain = self.dataset_name == 'Ross_noRain'
        has_date_args = all(x is not None for x in [self.start_point, self.train_point, self.test_start, self.test_end])
        if is_ross_no_rain and has_date_args:
            dt_series = pd.to_datetime(df_raw.iloc[:, 0], errors='coerce').astype(str)
            idx_map = pd.Series(np.arange(len(dt_series)), index=dt_series)
            try:
                idx_start = int(idx_map[str(self.start_point)])
                idx_train = int(idx_map[str(self.train_point)])
                idx_test_start = int(idx_map[str(self.test_start)])
                idx_test_end = int(idx_map[str(self.test_end)])
            except KeyError as exc:
                raise ValueError(f'Missing Ross_noRain split timestamp: {exc}')

            df_raw = df_raw.iloc[idx_start:idx_test_end + 1].reset_index(drop=True)
            ross_train_end = idx_train - idx_start + 1
            ross_test_begin = idx_test_start - idx_start
            ross_test_end = idx_test_end - idx_start + 1
        if (self.data_split[0] < 1):
            train_num = int(len(df_raw) * self.data_split[0])
            test_num = int(len(df_raw) * self.data_split[2])
            val_num = len(df_raw) - train_num - test_num
        else:
            train_num = self.data_split[0]
            val_num = self.data_split[1]
            test_num = self.data_split[2]
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num + val_num, train_num + val_num + test_num]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        # assume, first col = timestamp, last col = label
        cols_data = df_raw.columns[1:-1]  # all feature columns
        col_label = df_raw.columns[-1]  # the label column

        df_data = df_raw[cols_data]
        df_label = df_raw[[col_label]]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean=self.scale_statistic['mean'], std=self.scale_statistic['std'])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        data_label = df_label.values  # raw labels, no scaling

        if is_ross_no_rain and has_date_args:
            # Keep the full date-ranged timeline; sampled indices define train/val/test membership.
            self.data_x = data
            self.data_y = data
            self.data_label = data_label
        else:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.data_label = data_label[border1:border2]

        if is_ross_no_rain and has_date_args:
            sample_len = self.in_len + self.pred_len
            all_train_starts = np.arange(max(0, ross_train_end - sample_len + 1))

            rng_val = np.random.RandomState(0 if self.val_seed is None else int(self.val_seed))
            val_count = min(int(self.val_size) if self.val_size is not None else 120, len(all_train_starts))
            val_starts = rng_val.choice(all_train_starts, size=val_count, replace=False) if val_count > 0 else np.array([], dtype=int)

            train_pool = np.setdiff1d(all_train_starts, val_starts, assume_unique=False)
            rng_train = np.random.RandomState(0 if self.train_seed is None else int(self.train_seed))
            train_count = min(int(self.train_volume) if self.train_volume is not None else len(train_pool), len(train_pool))
            train_starts = rng_train.choice(train_pool, size=train_count, replace=False) if train_count > 0 else np.array([], dtype=int)

            first_test_start = ross_test_begin - self.in_len
            stride = max(1, int(self.test_stride))
            test_count = int((ross_test_end - ross_test_begin - self.pred_len) / stride)
            if test_count > 0:
                test_starts = first_test_start + np.arange(test_count) * stride
            else:
                test_starts = np.array([], dtype=int)

            if self.flag == 'train':
                self.sample_indices = np.sort(train_starts.astype(int))
            elif self.flag == 'val':
                self.sample_indices = np.sort(val_starts.astype(int))
            else:
                self.sample_indices = test_starts.astype(int)

    def __getitem__(self, index):
        s_begin = int(self.sample_indices[index]) if self.sample_indices is not None else index
        s_end = s_begin + self.in_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        label_y = self.data_label[s_begin:s_end]  # corresponding labels

        seq_x_mark = 6.5
        seq_y_mark = 5.3
        cycle_index = 6
        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, label_y

    def __len__(self):
        if self.sample_indices is not None:
            return len(self.sample_indices)
        return len(self.data_x) - self.in_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # split features + label
        cols_feat = df_raw.columns[1:-1]  # everything except date and label
        col_label = df_raw.columns[-1]  # last column is label
        df_feat = df_raw[cols_feat]
        df_label = df_raw[[col_label]]
        data_label = df_label.values  # no scaling

        if self.features == 'M' or self.features == 'MS':
            # cols_data = df_raw.columns[1:]
            # df_data = df_raw[cols_data]
            df_data = df_feat
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # NEW: keep labels aligned with current split borders
        self.data_label = data_label[border1:border2]

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor(self.cycle_index[s_end])

        # NEW: label_y aligned with input window
        label_y = self.data_label[s_begin:s_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, label_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # split features + label
        cols_feat = df_raw.columns[1:-1]  # everything except date and label
        col_label = df_raw.columns[-1]  # last column is label
        df_feat = df_raw[cols_feat]
        df_label = df_raw[[col_label]]
        data_label = df_label.values  # no scaling

        if self.features == 'M' or self.features == 'MS':
            # cols_data = df_raw.columns[1:]
            # df_data = df_raw[cols_data]
            df_data = df_feat
        elif self.features == 'S':
            df_data = df_raw[[self.target]]



        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values


        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # NEW: keep labels aligned with current split borders
        self.data_label = data_label[border1:border2]

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor(self.cycle_index[s_end])
        # NEW: label_y aligned with input window
        label_y = self.data_label[s_begin:s_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, label_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
