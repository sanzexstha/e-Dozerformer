import os
from logging import root

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler
from utils.timefeatures import time_features
from utils.tools import get_statistical, get_statistical_dan
from utils.scale import StandardNorm
from utils.ext_utils import log_std_denorm_dataset
from utils.misc import fprint


import warnings

warnings.filterwarnings('ignore')


class Dataset_MTS(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None, features='M',
                 data_split=[0.7, 0.1, 0.2], scale=True, scale_statistic=None, target='OT', timeenc=0, freq='h', cycle=None):
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
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
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

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_label = data_label[border1:border2]

    def __getitem__(self, index):
        s_begin = index
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
        return len(self.data_x) - self.in_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Ross(Dataset):
    """
    Dozerformer-style dataset for Ross_S_fixed.csv (no rain/watershed).

    Matches DAN's (DS.py) preprocessing exactly for a fair comparison:
      - Same timestamp-based train/val/test splits
      - Same log + z-score normalization (fit on training data only)
      - Same hydro-year filter: prediction target must fall in Sep-May (excludes Jun/Jul/Aug)
      - Same NaN rejection
      - Same train_volume (30000) randomly sampled from eligible windows
      - Same train_seed (1010)
    """

    def __init__(
        self,
        root_path,
        data_path,
        flag="train",
        size=None,          # [in_len, label_len, pred_len] — matches Dozerformer interface
        start_point="1988-01-01 14:30:00",
        val_point="2020-09-01 00:30:00",    # start of val period (carved from training data)
        train_point="2021-08-31 23:30:00",  # end of val / end of train+val combined
        test_start="2021-09-01 00:30:00",
        test_end="2022-05-31 23:30:00",
        scale=True,
        scale_statistic=None,
        train_volume=30000,
        train_seed=1010,
        test_stride=16,
        features='S', target='OT', timeenc=0, freq='h', cycle=None
    # mirrors DAN's gen_test_data stride of 16 steps (every 4 hours)
    ):
        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.in_len    = size[0] if size is not None else 1440
        self.label_len = size[1] if size is not None else 0
        self.pred_len  = size[2] if size is not None else 288
        self.scale = scale
        self.scale_statistic = scale_statistic
        self.train_volume = train_volume
        self.train_seed = train_seed
        self.test_stride = test_stride

        self.start_point = start_point
        self.val_point   = val_point
        self.train_point = train_point
        self.test_start  = test_start
        self.test_end    = test_end

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), sep="\t")
        df_raw.columns = ["id", "datetime", "value"]
        df_raw.sort_values("datetime", inplace=True)
        df_raw.reset_index(drop=True, inplace=True)

        # --- Timestamp-based split boundaries ---
        start_idx      = df_raw[df_raw["datetime"] == self.start_point].index[0]
        val_start_idx  = df_raw[df_raw["datetime"] == self.val_point].index[0]
        train_end_idx  = df_raw[df_raw["datetime"] == self.train_point].index[0]
        test_start_idx = df_raw[df_raw["datetime"] == self.test_start].index[0]
        test_end_idx   = df_raw[df_raw["datetime"] == self.test_end].index[0]

        # train : start_point → val_point
        # val   : val_point - in_len → train_point  (in_len context + val period)
        # test  : test_start - in_len → test_end
        border1s = [
            start_idx,
            val_start_idx  - self.in_len,
            test_start_idx - self.in_len,
        ]
        border2s = [
            val_start_idx,
            train_end_idx,
            test_end_idx,
        ]

        flag_map = {"train": 0, "val": 1, "test": 2}
        idx = flag_map[self.flag]
        b1, b2 = border1s[idx], border2s[idx]

        raw_values = df_raw["value"].values.astype(np.float32)

        # --- Log + z-score normalization (mirrors DS.py log_std_normalization) ---
        if self.scale:
            train_raw = raw_values[border1s[0]: border2s[0]]
            if self.scale_statistic is None:
                log_train = np.log(train_raw + 1)
                self.mean = float(np.nanmean(log_train))
                self.std  = float(np.nanstd(log_train))
            else:
                self.mean = self.scale_statistic["mean"]
                self.std  = self.scale_statistic["std"]
            data = (np.log(raw_values + 1) - self.mean) / self.std
        else:
            data = raw_values
            self.mean, self.std = 0.0, 1.0

        self.data_x    = data[b1:b2].astype(np.float32)
        self.data_y    = data[b1:b2].astype(np.float32)
        self.data_time = df_raw["datetime"].values[b1:b2]

        # --- Build eligible indices ---
        all_valid = []
        total = len(self.data_x) - self.in_len - self.pred_len + 1
        for i in range(total):
            window = self.data_x[i: i + self.in_len + self.pred_len]

            # 1. Reject NaN windows (mirrors DS.py NaN check)
            if np.isnan(window).any():
                continue

            # 2. Hydro-year filter: prediction target month must be Sep-May
            #    (mirrors DS.py: tag <= -9 or -6 < tag < 0, i.e. exclude Jun/Jul/Aug)
            #    The prediction start is at index i + in_len
            pred_datetime = self.data_time[i + self.in_len]
            month = int(pred_datetime[5:7])
            if month in (6, 7, 8):
                continue

            all_valid.append(i)

        # 3. For training, subsample to train_volume (mirrors DS.py train_volume=30000)
        if self.flag == "train" and self.train_volume is not None:
            rng = np.random.default_rng(self.train_seed)
            size = min(self.train_volume, len(all_valid))
            self.indices = rng.choice(all_valid, size=size, replace=False).tolist()
        elif self.flag == "test":
            # Mirrors DAN's gen_test_data exactly:
            #   begin_num = test_start offset; data_x starts in_len before test_start
            #   so s_begin = i * stride, num_windows = int((len-in_len-pred_len) / stride)
            #   NaN check covers the full window [s_begin : s_begin + in_len + pred_len]
            #   No hydro-year filter (DAN doesn't apply it to test)
            num_windows = int((len(self.data_x) - self.in_len - self.pred_len) / self.test_stride)
            self.indices = []
            for i in range(num_windows):
                s_begin = i * self.test_stride
                if not np.isnan(self.data_x[s_begin: s_begin + self.in_len + self.pred_len]).any():
                    self.indices.append(s_begin)
        else:
            self.indices = all_valid

    def __getitem__(self, index):
        s_begin = self.indices[index]
        s_end   = s_begin + self.in_len
        r_begin = s_end - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end].reshape(-1, 1)   # (in_len, 1)
        seq_y = self.data_y[r_begin:r_end].reshape(-1, 1)   # (label_len + pred_len, 1)

        # Dummy time-mark and cycle features — zeros since we use log+z-score only
        seq_x_mark  = np.zeros((self.in_len, 4),                       dtype=np.float32)
        seq_y_mark  = np.zeros((self.label_len + self.pred_len, 4),    dtype=np.float32)
        cycle_index = np.zeros(1,                                       dtype=np.int64)
        label_y     = self.data_y[s_begin:s_end].reshape(-1, 1)  # (label_len + pred_len, 1) — same as seq_y for now

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, label_y

    def __len__(self):
        return len(self.indices)

    def get_scale_statistic(self):
        return {"mean": self.mean, "std": self.std}

    def inverse_transform(self, data):
        """Undo z-score then log — returns original scale values."""
        return np.exp(np.array(data) * self.std + self.mean) - 1


# =============================================================================
# NORMALIZATION UTILITIES
# =============================================================================
def standard_denormalization(norm_data, mean, std):
    """Reverse the standard normalization."""
    norm_data = np.array(norm_data)
    return norm_data * std + mean


def get_statistical(file_path):
    """Load pre-computed mean and std from a saved .pt file."""
    stats_path = os.path.join(file_path, "mean_std_mini.pt")
    try:
        statistics_data = torch.load(stats_path, map_location='cpu', weights_only=False)
    except TypeError:
        statistics_data = torch.load(stats_path, map_location='cpu')

    train_mean = statistics_data['stdn_mean']
    train_std = statistics_data['stdn_std']
    return train_mean, train_std


class Dataset_Reservoir(Dataset):
    def __init__(self, root_path, data_path='', flag='train', size=None,
                 scale_statistic=None, features='M',
                 scale=True,target='OT', timeenc=0, freq='h', cycle=None):
        """
        Args:
            root_path (str): Base directory
            data_path (str): Subdirectory under root_path (can be '')
            flag (str): One of 'train', 'val', 'test'
            size (list): [seq_len, label_len, pred_len]
            scale_statistic (dict): Not used (kept for API compatibility).
        """
        assert flag in ['train', 'val', 'test'], "flag must be 'train', 'val', or 'test'"
        assert size is not None and len(size) == 3, "size must be [seq_len, label_len, pred_len]"

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path

        # These are set differently depending on flag
        self.data_x = None       # For test: pre-windowed (N, seq_len, 1)
        self.data_y = None       # For test: pre-windowed (N, label_len+pred_len, 1)
        self.data_flat = None    # For train/val: flat array (T, 1)
        self.prewindowed = False

        self.mean = None
        self.std = None
        self.scaler = None

        self.__read_data__()

    def __read_data__(self):
        base_dir = os.path.join(self.root_path, self.data_path,
                                f'in{self.seq_len}_out{self.pred_len}')

        # --- Load normalization statistics ---
        self.mean, self.std = get_statistical(base_dir)

        # --- Reconstruct StandardScaler ---
        stat = torch.load(os.path.join(base_dir, 'mean_std_mini.pt'),
                          map_location='cpu', weights_only=False)
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array([stat['scaler_mean']])
        self.scaler.scale_ = np.array([stat['scaler_scale']])
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = 1

        # --- Load data based on flag ---
        if self.flag == 'test':
            # Test: pre-windowed .npy (matches MCANN's 1086 test points)
            self.data_x = np.load(os.path.join(base_dir, 'test_x.npy'))
            self.data_y = np.load(os.path.join(base_dir, 'test_y.npy'))
            self.prewindowed = True
        else:
            # Train/Val: flat scaled .npy (slice on the fly, like Dataset_MTS)
            file_map = {'train': 'train_data.npy', 'val': 'val_data.npy'}
            self.data_flat = np.load(os.path.join(base_dir, file_map[self.flag]))
            self.prewindowed = False

    def __getitem__(self, index):
        if self.prewindowed:
            # Test: instant array lookup
            seq_x = self.data_x[index]
            seq_y = self.data_y[index]
        else:
            # Train/Val: slice window on the fly (like Dataset_MTS)
            s_begin = index
            s_end   = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end   = r_begin + self.label_len + self.pred_len

            seq_x = self.data_flat[s_begin:s_end]
            seq_y = self.data_flat[r_begin:r_end]

        seq_x_mark = 6.5
        seq_y_mark = 5.3
        cycle_index = 6
        label_y = np.zeros((self.seq_len, 1), dtype=np.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, label_y  # No label for reservoir dataset

    def __len__(self):
        if self.prewindowed:
            return self.data_x.shape[0]
        else:
            return len(self.data_flat) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """Inverse using self.mean and self.std (nanmean/nanstd based)."""
        return standard_denormalization(data, self.mean, self.std)

class Dataset_MTS_NPY(Dataset):
    """
    NPY loader for pre-windowed files generated by data_processing.py.
    Expected files under {root_path}/{data_path}:
        train_x.npy, train_y.npy, val_x.npy, val_y.npy, test_x.npy, test_y.npy
    """

    def __init__(self, root_path, data_path, flag='train', size=None, features='M',
                 target='OT', timeenc=0, freq='h', cycle=None, norm_type='std',
                 merge_to_series=False, scale_statistic=None, Scale=None):
        assert flag in ['train', 'val', 'test']
        assert size is not None and len(size) == 3, "size must be [seq_len, label_len, pred_len]"

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
        self.norm_type = norm_type
        self.merge_to_series = merge_to_series
        self.scale_statistic = scale_statistic

        self.data_x = None
        self.data_y_full = None
        self.data_x_sel = None
        self.data_y_target = None
        self.series_x = None
        self.series_y = None
        self.series_label = None
        self.series_cycle = None
        self.anomaly = None
        self.mean = None
        self.std = None

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        base_dir = os.path.join(self.root_path, self.data_path)
        x_path = os.path.join(base_dir, f"{self.flag}_x.npy")
        y_path = os.path.join(base_dir, f"{self.flag}_y.npy")

        self.data_x = np.load(x_path).astype(np.float32)      # (N, seq_len, Cx)
        self.data_y_full = np.load(y_path).astype(np.float32) # (N, out_len, Cy)

        # y channel 4 = raw ground-truth in original extreme pipeline; fallback to channel 0 for manual labels.
        y_target_col = 4
        self.data_y_target = self.data_y_full[:, :, [y_target_col]]

        if self.norm_type == 'std':
            if self.scale_statistic is None:
                stat_file = os.path.join(base_dir, "mean_std_mini.pt")
                if os.path.isfile(stat_file):
                    _, _, _, train_mean, train_std = get_statistical(base_dir)
                    self.scale_norm = StandardNorm(mean=train_mean, std=train_std)
                    self.mean = train_mean
                    self.std = train_std

        # x channel mapping from original get_data: ori->6, std->5, all->all.
        if self.norm_type == 'ori':
            x_col = 6
            self.data_x_sel = self.data_x[:, :, [x_col]]
        elif self.norm_type == 'std':
            x_col = 5
            self.data_x_sel = self.data_x[:, :, [x_col]]
            self.data_y_target = self.scale_norm.transform(self.data_y_target)
        else:
            self.data_x_sel = self.data_x

        # create anomaly flag from dim 1 (prob_like_outlier)
        self.anomaly = (self.data_x[:, :, 1:2] > 0.9).astype(np.float32)  # (N, input_len, 1)
        # Anomaly stats
        fprint("Anomaly Stats", {
            "Flag": self.flag,
            "Number of 1s": int(self.anomaly.sum()),
            "Total Elements": self.anomaly.size,
            "Percentage": f"{self.anomaly.mean() * 100:.2f}%"
        })

    def __getitem__(self, index):
        seq_x = self.data_x_sel[index]   # (360, 1)
        seq_y = self.data_y_target[index]   # (72, 1)
        # dummy values
        seq_x_mark = 6.5
        seq_y_mark = 5.3
        cycle_index = 6
        label_y = self.anomaly[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, label_y

    def __len__(self):
        return self.data_x_sel.shape[0]

    def inverse_transform(self, data, norm_type=None, part='test', obj='predict'):
        if self.norm_type == 'std' and self.scale_norm is not None:
            use_norm_type = self.norm_type if norm_type is None else norm_type
            return self.scale_norm.inverse_transform(data, use_norm_type, part, obj)
        return data



class Dataset_DAN_Watershed(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None, features='M',
                 target='OT', timeenc=0, freq='h', cycle=None, dan_norm_type='std',
                 merge_to_series=False, scale_statistic=None, Scale=None):
        assert flag in ['train', 'val', 'test']
        assert size is not None and len(size) == 3, "size must be [seq_len, label_len, pred_len]"

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
        self.dan_norm_type = dan_norm_type
        self.merge_to_series = merge_to_series
        self.scale_statistic = scale_statistic

        self.data_x = None
        self.data_y_full = None
        self.data_x_sel = None
        self.data_y_target = None
        self.series_x = None
        self.series_y = None
        self.series_label = None
        self.series_cycle = None
        self.anomaly = None
        self.mean = None
        self.std = None

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        base_dir = os.path.join(self.root_path, self.data_path, f'in{self.seq_len}_out{self.pred_len}')
        x_path = os.path.join(base_dir, f"{self.flag}_x.npy")
        y_path = os.path.join(base_dir, f"{self.flag}_y.npy")

        self.data_x = np.load(x_path).astype(np.float32)      # (N, seq_len, Cx)
        self.data_y_full = np.load(y_path).astype(np.float32) # (N, out_len, Cy)

        # y channel 4 = raw ground-truth in original extreme pipeline; fallback to channel 0 for manual labels.
        # 8 standard normalized GT ← new, 0 log-std normalized GT
        y_target_col = None
        if self.dan_norm_type == 'std':
            y_target_col = 4
        elif self.dan_norm_type == 'log-std':
            y_target_col = 0

        self.data_y_target = self.data_y_full[:, :, [y_target_col]]
        print("norm type:", self.dan_norm_type)
        #
        # if self.dan_norm_type == 'std':
        #     if self.scale_statistic is None:
        stat_file = os.path.join(base_dir, "mean_std_mini.pt")
        if os.path.isfile(stat_file):
            train_mean, train_std = get_statistical_dan(base_dir, self.dan_norm_type)
            self.scale_norm = StandardNorm(mean=train_mean, std=train_std)
            self.mean = train_mean
            self.std = train_std

        # x channel mapping from original get_data: ori->6, std->5, all->all, 0-> log-std
        x_col = None
        if self.dan_norm_type == 'ori':
            x_col = 6
        elif self.dan_norm_type == 'std':
            x_col = 5
            self.data_y_target = self.scale_norm.transform(self.data_y_target)
        elif self.dan_norm_type == 'log-std':
            x_col = 0
        else:
            self.data_x_sel = self.data_x

        if x_col is not None:
            self.data_x_sel = self.data_x[:, :, [x_col]]

        # create anomaly flag from dim 1 (prob_like_outlier)
        self.anomaly = (self.data_x[:, :, 1:2] > 0.9).astype(np.float32)  # (N, input_len, 1)
        print(f"flag: {self.flag}")
        print(f"Number of 1s: {int(self.anomaly.sum())}")
        print(f"Total elements: {self.anomaly.size}")
        print(f"Percentage: {self.anomaly.mean() * 100:.2f}%")


    def __getitem__(self, index):
        seq_x = self.data_x_sel[index]   # (seq_len, 1)
        seq_y = self.data_y_target[index]   # (pred_len, 1)
        # dummy values
        seq_x_mark = 6.5
        seq_y_mark = 5.3
        cycle_index = 6
        label_y = self.anomaly[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, label_y

    def __len__(self):
        return self.data_x_sel.shape[0]

    def inverse_transform(self, data, norm_type=None, part='test', obj='predict'):
        if self.dan_norm_type == 'std' and self.scale_norm is not None:
            use_norm_type = self.dan_norm_type if norm_type is None else norm_type
            return self.scale_norm.inverse_transform(data, use_norm_type, part, obj)
        else:
            return log_std_denorm_dataset(self.mean, self.std, data)


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
