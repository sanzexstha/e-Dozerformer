import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler
from utils.timefeatures import time_features
from utils.tools import get_statistical
from utils.scale import StandardNorm
from sklearn.mixture import GaussianMixture
from scipy import stats

import warnings

warnings.filterwarnings('ignore')


def _dan_log_std_normalization(sensor_data_val):
    data = np.log(np.array(sensor_data_val, dtype=np.float64) + 1.0)
    mean = np.nanmean(data)
    std = np.nanstd(data)
    if std == 0 or np.isnan(std):
        std = 1.0
    data = (data - mean) / std
    return data.astype(np.float32), float(mean), float(std)


def _dan_log_std_normalization_with_stats(sensor_data_val, mean, std):
    data = np.log(np.array(sensor_data_val, dtype=np.float64) + 1.0)
    std = 1.0 if std == 0 or np.isnan(std) else std
    data = (data - mean) / std
    return data.astype(np.float32)


def _dan_log_std_inverse(sensor_data_val, mean, std):
    if torch.is_tensor(sensor_data_val):
        mean_t = torch.as_tensor(mean, dtype=sensor_data_val.dtype, device=sensor_data_val.device)
        std_t = torch.as_tensor(std, dtype=sensor_data_val.dtype, device=sensor_data_val.device)
        return torch.exp(sensor_data_val * std_t + mean_t) - 1.0
    return np.exp(np.array(sensor_data_val) * std + mean) - 1.0


def _dan_standard_normalization(sensor_data_val):
    data = np.array(sensor_data_val, dtype=np.float64)
    mean = np.nanmean(data)
    std = np.nanstd(data)
    if std == 0 or np.isnan(std):
        std = 1.0
    data = (data - mean) / std
    return data.astype(np.float32), float(mean), float(std)


def _dan_standard_normalization_with_stats(sensor_data_val, mean, std):
    data = np.array(sensor_data_val, dtype=np.float64)
    std = 1.0 if std == 0 or np.isnan(std) else std
    data = (data - mean) / std
    return data.astype(np.float32)


def _dan_standard_inverse(sensor_data_val, mean, std):
    if torch.is_tensor(sensor_data_val):
        mean_t = torch.as_tensor(mean, dtype=sensor_data_val.dtype, device=sensor_data_val.device)
        std_t = torch.as_tensor(std, dtype=sensor_data_val.dtype, device=sensor_data_val.device)
        return sensor_data_val * std_t + mean_t
    return np.array(sensor_data_val) * std + mean


def _load_dan_sensor_file(file_path):
    df = pd.read_csv(file_path, sep='\t')
    if 'datetime' not in df.columns:
        if df.shape[1] < 2:
            raise ValueError(f'Unexpected Dan dataset format in {file_path}')
        datetime_col = df.columns[1]
    else:
        datetime_col = 'datetime'

    value_col = 'value' if 'value' in df.columns else df.columns[-1]
    out = df[[datetime_col, value_col]].copy()
    out.columns = ['datetime', 'value']
    out['datetime'] = out['datetime'].astype(str)
    out['value'] = pd.to_numeric(out['value'], errors='coerce')
    out.sort_values('datetime', inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def _find_exact_time_index(df, timestamp, source_path):
    idx = df.index[df['datetime'] == str(timestamp)].tolist()
    if not idx:
        min_time = df['datetime'].iloc[0]
        max_time = df['datetime'].iloc[-1]
        raise ValueError(
            f"Timestamp '{timestamp}' was not found in {source_path}. "
            f'Available range: [{min_time}, {max_time}]'
        )
    return idx[0]


def _is_hydro_month(tag_value):
    # Match Dan's month filtering logic exactly.
    return (
        (tag_value <= -9)
        or (-6 < tag_value < 0)
        or (2 <= tag_value <= 3)
    )


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

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        base_dir = os.path.join(self.root_path, self.data_path)
        x_path = os.path.join(base_dir, f"{self.flag}_x.npy")
        y_path = os.path.join(base_dir, f"{self.flag}_y.npy")

        self.data_x = np.load(x_path).astype(np.float32)      # (N, seq_len, Cx)
        self.data_y_full = np.load(y_path).astype(np.float32) # (N, out_len, Cy)

        # y channel 4 = raw ground-truth in original extreme pipeline; fallback to channel 0 for manual labels.
        y_target_col = 4 if self.data_y_full.shape[2] > 4 else 0
        self.data_y_target = self.data_y_full[:, :, [y_target_col]]

        if self.norm_type == 'std':
            if self.scale_statistic is None:
                stat_file = os.path.join(base_dir, "mean_std_mini.pt")
                if os.path.isfile(stat_file):
                    _, _, _, train_mean, train_std = get_statistical(base_dir)
                    self.scale_norm = StandardNorm(mean=train_mean, std=train_std)

        # x channel mapping from original get_data: ori->6, std->5, all->all.
        if self.norm_type == 'ori':
            x_col = 6 if self.data_x.shape[2] > 6 else 0
            self.data_x_sel = self.data_x[:, :, [x_col]]
        elif self.norm_type == 'std':
            x_col = 5 if self.data_x.shape[2] > 5 else 0
            self.data_x_sel = self.data_x[:, :, [x_col]]
            self.data_y_target = self.scale_norm.transform(self.data_y_target)
        else:
            self.data_x_sel = self.data_x

        # create anomaly flag from dim 1 (prob_like_outlier)
        self.anomaly = (self.data_x[:, :, 1:2] > 0.5).astype(np.float32)  # (N, input_len, 1)

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
    """
    Dan-style loader for watershed datasets such as:
      Ross_S_fixed.csv / Ross_R_fixed.csv
      Saratoga_S_fixed.csv / Saratoga_R_fixed.csv
      SFC_S_fixed.csv / SFC_R_fixed.csv
      UpperPen_S_fixed.csv / UpperPen_R_fixed.csv
    """

    def __init__(self, root_path, data_path, flag='train', size=None, features='M',
                 target='OT', timeenc=0, freq='h', cycle=None, dataset_name=None,
                 start_point=None, train_point=None, test_start=None, test_end=None,
                 train_seed=1010, train_volume=30000, val_seed=2007, val_size=120, test_stride=16,
                 watershed=1, rain_data_path=None, oversampling=80, event_focus_level=18,
                 dan_norm_type='logstd', **kwargs):
        assert flag in ['train', 'val', 'test']
        assert size is not None and len(size) == 3, 'size must be [seq_len, label_len, pred_len]'

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.dataset_name = dataset_name if dataset_name is not None else data_path
        self.cycle = 24 if cycle is None else cycle

        self.start_point = start_point or '1988-01-01 14:30:00'
        self.train_point = train_point or '2021-08-31 23:30:00'
        self.test_start = test_start or '2021-09-01 00:30:00'
        self.test_end = test_end or '2022-05-31 23:30:00'

        self.train_seed = int(train_seed)
        self.train_volume = int(train_volume)
        self.val_seed = int(val_seed)
        self.val_size = int(val_size)
        self.test_stride = int(test_stride)
        self.watershed = int(watershed)
        self.rain_data_path = rain_data_path
        self.oversampling = float(oversampling)
        self.event_focus_level = int(event_focus_level)

        self.dan_norm_type = str(dan_norm_type).lower()
        if self.dan_norm_type in ['std', 'zscore', 'standard']:
            self.dan_norm_type = 'zscore'
        elif self.dan_norm_type in ['logstd', 'log_std']:
            self.dan_norm_type = 'logstd'
        else:
            raise ValueError("dan_norm_type must be one of: ['logstd', 'zscore']")

        self.logstd_mean = 0.0
        self.logstd_std = 1.0
        self.zscore_mean = 0.0
        self.zscore_std = 1.0
        self.stream_mean = 0.0
        self.stream_std = 1.0

        self.samples_x = []
        self.samples_y = []
        self.samples_label = []
        self.samples_cycle = []

        self.__read_data__()

    def _resolve_rain_path(self, stream_path):
        if self.rain_data_path is not None:
            return os.path.join(self.root_path, self.rain_data_path)
        if '_S_fixed' in self.data_path:
            inferred = self.data_path.replace('_S_fixed', '_R_fixed')
            inferred_path = os.path.join(self.root_path, inferred)
            if os.path.isfile(inferred_path):
                return inferred_path
        if '_S_fixed' in stream_path:
            inferred_path = stream_path.replace('_S_fixed', '_R_fixed')
            if os.path.isfile(inferred_path):
                return inferred_path
        return None

    def _build_outlier_indicator(self, values, gmm):
        values = np.array(values, dtype=np.float32)
        clean_mask = ~np.isnan(values)
        if clean_mask.sum() == 0:
            return np.full(values.shape[0], 0.5, dtype=np.float32)

        clean_values = values[clean_mask].reshape(-1, 1)
        data_prob = gmm.predict_proba(clean_values)
        weights = gmm.weights_
        prob_in_distribution = np.sum(data_prob * weights, axis=1)
        prob_like_outlier = 1.0 - prob_in_distribution

        recover = np.full(values.shape[0], 0.5, dtype=np.float32)
        recover[clean_mask] = prob_like_outlier.astype(np.float32)
        return recover

    def _sample_val_indices(self, stream_raw, rain_raw, tag):
        lens = self.seq_len + self.pred_len + 1
        left = self.pred_len
        right = len(stream_raw) - lens - 1
        if right <= left:
            return [], tag

        tag = np.array(tag, copy=True)
        near_len = self.pred_len
        rng = random.Random(self.val_seed)
        indices = []
        attempts = 0
        max_attempts = max(self.val_size * 200, len(stream_raw) * 6)

        while len(indices) < self.val_size and attempts < max_attempts:
            attempts += 1
            i = rng.randint(left, right)
            c = i + self.seq_len

            if c + self.pred_len >= len(stream_raw):
                continue

            if (
                (not np.isnan(stream_raw[i:i + lens]).any())
                and (not np.isnan(rain_raw[i:i + lens]).any())
                and _is_hydro_month(tag[c])
            ):
                tag[c] = 2
                for k in range(near_len):
                    l_idx = c - k
                    r_idx = c + k
                    if 0 <= l_idx < len(tag):
                        tag[l_idx] = 3
                    if 0 <= r_idx < len(tag):
                        tag[r_idx] = 3
                indices.append(i)

        if len(indices) < self.val_size:
            print(f'[Dataset_DAN_Watershed] Validation samples: requested {self.val_size}, got {len(indices)}')
        return indices, tag

    def _kruskal_h(self, future_raw):
        if len(future_raw) < 4:
            return 0.0
        groups = np.array_split(np.array(future_raw, dtype=np.float64), 4)
        if any(g.size == 0 for g in groups):
            return 0.0
        if all(np.array_equal(groups[0], g) for g in groups[1:]):
            return 0.0
        try:
            h_val, _ = stats.kruskal(*groups)
        except Exception:
            h_val = 0.0
        if np.isnan(h_val):
            return 0.0
        return float(h_val)

    def _sample_train_indices(self, stream_raw, rain_raw, tag):
        lens = self.seq_len + self.pred_len + 1
        left = self.pred_len
        right = len(stream_raw) - 31 * self.pred_len - 1
        fallback_right = len(stream_raw) - lens - 1
        right = right if right > left else fallback_right
        if right <= left:
            return []

        tag = np.array(tag, copy=True)
        rng = random.Random(self.train_seed)
        indices = []
        attempts = 0
        max_attempts = max(self.train_volume * 200, len(stream_raw) * 12)

        while len(indices) < self.train_volume and attempts < max_attempts:
            attempts += 1
            i = rng.randint(left, right)
            c = i + self.seq_len
            if c + self.pred_len >= len(stream_raw):
                continue

            if (
                (not np.isnan(stream_raw[i:i + lens]).any())
                and (not np.isnan(rain_raw[i:i + lens]).any())
                and _is_hydro_month(tag[c])
            ):
                future_raw = stream_raw[c:c + self.pred_len]
                h_val = self._kruskal_h(future_raw)
                if (h_val > self.oversampling) or (rng.randint(0, 99) <= self.event_focus_level):
                    indices.append(i)
                    tag[c] = 4

        if len(indices) < self.train_volume:
            print(f'[Dataset_DAN_Watershed] Train samples: requested {self.train_volume}, got {len(indices)}')
        return indices

    def _append_sample(self, stream_norm, outlier_prob, time_values, x_start, y_start):
        x_end = y_start
        y_end = y_start + self.pred_len

        if x_start < 0 or y_end > len(stream_norm):
            return

        seq_x = stream_norm[x_start:x_end]
        seq_y = stream_norm[y_start:y_end]
        label_y = (outlier_prob[x_start:x_end] > 0.5).astype(np.float32)

        if len(seq_x) != self.seq_len or len(seq_y) != self.pred_len:
            return

        cycle_index = 0
        if self.cycle and self.cycle > 0:
            try:
                hour = pd.Timestamp(time_values[y_start]).hour
                cycle_index = int(hour % self.cycle)
            except Exception:
                cycle_index = int(y_start % self.cycle)

        self.samples_x.append(seq_x.reshape(self.seq_len, 1).astype(np.float32))
        self.samples_y.append(seq_y.reshape(self.pred_len, 1).astype(np.float32))
        self.samples_label.append(label_y.reshape(self.seq_len, 1).astype(np.float32))
        self.samples_cycle.append(cycle_index)

    def __read_data__(self):
        stream_path = os.path.join(self.root_path, self.data_path)
        if not os.path.isfile(stream_path):
            raise FileNotFoundError(f'Stream dataset not found: {stream_path}')
        stream_df = _load_dan_sensor_file(stream_path)

        rain_df = None
        if self.watershed >= 1:
            rain_path = self._resolve_rain_path(stream_path)
            if rain_path is None or not os.path.isfile(rain_path):
                raise FileNotFoundError(
                    f'Rain dataset is required (watershed=1), but no matching file was found for {stream_path}'
                )
            rain_df = _load_dan_sensor_file(rain_path)

        s_start = _find_exact_time_index(stream_df, self.start_point, stream_path)
        s_train = _find_exact_time_index(stream_df, self.train_point, stream_path)
        s_test_start = _find_exact_time_index(stream_df, self.test_start, stream_path)
        s_test_end = _find_exact_time_index(stream_df, self.test_end, stream_path)

        if not (s_start < s_train <= s_test_start < s_test_end):
            raise ValueError(
                'Invalid split order. Expected start_point < train_point <= test_start < test_end'
            )

        stream_train_df = stream_df.iloc[s_start:s_train].copy()
        stream_full_df = stream_df.iloc[s_start:s_test_end].copy()

        stream_train_raw = stream_train_df['value'].to_numpy(dtype=np.float32)
        stream_full_raw = stream_full_df['value'].to_numpy(dtype=np.float32)
        stream_train_time = stream_train_df['datetime'].to_numpy()
        stream_full_time = stream_full_df['datetime'].to_numpy()

        stream_train_logstd, self.logstd_mean, self.logstd_std = _dan_log_std_normalization(stream_train_raw)
        stream_full_logstd = _dan_log_std_normalization_with_stats(
            stream_full_raw, self.logstd_mean, self.logstd_std
        )
        stream_train_zscore, self.zscore_mean, self.zscore_std = _dan_standard_normalization(stream_train_raw)
        stream_full_zscore = _dan_standard_normalization_with_stats(
            stream_full_raw, self.zscore_mean, self.zscore_std
        )

        if self.dan_norm_type == 'zscore':
            stream_train_norm = stream_train_zscore
            stream_full_norm = stream_full_zscore
            self.stream_mean, self.stream_std = self.zscore_mean, self.zscore_std
        else:
            stream_train_norm = stream_train_logstd
            stream_full_norm = stream_full_logstd
            self.stream_mean, self.stream_std = self.logstd_mean, self.logstd_std

        clean_train = stream_train_raw[~np.isnan(stream_train_raw)]
        if clean_train.size >= 3:
            gmm = GaussianMixture(n_components=3)
            gmm.fit(clean_train.reshape(-1, 1))
            outlier_train = self._build_outlier_indicator(stream_train_raw, gmm)
            outlier_full = self._build_outlier_indicator(stream_full_raw, gmm)
        else:
            outlier_train = np.full(stream_train_raw.shape[0], 0.5, dtype=np.float32)
            outlier_full = np.full(stream_full_raw.shape[0], 0.5, dtype=np.float32)

        if rain_df is not None:
            r_path = self._resolve_rain_path(stream_path)
            r_start = _find_exact_time_index(rain_df, self.start_point, r_path)
            r_train = _find_exact_time_index(rain_df, self.train_point, r_path)
            rain_train_raw = rain_df.iloc[r_start:r_train]['value'].to_numpy(dtype=np.float32)
        else:
            rain_train_raw = outlier_train

        # Dan training/validation logic uses both stream and rain arrays by shared index.
        train_common_len = min(len(stream_train_raw), len(rain_train_raw))
        stream_train_raw = stream_train_raw[:train_common_len]
        stream_train_norm = stream_train_norm[:train_common_len]
        outlier_train = outlier_train[:train_common_len]
        rain_train_raw = rain_train_raw[:train_common_len]
        stream_train_time = stream_train_time[:train_common_len]

        train_month = pd.to_datetime(stream_train_time).month.values.astype(np.int16)
        tag = -1 * train_month

        if self.flag == 'val':
            val_indices, _ = self._sample_val_indices(stream_train_raw, rain_train_raw, tag)
            for i in val_indices:
                self._append_sample(
                    stream_train_norm, outlier_train, stream_train_time, i, i + self.seq_len
                )
        elif self.flag == 'train':
            _, tag_after_val = self._sample_val_indices(stream_train_raw, rain_train_raw, tag)
            train_indices = self._sample_train_indices(stream_train_raw, rain_train_raw, tag_after_val)
            for i in train_indices:
                self._append_sample(
                    stream_train_norm, outlier_train, stream_train_time, i, i + self.seq_len
                )
        else:
            begin_num = s_test_start - s_start
            end_num = s_test_end - s_start
            total = int((end_num - begin_num - self.pred_len) / self.test_stride)
            for i in range(total):
                y_start = begin_num + i * self.test_stride
                x_start = y_start - self.seq_len
                x_end = y_start
                y_end = y_start + self.pred_len
                if x_start < 0 or y_end > len(stream_full_raw):
                    continue
                if np.isnan(stream_full_raw[x_start:y_end]).any():
                    continue
                self._append_sample(
                    stream_full_norm, outlier_full, stream_full_time, x_start, y_start
                )

    def __getitem__(self, index):
        seq_x = self.samples_x[index]
        seq_y = self.samples_y[index]
        label_y = self.samples_label[index]
        cycle_index = self.samples_cycle[index]
        seq_x_mark = 6.5
        seq_y_mark = 5.3
        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, label_y

    def __len__(self):
        return len(self.samples_x)

    def inverse_transform(self, data, norm_type=None, part='test', obj='predict'):
        if self.dan_norm_type == 'zscore':
            return _dan_standard_inverse(data, self.stream_mean, self.stream_std)
        return _dan_log_std_inverse(data, self.stream_mean, self.stream_std)


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
