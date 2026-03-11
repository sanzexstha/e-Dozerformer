import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class StandardScaler:
    """Z-score scaler (mean=0, std=1)."""
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        self.std[self.std == 0] = 1.0  # avoid division by zero

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def _preprocess_watershed_dataframe(file_path):
    """Load watershed TSV files and convert to [id, datetime, value] schema.

    Matches the reference preprocessing:
        trainX = pd.read_csv(..., sep="\\t")
        trainX.columns = ["id", "datetime", "value"]
        trainX.sort_values("datetime", inplace=True)
    """
    df_raw = pd.read_csv(file_path, sep='\t')

    # Handle files that may have been read as a single column.
    if df_raw.shape[1] == 1:
        expanded = df_raw.iloc[:, 0].astype(str).str.split('\t', expand=True)
        if expanded.shape[1] >= 3:
            expanded = expanded.iloc[:, -3:]
            expanded.columns = ['id', 'datetime', 'value']
            df_raw = expanded

    # Normalise column names to match reference: ["id", "datetime", "value"] + optional "label"
    expected_3 = ['id', 'datetime', 'value']
    expected_4 = ['id', 'datetime', 'value', 'label']
    if list(df_raw.columns) not in [expected_3, expected_4]:
        # Remove unnamed index columns that pandas may export.
        df_raw = df_raw.drop(
            columns=[c for c in df_raw.columns if str(c).lower().startswith('unnamed')],
            errors='ignore',
        )
        if df_raw.shape[1] == 4:
            df_raw.columns = ['id', 'datetime', 'value', 'label']
        elif df_raw.shape[1] == 3:
            df_raw.columns = ['id', 'datetime', 'value']
        elif df_raw.shape[1] == 2:
            # Assume [datetime, value]; synthesise an id column.
            df_raw.columns = ['datetime', 'value']
            df_raw.insert(0, 'id', range(len(df_raw)))
        else:
            raise ValueError(
                f'Expected 2-4 columns in watershed file, got {df_raw.shape[1]}: {file_path}'
            )

    # Sort by datetime – matches reference: trainX.sort_values("datetime", inplace=True)
    df_raw.sort_values('datetime', inplace=True)
    df_raw.reset_index(drop=True, inplace=True)

    # Coerce value to numeric; fill NaN same as reference (fillna(np.nan) is a no-op there,
    # but downstream the normalization handles NaN).
    df_raw['value'] = pd.to_numeric(df_raw['value'], errors='coerce')

    return df_raw


class Dataset_MTS_ross(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None, features='M',
                 data_split=[0.7, 0.1, 0.2], scale=True, scale_statistic=None, target='OT', timeenc=0, freq='h',
                 cycle=None, dataset_name=None, start_point=None, train_point=None, test_start=None, test_end=None,
                 train_seed=None, train_volume=None, val_seed=None, val_size=None, test_stride=16):
        # size [seq_len, label_len, pred_len]
        self.in_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.scale = scale
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

        is_ross_no_rain = self.dataset_name == 'Ross_noRain'
        has_date_args = all(
            x is not None for x in [self.start_point, self.train_point, self.test_start, self.test_end]
        )

        # ── Ross_noRain path: TSV with reference-matching preprocessing ──
        if is_ross_no_rain and has_date_args:
            # 1) Read TSV, set columns, sort – identical to reference.
            trainX = _preprocess_watershed_dataframe(file_path)
            # trainX now has columns ["id", "datetime", "value"], sorted by datetime.

            # 2) Locate split indices – mirrors reference exactly:
            #      start_num  = index where datetime == start_point
            #      train_end  = (index where datetime == train_point) - start_num
            try:
                start_num = trainX[trainX['datetime'] == str(self.start_point)].index.values[0]
            except IndexError:
                raise ValueError(f'start_point {self.start_point!r} not found in datetime column')
            try:
                train_point_idx = trainX[trainX['datetime'] == str(self.train_point)].index.values[0]
            except IndexError:
                raise ValueError(f'train_point {self.train_point!r} not found in datetime column')
            try:
                test_start_idx = trainX[trainX['datetime'] == str(self.test_start)].index.values[0]
            except IndexError:
                raise ValueError(f'test_start {self.test_start!r} not found in datetime column')
            try:
                test_end_idx = trainX[trainX['datetime'] == str(self.test_end)].index.values[0]
            except IndexError:
                raise ValueError(f'test_end {self.test_end!r} not found in datetime column')

            print(f"for sensor {self.data_path} start_num is: {start_num}")
            train_end = train_point_idx - start_num  # relative offset
            print(f"train set length is : {train_end}")

            # 3) Extract the full usable range: start_point → test_end (inclusive).
            full_data = trainX.loc[start_num: test_end_idx].reset_index(drop=True)
            values_all = np.array(full_data['value'].values, dtype=np.float64)

            # Extract label column (0=normal, 1=extreme) if present; else default to zeros.
            if 'label' in full_data.columns:
                labels_all = np.array(full_data['label'].values, dtype=np.float32)
            else:
                labels_all = np.zeros(len(full_data), dtype=np.float32)

            # Relative indices within full_data (which starts at start_num):
            ross_train_end = train_end                        # exclusive end of training window
            ross_test_begin = test_start_idx - start_num      # first test timestep
            ross_test_end = test_end_idx - start_num + 1      # exclusive end of test window

            # 4) Normalise using training portion only – matches reference:
            #      sensor_data = trainX[start_num : train_end + start_num]
            #      sensor_data_norm, mean, std = standar_normalization(data)
            train_values = values_all[:ross_train_end].copy()

            if self.scale:
                if self.scale_statistic is None:
                    self.scaler = StandardScaler()
                    # Reshape to (N, 1) so scaler stores per-column stats.
                    # nanmean/nanstd in fit() ignore NaNs, so stats reflect real measurements only.
                    self.scaler.fit(train_values.reshape(-1, 1))
                else:
                    self.scaler = StandardScaler(
                        mean=np.array(self.scale_statistic['mean']),
                        std=np.array(self.scale_statistic['std']),
                    )

                # Fill NaNs AFTER fitting (so fills don't bias mean/std), BEFORE transform.
                values_all = pd.Series(values_all).ffill().bfill().fillna(0.0).values

                data = self.scaler.transform(values_all.reshape(-1, 1)).astype(np.float32)
            else:
                # Fill NaNs even when not scaling.
                values_all = pd.Series(values_all).ffill().bfill().fillna(0.0).values
                data = values_all.reshape(-1, 1).astype(np.float32)

            # Labels: binary anomaly labels from the file (0=normal, 1=extreme).
            data_label = labels_all.reshape(-1, 1)

            self.data_x = data
            self.data_y = data
            self.data_label = data_label

            # 5) Build sample indices for train / val / test splits.
            sample_len = self.in_len + self.pred_len
            all_train_starts = np.arange(max(0, ross_train_end - sample_len + 1))

            # Validation: random subset of training windows.
            rng_val = np.random.RandomState(0 if self.val_seed is None else int(self.val_seed))
            val_count = min(
                int(self.val_size) if self.val_size is not None else 120,
                len(all_train_starts),
            )
            val_starts = (
                rng_val.choice(all_train_starts, size=val_count, replace=False)
                if val_count > 0 else np.array([], dtype=int)
            )

            # Training: remaining windows (optionally sub-sampled).
            train_pool = np.setdiff1d(all_train_starts, val_starts, assume_unique=False)
            rng_train = np.random.RandomState(0 if self.train_seed is None else int(self.train_seed))
            train_count = min(
                int(self.train_volume) if self.train_volume is not None else len(train_pool),
                len(train_pool),
            )
            train_starts = (
                rng_train.choice(train_pool, size=train_count, replace=False)
                if train_count > 0 else np.array([], dtype=int)
            )

            # Test: strided windows starting in the test region.
            first_test_start = ross_test_begin - self.in_len
            stride = max(1, int(self.test_stride))
            test_count = int((ross_test_end - ross_test_begin - self.pred_len) / stride)
            test_starts = (
                first_test_start + np.arange(test_count) * stride
                if test_count > 0 else np.array([], dtype=int)
            )

            if self.flag == 'train':
                self.sample_indices = np.sort(train_starts.astype(int))
            elif self.flag == 'val':
                self.sample_indices = np.sort(val_starts.astype(int))
            else:
                self.sample_indices = test_starts.astype(int)

            return  # ← done for Ross_noRain

        # ── Generic CSV path (non-Ross datasets) ──
        df_raw = pd.read_csv(file_path)

        if self.data_split[0] < 1:
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

        cols_data = df_raw.columns[1:-1]   # feature columns
        col_label = df_raw.columns[-1]     # label column

        df_data = df_raw[cols_data]
        df_label = df_raw[[col_label]]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(
                    mean=self.scale_statistic['mean'],
                    std=self.scale_statistic['std'],
                )
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_label = df_label.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_label = data_label[border1:border2]

    def __getitem__(self, index):
        s_begin = int(self.sample_indices[index]) if self.sample_indices is not None else index
        s_end = s_begin + self.in_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        label_y = self.data_label[s_begin:s_end]

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
