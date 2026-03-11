"""
Reservoir Storage Data Processing Pipeline
===========================================
This script performs the following operations:
1. Loads the raw TSV dataset
2. Counts the total number of timesteps
3. Splits the data into Training, Validation, and Test sets
4. Exports all three sets as CSV files
"""

import pandas as pd
import os
import numpy as np
# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_FILE = os.path.join("..", "data/datasets/reservoir/raw", "reservoir_stor_4007_sof24.tsv")
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


TRAIN_START     = "1991-07-01 23:30:00"
TRAIN_END       = "2018-06-30 23:30:00"
TEST_PRED_START = "2018-07-01 00:30:00"   # First prediction point in test
TEST_PRED_END   = "2019-07-01 00:30:00"   # End of test period

SEQ_LEN   = 360    # input_len  (15 days * 24 hours)
LABEL_LEN = 48     # decoder label length
PRED_LEN  = 72     # output_len (3 days * 24 hours)
VAL_SIZE  = 1000

TEST_ROLL = 8      # stride for test windows (match MCANN roll=8)

SAVE_DIR = "./"    # output directory


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def standard_normalization(x):
    """Compute nanmean and nanstd of array x."""
    x = np.array(x)
    mean = np.nanmean(x)
    std = np.nanstd(x)
    return mean, std


def create_windows(data, seq_len, label_len, pred_len, roll=1):
    """
    Create pre-windowed arrays from a flat scaled array.

    Returns:
        X: np.array of shape (N, seq_len, 1)
        Y: np.array of shape (N, label_len + pred_len, 1)
    """
    n_samples = (len(data) - seq_len - pred_len) // roll + 1
    X = np.zeros((n_samples, seq_len, data.shape[1]), dtype=np.float32)
    Y = np.zeros((n_samples, label_len + pred_len, data.shape[1]), dtype=np.float32)

    for i in range(n_samples):
        s_begin = i * roll
        s_end   = s_begin + seq_len
        r_begin = s_end - label_len
        r_end   = r_begin + label_len + pred_len

        X[i] = data[s_begin:s_end]
        Y[i] = data[r_begin:r_end]

    return X, Y


# =============================================================================
# STEP 1: Load the raw TSV dataset
# =============================================================================
print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

df = pd.read_csv(INPUT_FILE, sep="\t")
df.columns = ["datetime", "value"]
df["datetime"] = pd.to_datetime(df["datetime"])
df.sort_values("datetime", inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"  File loaded: {INPUT_FILE}")
print(f"  Columns: {list(df.columns)}")
print(f"  Total timesteps: {len(df)}")
print(f"  Date range: {df['datetime'].min()} → {df['datetime'].max()}")
print()

# =============================================================================
# STEP 2: Split into Training, Validation, and Test pools
# =============================================================================
print("=" * 60)
print("STEP 2: Splitting into Train / Validation / Test pools")
print("=" * 60)

train_start     = pd.Timestamp(TRAIN_START)
train_end       = pd.Timestamp(TRAIN_END)
test_pred_start = pd.Timestamp(TEST_PRED_START)
test_pred_end   = pd.Timestamp(TEST_PRED_END)

# Test boundary: match MCANN's formula exactly
# n_test = int((test_hours - pred_len) / roll)
test_start_idx = df[df["datetime"] == test_pred_start].index.values[0]
test_end_idx   = df[df["datetime"] == test_pred_end].index.values[0]
test_hours = test_end_idx - test_start_idx
n_test = int((test_hours - PRED_LEN) / TEST_ROLL)

# Exact data range for test
test_data_rows = SEQ_LEN + (n_test - 1) * TEST_ROLL + PRED_LEN
test_data_start_idx = test_start_idx - SEQ_LEN
test_data_end_idx   = test_data_start_idx + test_data_rows

# --- Training + Validation pool ---
train_val_pool = df[
    (df["datetime"] >= train_start) &
    (df["datetime"] <= train_end)
].copy().reset_index(drop=True)

# --- Test pool ---
test_pool = df.iloc[test_data_start_idx:test_data_end_idx].copy().reset_index(drop=True)

# --- Sequential validation split ---
val_pool   = train_val_pool.iloc[-VAL_SIZE:].copy().reset_index(drop=True)
train_pool = train_val_pool.iloc[:-VAL_SIZE].copy().reset_index(drop=True)

print(f"  Train+Val pool:  {len(train_val_pool)} rows ({TRAIN_START} → {TRAIN_END})")
print(f"  Training pool:   {len(train_pool)} rows")
print(f"  Validation pool: {len(val_pool)} rows (last {VAL_SIZE} timesteps)")
print(f"  Test pool:       {len(test_pool)} rows ({test_pool['datetime'].iloc[0]} → {test_pool['datetime'].iloc[-1]})")
print(f"  Test samples:    {n_test} (matching MCANN's roll={TEST_ROLL})")
print()

# =============================================================================
# STEP 3: Compute and save normalization statistics
# =============================================================================
print("=" * 60)
print("STEP 3: Computing normalization statistics")
print("=" * 60)

train_values = np.array(train_pool['value'].fillna(np.nan))

# nanmean / nanstd (for compatibility with MCANN's denormalization)
stdn_mean, stdn_std = standard_normalization(train_values)

# StandardScaler (for data transformation)
scaler = StandardScaler()
scaler.fit(train_pool[['value']].values)

# Save everything to mean_std_mini.pt
out_dir = os.path.join(SAVE_DIR, f"in{SEQ_LEN}_out{PRED_LEN}")
os.makedirs(out_dir, exist_ok=True)

mean_std_mini = {
    'stdn_mean': stdn_mean,
    'stdn_std': stdn_std,
    'scaler_mean': float(scaler.mean_[0]),
    'scaler_scale': float(scaler.scale_[0]),
}
torch.save(mean_std_mini, os.path.join(out_dir, "mean_std_mini.pt"))

print(f"  Train nanmean:     {stdn_mean:.6f}")
print(f"  Train nanstd:      {stdn_std:.6f}")
print(f"  Scaler mean:       {scaler.mean_[0]:.6f}")
print(f"  Scaler scale:      {scaler.scale_[0]:.6f}")
print(f"  Saved: {out_dir}/mean_std_mini.pt")
print()

# =============================================================================
# STEP 4: Apply StandardScaler normalization
# =============================================================================
print("=" * 60)
print("STEP 4: Scaling data with StandardScaler (fit on train)")
print("=" * 60)

train_scaled = scaler.transform(train_pool[['value']].values).astype(np.float32)
val_scaled   = scaler.transform(val_pool[['value']].values).astype(np.float32)
test_scaled  = scaler.transform(test_pool[['value']].values).astype(np.float32)

print(f"  Train scaled: {train_scaled.shape}")
print(f"  Val   scaled: {val_scaled.shape}")
print(f"  Test  scaled: {test_scaled.shape}")
print()

# =============================================================================
# STEP 5: Save .npy files
#   - Train/Val: flat scaled arrays (sliced on the fly in DataLoader, like Dataset_MTS)
#   - Test: pre-windowed with roll=8 (to match MCANN's 1086 test points)
# =============================================================================
print("=" * 60)
print("STEP 5: Saving .npy files")
print("=" * 60)

# Train/Val: save flat arrays — DataLoader slices windows on the fly
np.save(os.path.join(out_dir, "train_data.npy"), train_scaled)
np.save(os.path.join(out_dir, "val_data.npy"),   val_scaled)

train_samples = len(train_scaled) - SEQ_LEN - PRED_LEN + 1
val_samples   = len(val_scaled) - SEQ_LEN - PRED_LEN + 1

print(f"  train_data.npy: {train_scaled.shape} → {train_samples} samples (stride=1)")
print(f"  val_data.npy:   {val_scaled.shape}   → {val_samples} samples (stride=1)")

# Test: pre-windowed to guarantee exactly n_test samples with roll=8
test_x, test_y = create_windows(test_scaled, SEQ_LEN, LABEL_LEN, PRED_LEN, roll=TEST_ROLL)
np.save(os.path.join(out_dir, "test_x.npy"), test_x)
np.save(os.path.join(out_dir, "test_y.npy"), test_y)

print(f"  test_x.npy:     {test_x.shape}   (pre-windowed, roll={TEST_ROLL})")
print(f"  test_y.npy:     {test_y.shape}   (pre-windowed, roll={TEST_ROLL})")
print(f"  Saved to: {out_dir}/")
print()

# =============================================================================
# STEP 6: Export CSV files for reference
# =============================================================================
print("=" * 60)
print("STEP 6: Exporting CSV files (reference)")
print("=" * 60)

train_pool.to_csv(os.path.join(out_dir, "train_set.csv"), index=False)
val_pool.to_csv(os.path.join(out_dir, "val_set.csv"), index=False)
test_pool.to_csv(os.path.join(out_dir, "test_set.csv"), index=False)

print(f"  Saved: train_set.csv  ({len(train_pool)} rows)")
print(f"  Saved: val_set.csv    ({len(val_pool)} rows)")
print(f"  Saved: test_set.csv   ({len(test_pool)} rows)")
print()

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Output directory: {out_dir}/")
print()
print(f"  {'Set':<12} {'Raw Rows':>10} {'Samples':>10}  Loading")
print(f"  {'-'*12} {'-'*10} {'-'*10}  {'-'*30}")
print(f"  {'Train':<12} {len(train_pool):>10,} {train_samples:>10,}  Flat .npy, slice on the fly")
print(f"  {'Validation':<12} {len(val_pool):>10,} {val_samples:>10,}  Flat .npy, slice on the fly")
print(f"  {'Test':<12} {len(test_pool):>10,} {test_x.shape[0]:>10,}  Pre-windowed .npy (roll={TEST_ROLL})")
print()
print(f"  Seq_len={SEQ_LEN}, Label_len={LABEL_LEN}, Pred_len={PRED_LEN}")
print("=" * 60)
print("Done!")