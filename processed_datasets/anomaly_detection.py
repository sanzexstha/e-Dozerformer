import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# =============================================================================
# Configuration
# =============================================================================
WINDOW = 24 * 7          # 7-day rolling window (assuming hourly data)
CONTAMINATION = 0.05     # expected fraction of anomalies (~5%)
N_ESTIMATORS = 200       # number of trees in Isolation Forest
RANDOM_STATE = 42

TRAIN_PATH = "train.csv"
VAL_PATH   = "val.csv"
TEST_PATH  = "test.csv"


# =============================================================================
# 1. Load data
# =============================================================================
train = pd.read_csv(TRAIN_PATH)
val   = pd.read_csv(VAL_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"Loaded  train: {len(train)} rows | val: {len(val)} rows | test: {len(test)} rows")


# =============================================================================
# 2. Global z-score normalisation (fit on train, apply to all)
# =============================================================================
train_mean = train["value"].mean()
train_std  = train["value"].std()

train["value_zscore"] = (train["value"] - train_mean) / train_std
val["value_zscore"]   = (val["value"]   - train_mean) / train_std
test["value_zscore"]  = (test["value"]  - train_mean) / train_std

print(f"Z-score stats  —  mean: {train_mean:.4f}  std: {train_std:.4f}")


# =============================================================================
# 3. Sliding-window local z-score (captures sudden spikes / drops)
# =============================================================================
def add_local_zscore(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    roll_mean = df["value"].rolling(window=window, min_periods=1, center=False).mean()
    roll_std  = df["value"].rolling(window=window, min_periods=1, center=False).std().replace(0, np.nan)
    df["local_zscore"] = (df["value"] - roll_mean) / roll_std
    df["local_zscore"] = df["local_zscore"].fillna(0)
    return df

train = add_local_zscore(train)
val   = add_local_zscore(val)
test  = add_local_zscore(test)


# =============================================================================
# 4. Anomaly detection — Isolation Forest (fit on train local z-scores)
# =============================================================================
model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
)
model.fit(train[["local_zscore"]])


# =============================================================================
# 5. Predict & label  (0 = normal, 1 = extreme)
# =============================================================================
for df, name in [(train, "train"), (val, "val"), (test, "test")]:
    preds = model.predict(df[["local_zscore"]])
    df["anomaly_label"] = (preds == -1).astype(int)

    n_anom = df["anomaly_label"].sum()
    print(f"  {name:>5s}: {n_anom:>6d} anomalies / {len(df)} rows  ({100 * n_anom / len(df):.2f}%)")


# =============================================================================
# 6. Save — original columns + global z-score + anomaly label
# =============================================================================
output_cols = ["datetime", "value", "value_zscore", "anomaly_label"]

train[output_cols].to_csv("train_labeled.csv", index=False)
val[output_cols].to_csv("val_labeled.csv",     index=False)
test[output_cols].to_csv("test_labeled.csv",   index=False)

print("\nSaved: train_labeled.csv | val_labeled.csv | test_labeled.csv")
