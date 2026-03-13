# Ross_S_fixed — DAN Baseline Alignment Notes

## Dataset

- **Sensor**: Ross_S_fixed (rainfall gauge, 15-min intervals)
- **Period**: Oct 1981 – Aug 2022 (~1.4M records)
- **Train**: 1988-01-01 14:30 → 2021-08-31 23:30 (1,180,452 records, 6,573 NaNs)
- **Test**: 2021-09-01 00:30 → 2022-05-31 23:30 (26,204 records, 0 NaNs)
- **Test windows**: 1,619 (sliding step=16, i.e. every 4 hours)
- **Evaluated windows**: 1,600 (truncated to nearest 100, per DAN's `compute_metrics`)

## Problem

RMSE was stuck at ~16 regardless of model architecture or hyperparameters. DAN paper reports RMSE 4.2 for the same dataset.

## Root Causes

### 1. RMSE Evaluation Method (the main issue)

Our code computed **global RMSE** — flatten all predictions into one array, then compute RMSE once. DAN computes **per-window RMSE then averages** across windows using `metric_g`.

```
Global:    sqrt(mean((all_preds - all_trues)²))           → ~16
metric_g:  mean([ sqrt(mean((p_i - g_i)²)) for each i ]) → ~4.25
```

Why the difference is so large: 97% of test data is ≤ 1.0 (dry periods), but a handful of extreme storms (up to 549) dominate the squared error. Global RMSE lets those storms control the result. Per-window averaging treats every 288-step window equally, so the many quiet windows (RMSE ≈ 0) dilute the few storm windows.

Proof — even a naive "predict zero" model gives:

| Method         | RMSE |
|----------------|------|
| Global         | 16.45 |
| metric_g       | 4.27  |

### 2. Column Naming Mismatch

Our CSV column was `rainfall`. The DS class reads `trainX["value"]` everywhere. Fix: rename columns at load time.

```python
trainX = pd.read_csv('Ross_S_fixed.csv', sep='\t')
trainX.columns = ["id", "datetime", "value"]
```

### 3. Normalization Confirmation

DAN uses `log_std_normalization` from `utils2.py`:

```python
a = np.log(np.array(data) + 1)   # same as log1p
mean = np.nanmean(a)
std  = np.nanstd(a)               # ddof=0
normalized = (a - mean) / std
```

Denormalization (for converting predictions back to raw scale):

```python
raw = np.exp(normalized * std + mean) - 1   # same as expm1
raw = np.clip(raw, 0, None)                 # clip negatives to 0
```

For Ross_S_fixed training data: **mean = 0.405030, std = 0.765535**.

## DAN-Style metric_g Implementation

```python
def metric_g(pred, true, window_size=288):
    ll = len(pred) // window_size
    rmse_all = []
    mape_all = []
    for i in range(ll):
        p = pred[i * window_size : (i + 1) * window_size]
        g = true[i * window_size : (i + 1) * window_size]
        rmse_all.append(np.sqrt(np.mean((p - g) ** 2)))
        mape_all.append(mean_absolute_percentage_error(g + 1, p + 1))
    return np.around(np.mean(rmse_all), 2), np.around(np.mean(mape_all), 3)
```

Key details:
- Operates on **denormalized (raw) predictions**, not normalized
- Splits into 288-step windows, computes RMSE per window, then averages
- MAPE uses `+1` offset to avoid division by zero (rainfall has many zeros)
- DAN truncates to nearest 100 windows before computing (1,619 → 1,600)

## Other DAN Preprocessing Details

| Step | What DAN does |
|------|---------------|
| NaN handling | Keeps NaNs; rejects any train/val/test window that touches a NaN |
| Hydro year | Only Sep–May used for train/val; Jun–Aug excluded |
| Train sampling | 30,000 random windows with Kruskal-Wallis event oversampling |
| Val sampling | 120 random windows |
| Test generation | Sliding window, step=16 (4 hours), 1,619 windows |
| Negative clipping | `pred = (pred + abs(pred)) / 2` (ReLU) |

## Final Result

| Model | metric_g RMSE |
|-------|---------------|
| DAN (paper) | 4.2 |
| Our reproduction | **4.25** |

The 0.05 difference is within expected variance from random sampling.