## exp_extreme_dataset

**Date:** 2026-03-11 -  
**Goal:** To evaluate the performance of different sparse masks in those datasets that are specifically related to extreme event time series forecasting

### Sparse Mask Compared
- `dozer_ext_0`
- `dozer_v1`

### Tasks
- [x] Added data loading support for two groups of datasets into `Data_loader` class:
  - Watershed:  e.g `Ross_S_fixed`
    - `seq_len = 1440`, `pred_len = 288`
  - Reservoir: e.g. `Lexington`
    - `seq_len= 360`, `pred_len= 72`
  - Preprocessing of MCAAN and DAN paper-related:
    - oversampling
    - normalization (log-std)
    - GMM clustering for extreme event detection
    - Two types of experimental setups: MCAAN and DAN paper-related and the original Dozer paper-related setup
      -  random sampling vs temporal split in Dozerformer
      - For a fair comparision, we use the same training, val and testing period with the same `seq_len` and `pred_len`
      - For lexington, the same number of testing samples is used `1086`. Training period: `(235234, 1)` MCANN training shape: `(26000, 360, 1)`
- [x] Converted the normalized data into the original space for a fair comparison with MCANN and DAN, added into `metrics`
- [ ] rmse stuck around 16 in `Ross_noRain`, both for Dozer and informer (with d_model `256 <- 512`, where in paper, informer has around 9 

### Experimental Setup
- Full Pipeline DS Style
  - [x] NaN windows ignore
  - [x] Hydro year in training data
  - [x] log1p + z-score normalization
  - [x] Train/val randomly sampled (30,000 / 120)
  - [x] Test sliding with step=16 (every 4 hours)
  - [x] Normalization params saved in `Norm.txt`
  - [x] `inverse_transform` uses `expm1(x * std + mean)`

### Dataset Splits
 
| Split | Sampling Strategy | Shape | Notes |
|-------|-------------------|-------|-------|
| **Train** | Random sampling | `(30000, 1440, 1)` | 30,000 windows of 1,440 timesteps (≈ 60 days at 15-min resolution), 1 feature |
| **Validation** | Random sampling | `(120, 1440, 1)` | 120 windows, same length as training |
| **Test** | Sliding window, step = 16 | `(1619, 288, 1)` | 1,619 windows of 288 timesteps (≈ 3 days / 72 hours at 15-min resolution) |

### Extreme events datasets
#### Watershred

| Dataset | Stream Sensor        | Rain Sensor          | 
|---------|----------------------|----------------------|
| **Ross** | `Ross_S_fixed.csv`     | `Ross_R_fixed.csv`     |
| **Saratoga** | `Saratoga_S_fixed.csv` | `Saratoga_R_fixed.csv` |
| **SFC** | `SFC_S_fixed.csv`      | `SFC_R_fixed.csv`      |
| **UpperPen** | `UpperPen_S_fixed.csv` | `UpperPen_R_fixed.csv` |

#### Reservoir
The reservoir storage datasets are located at the `data/datasets/reservoir` directory.

| Dataset File | Reservoir Name |
|---|---|
| `reservoir_stor_4001_sof24.tsv` | Almaden |
| `reservoir_stor_4005_sof24.tsv` | Coyote |
| `reservoir_stor_4007_sof24.tsv` | Lexington |
| `reservoir_stor_4009_sof24.tsv` | Stevens Creek |
| `reservoir_stor_4011_sof24.tsv` | Vasona |

These univarite time series datasets contain storage data for five different reservoirs in the Santa Clara Valley region. Each TSV file corresponds to a specific reservoir and is referenced by its dataset ID (4001, 4005, 4007, 4009, 4011).

