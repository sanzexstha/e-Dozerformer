## exp_extreme_dataset

**Date:** 2026-03-11  
**Goal:** To evaluate the performance of different sparse masks in those datasets that are specifically related to extreme event time series forecasting

### Sparse Mask Compared
- `dozer_ext_0`
- `dozer_v1`

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

