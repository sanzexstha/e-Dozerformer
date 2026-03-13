## Hyperparameter Search Report

---
Exp Name: `ross_s_metric_g`

- Dataset: `Ross_S_fixed`
- Extreme Label: `GMM`
- Decomposition: `None` | `True`

### Search Space

| Parameter       | Option Range    |
|-----------------|-----------------|
| Patch size      | {24, 48, 60}    |
| Patch Threshold | {1, 2, ..., 15} |


---
Exp 2 Name:`ross_withrain_metric_g`

| Parameter          | Value               |
|--------------------|---------------------|
| Patch size         | {8, 12, 24, 48, 60} |



**Results:**

| Metric | Value |
|--------|-------|
| MSE    |       |
| MAE    |       |

**Notes:**
> _[Observations, convergence behavior, training time, etc.]_

---

### Experiment 2 — _[Short Description]_

- **Date:** _YYYY-MM-DD_
- **Dataset:** _[Dataset Name]_
- **Forecast horizon:** _[e.g., 96, 192, 336, 720]_
- **Seed:** _[e.g., 42]_

**Hyperparameters:**

| Parameter          | Value |
|--------------------|-------|
| Patch size         |       |
| Stride size        |       |
| Hidden dimension   |       |
| Number of heads    |       |
| Number of layers   |       |

**Results:**

| Metric | Value |
|--------|-------|
| MSE    |       |
| MAE    |       |

**Notes:**
> _[Observations, convergence behavior, training time, etc.]_

---

### Experiment 3 — _[Short Description]_

- **Date:** _YYYY-MM-DD_
- **Dataset:** _[Dataset Name]_
- **Forecast horizon:** _[e.g., 96, 192, 336, 720]_
- **Seed:** _[e.g., 42]_

**Hyperparameters:**

| Parameter          | Value |
|--------------------|-------|
| Patch size         |       |
| Stride size        |       |
| Hidden dimension   |       |
| Number of heads    |       |
| Number of layers   |       |

**Results:**

| Metric | Value |
|--------|-------|
| MSE    |       |
| MAE    |       |

**Notes:**
> _[Observations, convergence behavior, training time, etc.]_

---

## Summary of Best Configurations

| Rank | Experiment | Patch | Stride | Hidden Dim | Heads | Layers | MSE   | MAE   |
|------|------------|-------|--------|------------|-------|--------|-------|-------|
| 1    |            |       |        |            |       |        |       |       |
| 2    |            |       |        |            |       |        |       |       |
| 3    |            |       |        |            |       |        |       |       |

---

## Key Findings

- _[Which parameters had the most impact?]_
- _[Any parameter interactions observed?]_
- _[Diminishing returns beyond a certain model size?]_

## Next Steps

- [ ] _[e.g., Narrow search around best config]_
- [ ] _[e.g., Test on additional datasets]_
- [ ] _[e.g., Run ablation study on top-k configs]_