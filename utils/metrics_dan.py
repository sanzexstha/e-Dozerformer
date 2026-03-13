import numpy as np
from sklearn.metrics import mean_absolute_percentage_error


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    pred = np.squeeze(pred)
    true = np.squeeze(true)
    return mean_absolute_percentage_error(np.array(true) + 1, np.array(pred) + 1)


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    corr = CORR(pred, true)
    return mae, mse, rmse, mape, mspe, corr


# ── DAN exact replication ─────────────────────────────────────
def _metric_per_window(pred, true):
    """DAN's metric() — per-window stats (exact copy from DAN metric.py)"""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    return mae, mse, rmse, mape


def metric_g(pred, true, window_size=288):
    """DAN's metric_g — exact replication.

    Splits pred/true into windows of window_size,
    computes RMSE and MAPE per window, then averages.

    Args:
        pred: flattened predictions in RAW space, shape (N*window_size,)
        true: flattened ground truth in RAW space, shape (N*window_size,)
        window_size: prediction length (default 288 = 3 days of 15-min)

    Returns:
        (rmse, mape): averaged across windows, rounded to 2 and 3 decimals
    """
    pred = np.array(pred)
    true = np.array(true)
    ll = int(len(pred) / window_size)
    rmse_all = []
    mape_all = []
    for i in range(ll):
        p = pred[i * window_size: (i + 1) * window_size]
        g = true[i * window_size: (i + 1) * window_size]
        mae, mse, rmse, mape = _metric_per_window(p, g)
        rmse_all.append(rmse)
        mape_all.append(mape)
    return np.around(np.mean(np.array(rmse_all)), 2), np.around(np.mean(np.array(mape_all)), 3)


def truncate_to_dan(pred, true, window_size=288):
    """DAN's compute_metrics truncation — round down to nearest 100 windows.

    DAN does: if ind >= count - count % 100: break
    This drops the last (count % 100) windows.

    Args:
        pred: array shape (N, window_size, C) or (N*window_size,)
        true: array shape (N, window_size, C) or (N*window_size,)
        window_size: prediction length (default 288)

    Returns:
        (pred_flat, true_flat): truncated and flattened
    """
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)
    n_windows = len(pred_flat) // window_size
    n_use = n_windows - n_windows % 100
    n_values = n_use * window_size
    return pred_flat[:n_values], true_flat[:n_values]


def compute_metrics_dan(pred_raws, true_raws, window_size=288):
    """Full DAN compute_metrics pipeline — single call.

    1. Clip negative predictions to 0
    2. Truncate to nearest 100 windows
    3. Compute per-window RMSE and MAPE then average

    Args:
        pred_raws: denormalized predictions, shape (N, window_size, C)
        true_raws: denormalized ground truth, shape (N, window_size, C)
        window_size: prediction length (default 288)

    Returns:
        dict with 'rmse' and 'mape'
    """
    # Step 2: truncate to nearest 100 windows
    pred_flat, true_flat = truncate_to_dan(pred_raws, true_raws, window_size)

    # Step 3: per-window RMSE and MAPE
    rmse, mape = metric_g(pred_flat, true_flat, window_size)

    return {'rmse': rmse, 'mape': mape}