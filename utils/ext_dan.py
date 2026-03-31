import pandas as pd
import numpy as np
import os

base_dir = '/home/SGF.EDUBEAR.NET/ss472s/e-Dozerformer/'
val_path = os.path.join(base_dir, "data", "datasets", "watershed/raw", "test_timestamps_24avg.tsv")
train_path = os.path.join(base_dir, "data", "datasets", "watershed/raw", "UpperPen_S_fixed.csv")


def compute_metrics_same(aa):
    val_set = pd.read_csv(
        val_path, sep="\t"
    )
    val_points = val_set["Hold Out Start"]
    trainX = pd.read_csv(
        train_path, sep="\t"
    )
    trainX.columns = ["id", "datetime", "value"]
    count = 0
    for test_point in val_points:
        point = trainX[trainX["datetime"] == test_point].index.values[0]
        NN = np.isnan(
            trainX[point - 1440: point + 288]["value"]
        ).any()
        if not NN:
            count += 1
    vals4 = aa
    # compute metrics
    all_GT = []
    all_DAN = []
    loop = 0
    ind = 0
    while loop < len(val_points):
        ii = val_points[loop]
        point = trainX[trainX["datetime"] == ii].index.values[0]
        x = trainX[point - 1440: point + 288][
            "value"
        ].values.tolist()
        if np.isnan(np.array(x)).any():
            loop = loop + 1  # id for time list
            continue
        loop = loop + 1
        if ind >= count - count % 100:
            break
        ind += 1
        temp_vals4 = list(vals4[ind - 1])
        all_GT.extend(x[1440:])
        all_DAN.extend(temp_vals4)
    metrics = metric_g("DAN", np.array(all_DAN), np.array(all_GT))
    return metrics


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


def metric(model, pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)

    return mae, mse, rmse, mape  # , mspe


def metric_g(name, pre, gt):
    pre = np.array(pre)
    gt = np.array(gt)
    ll = int(len(pre) / 288)
    mae_all = []  # unused?
    mse_all = []  # unused?
    rmse_all = []
    mape_all = []
    l2 = []
    l3 = []
    lll = []
    for i in range(ll):
        mae, mse, rmse, mape = metric(
            name, pre[i * 288: (i + 1) * 288], gt[i * 288: (i + 1) * 288]
        )
        rmse_all.append(rmse)
        mape_all.append(mape)
    l2.append(np.around(np.mean(np.array(rmse_all)), 2))
    l3.append(np.around(np.mean(np.array(mape_all)), 3))
    lll.append(l2)
    lll.append(l3)
    return lll
