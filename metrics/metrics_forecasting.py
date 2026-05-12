import time

import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))



def RSE_reward(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR_reward(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE_reward(pred, true):
    return np.abs(pred - true)


def MSE_reward(pred, true):
    return (pred - true) ** 2


def RMSE_reward(pred, true):
    return np.sqrt(MSE_reward(pred, true))


def MAPE_reward(pred, true):
    return np.abs((pred - true) / true)


def MSPE_reward(pred, true):
    return np.square((pred - true) / true)


import torch



def metric_reward(pred, true):
    mae = MAE_reward(pred, true)
    mae=np.mean(mae,axis=-2)


    mse = MSE_reward(pred, true)
    mse = np.mean(mse, axis=-2)


    rmse = RMSE_reward(pred, true)

    rmse = np.mean(rmse, axis=-2)



    mape = MAPE_reward(pred, true)
    mape = np.mean(mape, axis=-2)


    return mae*-1, mse*-1, rmse*-1, mape*-1

def metric_tfs(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr
