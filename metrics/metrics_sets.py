# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time

import numpy as np
# from utils_kovae.data_utils import test_data_loading
from metrics.feature_distance_eval import get_mdd_eval, mmd_metric, get_flat_distance


# data_root = os.environ['DATA_ROOT']
data_root='./data'

def calculate_one(gen_data, scaled_ori, model_name, repeat, data_name, seq_len, uni_data_sub, uni_data_div, n_samples):
    this_metrics = {}
    print(model_name, gen_data.shape)
    scaled_gen = (gen_data - uni_data_sub) / uni_data_div
    scaled_gen = scaled_gen[:n_samples, :, None]
    this_metrics = update_metrics_dict(this_metrics, model_name, data_name, seq_len, scaled_ori, scaled_gen, repeat_id=repeat)
    return this_metrics

def update_metrics_dict(the_dict, key, data_name, seq_len, ori_data, gen_data, repeat_id=0):
    if (key, data_name, seq_len, repeat_id) in the_dict:
        print(f'{key} {data_name} {seq_len} {repeat_id} already in the dict, skip!')
        return the_dict

    mdd = get_mdd_eval(ori_data, gen_data)
    the_dict[(key, data_name, seq_len, repeat_id)] = {
        'mdd': mdd,
    }
    flat_sk_result = get_flat_distance(ori_data, gen_data)
    the_dict[(key, data_name, seq_len, repeat_id)].update(flat_sk_result)
    the_dict[(key, data_name, seq_len, repeat_id)].update(mmd_metric(ori_data, gen_data))
    return the_dict

def divide_x_y(x_y):
    x_y=x_y.transpose(0,2,1)
    data_x = x_y[..., :-1, :]
    labels_y = x_y[..., -1, :]
    labels_y = labels_y[:, 0]
    return data_x, labels_y
def run_metrics(data_name, seq_len, model_name, gen_data, scale='zscore', exist_dict={}, repeat_id=0,data=None,gen_y=None):
    extend_metrics = exist_dict
    # print(data.norm_val_dict['natops'].shape)
    # test_data=data.norm_val_dict['natops']
    test_data=data.norm_train_dict['natops']
    uni_ori_data, y = divide_x_y(test_data)
    sel_y=5
    uni_ori_data=uni_ori_data[y==sel_y]
    # print(uni_ori_data.shape,gen_data.shape,gen_y.shape)
    gen_data=gen_data.transpose(0,2,1)
    gen_data=gen_data[gen_y==sel_y] #[:uni_ori_data.shape[0]]
    # print(uni_ori_data.shape,gen_data.shape)
    # time.sleep(500)
    # uni_ori_data = test_data_loading(data_name, seq_len, stride=seq_len, univar=True)
    uni_data_min, uni_data_max = np.min(uni_ori_data), np.max(uni_ori_data)
    uni_data_mean, uni_data_std = np.mean(uni_ori_data), np.std(uni_ori_data)


    if scale == 'minmax':
        uni_data_sub, uni_data_div = uni_data_min, uni_data_max - uni_data_min + 1e-7
    elif scale == 'zscore':
        uni_data_sub, uni_data_div = uni_data_mean, uni_data_std + 1e-7
    elif scale == 'raw':
        uni_data_sub, uni_data_div = 0, 1
    elif scale == 'robust_zscore':
        median = np.median(uni_ori_data)
        mad = np.median(np.abs(uni_ori_data - median))
        uni_data_sub, uni_data_div = median, 1.4826 * mad + 1e-7
    uni_scaled_ori = (uni_ori_data - uni_data_sub) / uni_data_div
    print(data_name, 'univar', uni_scaled_ori.shape)
    scaled_ori = uni_scaled_ori
    scaled_gen = (gen_data - uni_data_sub) / uni_data_div
    extend_metrics = update_metrics_dict(extend_metrics, model_name, data_name, seq_len, scaled_ori, scaled_gen, repeat_id=repeat_id)
    return extend_metrics
