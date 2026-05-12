# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time

import numpy as np
import torch
from pathlib import Path
from metrics.metrics_sets import run_metrics, calculate_one
from ldm.data.tsg_dataset import TSGDataset
import os
import re

import json
import random

# data_root = os.environ['DATA_ROOT']
data_root='./data'

from metrics.feature_distance_eval import get_mdd_eval, mmd_metric, get_flat_distance

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


def evaluation(args,ori_data,generated_data):

    from metrics.discriminative_torch import discriminative_score_metrics
    # deterministic eval
    met_dict_final = {}
    disc_res = []
    for ii in range(10):
        dsc = discriminative_score_metrics(ori_data, generated_data, args)
        disc_res.append(dsc)
    disc_mean, disc_std = np.round(np.mean(disc_res), 4), np.round(np.std(disc_res), 4)

    metric_1={}
    print('test/disc_mean: ', disc_mean)
    print('test/disc_std: ', disc_std)
    metric_1['disc_mean']=disc_mean
    metric_1['disc_std']=disc_std



    met_dict_final['disc']=metric_1
    tmp_name = f'{args.dataset}_{args.seq_len}_generation'
    data_name=args.dataset

    # metric_2[]
    extend_metrics = update_metrics_dict({}, tmp_name, data_name, args.seq_len, ori_data, generated_data)
    met_dict_final['extend'] = extend_metrics[list(extend_metrics.keys())[0]]

    json_record_loss_test = json.dumps(met_dict_final, indent=4)
    with open(args.log_dir + '/record_all_loss_test'+args.tag + '.json', 'w') as json_file:
        json_file.write(json_record_loss_test)

    with open(args.log_dir +'/args'+args.tag+'.json' , 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)


device='cuda:0'
def test_model_with_diffcde(model, data, trainer, opt, logdir,train_loader):

    checkpoints=os.listdir(opt.ckptdir)

    valid_ckpts = [(c, float(c.split('-')[1].split('.ckpt')[0]))
                   for c in checkpoints if 'last.ckpt' not in c]

    if valid_ckpts:
        min_ckpt, min_loss = min(valid_ckpts, key=lambda x: x[1])
    best_ckpt_path=os.path.join(opt.ckptdir,min_ckpt)


    checkpoint = torch.load(best_ckpt_path,weights_only=False)

    print(checkpoint.keys())
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    save_path = Path(logdir) / 'generated_samples'
    save_path.mkdir(exist_ok=True, parents=True)
    seq_len = data.window
    num_dp = 100  # number of samples for constructingdomain prompts
    all_metrics = {}
    # print(data.norm_train_dict['natops'].shape) (324, 51, 25)
    all_gen = []
    for batch_idx, batch in enumerate(train_loader):
        n_sample = batch['context'].shape[0]
        x = torch.tensor(batch['context']).to(device).float()
        c, mask = model.get_learned_conditioning(x, return_mask=True)
        samples, _ = model.sample_log(cond=c, y=None, batch_size=n_sample if not opt.debug else 100, ddim=False, cfg_scale=1,
                                      mask=mask)
        norm_samples = model.decode_first_stage(samples).detach().cpu().numpy()
        # print(norm_samples.shape)
        all_gen.append(norm_samples)
    generated_data = np.vstack(all_gen).transpose(0, 2, 1)

    tmp_name = f'{opt.d_name}_{seq_len}_generation_' + str(min_loss)
    np.save(save_path / f'{tmp_name}.npy', generated_data)


    

    
def zero_shot_k_repeat(samples, model, train_data_module, num_gen_samples=1000):
    data = train_data_module
    k_samples = samples.transpose(0,2,1)
    k = k_samples.shape[0]
    normalizer = data.fit_normalizer(k_samples)

    norm_k_samples = data.transform(k_samples, normalizer=normalizer)

    x = torch.tensor(norm_k_samples).float().to('cuda')
    c, mask = model.get_learned_conditioning(x, return_mask=True)

    repeats = int(num_gen_samples / k)
    extra = num_gen_samples - repeats * k
    
    cond = torch.repeat_interleave(c, repeats, dim=0)
    cond = torch.cat([cond, c[:extra]], dim=0)
    mask_repeat = torch.repeat_interleave(mask, repeats, dim=0)
    mask_repeat = torch.cat([mask_repeat, mask[:extra]], dim=0)
    
    samples, z_denoise_row = model.sample_log(cond=cond, batch_size=cond.shape[0], ddim=False, cfg_scale=1, mask=mask_repeat)
    norm_samples = model.decode_first_stage(samples).detach().cpu().numpy()
    inv_samples = data.inverse_transform(norm_samples, normalizer=normalizer)
    gen_data = inv_samples.transpose(0,2,1)
    
    return gen_data, k_samples.transpose(0,2,1)

def merge_dicts(dicts):
    result = {}
    for d in dicts:
        for k, v in d.items():
            result[k] = v
    return result


