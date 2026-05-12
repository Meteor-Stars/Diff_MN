import os

os.environ['WANDB_MODE'] = 'disabled'
import os, sys
import time
import numpy as np
from pytorch_lightning.trainer import Trainer

from types import SimpleNamespace
import os
import json
import time

import torch.utils.data as Data
import torch.nn.init
import numpy as np
# import tensorflow as tf
# import neptune.new as neptune
import random
import argparse
import torchcde
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from utils_diffmn.utils_data_continues import TimeDataset_irregular
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.feature_distance_eval import get_mdd_eval, mmd_metric, get_flat_distance

from metrics.downstream_forecasting import MTS_forecasting_eval

from models.moe_ncde.moe_neural_cde import NeuralCDE_Continues,MoECDEFunc,ChannelWiseAutoencoder

def get_args():
    p = argparse.ArgumentParser(description="Neural CDE with MoE Long-Expert options")

    # General
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=600)
    # p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument("--dataset", default='stock')
    p.add_argument('--batch_size', type=int, default=256, metavar='N')
    p.add_argument('--seq_len', type=int, default=24, metavar='N')
    p.add_argument('--inp_dim', type=int, default=6, metavar='N')
    p.add_argument('--missing_value', type=float, default=0.3)

    # Model
    p.add_argument('--hidden_channels', type=int, default=64)
    p.add_argument('--interpolation', type=str, default='cubic', choices=['linear', 'cubic'])


    p.add_argument('--batch_norm', type=bool, default=True)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--z_dim', type=int, default=16)
    p.add_argument('--hidden_dim', type=int, default=64,
                        help='the hidden dimension of the output decoder lstm')

    # MoE
    p.add_argument('--moe_activate', action='store_true')
    p.add_argument('--moe_num_experts', type=int, default=4)
    p.add_argument('--moe_topk', type=int, default=None)
    p.add_argument('--moe_ema_alpha', type=float, default=None)
    p.add_argument('--moe_load_balance_weight', type=float, default=0.0)

    # Long expert options
    p.add_argument('--moe_long_expert', action='store_true', help='Enable long expert at index 0')
    p.add_argument('--moe_long_alpha_init', type=float, default=0.2, help='init of global alpha before sigmoid')
    p.add_argument('--moe_long_min', type=float, default=0.0, help='min clamp of alpha after sigmoid')
    p.add_argument('--moe_long_max', type=float, default=1.0, help='max clamp of alpha after sigmoid')
    p.add_argument('--moe_long_target_alpha', type=float, default=0.2, help='target alpha for regularizer')
    p.add_argument('--moe_long_alpha_reg', type=float, default=0.0, help='lambda for (alpha - target)^2')

    p.add_argument('--moe_long_use_lowpass', action='store_true', help='apply depthwise low-pass on long expert input')
    p.add_argument('--moe_long_kernel', type=int, default=5, help='kernel size for low-pass (odd recommended)')
    p.add_argument('--moe_long_freeze_epochs', type=int, default=0, help='freeze low-pass weights for first K epochs')

    return p.parse_args()

def set_seed_device(seed,gpu_):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    # tf.keras.utils_kovae.set_random_seed(seed)

    # Use cuda if available
    if torch.cuda.is_available():
        device = torch.device(gpu_)
        print('cuda is available')
    else:
        device = torch.device("cpu")
    return gpu_

def agg_losses(LOSSES, losses):
    if not LOSSES:
        LOSSES = [[] for _ in range(len(losses))]
    for jj, loss in enumerate(losses):
        LOSSES[jj].append(loss.item())
    return LOSSES

def log_losses(epoch, losses_tr, names):
    losses_avg_tr = []

    for loss in losses_tr:
        losses_avg_tr.append(np.mean(loss))

    loss_str_tr = 'Epoch {}, TRAIN: '.format(epoch + 1)
    for jj, loss in enumerate(losses_avg_tr):
        loss_str_tr += '{}={:.3e}, \t'.format(names[jj], loss)
    logging.info(loss_str_tr)

    logging.info('#'*30)
    return losses_avg_tr[0]


# parser = define_args()
# args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

    extend_metrics = update_metrics_dict({}, tmp_name, data_name, args.seq_len, ori_data, generated_data)

    met_dict_final['extend'] = extend_metrics[list(extend_metrics.keys())[0]]

    json_record_loss_test = json.dumps(met_dict_final, indent=4)
    with open(args.log_dir + '/record_all_loss_test'+args.tag + '.json', 'w') as json_file:
        json_file.write(json_record_loss_test)

    with open(args.log_dir +'/args'+args.tag+'.json' , 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)




def expand_time_steps(n):

    original_steps = list(range(n))

    new_steps = []

    for i in range(n - 1):
        new_steps.append(original_steps[i])
        new_steps.append((original_steps[i] + original_steps[i + 1]) / 2)

    new_steps.append(original_steps[-1])

    return new_steps


def evaluate_moe_neuralCDE(args,train_loader,channel_wise_autoencoder):
    print("="*50)
    print("STEP 3: Evaluating MOE_neuralCDE")
    print("="*50)
    generated_sequence_continues_all=[]
    generated_sequence_ori_all=[]
    generated_sequence_continues_all_groundtruth=[]
    ground_truth_data=[]

    moe_cde_model_path = args.log_dir + f"/moe_neuralcde_fix_{args.dataset}.pth"
    if args.model_name != 'Ours':
        args.moe_activate = False
        args.use_channel_wise_autoencoder = False
        # moe_cde_model_path = f'/data_new/daroms/paroms/KOVAE/logs_irgen_moencde_finalv3/{args.dataset}/MoeNcdeIrreg-seed=42-miss={args.missing_value}seqlen={args.seq_len}epochs=30/neuralcde_fix_{args.dataset}_woautenc.pth'

    else:
        # moe_cde_model_path = f'/data_new/daroms/paroms/KOVAE/logs_irgen_moencde_final/{args.dataset}/MoeNcdeIrreg-seed=42-miss={args.missing_value}seqlen={args.seq_len}epochs=30/moe_neuralcde_fix_{args.dataset}.pth'
        if args.dataset=='polynomial':
            moe_cde_model_path=f'/data_new/daroms/paroms/KOVAE/logs_irgen_moencde_final_polynomial/polynomial/MoeNcdeIrreg-seed=42-miss={args.missing_value}seqlen=24epochs=30/moe_neuralcde_fix_polynomial.pth'
        elif args.dataset in args.medical_datasets:
            if args.model_name != 'Ours':
                args.moe_activate = False
                args.use_channel_wise_autoencoder = False
                moe_cde_model_path = f'/data_new/daroms/paroms/KOVAE/logs_irgen_moencde_final_ECG_womoe_woautoenc/{args.dataset}/MoeNcdeIrreg-seed=42-miss={args.missing_value}seqlen=24epochs=30/neuralcde_fix_{args.dataset}_woautenc.pth'
            else:
                moe_cde_model_path=f'/data_new/daroms/paroms/KOVAE/logs_irgen_moencde_final_ECG/{args.dataset}/MoeNcdeIrreg-seed=42-miss={args.missing_value}seqlen=24epochs=30/moe_neuralcde_fix_{args.dataset}.pth'
        else:
            if args.model_name != 'Ours':
                args.moe_activate = False
                args.use_channel_wise_autoencoder = False
                moe_cde_model_path = f'/data_new/daroms/paroms/KOVAE/logs_irgen_moencde_finalv3/{args.dataset}/MoeNcdeIrreg-seed=42-miss={args.missing_value}seqlen={args.seq_len}epochs=30/neuralcde_fix_{args.dataset}_woautenc.pth'
            else:
                moe_cde_model_path = f'/data_new/daroms/paroms/KOVAE/logs_irgen_moencde_final/{args.dataset}/MoeNcdeIrreg-seed=42-miss={args.missing_value}seqlen={args.seq_len}epochs=30/moe_neuralcde_fix_{args.dataset}.pth'
    model = NeuralCDE_Continues(
        args.inp_dim,
        args.hidden_channels,
        args.interpolation,
        pretrained_encoder=channel_wise_autoencoder.encoder,
        pretrained_decoder=channel_wise_autoencoder.decoder,
        moe_activate=args.moe_activate,
        args=args
    ).to(args.device)

    model.freeze_encoder_decoder()
    print('loading model from ',moe_cde_model_path)
    model.load_state_dict(torch.load(moe_cde_model_path, map_location=args.device))

    moe_experts_all = []
    samples_all = []
    filled=False
    sample_count=0
    generated_sequence_ori_all=[]
    generated_sequence_continues_all_groundtruth = []
    generated_sequence_continues_all = []
    ground_truth_data=[]
    for i, data in enumerate(train_loader, 1):
        with torch.no_grad():
            x = data['data'].to(args.device).float()
            ori_data=data['original_data'].to(args.device).float()
            ground_truth_data.append(ori_data)

            moe_weights=[]
            generated_sequence=x.clone()
            if x.shape[1]>args.seq_len:
                generated_sequence=x[..., :-4, :]
                moe_weights = x[..., -4:, :]

            generated_sequence_ori_all.append(generated_sequence)

            sample_count += x.shape[0]
            t = torch.FloatTensor(list(range(args.seq_len))).to(args.device)

            train_coeffs = data['inter']  # .to(device)
            train_coeffs = torch.cat(train_coeffs, dim=-1)
            x_intp_full = model(t, train_coeffs,moe_weights=moe_weights)  # (len, 1)

            if args.dataset in args.medical_datasets or args.dataset == 'polynomial':
                x_intp_full = x_intp_full.unsqueeze(-1)
                integer_indices = np.arange(0, args.seq_len * 2 - 1, 2)
                x_intp_full[:, integer_indices, :] = generated_sequence
                if args.dataset == 'polynomial':
                    continue
            else:
                integer_indices = np.arange(0, args.seq_len * 2 - 1, 2)
                x_intp_full[:, integer_indices, :] = generated_sequence
            train_coeffs_ori = data['inter_ori']  # .to(device)
            train_coeffs_ori = torch.cat(train_coeffs_ori, dim=-1)
            x_intp_full_ori = model(t, train_coeffs_ori)  # (len, 1)
            if args.dataset in args.medical_datasets or args.dataset == 'polynomial':
                x_intp_full_ori= x_intp_full_ori.unsqueeze(-1)
            integer_indices = np.arange(0, args.seq_len * 2 - 1, 2)
            x_intp_full_ori[:, integer_indices, :] = ori_data

            generated_sequence_continues_all_groundtruth.append(x_intp_full_ori)


            generated_sequence_continues_all.append(x_intp_full)

            samples_all.append(x_intp_full)


    generated_sequence_continues_all = torch.cat(generated_sequence_continues_all, dim=0).cpu().numpy()
    if args.dataset == 'polynomial':
        #Used solely for closed-form solution evaluation
        np.save(args.log_dir+'/continues_Y.npy',generated_sequence_continues_all)
        return

    generated_sequence_ori_all = torch.cat(generated_sequence_ori_all, dim=0).cpu().numpy()
    generated_sequence_continues_all_groundtruth=torch.cat(generated_sequence_continues_all_groundtruth, dim=0).cpu().numpy()
    ground_truth_data=torch.cat(ground_truth_data, dim=0).cpu().numpy()

    print(f"Neural CDE model for {args.dataset} saved!")
    return ground_truth_data,generated_sequence_continues_all_groundtruth,generated_sequence_continues_all,generated_sequence_ori_all


def main(args,seq_,miss_,d_name,gpu_,model_name):
    args.seed=2009
    args.model_name=model_name

    args.device = set_seed_device(args.seed,gpu_)

    args.seq_len=seq_
    args.missing_value=miss_
    args.dataset=d_name

    args.log_dir = './logs_irgen_moencde_final_continues_debug'#函数连续生成

    args.epochs=30

    name = 'MoeNcdeIrreg-miss={}seqlen={}epochs={}'.format(args.missing_value, args.seq_len,args.epochs)

    args.tag=''
    if args.use_channel_wise_autoencoder:
        args.tag += '-autoenc'



    args.normal_datasets=['sine','stock','energy','mujoco']
    args.medical_datasets=['ECG200','ECG5000','TwoLeadECG','ECGFiveDays']

    args.log_dir = '%s/%s/%s' % (args.log_dir, args.dataset, name)

    if d_name=='sine':
        args.inp_dim=5
    elif d_name=='stock':
        args.inp_dim = 6
    elif d_name=='energy':
        args.inp_dim = 28
    elif d_name=='mujoco':
        args.inp_dim = 14
    elif d_name=='polynomial':
        args.inp_dim = 1
    elif d_name in args.medical_datasets:
        args.inp_dim = 1


    args.seq_len=seq_
    args.missing_value=miss_
    args.dataset=d_name

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    args.evaluation_continues=True

    dataset = TimeDataset_irregular(args.seq_len, args.dataset, args.missing_value,args)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                   worker_init_fn=seed_worker, generator=g)

    logging.info(args.dataset + ' dataset is ready.')


    if d_name =='polynomial':
        channel_wise_autoencoder_path=f'/data_new/daroms/paroms/KOVAE/logs_irgen_moencde_final_polynomial/polynomial/MoeNcdeIrreg-seed=42-miss={miss_}seqlen=24epochs=30/channel_wise_autoencoder_1d_polynomial.pth'
    elif d_name in args.medical_datasets:
        channel_wise_autoencoder_path=f'/data_new/daroms/paroms/KOVAE/logs_irgen_moencde_final_ECG/{d_name}/MoeNcdeIrreg-seed=42-miss={miss_}seqlen=24epochs=30/channel_wise_autoencoder_1d_{d_name}.pth'
    else:
        channel_wise_autoencoder_path = f'/data_new/daroms/paroms/KOVAE/logs_irgen_moencde_final/{args.dataset}/MoeNcdeIrreg-seed=42-miss={miss_}seqlen={args.seq_len}epochs=30/channel_wise_autoencoder_1d_{args.dataset}.pth'

    input_dim = args.inp_dim
    hidden_dim = args.hidden_channels
    channel_wise_autoencoder = ChannelWiseAutoencoder(input_dim, hidden_dim).to(args.device)

    if os.path.exists(channel_wise_autoencoder_path):
        print("Loading existing 1D channel_wise_autoencoder...")
        print(channel_wise_autoencoder_path)
        input_dim = args.inp_dim
        hidden_dim = args.hidden_channels
        channel_wise_autoencoder = ChannelWiseAutoencoder(input_dim, hidden_dim).to(args.device)
        channel_wise_autoencoder.load_state_dict(torch.load(channel_wise_autoencoder_path, map_location=args.device))


    train_loader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                   worker_init_fn=seed_worker, generator=g)


    if args.dataset == 'polynomial':
        #Used solely for closed-form solution evaluation
        evaluate_moe_neuralCDE(args, train_loader, channel_wise_autoencoder)
        return
    ground_truth_data,generated_sequence_continues_all_groundtruth,generated_sequence_continues_all,generated_sequence_ori_all=evaluate_moe_neuralCDE(args, train_loader, channel_wise_autoencoder)

    print(generated_sequence_continues_all.shape,generated_sequence_ori_all.shape)
    time.sleep(500)
    ori_data_path = f'/data_new/daroms/paroms/KOVAE/datasets/{args.dataset}{args.missing_value}-{args.seq_len}/original_data.pt'
    ori_data = torch.load(ori_data_path, map_location='cpu').numpy()


    metric_continues=MTS_forecasting_eval(generated_sequence_continues_all_groundtruth, generated_sequence_continues_all, args)

    if args.model_name=='Ours':
        if args.use_gen_moe:
            json_record_loss_test = json.dumps(metric_continues, indent=4)
            with open(
                    args.log_dir + f'/seed={args.seed}_{args.predictor}_record_all_loss_test_{args.model_name}_continues_genmoe' + '.json',
                    'w') as json_file:
                json_file.write(json_record_loss_test)
            return

    json_record_loss_test = json.dumps(metric_continues, indent=4)
    with open(args.log_dir + f'/seed={args.seed}_{args.predictor}_record_all_loss_test_{args.model_name}_continues' + '.json', 'w') as json_file:
        json_file.write(json_record_loss_test)


    metric_wocontinues=MTS_forecasting_eval(ori_data,generated_sequence_ori_all,args)

    json_record_loss_test = json.dumps(metric_wocontinues, indent=4)
    with open(args.log_dir + f'/seed={args.seed}_{args.predictor}_record_all_loss_test_{args.model_name}_wocontinues' + '.json', 'w') as json_file:
        json_file.write(json_record_loss_test)



if __name__ == '__main__':
    args = get_args()

    args.use_gen_moe = True

    args.moe_activate = True
    args.use_channel_wise_autoencoder=True
    seq_lens_all = [24]
    missing_values = [0.3,0.5, 0.7]
    missing_values = [0.3]
    # missing_values = [0.3,0.7]
    missing_values = [0.5]
    missing_values = [0.7]
    datasets_all=['polynomial']
    gpu_ = "cuda:6"
    args.use_gen_moe = False


    datasets_all=['ECG200'] #['stock','mujoco','sine','energy','polynomial','ECG200','ECG5000','TwoLeadECG','ECGFiveDays',]
    # datasets_all = ['stock']
    gpu_ = "cuda:0"
    args.use_gen_moe = False #whether using generated moe weights through diffusion model
    model_names=['Ours','KOVAE','GTGAN']

    for model_name in model_names:
            for d_name in datasets_all:
                for seq_ in seq_lens_all:
                    if d_name=='ECG200':
                        seq_=96
                    elif d_name=='ECG5000':
                        seq_=140
                    elif d_name=='ECGFiveDays':
                        seq_=136
                    elif d_name=='TwoLeadECG':
                        seq_=82
                    for miss_ in missing_values:
                        main(args,seq_,miss_,d_name,gpu_,model_name)
