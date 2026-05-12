import os
os.environ['WANDB_MODE'] = 'disabled'

import os
import json
import time

import torch.utils.data as Data
import torch.nn.init
import numpy as np

import random
import argparse
import torchcde
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kovae import KoVAE
import torch.optim as optim
import logging
from utils_kovae.utils_data import TimeDataset_irregular
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.feature_distance_eval import get_mdd_eval, mmd_metric, get_flat_distance


from models.moe_ncde.moe_neural_cde import NeuralCDE,MoECDEFunc,ChannelWiseAutoencoder

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




def train_channel_wise_autoencoder(args,train_loader):
    print("="*50)
    print("STEP 1: Training Channle-wise Autoencoder")
    print("="*50)

    input_dim = args.inp_dim
    hidden_dim = args.hidden_channels
    channel_wise_autoencoder = ChannelWiseAutoencoder(input_dim, hidden_dim).to(args.device)
    opt = torch.optim.Adam(channel_wise_autoencoder.parameters(), lr=1e-4)

    epochs = 1000
    epochs = 500
    # epochs = 10
    for epoch in range(0, epochs):
        logging.info("Running Epoch : {}".format(epoch + 1))
        channel_wise_autoencoder.train()
        losses_agg_tr = []
        for i, data in enumerate(train_loader, 1):
            opt.zero_grad()
            x = data['data'].to(args.device).float()
            # print(x.shape) #torch.Size([64, 24, 7])
            if args.dataset in ['stock','energy']:
                x = x[:, :, :-1]
            t = torch.FloatTensor(list(range(args.seq_len))).to(args.device)
            train_coeffs = data['inter'] # .to(device)
            train_coeffs=torch.cat(train_coeffs,dim=-1)
            int_func = torchcde.CubicSpline(train_coeffs)
            x_intp_full = int_func.evaluate(t)
            # print(x_intp_full.shape,x.shape)
            # time.sleep(500)
            x_filled = x.clone()  # 保留原始值
            missing_mask = torch.isnan(x)  # 或根据你的缺失标记
            x_filled[missing_mask] = x_intp_full[missing_mask]


            x_filled = x_filled.reshape(-1,args.inp_dim)
            x = x.reshape(-1,args.inp_dim)
            recon = channel_wise_autoencoder(x_filled)
            x_no_nan = x_filled[~torch.isnan(x)]
            x_rec_no_nan = recon[~torch.isnan(x)]

            loss = F.mse_loss(x_rec_no_nan, x_no_nan)
            loss.backward()
            opt.step()
            if epoch % 10 == 0:
                print(f"Channel-wise Autoencoder Epoch {epoch}, Loss: {loss.item():.6f}")


    print(f"Channel-wise Autoencoder training completed! Final loss: {loss.item():.6f}")
    torch.save(channel_wise_autoencoder.state_dict(), args.log_dir+"/channel_wise_autoencoder_1d_{}".format(args.dataset)+".pth")
    print("1D Channel-wise Autoencoder saved!")
    return channel_wise_autoencoder

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

    extend_metrics = update_metrics_dict({}, tmp_name, data_name, args.seq_len, ori_data, generated_data)
    met_dict_final['extend'] = extend_metrics[list(extend_metrics.keys())[0]]

    json_record_loss_test = json.dumps(met_dict_final, indent=4)
    with open(args.log_dir + '/record_all_loss_test'+args.tag + '.json', 'w') as json_file:
        json_file.write(json_record_loss_test)

    with open(args.log_dir +'/args'+args.tag+'.json' , 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)




def train_moe_neuralCDE(args,train_loader,channel_wise_autoencoder):
    print("="*50)
    print("STEP 2: Training MOE_neuralCDE")
    print("="*50)


    model = NeuralCDE(
        args.inp_dim,
        args.hidden_channels,
        args.interpolation,
        pretrained_encoder=channel_wise_autoencoder.encoder,
        pretrained_decoder=channel_wise_autoencoder.decoder,
        moe_activate=args.moe_activate,
        args=args
    ).to(args.device)

    model.freeze_encoder_decoder()

    # 只训练 func（包含 MoE/路由/alpha/低通）
    if args.use_channel_wise_autoencoder:
        opt = torch.optim.Adam(model.func.parameters(), lr=args.learning_rate)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epochs = args.epochs
    # print(epochs)
    # time.sleep(500)
    # epochs = 10
    loss_all_dict={}
    for epoch in range(0, epochs):
        t0 = time.time()
        logging.info("Running Epoch : {}".format(epoch + 1))
        channel_wise_autoencoder.train()
        losses_agg_tr = []

        epoch_loss = 0.0
        epoch_lb = 0.0
        epoch_alpha_reg = 0.0

        # 可选：若使用低通且需要冻结若干 epoch
        if args.moe_activate and isinstance(model.func, MoECDEFunc) and model.func.use_long_lowpass:
            # 是否需要冻结/解冻低通
            if epoch < model.func.long_lowpass_freeze_epochs:
                for p in model.func.long_lowpass.parameters():
                    p.requires_grad = False
            else:
                for p in model.func.long_lowpass.parameters():
                    p.requires_grad = True
        sample_count=0
        for i, data in enumerate(train_loader, 1):
            opt.zero_grad()
            x = data['data'].to(args.device).float()
            # print(x.shape) #torch.Size([64, 24, 7])
            if args.dataset in ['stock','energy']:
                x = x[:, :, :-1]
            sample_count+=x.shape[0]
            t = torch.FloatTensor(list(range(args.seq_len))).to(args.device)
            train_coeffs = data['inter'] # .to(device)
            train_coeffs=torch.cat(train_coeffs,dim=-1)
            recon = model(t,train_coeffs)  # (len, 1)

            x_no_nan = x[~torch.isnan(x)]
            x_rec_no_nan = recon[~torch.isnan(x)]
            recon = F.mse_loss(x_rec_no_nan, x_no_nan)  # 主重建损失

            lb_loss = torch.tensor(0.0, device=args.device)
            alpha_reg = torch.tensor(0.0, device=args.device)
            # t1 = time.time()
            if args.moe_activate and isinstance(model.func, MoECDEFunc):
                lb_loss = model.func.get_load_balance_loss()
                alpha_reg = model.func.alpha_regularizer()
                # print('3333333333333333')
            # print('****************alpha_reg cal costs******************', time.time() - t1)
            t1 = time.time()
            total = recon + args.moe_load_balance_weight * lb_loss + alpha_reg
            # print(total,lb_loss,alpha_reg,args.moe_load_balance_weight,recon)
            # time.sleep(500)
            total.backward()
            opt.step()
            # print('****************backward cal costs******************', time.time() - t1)
            if args.moe_activate and isinstance(model.func, MoECDEFunc):
                model.func.clear_expert_weights()

            epoch_loss += recon.item()
            epoch_lb += lb_loss.item()
            epoch_alpha_reg += alpha_reg.item()

        if args.moe_activate and isinstance(model.func, MoECDEFunc):
            model.func.bump_epoch()


        avg_recon = epoch_loss / sample_count
        avg_lb = epoch_lb / sample_count
        avg_alpha_reg = epoch_alpha_reg / sample_count

        loss_all_dict[epoch]=avg_recon
        print('***************epoch cost***************',time.time()-t0)
        if args.moe_activate and args.moe_load_balance_weight > 0:
            print(f"[{args.dataset}] Epoch {epoch} | Recon: {avg_recon:.6f} | LB: {avg_lb:.6f} | AlphaReg: {avg_alpha_reg:.6f} | Total: {(avg_recon + args.moe_load_balance_weight*avg_lb + avg_alpha_reg):.6f}")
        else:
            print(f"[{args.dataset}] Epoch {epoch} | Recon: {avg_recon:.6f}")


        # 每 10 个 epoch 打印一次专家权重概况
        if args.moe_activate and isinstance(model.func, MoECDEFunc) and epoch % 10 == 0:
            # test_coeffs = prepare_data_for_cde(samples[0], t, interpolation_method, device)
            with torch.no_grad():
                _ = model(t,train_coeffs)
                stats = model.func.get_expert_utilization_stats()
                if stats:
                    ew = np.asarray(stats['expert_weights'])
                    # ew_fmt = [f"{float(w):.3f}" for w in ew]
                    ew_fmt=ew
                    alpha = model.func._compute_alpha().item() if model.func.use_long_expert else 0.0
                    print(f"  Expert weights: {ew_fmt} (alpha≈{alpha:.3f})")
                    print(f"  Active experts: {int(stats['num_active_experts'])}/{args.moe_num_experts}")
            model.func.clear_expert_weights()

    # 保存最终样本和专家权重
    moe_experts_all = []
    samples_all = []
    samples_all_filled = []
    filled=True
    for i, data in enumerate(train_loader, 1):
        with torch.no_grad():
            x = data['data'].to(args.device).float()
            # print(x.shape) #torch.Size([64, 24, 7])
            if args.dataset in ['stock','energy']:
                x = x[:, :, :-1]
            sample_count += x.shape[0]
            t = torch.FloatTensor(list(range(args.seq_len))).to(args.device)

            train_coeffs = data['inter']  # .to(device)
            train_coeffs = torch.cat(train_coeffs, dim=-1)
            x_intp_full = model(t, train_coeffs)  # (len, 1)

            if filled:
                x_filled = x.clone()  # 保留原始值
                missing_mask = torch.isnan(x)  # 或根据你的缺失标记
                x_filled[missing_mask] = x_intp_full[missing_mask]

            if args.moe_activate:
                stats = model.func.get_expert_utilization_stats()
                # print(stats['expert_weights'].shape)
                moe_experts_all.append(stats['expert_weights'])

            samples_all_filled.append(x_filled)
            samples_all.append(x_intp_full)

    samples_all = torch.cat(samples_all, dim=0).cpu().numpy()
    samples_all_filled = torch.cat(samples_all_filled, dim=0).cpu().numpy()

    # moe_experts_all = torch.cat(moe_experts_all, dim=0).cpu().numpy()
    if args.moe_activate:
        moe_experts_all = np.concatenate(moe_experts_all, axis=0)
    ori_data = list()
    for data in train_loader:
        ori_data.append(data['original_data'].detach().cpu().numpy())
    ori_data = np.vstack(ori_data)

    json_record_loss_test = json.dumps(loss_all_dict, indent=4)
    with open(args.log_dir + '/loss_train_dict_'+args.tag + '.json', 'w') as json_file:
        json_file.write(json_record_loss_test)

    evaluation(args,ori_data,samples_all)


    if args.moe_activate:
        if args.use_channel_wise_autoencoder:
            np.save(args.log_dir + f"/moe_samples_all_{args.dataset}_replace.npy", samples_all_filled)
            np.save(args.log_dir + f"/moe_samples_all_{args.dataset}.npy", samples_all)
            np.save(args.log_dir + f"/moe_experts_all_{args.dataset}.npy", moe_experts_all)
            torch.save(model.state_dict(), args.log_dir + f"/moe_neuralcde_fix_{args.dataset}.pth")
        else:
            np.save(args.log_dir + f"/moe_samples_all_{args.dataset}_replace_woautenc.npy", samples_all_filled)
            np.save(args.log_dir + f"/moe_samples_all_{args.dataset}_woautenc.npy", samples_all)
            np.save(args.log_dir + f"/moe_experts_all_{args.dataset}_woautenc.npy", moe_experts_all)
            torch.save(model.state_dict(), args.log_dir + f"/moe_neuralcde_fix_{args.dataset}_woautenc.pth")
    else:
        if args.use_channel_wise_autoencoder:
            np.save(args.log_dir + f"/samples_all_{args.dataset}_replace.npy", samples_all_filled)
            np.save(args.log_dir + f"/samples_all_{args.dataset}.npy", samples_all)
            # np.save(args.log_dir + f"/moe_experts_all_{args.dataset}.npy", moe_experts_all)
            torch.save(model.state_dict(), args.log_dir + f"/neuralcde_fix_{args.dataset}.pth")
        else:
            np.save(args.log_dir + f"/samples_all_{args.dataset}_replace_woautenc.npy", samples_all_filled)
            np.save(args.log_dir + f"/samples_all_{args.dataset}_woautenc.npy", samples_all)
            # np.save(args.log_dir + f"/moe_experts_all_{args.dataset}.npy", moe_experts_all)
            torch.save(model.state_dict(), args.log_dir + f"/neuralcde_fix_{args.dataset}_woautenc.pth")
    print(f"Neural CDE model for {args.dataset} saved!")
    return model




def evaluate_moe_neuralCDE(args,train_loader,channel_wise_autoencoder):
    print("="*50)
    print("STEP 3: Evaluating MOE_neuralCDE")
    print("="*50)


    model = NeuralCDE(
        args.inp_dim,
        args.hidden_channels,
        args.interpolation,
        pretrained_encoder=channel_wise_autoencoder.encoder,
        pretrained_decoder=channel_wise_autoencoder.decoder,
        moe_activate=args.moe_activate,
        args=args
    ).to(args.device)

    model.freeze_encoder_decoder()
    moe_cde_model_path=args.log_dir + f"/moe_neuralcde_fix_{args.dataset}.pth"
    model.load_state_dict(torch.load(moe_cde_model_path, map_location=args.device))

    # 保存最终样本和专家权重
    moe_experts_all = []
    samples_all = []
    filled=False
    sample_count=0
    for i, data in enumerate(train_loader, 1):
        with torch.no_grad():
            x = data['data'].to(args.device).float()
            # print(x.shape) #torch.Size([64, 24, 7])
            if args.dataset in ['stock', 'energy']:
                x = x[:, :, :-1]
            sample_count += x.shape[0]
            t = torch.FloatTensor(list(range(args.seq_len))).to(args.device)

            train_coeffs = data['inter']  # .to(device)
            train_coeffs = torch.cat(train_coeffs, dim=-1)
            x_intp_full = model(t, train_coeffs)  # (len, 1)

            if filled:
                x_filled = x.clone()  # 保留原始值
                missing_mask = torch.isnan(x)  # 或根据你的缺失标记
                x_filled[missing_mask] = x_intp_full[missing_mask]


            stats = model.func.get_expert_utilization_stats()
            # print(stats['expert_weights'].shape)
            moe_experts_all.append(stats['expert_weights'])
            if filled:
                samples_all.append(x_filled)
            else:
                samples_all.append(x_intp_full)

    samples_all = torch.cat(samples_all, dim=0).cpu().numpy()
    # moe_experts_all = torch.cat(moe_experts_all, dim=0).cpu().numpy()
    moe_experts_all = np.concatenate(moe_experts_all, axis=0)
    np.save(args.log_dir + f"/samples_all_{args.dataset}.npy", samples_all)
    if filled:
        np.save(args.log_dir + f"/moe_experts_all_{args.dataset}_replace.npy", moe_experts_all)
    else:
        np.save(args.log_dir + f"/moe_experts_all_{args.dataset}.npy", moe_experts_all)
    # print(samples_all.shape,moe_experts_all.shape)

    if args.moe_activate:
        torch.save(model.state_dict(), args.log_dir + f"/moe_neuralcde_fix_{args.dataset}.pth")
    else:
        torch.save(model.state_dict(), args.log_dir + f"/neuralcde_fix_{args.dataset}.pth")
    print(f"Neural CDE model for {args.dataset} saved!")
    return model


def main(args,seq_,miss_,d_name,gpu_,n_expert):
    args.device = set_seed_device(args.seed,gpu_)
    args.seq_len=seq_
    args.missing_value=miss_
    args.dataset=d_name

    args.log_dir = './logs_irgen_moencde_final_debug'
    args.normal_datasets=['sine','stock','energy','mujoco']
    args.medical_datasets=['ECG200','ECG5000','TwoLeadECG','ECGFiveDays']

    args.epochs = 30

    args.moe_num_experts=n_expert
    name = 'MoeNcdeIrreg-seed={}-miss={}seqlen={}epochs={}'.format(args.seed, args.missing_value, args.seq_len,args.epochs)

    args.tag=''
    if args.use_channel_wise_autoencoder:
        args.tag += '-autoenc'


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
        args.inp_dim=1
    args.seq_len=seq_
    args.missing_value=miss_
    args.dataset=d_name


    check_path=args.log_dir + '/record_all_loss_test'+args.tag + '.json'
    if os.path.exists(check_path):
        return
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if d_name=='polynomial':
        args.n_samples=5000
    dataset = TimeDataset_irregular(args.seq_len, args.dataset, args.missing_value,args)


    if d_name in args.medical_datasets:
        args.seq_len = dataset.seq_len
        args.inp_dim = dataset.inp_dim

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                   worker_init_fn=seed_worker, generator=g)

    logging.info(args.dataset + ' dataset is ready.')

    #Training channel_wise_autoencoder
    channel_wise_autoencoder_path = args.log_dir+"/channel_wise_autoencoder_1d_{}".format(args.dataset)+".pth"

    if not args.use_channel_wise_autoencoder:
        input_dim = args.inp_dim
        hidden_dim = args.hidden_channels
        channel_wise_autoencoder = ChannelWiseAutoencoder(input_dim, hidden_dim).to(args.device)
        #In practice, in the NeuralCDE of `moe_neural_cde.py` script, replace the pre-trained encoder and
        # decoder with learnable linear layers.
    elif os.path.exists(channel_wise_autoencoder_path):

        print("Loading existing 1D channel_wise_autoencoder...")
        print(channel_wise_autoencoder_path)
        input_dim = args.inp_dim
        hidden_dim = args.hidden_channels
        channel_wise_autoencoder = ChannelWiseAutoencoder(input_dim, hidden_dim).to(args.device)
        channel_wise_autoencoder.load_state_dict(torch.load(channel_wise_autoencoder_path, map_location=args.device))
    else:
        channel_wise_autoencoder=train_channel_wise_autoencoder(args, train_loader)

    # Training MOE neuralCDE
    train_loader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                   worker_init_fn=seed_worker, generator=g)


    train_moe_neuralCDE(args, train_loader, channel_wise_autoencoder)

if __name__ == '__main__':
    args = get_args()
    args.moe_activate = False
    args.use_channel_wise_autoencoder=True
    args.use_channel_wise_autoencoder=False
    datasets_all = ['polynomial','ECG200','ECG5000','TwoLeadECG','ECGFiveDays','stock','mujoco','sine','energy']
    seq_lens_all=[36]
    missing_values=[0.5]
    gpu_ = "cuda:0"
    n_expert=4
    for seq_ in seq_lens_all:
        for d_name in datasets_all:
            for miss_ in missing_values:
                main(args,seq_,miss_,d_name,gpu_,n_expert)
