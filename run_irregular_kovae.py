import time

import torch.utils.data as Data
import torch.nn.init
import numpy as np

import random
import argparse

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kovae import KoVAE
import torch.optim as optim
import logging
from utils_kovae.utils_data import TimeDataset_irregular

from metrics.feature_distance_eval import get_mdd_eval, mmd_metric, get_flat_distance

import json
def define_args():
    parser = argparse.ArgumentParser(description="KoVAE")

    # general
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--pinv_solver', type=bool, default=False)
    parser.add_argument('--neptune', default='debug', help='async runs as usual, debug prevents logging')
    parser.add_argument('--tag', default='sine, alpha_beta_sens')

    # data
    parser.add_argument("--dataset", default='stock')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N')
    parser.add_argument('--seq_len', type=int, default=24, metavar='N')
    parser.add_argument('--inp_dim', type=int, default=6, metavar='N')
    parser.add_argument('--missing_value', type=float, default=0.3)

    # model
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=20,
                        help='the hidden dimension of the output decoder lstm')

    # loss params
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--w_rec', type=float, default=1.)
    parser.add_argument('--w_kl', type=float, default=.1)
    parser.add_argument('--w_pred_prior', type=float, default=0.0005)

    return parser

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


parser = define_args()
args = parser.parse_args()
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

def main(args,seq_,miss_,d_name,gpu_):

    args.device = set_seed_device(args.seed,gpu_)

    args.log_dir = './logs_irgen_baselines'

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

    args.batch_size=256


    args.normal_datasets=['sine','stock','energy','mujoco']
    args.medical_datasets=['ECG200','ECG5000','TwoLeadECG','ECGFiveDays']
    args.n_samples=2500

    name = 'KOVAE-seed={}-miss={}seqlen={}'.format(args.seed,args.missing_value,args.seq_len)

    args.log_dir = '%s/%s/%s' % (args.log_dir, args.dataset, name)




    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    path=args.log_dir + '/record_all_loss_test' + '.json'
    # print(path)
    if os.path.exists(path):
        return
    print('***********************{}************************'.format(args.log_dir))

    dataset = TimeDataset_irregular(args.seq_len, args.dataset, args.missing_value,args)

    if d_name in args.medical_datasets:
        args.seq_len=dataset.seq_len
        args.inp_dim=dataset.inp_dim

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                   worker_init_fn=seed_worker, generator=g)

    logging.info(args.dataset + ' dataset is ready.')

    # create model
    model = KoVAE(args).to(device=args.device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)


    params_num = sum(param.numel() for param in model.parameters())

    logging.info(args)
    logging.info("number of model parameters: {}".format(params_num))
    print("number of model parameters: {}".format(params_num))

    logging.info("Starting training loop at step %d." % (0,))


    for epoch in range(0, args.epochs):
        logging.info("Running Epoch : {}".format(epoch + 1))

        model.train()
        losses_agg_tr = []
        count=0
        for i, data in enumerate(train_loader, 1):

            x = data['data'].to(args.device).float()

            count+=x.shape[0]

            train_coeffs = data['inter'] # .to(device)
            time = torch.FloatTensor(list(range(args.seq_len))).to(args.device)
            final_index = (torch.ones(x.shape[0]) * (args.seq_len-1)).to(args.device).float()#64个样本的最后一个时间步值

            if d_name in ['stock','energy']:
                x = x[:, :, :-1]

            optimizer.zero_grad()
            x_rec, Z_enc, Z_enc_prior = model(train_coeffs, time, final_index)

            x_no_nan = x[~torch.isnan(x)]
            x_rec_no_nan = x_rec[~torch.isnan(x)]

            losses = model.loss(x_no_nan, x_rec_no_nan, Z_enc, Z_enc_prior)  # x_rec, x_pred_rec, z, z_pred_, Ct
            losses[0].backward()
            optimizer.step()

            losses_agg_tr = agg_losses(losses_agg_tr, losses)

        log_losses(epoch, losses_agg_tr, model.names)

    logging.info("Training is complete")

    torch.save(model.state_dict(), args.log_dir+'/model.pth')

    # generate datasets:
    args.device = set_seed_device(args.seed,gpu_)
    model.eval()
    with torch.no_grad():
        generated_data = []
        for data in train_loader:
            n_sample = data['original_data'].shape[0]
            generated_data.append(model.sample_data(n_sample).detach().cpu().numpy())
    generated_data = np.vstack(generated_data)

    tmp_name = f'{args.dataset}_{args.seq_len}_{args.missing_value}_generation'
    np.save(args.log_dir+'/'+ f'{tmp_name}.npy', generated_data)


    logging.info("Data generation is complete")

    ori_data = list()
    for data in train_loader:
        ori_data.append(data['original_data'].detach().cpu().numpy())
    # print(ori_data[0].shape)
    ori_data = np.vstack(ori_data)

    import time
    print('ok')
    met_dict_final={}

    from metrics.discriminative_torch import discriminative_score_metrics
    # deterministic eval
    args.device = set_seed_device(args.seed,gpu_)
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


    tmp_name = f'{dataset}_{args.seq_len}_generation'
    data_name=args.dataset

    extend_metrics = update_metrics_dict({}, tmp_name, data_name, args.seq_len, ori_data, generated_data)

    met_dict_final['extend'] = extend_metrics[list(extend_metrics.keys())[0]]
    met_dict_final['epoch']=epoch
    json_record_loss_test = json.dumps(met_dict_final, indent=4)
    with open(args.log_dir + '/record_all_loss_test' + '.json', 'w') as json_file:
        json_file.write(json_record_loss_test)

    with open(args.log_dir +'/args.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    datasets_all = ['stock','mujoco','sine','energy','polynomial','ECG200','ECG5000','TwoLeadECG','ECGFiveDays',]
    seq_lens_all=[36] #24 12
    missing_values=[0.5]#0.3 0.7
    gpu_='cuda:0'
    for d_name in datasets_all:
        for seq_ in seq_lens_all:
            for miss_ in missing_values:
                main(args,seq_,miss_,d_name,gpu_)
