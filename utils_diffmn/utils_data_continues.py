"""
data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine datasets
(2) real_data_loading: Load and preprocess real datasets
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""
import copy
import time

## Necessary Packages
import numpy as np
import os
import torch
import controldiffeq
import pathlib
from utils_kovae import datautils

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent


def to_tensor(data):
    return torch.from_numpy(data).float()


def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - datasets: original datasets

    Returns:
      - norm_data: normalized datasets
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def pendulum_nonlinear(num_points, noise, theta=2.4):
    from matplotlib import pylab as plt
    from scipy.special import ellipj, ellipk
    np.random.seed(1)

    def sol(t, theta0):
        S = np.sin(0.5 * (theta0))
        K_S = ellipk(S ** 2)
        omega_0 = np.sqrt(9.81)
        sn, cn, dn, ph = ellipj(K_S - omega_0 * t, S ** 2)
        theta = 2.0 * np.arcsin(S * sn)
        d_sn_du = cn * dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0 * S * d_sn_dt / np.sqrt(1.0 - (S * sn) ** 2)
        return np.stack([theta, d_theta_dt], axis=1)

    anal_ts = np.arange(0, 170 * 0.1, 0.1)
    # Generate random angles in radians
    # angles = np.random.uniform(1, 3.5, num_points)
    angles = np.random.uniform(.5, 2.7, num_points)
    X = []
    for theta in angles:
     X.append(sol(anal_ts, theta))

    # X = X.T
    # Xclean = X.copy()
    X = np.array(X)
    X += np.random.standard_normal(X.shape) * noise


    X = MinMaxScaler(X)

    return X

def sine_data_generation(no, seq_len, dim):
    """Sine datasets generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - datasets: generated datasets
    """
    # Initialize the output
    data = list()

    # Generate sine datasets
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated datasets
        data.append(np.expand_dims(temp,axis=0) )

    return data


def real_data_loading(data_name, seq_len):
    """Load and preprocess real-world datasets.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length

    Returns:
      - datasets: preprocessed datasets.
    """
    assert data_name in ['stock', 'energy', 'metro']

    if data_name == 'stock':
        ori_data = np.loadtxt('./datasets/stock_data.csv', delimiter=",", skiprows=1)
    elif data_name == 'energy':
        ori_data = np.loadtxt('./datasets/energy_data.csv', delimiter=",", skiprows=1)
    elif data_name == 'metro':
        ori_data = np.loadtxt('./datasets/metro_data.csv', delimiter=",", skiprows=1)

    # Flip the datasets to make chronological datasets
    ori_data = ori_data[::-1]
    # Normalize the datasets
    ori_data = MinMaxScaler(ori_data)

    # Preprocess the datasets
    temp_data = []
    # Cut datasets by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data


class TimeDataset_irregular(torch.utils.data.Dataset):
    def __init__(self, seq_len, data_name, missing_rate=0.0,args=None):
        self.args=args
        self.device=self.args.device
        SEED = 56789
        base_loc = PROJECT_DIR / 'datasets_continues'
        loc = PROJECT_DIR / 'datasets_continues' / (data_name + str(missing_rate)+'-'+str(args.seq_len))


        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)

        if data_name == 'polynomial':
            ori_data=np.load('./logs_irgen_moencde_final_polynomial/polynomial/MoeNcdeIrreg-seed=42-miss=0.3seqlen=24epochs=30/polynomial_Y_gt.npy')
            ori_data=np.expand_dims(ori_data,axis=-1)
            pass
        elif data_name in args.medical_datasets:
            ori_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
            norm_data = ori_data

            self.seq_len = norm_data.shape[1]
            self.inp_dim = norm_data.shape[2]

        else:
            ori_data_path = f'./datasets/{args.dataset}{args.missing_value}-{args.seq_len}/original_data.pt'
            ori_data = torch.load(ori_data_path, map_location='cpu').numpy()


        if args.model_name=='KOVAE':
            generated_path_kovae = f'./logs_irgen_baselines_generation/{args.dataset}/KOVAE-seed=10-miss={args.missing_value}seqlen={args.seq_len}/{args.dataset}_{args.seq_len}_{args.missing_value}_generation.npy'
            generated_data = np.load(generated_path_kovae)
            moe_weights=[]
        elif args.model_name=='GTGAN':

            generated_path_kovae = f'/data_new/daroms/paroms/GT-GAN/logs_irgen_generation/{args.dataset}/gtgan-seed=10-miss={args.missing_value}seqlen={args.seq_len}/samples_generation.npy'
            generated_data = np.load(generated_path_kovae)
            moe_weights=[]
        elif args.model_name=='Ours':
            if data_name=='polynomial':
                generated_data_all = np.load(
                    f'/data_new/daroms/paroms/KOVAE/logs/MOEDiff_Final_polynomial/dataset=polynomial-miss={args.missing_value}seqlen={args.seq_len}-replace=False_pam_seed0/generated_samples/polynomial_{args.seq_len}_generation.npy')

            elif data_name in args.medical_datasets:
                generated_path_ours = f'/data_new/daroms/paroms/KOVAE/logs/MOEDiff_Final_ECG/dataset={args.dataset}-miss={args.missing_value}seqlen={args.seq_len}-replace=False_pam_seed0/generated_samples'
                files_all = os.listdir(generated_path_ours)
                for file_path in files_all:
                    if 'last' in file_path:
                        continue
                    temp_path = os.path.join(generated_path_ours, file_path)
                    generated_data_all = np.load(temp_path)

            else:
                generated_path_ours = f'/data_new/daroms/paroms/KOVAE/logs/MOEDiff_Final_v2/dataset={args.dataset}-miss={args.missing_value}seqlen={args.seq_len}-replace=False_pam_seed0/generated_samples'
                files_all = os.listdir(generated_path_ours)
                for file_path in files_all:
                    if 'last' in file_path:
                        continue
                    temp_path = os.path.join(generated_path_ours, file_path)
                    generated_data_all = np.load(temp_path)

            generated_data = generated_data_all[..., :-4, :]
            moe_weights = generated_data_all[..., -4:, :]

        norm_data=generated_data
        self.original_sample=copy.deepcopy(ori_data)
        self.samples=copy.deepcopy(norm_data)
        norm_data_tensor = torch.Tensor(norm_data).float().to(self.device)
        ori_data_tensor= torch.Tensor(ori_data).float().to(self.device)

        time = torch.FloatTensor(list(range(norm_data_tensor.size(1)))).to(self.device)

        self.train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time, norm_data_tensor)

        self.train_coeffs_ori=controldiffeq.natural_cubic_spline_coeffs(time,ori_data_tensor)

        self.original_sample = torch.tensor(self.original_sample).to(self.device)
        self.samples = torch.tensor(self.samples).to(self.device)

        if len(moe_weights)!=None:
            moe_weights=torch.Tensor(moe_weights).float().to(self.device)
            # self.original_sample=torch.cat([self.original_sample,moe_weights],dim=1)
            self.samples = torch.cat([self.samples, moe_weights], dim=1)

        self.original_sample = np.array(self.original_sample.cpu())
        self.samples = np.array(self.samples.cpu())
        self.size = len(self.samples)

    def __getitem__(self, index):
        batch_coeff = (self.train_coeffs[0][index].float().to(self.device),
                       self.train_coeffs[1][index].float().to(self.device),
                       self.train_coeffs[2][index].float().to(self.device),
                       self.train_coeffs[3][index].float().to(self.device))

        batch_coeff_ori = (self.train_coeffs_ori[0][index].float().to(self.device),
                       self.train_coeffs_ori[1][index].float().to(self.device),
                       self.train_coeffs_ori[2][index].float().to(self.device),
                       self.train_coeffs_ori[3][index].float().to(self.device))

        self.sample = {'data': self.samples[index], 'inter': batch_coeff, 'inter_ori':batch_coeff_ori,'original_data': self.original_sample[index]}

        return self.sample  # self.samples[index]

    def __len__(self):
        return len(self.samples)


def load_data(dir,args):
    # print(dir)
    # time.sleep(500)
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename),map_location='cpu')
            tensors[tensor_name] = tensor_value

    return tensors

def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')