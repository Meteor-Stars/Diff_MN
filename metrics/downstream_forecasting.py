"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: August 21st 2025
Code adapted to PyTorch from original TensorFlow implementation
"""
import time

# Necessary Packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from .metrics_forecasting import metric_tfs


def MTS_forecasting_eval(ori_data, generated_data,args=None):
    """Report the performance of Post-hoc RNN one-step ahead prediction.

    Args:
        - ori_data: original datasets
        - generated_data: generated synthetic datasets

    Returns:
        - predictive_score: MAE of the predictions on the original datasets
    """
    # 基本参数
    print('data shape',ori_data.shape,generated_data.shape)

    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)
    no, seq_len, dim = ori_data.shape

    device=args.device
    # 网络参数
    hidden_dim = int(dim / 2)

    iterations = 5000

    batch_size = 128

    class Predictor(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(Predictor, self).__init__()

            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            """
            Args:
                x: shape (batch_size, 11, input_dim)
            Returns:
                out: shape (batch_size, output_dim) → 预测第12步的5个变量
            """

            out, hidden = self.gru(x)

            last_output = out[:, -1, :]  #

            prediction = self.fc(last_output)  # (B, output_dim)

            return prediction
    args.pred_len=1
    args.individual=False
    args.enc_in=args.inp_dim
    args.seq_len=seq_len-1


    predictor = Predictor(input_dim=dim, hidden_dim=hidden_dim*4, output_dim=args.inp_dim).to(args.device)


    criterion = nn.L1Loss()  #

    optimizer = optim.Adam(predictor.parameters(),lr=1e-4)

    def prepare_batch(data, batch_indices):
        batch_data = [data[i] for i in batch_indices]
        X = torch.stack([torch.FloatTensor(seq[:seq_len-1, :]).to(device) for seq in batch_data])
        Y = torch.stack([torch.FloatTensor(seq[seq_len-1, :]).to(device) for seq in batch_data])
        return X, Y

    predictor.train()
    for itt in range(iterations):
        if itt%100==0:
            print('iterations',itt)
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]

        X_mb, Y_mb = prepare_batch(generated_data, train_idx)

        optimizer.zero_grad()
        y_pred = predictor(X_mb)

        loss = criterion(y_pred, Y_mb)
        loss.backward()
        optimizer.step()

    predictor.eval()
    idx = np.random.permutation(len(ori_data))
    train_idx = idx[:no]

    X_mb, Y_mb = prepare_batch(ori_data, train_idx)


    with torch.no_grad():
        pred_Y_curr = predictor(X_mb)

    met_dict={}
    mae, mse, rmse, mape, mspe, rse, corr=metric_tfs(np.expand_dims(pred_Y_curr.cpu().numpy(),axis=1) ,np.expand_dims(Y_mb.cpu().numpy(),axis=1) )

    if len(corr)!=1:
        print('problem corr',corr)
    met_dict['mae']=float(mae)
    met_dict['mse']=float(mse)
    met_dict['rmse']=float(rmse)
    met_dict['corr']=float(corr.tolist()[0])
    print(met_dict)
    return met_dict


