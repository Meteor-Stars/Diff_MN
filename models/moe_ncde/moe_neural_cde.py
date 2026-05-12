import os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcde

class ChannelWiseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        # print(y.shape)
        # time.sleep(500)
        return y

class CDEFunc(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.lin1 = nn.Linear(hidden_channels, 50)
        self.lin2 = nn.Linear(50, hidden_channels * input_channels)

    def forward(self, t, z):

        out = torch.tanh(self.lin2(F.relu(self.lin1(z))))

        return out.view(-1, self.hidden_channels, self.input_channels)

# -------------------------
# MoE CDE func with Long Expert (expert 0)
# -------------------------
class MoECDEFunc(nn.Module):
    def __init__(self, input_channels, hidden_channels, args):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_experts = args.moe_num_experts
        self.args=args
        # gating flags
        self.use_topk = args.moe_topk is not None
        self.topk = args.moe_topk if self.use_topk else 1

        self.use_ema = args.moe_ema_alpha is not None
        self.ema_alpha = args.moe_ema_alpha if self.use_ema else 0.9

        # ----- Long Expert controls -----
        self.use_long_expert = args.moe_long_expert
        # global alpha (scalar) for expert-0
        self.global_alpha = nn.Parameter(torch.tensor(args.moe_long_alpha_init, dtype=torch.float32))
        self.long_min = float(args.moe_long_min)        # hard floor for alpha after sigmoid
        self.long_max = float(args.moe_long_max)        # hard cap
        self.long_target_alpha = float(args.moe_long_target_alpha)  # for regularizer
        self.alpha_reg_weight = float(args.moe_long_alpha_reg)      # lambda for (alpha - target)^2
        # low-pass for long expert
        self.use_long_lowpass = args.moe_long_use_lowpass
        self.long_lowpass_kernel = int(args.moe_long_kernel)
        self.long_lowpass_freeze_epochs = int(args.moe_long_freeze_epochs)

        self.experts = nn.ModuleList([
            CDEFunc(input_channels,hidden_channels) for _ in range(self.num_experts)
        ])

        if self.use_long_lowpass:
            k = max(1, self.long_lowpass_kernel)
            pad = (k - 1) // 2
            self.long_lowpass = nn.Conv1d(
                hidden_channels, hidden_channels,
                kernel_size=k, padding=pad, groups=hidden_channels, bias=False
            )
            with torch.no_grad():
                w = torch.ones(hidden_channels, 1, k) / k
                self.long_lowpass.weight.copy_(w)
        else:
            self.long_lowpass = None

        # router
        self.router = nn.Sequential(
            nn.Linear(args.inp_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_experts)
        )

        self.input_fc = nn.Linear(1, 64)

        # load balance
        self.use_load_balance = args.moe_load_balance_weight > 0.0
        self.load_balance_weight = args.moe_load_balance_weight

        # runtime caches
        self.expert_weights = None
        self.expert_weights_for_loss = None
        self._ema_w = None
        self._epoch = 0  # for optional lowpass freeze scheduling

    def bump_epoch(self):
        self._epoch += 1

    def _compute_alpha(self):
        # alpha in (0,1) via sigmoid, then clamp
        alpha = torch.sigmoid(self.global_alpha)
        alpha = torch.clamp(alpha, min=self.long_min, max=self.long_max)
        return alpha

    def set_expert_weights_from_observation(self, observation_data,generated_moe_weights=[]):
        # observation_data: (seq_len, input_channels)
        # assert observation_data.dim() == 3 and observation_data.size(-1) == self.input_channels, \
        #     f"observation_data shape {tuple(observation_data.shape)} invalid; expect (seq_len, {self.input_channels})"
        self.moe_weights=generated_moe_weights

        if len(generated_moe_weights)!=0:
            # print('*************utilize generated moe weights****************')
            self.moe_weights=generated_moe_weights
            generated_moe_weights=generated_moe_weights.transpose(1,2)
            self.expert_weights = generated_moe_weights
            self.expert_weights_for_loss = generated_moe_weights.clone()
            return self.expert_weights

        obs_features = torch.mean(observation_data, dim=1)  # (bathsize,channel)
        logits = self.router(obs_features)  # (1,E)

        if self.num_experts <= 1:
            weights = torch.ones(1, 1, device=logits.device)
        else:
            # distribute residual among experts 1..E-1
            residual_logits = logits[:, 1:]
            even_weights=F.softmax(logits, dim=-1)
            if self.use_topk:
                k = min(self.topk, self.num_experts - 1)
                topv, topi = torch.topk(residual_logits, k, dim=-1)
                residual = torch.zeros_like(residual_logits)
                residual.scatter_(1, topi, F.softmax(topv, dim=-1))
            else:
                residual = F.softmax(residual_logits, dim=-1)  # (1, E-1)

            if self.use_long_expert:
                alpha = self._compute_alpha().view(1, 1)       # (1,1)
                alpha=alpha.repeat(residual.shape[0],1)
            else:
                alpha = torch.zeros(1, 1, device=logits.device)
                alpha = alpha.repeat(residual.shape[0], 1)

            weights = torch.cat([alpha, (1 - alpha) * residual], dim=-1)  # (1,E)

        w = weights.reshape(weights.shape[0],-1)  # (batch size, E,)

        # optional EMA smoothing across sequences (rarely needed since weights per-sequence fixed)
        if self.use_ema:
            if self._ema_w is None or self._ema_w.shape != w.shape:
                self._ema_w = w.detach()
            self._ema_w = self.ema_alpha * self._ema_w + (1 - self.ema_alpha) * w.detach()
            w = self._ema_w


        if self.use_long_expert:
            self.expert_weights = w
            self.expert_weights_for_loss = w.detach().clone()
        else:
            self.expert_weights = even_weights
            self.expert_weights_for_loss = even_weights.detach().clone()

        return self.expert_weights

    def forward(self, t, z):
        if self.expert_weights is None:
            raise ValueError("Expert weights not set! Call set_expert_weights_from_observation first.")

        batch = z.shape[0]
        expert_outs = []

        # expert 0 as long expert: optionally apply lowpass on z
        for idx, expert in enumerate(self.experts):
            if idx == 0 and self.use_long_expert and self.long_lowpass is not None:
                # shape transform for 1D conv over "channel-length"
                # z: (B, hidden) -> (B, hidden, 1), depthwise conv as simple smoothing along a fake length dim
                z_lp = z.unsqueeze(-1)                              # (B, H, 1)
                # If you later switch to time-dependent routing, you can lowpass along time dim instead.
                z_proc = self.long_lowpass(z_lp).squeeze(-1)        # (B, H)
                out = expert(t,z_proc)
            else:

                out = expert(t,z)  # (B, H*I)
            out = out.view(batch, self.hidden_channels, self.input_channels)
            expert_outs.append(out)

        expert_outs = torch.stack(expert_outs, dim=-1)  # (B, H, I, E)

        if len(self.moe_weights) !=0:
            #Directly using the generated expert weights and avoid recalculation
            w=self.expert_weights.unsqueeze(1).repeat(1,expert_outs.shape[1],1,1)
            return torch.sum(w * expert_outs, dim=-1)

        w = self.expert_weights.view(expert_outs.shape[0], 1, 1, -1).expand_as(expert_outs)  # broadcast

        return torch.sum(w * expert_outs, dim=-1)  # (B, H, I)

    def alpha_regularizer(self):
        if not self.use_long_expert or self.alpha_reg_weight <= 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        alpha = self._compute_alpha()
        return self.alpha_reg_weight * (alpha - self.long_target_alpha) ** 2

    def get_load_balance_loss(self):
        if not self.use_load_balance or self.expert_weights_for_loss is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        w = self.expert_weights_for_loss
        if w.numel() <= 1:
            return torch.tensor(0.0, device=w.device)
        # exclude long expert 0 from variance
        return torch.var(w[1:]) if self.use_long_expert and w.numel() > 1 else torch.var(w)

    def clear_expert_weights(self):
        self.expert_weights = None
        self.expert_weights_for_loss = None

    def get_expert_utilization_stats(self):
        if self.expert_weights_for_loss is None:
            return None
        ew = self.expert_weights_for_loss
        stats = {
            'expert_weights': ew.detach().cpu().numpy(),
            'max_weight': torch.max(ew).item(),
            'min_weight': torch.min(ew).item(),
            'entropy': -torch.sum(ew * torch.log(ew + 1e-8)).item(),
            'num_active_experts': torch.sum(ew > 0.01).item()
        }
        return stats


# -------------------------
# Neural CDE wrapper
# -------------------------
class NeuralCDE(nn.Module):
    def __init__(self, input_channels, hidden_channels, interpolation="linear",
                 pretrained_encoder=None, pretrained_decoder=None, moe_activate=False, args=None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.interpolation = interpolation

        self.moe_activate = moe_activate
        self.args = args


        self.inp_dim = self.args.inp_dim
        self.hidden_dim = self.args.hidden_dim
        self.batch_norm = self.args.batch_norm
        self.num_layers = self.args.num_layers

        self.func = MoECDEFunc(input_channels, hidden_channels, args) if moe_activate \
                    else CDEFunc(input_channels, hidden_channels)

        if not self.args.use_channel_wise_autoencoder:

            pretrained_encoder=None
            pretrained_decoder=None
        self.initial = pretrained_encoder if pretrained_encoder is not None \
                       else nn.Linear(input_channels, hidden_channels)

        self.readout = pretrained_decoder if pretrained_decoder is not None \
                       else nn.Linear(hidden_channels, input_channels)
        self.activation_fn = torch.sigmoid
    def freeze_encoder_decoder(self):
        for p in self.initial.parameters():
            p.requires_grad = False
        for p in self.readout.parameters():
            p.requires_grad = False
        print("Encoder and decoder layers frozen!")

    def forward(self, t_span,coeffs,x=None,mask=None):

        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        # set MoE weights once per sequence
        if self.moe_activate and isinstance(self.func, MoECDEFunc):
            if mask!=None:
                pass
            else:
                obs_all = X.evaluate(t_span)    # (batch, seq_len, channels)
                if len(obs_all.shape)==2:
                    obs_all=obs_all.unsqueeze(-1)# (batchsize,seq_len, channels)

            self.func.set_expert_weights_from_observation(obs_all)


        X0 = X.evaluate(X.interval[0])      # (batch, channels)
        # print(X0.shape)#torch.Size([1, 1])

        z0 = self.initial(X0)               # (batch, hidden)


        z_T = torchcde.cdeint(X=X, z0=z0, func=self.func, t=t_span)  # (len(t), batch, hidden) by default

        pred = self.readout(z_T)            # (len(t), batch, 1)

        return pred#.squeeze(-1)             # (len(t), batch)


class NeuralCDE_Continues(nn.Module):
    def __init__(self, input_channels, hidden_channels, interpolation="linear",
                 pretrained_encoder=None, pretrained_decoder=None, moe_activate=False, args=None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.interpolation = interpolation

        self.moe_activate = moe_activate
        self.args = args


        self.inp_dim = self.args.inp_dim
        self.hidden_dim = self.args.hidden_dim
        self.batch_norm = self.args.batch_norm
        self.num_layers = self.args.num_layers


        self.func = MoECDEFunc(input_channels, hidden_channels, args) if moe_activate \
                    else CDEFunc(input_channels, hidden_channels)


        if not self.args.use_channel_wise_autoencoder:
            # print(333)
            pretrained_encoder=None
            pretrained_decoder=None
        self.initial = pretrained_encoder if pretrained_encoder is not None \
                       else nn.Linear(input_channels, hidden_channels)
        print(self.initial)
        # time.sleep(500)
        self.readout = pretrained_decoder if pretrained_decoder is not None \
                       else nn.Linear(hidden_channels, input_channels)
        self.activation_fn = torch.sigmoid
    def freeze_encoder_decoder(self):
        for p in self.initial.parameters():
            p.requires_grad = False
        for p in self.readout.parameters():
            p.requires_grad = False
        print("Encoder and decoder layers frozen!")

    def expand_time_steps(self,n):
        # 原始时间步
        original_steps = list(range(n))

        # 新时间步列表
        new_steps = []

        for i in range(n - 1):
            new_steps.append(original_steps[i])
            new_steps.append((original_steps[i] + original_steps[i + 1]) / 2)

        # 添加最后一个时间步
        new_steps.append(original_steps[-1])

        return new_steps
    def forward(self, t_span,coeffs,moe_weights=[],mask=None):


        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        # set MoE weights once per sequence
        if self.moe_activate and isinstance(self.func, MoECDEFunc):
            if mask!=None:
                pass
            else:
                if self.args.use_gen_moe:

                    self.func.set_expert_weights_from_observation(None, generated_moe_weights=moe_weights)

                else:
                    obs_all = X.evaluate(t_span)    # (batch, seq_len, channels)

                    if len(obs_all.shape)==2:
                        obs_all=obs_all.unsqueeze(-1)# (batchsize,seq_len, channels)
            if self.args.use_gen_moe:
                pass
            else:
                self.func.set_expert_weights_from_observation(obs_all)


        X0 = X.evaluate(X.interval[0])      # (batch, channels)
        # print(X0.shape)#torch.Size([1, 1])

        z0 = self.initial(X0)               # (batch, hidden)



        new_time = np.arange(0, self.args.seq_len-1+self.args.interval, self.args.interval)



        new_time=torch.tensor(new_time).to(z0.device)


        z_T = torchcde.cdeint(X=X, z0=z0, func=self.func, t=new_time)  # (len(t), batch, hidden) by default

        pred = self.readout(z_T)            # (len(t), batch, 1)

        return pred.squeeze(-1)             # (len(t), batch)
