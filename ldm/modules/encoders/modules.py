# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time

import torch
import torch.nn as nn
from einops import repeat
import copy
       
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class ResBlockTime(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlockTime, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv1d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm1d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class DomainUnifiedEncoder(nn.Module):
    '''
    The input are encoded into two parts, invariant part and specific part. The specific part is generated attending to a random initialized latent vector pool.
    The length of the two part are equal in this implementation.
    '''
    def __init__(self, dim, window, num_channels=3, latent_dim=32, bn=True, **kwargs):
        super().__init__()
        dim_out = latent_dim
        flatten_dim = int(dim * window / 4)
        self.in_encoder = nn.Sequential(
            nn.Conv1d(num_channels, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
            )

        self.out_encoder = nn.Sequential(
            ResBlockTime(dim, dim, bn=bn),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            ResBlockTime(dim, dim, bn=bn),
            View((-1, flatten_dim)),                  # batch_size x 2048
            nn.Linear(flatten_dim, dim_out)
        )
            
    def forward(self, x):
        h = self.in_encoder(x)
        mask = None

        out = self.out_encoder(h)[:,None]   # b, 1, d
        return out, mask

class DomainUnifiedPrototyper(nn.Module):
    '''
    The input are encoded into two parts, invariant part and specific part.
    The specific part is generated attending to a random initialized latent vector pool.
    The length of the two part are equal in this implementation.
    '''
    def __init__(self, dim, window, num_latents=16, num_channels=3, latent_dim=32, bn=True, **kwargs):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        flatten_dim = dim * (window // 4)
        self.num_experts=4
        flatten_dim=self.num_experts*dim
        self.encoder = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.latents = nn.Parameter(torch.empty(num_latents, self.latent_dim), requires_grad=False)
        nn.init.orthogonal_(self.latents)
        self.init_latents = copy.deepcopy(self.latents.detach())
        self.mask_ffn = nn.Sequential(
            ResBlockTime(self.num_experts, self.num_experts, bn=bn),
            View((-1, flatten_dim)),                  # batch_size x 2048
            nn.Linear(flatten_dim, self.num_experts),
        )
        self.sigmoid = nn.Sigmoid()

        self.dims=[5,6,28,14,1]

        self.medical_datasets=['AtrialFibrillation','ECG200','ECG5000','TwoLeadECG','ECGFiveDays']
    def forward(self, x):
        # print(x.shape)
        # time.sleep(500)
        dim1=x.shape[1]
        dim2=x.shape[2]
        if dim1>dim2:
            x = x.transpose(1, 2)
        b = x.shape[0]
        # if x.shape[1]<=36:
        # x=x.transpose(1,2)
        # print(x.shape)
        # time.sleep(500)
        # if x.shape[1] not in self.dims:
        #     print('***********problem**************')
        #     time.sleep(500)
        # print(x.shape)#torch.Size([128, 1, 168])
        cond=x[:,0,-4:]
        cond=cond.unsqueeze(-1)
        h = self.encoder(cond)
        mask_logit = self.mask_ffn(h)
        # try:
        mask = mask_logit

        return h, mask

    # def forward(self, x):
    #     b = x.shape[0]
    #     # print(x.shape)#torch.Size([128, 1, 168])
    #     cond=x[:,0,-4:]
    #
    #     # try:
    #     h = self.encoder(cond)
    #     mask = None
    #
    #     return h, mask
        
