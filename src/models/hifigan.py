import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import weight_norm, spectral_norm
from typing import List, Tuple



def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


# GENERATOR ----------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: List[int] = [1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, 
                              dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, 
                              dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in dilation
        ])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)


    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


    def remove_weight_norm(self):
        for l in self.convs1:
            weight_norm.remove(l)
        for l in self.convs2:
            weight_norm.remove(l)


class MRF(nn.Module):
    def __init__(self, channels: int, kernel_sizes: List[int], dilations: List[List[int]]):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResBlock(channels, k, d) for k, d in zip(kernel_sizes, dilations)
        ])


    def forward(self, x):
        return sum([rb(x) for rb in self.resblocks]) / len(self.resblocks)


    def remove_weight_norm(self):
        for rb in self.resblocks:
            rb.remove_weight_norm()


class Generator(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.num_kernels = len(config['resblock_kernel_sizes'])
        self.num_upsamples = len(config['upsample_rates'])

        self.conv_pre = weight_norm(
            Conv1d(config['n_mels'], config['upsample_initial_channel'], 7, 1, padding=3)
        )
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config['upsample_rates'], config['upsample_kernel_sizes'])):
            self.ups.append(weight_norm(
                ConvTranspose1d(
                    config['upsample_initial_channel'] // (2**i),
                    config['upsample_initial_channel'] // (2**(i+1)),
                    k, u, padding=(k-u)//2
                )
            ))
        
        self.mrfs = nn.ModuleList()
        for i in range(len(self.ups)):
            self.mrfs.append(MRF(
                config['upsample_initial_channel'] // (2**(i+1)),
                config['resblock_kernel_sizes'],
                config['resblock_dilation_sizes']
            ))
        
        self.conv_post = weight_norm(Conv1d(
            config['upsample_initial_channel'] // (2**(self.num_upsamples)), 
            1, 7, 1, padding=3
        ))
        
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            x = self.mrfs[i](x)
        
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)  
        
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            weight_norm.remove(l)
        for l in self.mrfs:
            l.remove_weight_norm()
        weight_norm.remove(self.conv_pre)
        weight_norm.remove(self.conv_post)


# DISCRIMINATOR-------------------------------------------------------------------

class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        
        norm_f = weight_norm
        
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape

        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        
        x = x.view(b, c, t // self.period, self.period)
        
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return fmap, x


    def remove_weight_norm(self):
        for l in self.convs:
            weight_norm.remove(l)
        weight_norm.remove(self.conv_post)


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return fmap, x


    def remove_weight_norm(self):
        for l in self.convs:
            weight_norm.remove(l)
        weight_norm.remove(self.conv_post)


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(period) for period in [2, 3, 5, 7, 11]
        ])


    def forward(self, y, y_hat):
        y_d_rs = []
        fmap_rs = []

        y_d_gs = []
        fmap_gs = []
        
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


    def remove_weight_norm(self):
        for d in self.discriminators:
            d.remove_weight_norm()


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),  
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])


    def forward(self, y, y_hat):
        y_d_rs = []
        fmap_rs = []

        y_d_gs = []
        fmap_gs = []
        
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    def remove_weight_norm(self):
        for d in self.discriminators:
            d.remove_weight_norm()




def get_generator(config: dict) -> Generator:
    return Generator(config)


def get_discriminators() -> Tuple[MultiPeriodDiscriminator, MultiScaleDiscriminator]:
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    return mpd, msd