import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.mel_extraction import MelSpectrogram


class HiFiGANLoss(nn.Module):
    def __init__(self, config: dict, device: str = 'cuda'):
        super().__init__()
        self.config = config
        self.device = device
        self.lambda_fm = config.get('lambda_fm', 2.0)
        self.lambda_mel = config.get('lambda_mel', 45.0)
        self.mel_extractor = MelSpectrogram(config['mel_spec']).to(device)

    def feature_matching_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                min_len = min(rl.shape[-1], gl.shape[-1])
                rl = rl[..., :min_len]
                gl = gl[..., :min_len]
                loss += torch.mean(torch.abs(rl - gl))
        return loss

    def mel_spectrogram_loss(self, y, y_hat):
        y_mel = self.mel_extractor(y.squeeze(1))
        y_hat_mel = self.mel_extractor(y_hat.squeeze(1))
        min_len = min(y_mel.shape[-1], y_hat_mel.shape[-1])
        y_mel = y_mel[:, :, :min_len]
        y_hat_mel = y_hat_mel[:, :, :min_len]
        return F.l1_loss(y_mel, y_hat_mel)

    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            for r, g in zip(dr, dg):
                min_len = min(r.shape[-1], g.shape[-1])
                r = r[..., :min_len]
                g = g[..., :min_len]
                r_loss = torch.mean((1 - r) ** 2)
                g_loss = torch.mean(g ** 2)
                loss += (r_loss + g_loss)
        return loss

    def generator_loss(self, disc_outputs):
        loss = 0
        for dg in disc_outputs:
            for g in dg:
                loss += torch.mean((1 - g) ** 2)
        return loss

    def forward(self, y, y_hat, fmap_rs, fmap_gs, y_d_rs, y_d_gs):
        loss_disc = self.discriminator_loss(y_d_rs, y_d_gs)
        loss_gen_adv = self.generator_loss(y_d_gs)
        loss_gen_fm = self.feature_matching_loss(fmap_rs, fmap_gs)
        loss_gen_mel = self.mel_spectrogram_loss(y, y_hat)
        loss_gen = loss_gen_adv + self.lambda_fm * loss_gen_fm + self.lambda_mel * loss_gen_mel

        return {
            'loss_disc': loss_disc,  
            'loss_gen': loss_gen,
            'loss_gen_adv': loss_gen_adv,
            'loss_gen_fm': loss_gen_fm,
            'loss_gen_mel': loss_gen_mel
        }