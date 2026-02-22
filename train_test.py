import torch
from omegaconf import OmegaConf
from src.models.hifigan import get_generator, get_discriminators
from src.models.losses import HiFiGANLoss

def test_training_step():
    config = OmegaConf.load('configs/mel_config.yaml')
    device = 'cpu'
    
    generator = get_generator(config['generator']).to(device)
    mpd, msd = get_discriminators()
    mpd = mpd.to(device)
    msd = msd.to(device)
    loss_fn = HiFiGANLoss(config, device=device)
    
    mel = torch.randn(2, 80, 50).to(device)
    audio = torch.randn(2, 1, 12800).to(device)
    
    audio_hat = generator(mel)
    
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = mpd(audio, audio_hat)
    y_ds_rs, y_ds_gs, fmap_rs_s, fmap_gs_s = msd(audio, audio_hat)
    
    losses = loss_fn(
        audio, audio_hat,
        fmap_rs + fmap_rs_s, fmap_gs + fmap_gs_s,
        y_d_rs + y_ds_rs, y_d_gs + y_ds_gs
    )
    
    losses['loss_gen'].backward()
    
    print(f"✅ Training step test passed!")
    print(f"Loss: {losses['loss_gen'].item():.4f}")

if __name__ == '__main__':
    test_training_step()