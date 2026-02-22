# test_model.py
import torch
from omegaconf import OmegaConf
from src.models.hifigan import get_generator, get_discriminators
from src.models.losses import HiFiGANLoss

def test_hifigan():
    config = OmegaConf.load('configs/mel_config.yaml')
    
    # Generator
    generator = get_generator(config['generator'])
    print(f"Generator params: {sum(p.numel() for p in generator.parameters()) / 1e6:.2f}M")
    
    # Discriminators
    mpd, msd = get_discriminators()
    mpd_params = sum(p.numel() for p in mpd.parameters()) / 1e6
    msd_params = sum(p.numel() for p in msd.parameters()) / 1e6
    print(f"MPD params: {mpd_params:.2f}M, MSD params: {msd_params:.2f}M")
    
    # Loss
    loss_fn = HiFiGANLoss(config, device='cpu')
    
    # Тестовый forward pass
    batch_size = 2
    mel_len = 50
    
    mel = torch.randn(batch_size, config['mel_spec']['n_mels'], mel_len)
    
    # Generator forward
    audio_hat = generator(mel)
    print(f"Input mel shape: {mel.shape}")
    print(f"Output audio shape: {audio_hat.shape}")
    
    # Проверка upsampling factor
    # Вычисляем из upsample_rates: 8*8*2*2 = 256
    upsample_factor = 1
    for rate in config['generator']['upsample_rates']:
        upsample_factor *= rate
    
    expected_len = mel_len * upsample_factor
    actual_len = audio_hat.shape[-1]
    print(f"Upsampling factor: {upsample_factor}")
    print(f"Expected audio length: {expected_len}, Actual: {actual_len}")
    
    assert expected_len == actual_len, f"Length mismatch! Expected {expected_len}, got {actual_len}"
    
    # Discriminators forward
    audio_real = torch.randn(batch_size, 1, actual_len)
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = mpd(audio_real, audio_hat)
    print(f"MPD outputs: {len(y_d_rs)} sub-discriminators")
    
    # Loss forward
    losses = loss_fn(audio_real, audio_hat, fmap_rs, fmap_gs, y_d_rs, y_d_gs)
    print(f"Loss dict keys: {list(losses.keys())}")
    print(f"Total generator loss: {losses['loss_gen'].item():.4f}")
    
    print("\n✅ HiFi-GAN architecture test passed!")

if __name__ == '__main__':
    test_hifigan()