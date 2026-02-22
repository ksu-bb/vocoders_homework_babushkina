import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from omegaconf import OmegaConf
import hydra
import wandb
from tqdm import tqdm
from pathlib import Path
import gc

from src.datasets.ruslan import RUSLANVocoderDataset
from src.datasets.collate import vocoder_collate_fn
from src.models.hifigan import get_generator, get_discriminators
from src.models.losses import HiFiGANLoss


def save_checkpoint(generator, mpd, msd, optimizer_g, optimizer_d, epoch, path):
    torch.save({
        'generator': generator.state_dict(),
        'mpd': mpd.state_dict(),
        'msd': msd.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_d': optimizer_d.state_dict(),
        'epoch': epoch
    }, path)


def load_checkpoint(path, generator, mpd, msd, optimizer_g, optimizer_d):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    generator.load_state_dict(checkpoint['generator'])
    mpd.load_state_dict(checkpoint['mpd'])
    msd.load_state_dict(checkpoint['msd'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d'])
    return checkpoint['epoch']


@hydra.main(version_base=None, config_path='../configs', config_name='mel_config')
def train(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wandb.init(
        project='ruslan-vocoder',
        name=config.get('experiment_name', 'hifigan_baseline'),
        config=OmegaConf.to_container(config, resolve=True)
    )
    
    train_list = os.path.join(config['data']['filelist_path'], 'train.txt')
    val_list = os.path.join(config['data']['filelist_path'], 'val.txt')
    train_dataset = RUSLANVocoderDataset(train_list, config)
    val_dataset = RUSLANVocoderDataset(val_list, config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=vocoder_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=vocoder_collate_fn,
        pin_memory=True
    )

    generator = get_generator(config['generator']).to(device)
    mpd, msd = get_discriminators()
    mpd = mpd.to(device)
    msd = msd.to(device)

    optimizer_g = AdamW(
        generator.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['training']['betas'],
        weight_decay=config['training']['weight_decay']
    )
    
    optimizer_d = AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=config['training']['learning_rate'],
        betas=config['training']['betas'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler_g = ExponentialLR(optimizer_g, gamma=0.999)
    scheduler_d = ExponentialLR(optimizer_d, gamma=0.999)
    loss_fn = HiFiGANLoss(config, device=device)
    checkpoint_dir = Path(config['training']['checkpoint_path'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    global_step = 0
    latest_checkpoint = checkpoint_dir / 'g_latest.pth'
    
    if latest_checkpoint.exists():
        print(f'Found existing checkpoint: {latest_checkpoint}')
        start_epoch = load_checkpoint(
            str(latest_checkpoint),
            generator, mpd, msd,
            optimizer_g, optimizer_d
        )
    else:
        print('no checkpoint found')
    
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    print(f'Gradient accumulation steps: {accumulation_steps}')
    print(f'Effective batch size: {config.data.batch_size * accumulation_steps}')

    for epoch in range(start_epoch, config['training']['epochs']):
        generator.train()
        mpd.train()
        msd.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            mel = batch['mel'].to(device)
            audio = batch['audio'].unsqueeze(1).to(device)

            optimizer_g.zero_grad()
            audio_hat = generator(mel)

            y_d_rs, y_d_gs, fmap_rs, fmap_gs = mpd(audio, audio_hat)
            y_ds_rs, y_ds_gs, fmap_rs_s, fmap_gs_s = msd(audio, audio_hat)
            y_d_rs_combined = y_d_rs + y_ds_rs
            y_d_gs_combined = y_d_gs + y_ds_gs
            fmap_rs_combined = fmap_rs + fmap_rs_s
            fmap_gs_combined = fmap_gs + fmap_gs_s

            losses = loss_fn(
                audio, audio_hat,
                fmap_rs_combined, fmap_gs_combined,
                y_d_rs_combined, y_d_gs_combined
            )

            loss_gen_scaled = losses['loss_gen'] / accumulation_steps
            loss_gen_scaled.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer_g.step()

            optimizer_d.zero_grad()
            audio_hat_det = audio_hat.detach()
            
            y_d_rs, y_d_gs, _, _ = mpd(audio, audio_hat_det)
            y_ds_rs, y_ds_gs, _, _ = msd(audio, audio_hat_det)
            y_d_rs_combined = y_d_rs + y_ds_rs
            y_d_gs_combined = y_d_gs + y_ds_gs
            
            loss_disc = loss_fn.discriminator_loss(y_d_rs_combined, y_d_gs_combined)
            loss_disc_scaled = loss_disc / accumulation_steps
            loss_disc_scaled.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer_d.step()
            
            global_step += 1

            if global_step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            if global_step % config['training']['log_every'] == 0:
                wandb.log({
                    'train/loss_disc': loss_disc.item(),
                    'train/loss_gen': losses['loss_gen'].item(),
                    'train/loss_gen_adv': losses['loss_gen_adv'].item(),
                    'train/loss_gen_fm': losses['loss_gen_fm'].item(),
                    'train/loss_gen_mel': losses['loss_gen_mel'].item(),
                    'train/learning_rate': optimizer_g.param_groups[0]['lr'],
                    'global_step': global_step
                })
                
                pbar.set_postfix({
                    'loss_g': f"{losses['loss_gen'].item():.4f}",
                    'loss_d': f"{loss_disc.item():.4f}"
                })
        

        scheduler_g.step()
        scheduler_d.step()

        torch.cuda.empty_cache()
        gc.collect()
  
        if (epoch + 1) % 5 == 0:
            generator.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    mel = batch['mel'].to(device)
                    audio = batch['audio'].unsqueeze(1).to(device)
                    
                    audio_hat = generator(mel)
                    
                    y_d_rs, y_d_gs, fmap_rs, fmap_gs = mpd(audio, audio_hat)
                    y_ds_rs, y_ds_gs, fmap_rs_s, fmap_gs_s = msd(audio, audio_hat)
                    
                    losses = loss_fn(
                        audio, audio_hat,
                        fmap_rs + fmap_rs_s, fmap_gs + fmap_gs_s,
                        y_d_rs + y_ds_rs, y_d_gs + y_ds_gs
                    )
                    val_loss += losses['loss_gen'].item()
            
            val_loss /= len(val_loader)
            wandb.log({'val/loss_gen': val_loss, 'epoch': epoch + 1})

            wandb.log({
                'audio/real': wandb.Audio(audio[0].cpu().numpy(), sample_rate=config['mel_spec']['sr']),
                'audio/generated': wandb.Audio(audio_hat[0].cpu().numpy(), sample_rate=config['mel_spec']['sr']),
                'epoch': epoch + 1
            })
            
            generator.train()

        checkpoint_path = checkpoint_dir / f'g_{epoch+1:06d}.pth'
        save_checkpoint(
            generator, mpd, msd,
            optimizer_g, optimizer_d,
            epoch + 1,
            checkpoint_path
        )
        print(f'checkpoint saved: {checkpoint_path}')
        
        latest_path = checkpoint_dir / 'g_latest.pth'
        save_checkpoint(
            generator, mpd, msd,
            optimizer_g, optimizer_d,
            epoch + 1,
            latest_path
        )

    final_path = checkpoint_dir / 'g_final.pth'
    save_checkpoint(
        generator, mpd, msd,
        optimizer_g, optimizer_d,
        config['training']['epochs'],
        final_path
    )
    
    wandb.finish()
    print('Done...')


if __name__ == '__main__':
    train()