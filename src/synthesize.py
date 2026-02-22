import os
import torch
import torchaudio
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import hydra

from src.models.hifigan import get_generator
from src.datasets.custom_dir import CustomDirDataset
from src.utils.mel_extraction import MelSpectrogram


def load_checkpoint(checkpoint_path, generator):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'generator' in checkpoint:
        generator.load_state_dict(checkpoint['generator'])
    else:
        generator.load_state_dict(checkpoint)
    
    return generator


def synthesize_audio(mel, generator, device):
    generator.eval()
    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)
        audio = generator(mel)
        audio = audio.squeeze().cpu()
    return audio


@hydra.main(version_base=None, config_path='../configs', config_name='mel_config')
def synthesize(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    generator = get_generator(config['generator']).to(device)
    checkpoint_path = config.get('checkpoint_path', './checkpoints/g_final.pth')
    generator = load_checkpoint(checkpoint_path, generator)
    generator.remove_weight_norm()
    generator.eval()

    output_dir = Path(config.get('output_dir', './synthesized'))
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = config.get('audio_dir', './test_mos')
    dataset = CustomDirDataset(audio_dir, config)

    for item in tqdm(dataset, desc='Synthesizing'):
        audio_hat = synthesize_audio(item['mel'], generator, device)

        output_path = output_dir / f"{item['filename']}_synthesized.wav"
        torchaudio.save(
            str(output_path),
            audio_hat.unsqueeze(0),
            sample_rate=config['mel_spec']['sr']
        )
        

if __name__ == '__main__':
    synthesize()