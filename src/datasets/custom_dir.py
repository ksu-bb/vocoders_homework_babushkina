import os
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from src.utils.mel_extraction import MelSpectrogram
from omegaconf import OmegaConf


class CustomDirDataset(Dataset):
    def __init__(self, audio_dir: str, config: dict):
        self.audio_dir = Path(audio_dir)
        self.config = config

        self.audio_files = list(self.audio_dir.rglob('*.wav'))
        if len(self.audio_files) == 0:
            self.audio_files = list(Path(audio_dir).rglob('*.wav'))
        
        self.mel_extractor = MelSpectrogram(config['mel_spec'])
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio, sr = torchaudio.load(str(audio_path))
        target_sr = self.config['mel_spec']['sr']

        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)

        mel = self.mel_extractor(audio)
        
        audio = audio.mean(dim=0)  
        audio = audio / torch.max(torch.abs(audio))
        
        return {
            'mel': mel,
            'audio': audio,
            'path': str(audio_path),
            'filename': audio_path.stem
        }