import torch
from torch.utils.data import Dataset
import librosa
from torchaudio.transforms import Resample
import os
from src.utils.mel_extraction import MelSpectrogram

class RUSLANVocoderDataset(Dataset):
    def __init__(self, filelist_path: str, config: dict):
        self.config = config
        self.filepaths = []
        
        with open(filelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.filepaths.append(line.strip())
        
        self.mel_extractor = MelSpectrogram(config['mel_spec'])
        self.resampler = Resample(
            orig_freq=44100, 
            new_freq=config['mel_spec']['sr']
        )
    

    def __len__(self):
        return len(self.filepaths)
    
    
    def __getitem__(self, idx):
        audio_path = self.filepaths[idx]
        audio_np, sr = librosa.load(audio_path, sr=None)
        waveform = torch.from_numpy(audio_np).unsqueeze(0)  
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.config['mel_spec']['sr']:
            waveform = self.resampler(waveform)
 
        mel = self.mel_extractor(waveform)
        
        return {
            'audio': waveform.squeeze(0),  
            'mel': mel.squeeze(0),          
            'filepath': audio_path
        }