import torch
from torch import nn
import torchaudio
import librosa

class MelSpectrogram(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config['sr'],
            win_length=config['win_length'],
            hop_length=config['hop_length'],
            n_fft=config['n_fft'],
            f_min=config['f_min'],
            f_max=config['f_max'],
            n_mels=config['n_mels']
        )
        
        self.mel_spectrogram.spectrogram.power = config['power']

        mel_basis = librosa.filters.mel(
            sr=config['sr'],
            n_fft=config['n_fft'],
            n_mels=config['n_mels'],
            fmin=config['f_min'],
            fmax=config['f_max']
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))
    
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        
        mel = self.mel_spectrogram(audio).clamp_(min=1e-5).log_()
        
        return mel