import torch

def vocoder_collate_fn(batch):
    max_audio_len = max(item['audio'].shape[0] for item in batch)
    max_mel_len = max(item['mel'].shape[1] for item in batch)
    
    batch_size = len(batch)
    n_mels = batch[0]['mel'].shape[0]
    
    audio_batch = torch.zeros(batch_size, max_audio_len)
    mel_batch = torch.full(
        (batch_size, n_mels, max_mel_len), 
        fill_value=-11.5129251
    )
    
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    mel_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        audio_len = item['audio'].shape[0]
        mel_len = item['mel'].shape[1]
        
        audio_batch[i, :audio_len] = item['audio']
        mel_batch[i, :, :mel_len] = item['mel']
        
        audio_lengths[i] = audio_len
        mel_lengths[i] = mel_len
    
    return {
        'audio': audio_batch,
        'mel': mel_batch,
        'audio_lengths': audio_lengths,
        'mel_lengths': mel_lengths
    }