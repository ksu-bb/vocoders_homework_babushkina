import os
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from src.datasets.ruslan import RUSLANVocoderDataset
from src.datasets.collate import vocoder_collate_fn

def test():
    config = OmegaConf.load('configs/mel_config.yaml')
    
    train_list = os.path.join(config['data']['filelist_path'], 'train.txt')
    
    if not os.path.exists(train_list):
        print('error!!!')
        return
    
    train_dataset = RUSLANVocoderDataset(train_list, config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=vocoder_collate_fn
    )
    
    batch = next(iter(train_loader))
    
    print(f'audio shape: {batch['audio'].shape}', f'audio: {batch['audio_lengths']}')
    print(f'mel shape: {batch['mel'].shape}', f'lengths: {batch['mel_lengths']}')
    print('Я обязательно стану Хокаге!')

if __name__ == '__main__':
    test()