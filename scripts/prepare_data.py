import os
import random
from pathlib import Path
from omegaconf import OmegaConf

def scan_audio_files(root_dir, extension='.wav'):
    files = []
    for path in Path(root_dir).rglob(f'*{extension}'):
        files.append(str(path))
    return files

def create_filelists(config, train_ratio=0.95, seed=42):
    random.seed(seed)
    
    all_files = scan_audio_files(config['data']['audio_path'])
    print(f'всего файлов: {len(all_files)}')
    
    if len(all_files) == 0:
        raise FileNotFoundError('errorrr!')
    
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    os.makedirs(config['data']['filelist_path'], exist_ok=True)
    
    with open(os.path.join(config['data']['filelist_path'], 'train.txt'), 
              'w', encoding='utf-8') as f:
        for path in train_files:
            f.write(f'{path}\n')
    
    with open(os.path.join(config['data']['filelist_path'], 'val.txt'), 
              'w', encoding='utf-8') as f:
        for path in val_files:
            f.write(f'{path}\n')
    
    print(f'train: {len(train_files)}, val: {len(val_files)}')

if __name__ == '__main__':
    config = OmegaConf.load('configs/mel_config.yaml')
    create_filelists(config)