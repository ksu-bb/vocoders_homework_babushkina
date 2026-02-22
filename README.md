# HiFi-GAN Vocoder 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)

Реализация вокодера **HiFi-GAN** для синтеза речи. Модель обучена на датасете [RUSLAN](https://ruslan-corpus.github.io).


## Описание

HiFi-GAN  — это модель вокодера, преобразующая mel-спектрограммы в аудиосигналы. Данная реализация обучена на русскоязычном датасете RUSLAN для синтеза речи с частотой дискретизации 22050 Гц

### Особенности

- **Генератор**: Multi-Receptive Field Fusion (MRF) с Residual Blocks
- **Дискриминаторы**: Multi-Period (MPD) + Multi-Scale (MSD)
- **Функции потерь**: Adversarial + Feature Matching + Mel Spectrogram


### Гиперпараметры

| Sample Rate | 22050 Hz |
| n_fft | 1024 |
| hop_length | 256 |
| win_length | 1024 |
| n_mels | 80 |
| f_min | 0 Hz |
| f_max | 11025 Hz |
| Initial Channels | 512 |
| Learning Rate | 0.0002 |
| Batch Size | 4 (effective: 16 с gradient accumulation) |

## Установки

# Клонирование репозитория
git clone https://github.com/ksu-bb/vocoders_homework_babushkina.git
cd vocoders_homework_babushkina

# Создание виртуального окружения (рекомендуется)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt

torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
librosa>=0.10.0
soundfile>=0.12.0
audioread>=3.0.0
scipy>=1.10.0
hydra-core>=1.3.0
omegaconf>=2.3.0
wandb>=0.15.0
tqdm>=4.65.0
tensorboard>=2.12.0


Полный пайплайн доступен в демо-ноутбуке

# Структура репозитория

vocoders_homework_babushkina/
├── configs/
│   ├── mel_config.yaml          # конфигурация модели
│   └── default.yaml
├── data/
│   ├── RUSLAN/                  # датасет
│   └── filelists/               # Train/val списки
├── notebooks/
│   ├── demo_inference.ipynb     # демо для Colab
│   └── training_for_colab.ipynb # обучение в Colab
├── scripts/
│   ├── prepare_data.py          # подготовка данных
│   └── 
├── src/
│   ├── datasets/
│   │   ├── ruslan.py            # работа с RUSLAN 
│   │   ├── custom_dir.py        # Custom датасет
│   │   └── collate.py           # Collate функции
│   ├── models/
│   │   ├── hifigan.py           # архитектура моделм
│   │   └── losses.py            # функции потерь
│   ├── utils/
│   │   └── mel_extraction.py    # Mel-спектрограммы
│   ├── __init__.py
│   ├── train.py                 # скрипт обучения
│   └── synthesize.py            # скрипт синтеза
├── checkpoints/                 # сохранённые модели
├── synthesized/                 # синтезированные аудио
├── .gitignore
├── requirements.txt
├── README.md
└── report.md