# HiFi-GAN Vocoder 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)

Реализация вокодера HiFi-GAN для синтеза речи. Модель обучена на датасете [RUSLAN](https://ruslan-corpus.github.io).


## Описание

HiFi-GAN  — это модель, преобразующая mel-спектрограммы в аудиосигналы. В данном проекте обучение модели осуществлялось на русскоязычном датасете RUSLAN для синтеза речи 

## Особенности

- Генератор: Multi-Receptive Field Fusion (MRF) с Residual Blocks
- Дискриминаторы: Multi-Period (MPD) + Multi-Scale (MSD)
- Функции потерь: Adversarial + Feature Matching + Mel Spectrogram
