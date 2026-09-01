import os
import random
import numpy as np
import torch
import librosa
import scipy.io as sio
from sklearn.model_selection import train_test_split
import click

class Processor:
    """特征提取基类：提供 mel 频谱 + MFCC + delta 特征提取，子类可覆盖 process() 以自定义处理管线。"""

    def __init__(self, n_mels=64, n_mfcc=42, use_pcen=False, k=1):
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.use_pcen = use_pcen
        self.k = k 

    def __call__(self, signal, sr=42000):
        return self.process(signal, sr)

    def _extract_mel(self, signal, sr, n_mels):
        """提取预加重后的 mel 频谱（PCEN 或 dB 归一化）"""
        mel_spec = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_fft=4096 * self.k, hop_length=1024,
            win_length=4096 * self.k, window='hamming', n_mels=n_mels,
            fmin=1, fmax=sr / 2,
            power=1.0
        )
        if self.use_pcen:
            mel_spec = librosa.pcen(
                mel_spec / np.abs(np.max(mel_spec, axis=1, keepdims=True)) * 2 ** 31,
                sr=sr, hop_length=1024, gain=0.98, bias=2, power=0.5, time_constant=0.4
            )
        else:
            mel_spec = librosa.amplitude_to_db(mel_spec)
        return mel_spec

    def extract_features(self, signal, sr=42000):
        """提取 mel 频谱 + MFCC/Δ/Δ²，双分支模型用。"""
        signal = librosa.effects.preemphasis(signal, coef=0.97)
        # ── Mel 分支 ──
        mel_spec = self._extract_mel(signal, sr, n_mels=self.n_mels)

        # ── MFCC 分支 ──
        mfcc = librosa.feature.mfcc(
            y=signal, sr=sr, n_fft=4096 * self.k, hop_length=1024,
            win_length=4096 * self.k, window='hamming', n_mfcc=self.n_mfcc,
            fmin=1, fmax=sr / 2
        )

        delta_mfcc = librosa.feature.delta(mfcc, width=5)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=3)

        feature = np.concatenate([mel_spec, mfcc, delta_mfcc, delta2_mfcc], axis=0)
        return torch.tensor(feature, dtype=torch.float32)

    def process(self, signal, sr=42000):
        """默认处理：直接提取特征（无增强），子类可重写以添加增强管线"""
        return self.extract_features(signal, sr)

class AugProcessor(Processor):
    """数据增强处理器：在特征提取前/后随机应用信号级和频谱级增强。"""

    def __init__(self, aug_pro_dict=None, n_mels=64, n_mfcc=42, use_pcen=False):
        super().__init__(n_mels=n_mels, n_mfcc=n_mfcc, use_pcen=use_pcen)
        # 各项增强的触发概率（0~1 之间的浮点数）
        # aug_pro_dict=None 表示禁用所有增强；传入 dict 则按概率触发
        if aug_pro_dict is None:
            self.aug_pro_dict = {}       # 空字典 → 不做任何增强
        else:
            self.aug_pro_dict = aug_pro_dict

    # ---- 信号级增强 ----
    def add_noise(self, signal, snr_range=(35, 40)):
        """添加高斯噪声，SNR 范围为 snr_range（单位 dB）"""
        snr = np.random.uniform(*snr_range)
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
        return signal + noise

    def signal_translate(self, signal, shift_range=(-0.05, 0.05)):
        shift = int(np.random.uniform(*shift_range) * len(signal))
        return np.roll(signal, shift)

    def time_stretch(self, signal, rate_range=(0.95, 1.05)):
        rate = np.random.uniform(*rate_range)
        old_len = len(signal)
        new_signal = librosa.effects.time_stretch(signal, rate=rate)
        if len(new_signal) > old_len:
            return new_signal[:old_len]
        else:
            out = np.zeros(old_len, dtype=signal.dtype)
            out[:len(new_signal)] = new_signal
            return out

    def amplitude_scale(self, signal, scale_range=(0.95, 1.05)):
        scale = np.random.uniform(*scale_range)
        return signal * scale

    def pitch_shift(self, signal, sr=42000, n_steps_range=(-2, 2)):
        n_steps = np.random.uniform(*n_steps_range)
        return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)

    # ---- 主处理管线（覆盖父类） ----
    def process(self, signal, sr=42000, is_train=True):
        """对信号应用随机数据增强并提取特征"""

        # 非训练集或未配置任何增强 → 直接提取特征
        if not is_train or not self.aug_pro_dict:
            return self.extract_features(signal, sr)


        # 信号级增强方法调度表
        signal_augs = {
            'noise': self.add_noise,
            'stretch': self.time_stretch,
            'scale': self.amplitude_scale,
            'pitch': self.pitch_shift,
            'translate': self.signal_translate
        }

        # 按概率触发信号级增强
        for name, aug_fn in signal_augs.items():
            if random.random() < self.aug_pro_dict.get(name, 0):
                signal = aug_fn(signal)


        # 提取特征（调用父类方法）
        feature = self.extract_features(signal, sr)

        return feature
