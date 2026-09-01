import torch
from torch.utils.data import Dataset
import random
import numpy as np


class AugDataset(Dataset):
    """数据增强数据集"""

    def __init__(self, features, labels, min_aug_num=2, spec_aug_prob=0.5, flip_prob=0.5, noise_prob=0.5, cutout_prob=0.5):
        self.features = features
        self.labels = labels
        self.min_aug_num = min_aug_num
        self.spec_aug_prob = spec_aug_prob
        self.flip_prob = flip_prob
        self.noise_prob = noise_prob
        self.cutout_prob = cutout_prob

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        feature1 = self.transform(feature)
        # feature2 = self.transform(feature)

        # return (feature1, feature2), (label, label)  # 返回两个增强后的特征和对应的标签
        return feature1, label  # 返回增强后的特征和对应的标签


    def transform(self, feature):
        """对特征应用随机数据增强"""
        aug_num = 0 
        # if random.random() < self.noise_prob:
        #     feature = self.add_noise(feature)
        #     aug_num += 1
        if random.random() < self.spec_aug_prob:
            feature = self.spec_augment(feature)
            aug_num += 1
        # if random.random() < self.flip_prob:
            # feature = self.flip_time(feature)
            # aug_num += 1

        # if random.random() < self.cutout_prob:
        #     feature = self.cutout(feature)
        #     aug_num += 1

        # # 确保至少应用 min_aug_num 次增强
        # while aug_num < self.min_aug_num:
        #     feature = self.spec_augment(feature)
        #     aug_num += 1

        return feature

    # ---- 频谱级增强 ----
    def spec_augment(self, feature, num_frame_masks=2, num_freq_masks=2,
                     max_frame_mask=6, max_freq_mask=20):
        """频谱增强：随机遮盖部分时间帧和频率 bin"""
        feature = feature.clone()
        T, F = feature.shape[1], feature.shape[0]

        for _ in range(num_frame_masks):
            t = np.random.randint(0, max_frame_mask)
            t0 = np.random.randint(0, T - t + 1) if t > 0 else 0
            feature[:, t0:t0 + t] = 0

        for _ in range(num_freq_masks):
            f = np.random.randint(0, max_freq_mask)
            f0 = np.random.randint(0, F - f + 1) if f > 0 else 0
            feature[f0:f0 + f, :] = 0

        return feature

    def flip_time(self, feature):
        return torch.flip(feature, dims=[1])

    def add_noise(self, feature, noise_level=1.2, bias=-0.6):
        """添加高斯噪声"""
        noise = torch.randn_like(feature) * noise_level + bias
        return feature + noise

    def cutout(self, feature, num_cutouts=3, max_cutout_size=20):
        """随机遮挡部分频谱区域"""
        feature = feature.clone()
        T, F = feature.shape[1], feature.shape[0]

        for _ in range(num_cutouts):
            cutout_size = np.random.randint(1, max_cutout_size + 1)
            t0 = np.random.randint(0, T - cutout_size + 1)
            f0 = np.random.randint(0, F - cutout_size + 1)
            feature[f0:f0 + cutout_size, t0:t0 + cutout_size] = 0

        return feature

