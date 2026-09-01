import os
import numpy as np
import torch
import scipy.io as sio
from sklearn.model_selection import train_test_split
from src.dataset.processor import Processor, AugProcessor


def load_mat_signal(file_path):
    """读取 .mat 文件并返回信号数组（第二列）"""
    mat = sio.loadmat(file_path)
    if 'data' not in mat:
        raise ValueError(f"File {file_path} does not contain 'data' variable")
    return mat['data'][:, 1].astype(np.float64)


def read_rawdata(parent_dir):
    """遍历 parent_dir 下各故障类型子文件夹，读取所有 .mat 文件。

    目录结构要求:
        parent_dir/
        ├── BR/        # 故障类型名作为标签
        │   ├── *.mat
        │   └── ...
        ├── KA/
        ├── ...

    Returns:
        signals:   list of np.ndarray，每条信号
        labels:    list of int，每条信号对应的标签索引
        label_map: dict，{故障名: 索引}
    """
    if not os.path.exists(parent_dir):
        raise FileNotFoundError(f"文件夹不存在: {parent_dir}")

    # 获取所有子文件夹名，排序以保证标签映射稳定
    fault_types = sorted([
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ])

    if not fault_types:
        raise ValueError(f"未在 {parent_dir} 中找到任何子文件夹")

    label_map = {name: idx for idx, name in enumerate(fault_types)}

    signals = []
    labels = []

    for fault_name in fault_types:
        fault_dir = os.path.join(parent_dir, fault_name)
        mat_files = [f for f in os.listdir(fault_dir) if f.endswith('.mat')]

        for file_name in sorted(mat_files):
            file_path = os.path.join(fault_dir, file_name)
            try:
                signal = load_mat_signal(file_path)
                signals.append(signal)
                labels.append(label_map[fault_name])
            except Exception as e:
                print(f"警告: 跳过 {file_path} — {e}")

    return signals, labels, label_map


def slice_signal(signal, piece_sec, sr, overlap=0.5):
    """将长信号按指定秒数和重叠率切分成片段"""
    piece_len = int(piece_sec * sr)
    step = int(piece_len * (1 - overlap))
    segments = []
    for start in range(0, len(signal) - piece_len + 1, step):
        segments.append(signal[start:start + piece_len])
    return segments


def split_dataset(signals, labels, test_size=0.2, random_state=42):
    """按故障类型分层划分训练/测试集"""
    train_signals, test_signals, train_labels, test_labels = train_test_split(
        signals, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    datasets = {
        "train": (train_signals, train_labels),
        "test": (test_signals, test_labels),
    }
    return datasets


def build_features(datasets, sr, piece_sec, overlap, aug_pro_dict=None, label_names=None,
                   n_mels=64, n_mfcc=42, use_pcen=False, normalize=True):
    """对训练/测试集切分信号 → 提取特征（训练集额外做增强）。

    Args:
        datasets:  split_dataset 返回的 dict
        sr:        采样率
        piece_sec: 每段秒数
        overlap:   片段重叠率
        aug_pro_dict: AugProcessor 的增强概率配置（None 表示不做任何增强）
        label_names: label_map 的逆映射 {idx: name}，用于确定增强倍数
        normalize:  是否按通道做 z-score 归一化（统计量仅来自训练集）

    Returns:
        train_feats, train_labels, test_feats, test_labels: torch.Tensor
    """
    aug = AugProcessor(aug_pro_dict, n_mels=n_mels, n_mfcc=n_mfcc, use_pcen=use_pcen)
    aug_enabled = aug_pro_dict is not None

    all_features = {}
    all_labels = {}

    for phase, (raw_signals, raw_label_ids) in datasets.items():
        features = []
        labels_out = []

        for sig, label_id in zip(raw_signals, raw_label_ids):
            segments = slice_signal(sig, piece_sec, sr, overlap)

            for seg in segments:
                # 原始特征
                base_feat = aug.extract_features(seg, sr)
                features.append(base_feat)
                labels_out.append(label_id)

                # 训练阶段额外添加增强样本（仅在启用增强时）
                if phase == "train" and aug_enabled:
                    label_name = label_names[label_id] if label_names else None
                    times = _get_aug_times(label_name)
                    for _ in range(times):
                        aug_feat = aug.process(seg, sr)
                        features.append(aug_feat)
                        labels_out.append(label_id)

        all_features[phase] = torch.stack(features)
        all_labels[phase] = torch.tensor(labels_out, dtype=torch.long)

    train_feats, test_feats = all_features["train"], all_features["test"]

    if normalize:
        # 按通道归一化：mean/std 在 [样本, 时间帧] 上统计，形状 [1, C, 1]。
        # 只用训练集统计量，避免测试集信息泄露。
        eps = 1e-6
        mean = train_feats.mean(dim=(0, 2), keepdim=True)
        std = train_feats.std(dim=(0, 2), unbiased=False, keepdim=True).clamp_min(eps)
        train_feats = (train_feats - mean) / std
        test_feats = (test_feats - mean) / std

    return (train_feats, all_labels["train"],
            test_feats, all_labels["test"])


def _get_aug_times(label_name):
    """根据故障类型返回每段的增强份数"""
    if label_name in ('VU', 'SW'):
        return 2
    elif label_name in ('RM', 'RU', 'KA'):
        return 2
    elif label_name in ('BR',):
        return 2
    else:
        return 1


def build_all_features(signals, labels, sr, piece_sec, overlap, aug_pro_dict=None,
                       label_names=None, n_mels=64, n_mfcc=42, use_pcen=False,
                       normalize=False):
    """构建全量特征集（不做 train/test 划分），返回 features, labels, signal_ids。

    Args:
        signals:      list of np.ndarray，全部信号
        labels:       list of int，每条信号的标签
        sr:           采样率
        piece_sec:    每段秒数
        overlap:      片段重叠率
        aug_pro_dict: 增强概率配置（None 表示不做任何增强）
        label_names:  label_map 逆映射 {idx: name}
        normalize:    是否按通道 z-score（默认 False，因为交叉验证应在每折训练集上
                      单独计算统计量，避免验证折信息泄露；全量归一化仅供非 CV 场景）

    Returns:
        features:    torch.Tensor [N, C, T]
        labels:      torch.Tensor [N]
        signal_ids:  torch.Tensor [N]，每个样本来自哪条信号（供交叉验证按信号分组，
                     防止同一信号的相邻片段同时出现在训练/验证折）
    """
    aug = AugProcessor(aug_pro_dict, n_mels=n_mels, n_mfcc=n_mfcc, use_pcen=use_pcen)
    aug_enabled = aug_pro_dict is not None

    features, labels_out, signal_ids = [], [], []
    for sig_idx, (sig, label_id) in enumerate(zip(signals, labels)):
        segments = slice_signal(sig, piece_sec, sr, overlap)
        for seg in segments:
            base_feat = aug.extract_features(seg, sr)
            features.append(base_feat)
            labels_out.append(label_id)
            signal_ids.append(sig_idx)

            if aug_enabled:
                label_name = label_names[label_id] if label_names else None
                times = _get_aug_times(label_name)
                for _ in range(times):
                    aug_feat = aug.process(seg, sr)
                    features.append(aug_feat)
                    labels_out.append(label_id)
                    signal_ids.append(sig_idx)

    feats = torch.stack(features)
    labs = torch.tensor(labels_out, dtype=torch.long)
    sids = torch.tensor(signal_ids, dtype=torch.long)

    if normalize:
        eps = 1e-6
        mean = feats.mean(dim=(0, 2), keepdim=True)
        std = feats.std(dim=(0, 2), unbiased=False, keepdim=True).clamp_min(eps)
        feats = (feats - mean) / std

    return feats, labs, sids


def save_dataset(features, labels, output_dir, dataset_version, phase):
    """保存特征和标签到 .pt 文件"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{phase}_v{dataset_version}.pt")
    torch.save((features, labels), path)
    return path


def print_dataset_info(data_dir, dataset_version, slicelen=0):
    for phase in ["train", "test"]:
        path = os.path.join(data_dir, f"{phase}_v{dataset_version}.pt")
        if os.path.exists(path):
            data = torch.load(path, weights_only=False)
            features, labels = data[0], data[1]
            feat_shape = features.shape if features is not None else "empty"
            print(f"{phase.capitalize()}: {len(features)} samples, "
                  f"feature shape {feat_shape}, "
                  f"labels: {len(torch.unique(labels))} classes")
        else:
            print(f"{phase.capitalize()}: file not found ({path})")
