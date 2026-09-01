"""
test_feature.py —— 数据处理单元测试

归一性原理：数据读取、切分、增强、保存接口一致，
所有操作通过统一参数控制行为。
"""

import os
import tempfile
import numpy as np
import torch
import pytest
import scipy.io as sio
from src.dataset.feature import (
    load_mat_signal, read_rawdata, slice_signal,
    split_dataset, build_features, save_dataset, print_dataset_info
)


# ═══════════════════════════════════════════════════════════════
#  UT-11 ~ UT-14
# ═══════════════════════════════════════════════════════════════

class TestLoadMatSignal:
    """UT-11: .mat 文件读取"""

    def test_load_valid_mat(self):
        """创建临时 .mat 文件并验证读取。"""
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
            tmp_path = f.name
            data = np.random.randn(1000, 2)
            sio.savemat(tmp_path, {'data': data})

        try:
            signal = load_mat_signal(tmp_path)
            assert isinstance(signal, np.ndarray)
            assert len(signal) == 1000
            assert np.array_equal(signal, data[:, 1])
        finally:
            os.unlink(tmp_path)

    def test_missing_data_key(self):
        """缺少 'data' 键时应抛 ValueError。"""
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
            tmp_path = f.name
            sio.savemat(tmp_path, {'wrong_key': np.random.randn(10, 2)})

        try:
            with pytest.raises(ValueError, match="does not contain 'data'"):
                load_mat_signal(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestReadRawData:
    """UT-12: 目录遍历读取"""

    def test_read_from_data_dir(self):
        """从 data/ 目录读取真实数据（若 data/ 存在）。"""
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data'
        )
        if not os.path.isdir(data_dir):
            pytest.skip("data/ 目录不存在，跳过集成测试")

        signals, labels, label_map = read_rawdata(data_dir)
        assert len(signals) > 0
        assert len(labels) == len(signals)
        assert len(label_map) >= 2, f"至少应有 2 类故障, 实际 {len(label_map)}"

        # 标签应为连续整数 0 ~ n_classes-1
        unique_labels = sorted(set(labels))
        assert unique_labels == list(range(len(label_map)))


class TestSliceSignal:
    """UT-13: 信号切分"""

    @pytest.fixture
    def sr(self):
        return 42000

    def test_slice_count(self, sr):
        """UT-13: 片段数正确计算。"""
        # 2 秒信号, 1 秒片段, overlap=0 → 2 段
        signal = np.arange(2 * sr, dtype=np.float64)
        segments = slice_signal(signal, piece_sec=1.0, sr=sr, overlap=0.0)
        assert len(segments) == 2

    def test_slice_overlap(self, sr):
        """overlap=0.5 时片段数翻倍（近似）。"""
        signal = np.arange(2 * sr, dtype=np.float64)
        segments = slice_signal(signal, piece_sec=1.0, sr=sr, overlap=0.5)
        assert len(segments) >= 3, \
            f"overlap=0.5 时 2s/1s 应≥3 段, 实际 {len(segments)}"

    def test_each_segment_length(self, sr):
        """每段长度正确。"""
        signal = np.arange(2 * sr, dtype=np.float64)
        piece_len = int(1.0 * sr)
        segments = slice_signal(signal, piece_sec=1.0, sr=sr, overlap=0.0)
        for seg in segments:
            assert len(seg) == piece_len


class TestSplitDataset:
    """UT-14: 分层划分"""

    def test_stratify_preserves_ratios(self):
        """UT-14: 分层划分保持类别比例。"""
        signals = [np.random.randn(1000) for _ in range(200)]
        labels = [0] * 100 + [1] * 50 + [2] * 50

        datasets = split_dataset(signals, labels, test_size=0.2, random_state=42)
        train_sigs, train_labels = datasets["train"]
        test_sigs, test_labels = datasets["test"]

        # 检查各类比例
        for cls in [0, 1, 2]:
            train_ratio = train_labels.count(cls) / len(train_labels)
            test_ratio = test_labels.count(cls) / len(test_labels)
            assert abs(train_ratio - test_ratio) < 0.05, \
                f"类别 {cls}: train={train_ratio:.3f}, test={test_ratio:.3f}"


class TestBuildFeatures:
    """特征构建 (无增强)"""

    def test_no_aug(self):
        """--no_aug 时训练集样本数 = 测试集比例接近（不生成额外增强样本）。"""
        # 构造简单数据集
        sr = 42000
        piece_sec = 0.5  # 每段0.5秒，让片段数不同
        signals = [np.random.randn(2 * sr).astype(np.float64) for _ in range(16)]
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

        datasets = split_dataset(signals, labels, test_size=0.25, random_state=42)
        idx_to_name = {v: k for k, v in label_map.items()}

        # 不使用增强
        train_feats, train_labels, test_feats, test_labels = build_features(
            datasets, sr=sr, piece_sec=piece_sec, overlap=0.5,
            aug_pro_dict=None, label_names=idx_to_name,
            n_mels=32, n_mfcc=20, use_pcen=False
        )

        assert isinstance(train_feats, torch.Tensor)
        assert isinstance(test_feats, torch.Tensor)
        assert train_feats.shape[1] == 32 + 20 * 3  # mel + mfcc + Δ + Δ²
        assert len(train_labels) > 0
        assert len(test_labels) > 0
        # 无增强时训练样本数与测试样本数比例 ≈ 数据划分比例
        ratio = len(train_labels) / len(test_labels)
        assert 2.0 < ratio < 5.0, f"train/test 比例异常: {ratio:.1f}"

    def test_with_aug(self):
        """启用增强时训练样本数显著多于测试集（因为生成了增强副本）。"""
        sr = 42000
        piece_sec = 0.5
        signals = [np.random.randn(2 * sr).astype(np.float64) for _ in range(16)]
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        # label_map: {fault_name: idx}（模拟 read_rawdata 返回格式）
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        idx_to_name = {v: k for k, v in label_map.items()}  # {0: 'A', 1: 'B', ...}

        datasets = split_dataset(signals, labels, test_size=0.25, random_state=42)

        # 启用增强
        aug_pro_dict = {
            "noise": 0.7, "stretch": 0.5, "scale": 0.6,
            "pitch": 0.0, "specaug": 0.7, "flip": 0.7
        }
        train_feats_aug, train_labels_aug, test_feats, test_labels = build_features(
            datasets, sr=sr, piece_sec=piece_sec, overlap=0.5,
            aug_pro_dict=aug_pro_dict, label_names=idx_to_name,
            n_mels=32, n_mfcc=20, use_pcen=False
        )

        # 无增强版本作为对照
        train_feats_no, train_labels_no, _, _ = build_features(
            datasets, sr=sr, piece_sec=piece_sec, overlap=0.5,
            aug_pro_dict=None, label_names=idx_to_name,
            n_mels=32, n_mfcc=20, use_pcen=False
        )

        assert len(train_labels_aug) > len(train_labels_no), \
            "启用增强后训练样本数应增加"


class TestSaveDataset:
    """数据集保存与信息打印"""

    def test_save_and_load(self):
        features = torch.randn(100, 190, 50)
        labels = torch.randint(0, 4, (100,))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_dataset(features, labels, tmpdir, "1.0", "train")
            assert os.path.exists(path)

            loaded_feat, loaded_labels = torch.load(path, weights_only=False)
            assert torch.equal(loaded_feat, features)
            assert torch.equal(loaded_labels, labels)

    def test_print_dataset_info(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            features = torch.randn(100, 190, 50)
            labels = torch.randint(0, 4, (100,))
            save_dataset(features, labels, tmpdir, "1.0", "train")
            save_dataset(features, labels, tmpdir, "1.0", "test")

            print_dataset_info(tmpdir, "1.0")
            captured = capsys.readouterr()
            assert "Train:" in captured.out
            assert "Test:" in captured.out
            assert "100 samples" in captured.out
