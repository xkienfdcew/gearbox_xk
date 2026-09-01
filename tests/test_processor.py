"""
test_processor.py —— 特征提取与数据增强单元测试

归一性原理：Processor 和 AugProcessor 共享统一接口 (signal, sr) → feature，
增强概率通过 aug_pro_dict 统一控制，行为可预测。
"""

import numpy as np
import torch
import pytest
from src.dataset.processor import Processor, AugProcessor


class TestProcessor:
    """UT-03, UT-04: 基础特征提取"""

    @pytest.fixture
    def processor(self):
        return Processor(n_mels=64, n_mfcc=42, use_pcen=False)

    def test_extract_features_shape(self, processor, sample_signal_1s):
        """UT-03: 默认参数下输出 shape = [190, T] (64 mel + 42 mfcc + 42 Δ + 42 Δ²)。"""
        feat = processor.extract_features(sample_signal_1s, sr=42000)
        assert isinstance(feat, torch.Tensor)
        assert feat.ndim == 2, f"期望 2D tensor, 得到 {feat.ndim}D"
        assert feat.shape[0] == 190, f"期望 190 通道 (64+42×3), 实际 {feat.shape[0]}"
        assert feat.shape[1] > 0, "时间维度应为正"
        assert feat.dtype == torch.float32

    def test_pcen_differs_from_db(self, sample_signal_1s):
        """UT-04: use_pcen=True 产生的特征应与 dB 归一化不同。"""
        proc_db = Processor(n_mels=64, n_mfcc=42, use_pcen=False)
        proc_pcen = Processor(n_mels=64, n_mfcc=42, use_pcen=True)

        feat_db = proc_db.extract_features(sample_signal_1s, sr=42000)
        feat_pcen = proc_pcen.extract_features(sample_signal_1s, sr=42000)

        # PCEN 和 dB 归一化应产生不同的值
        # （比较 mel 部分前 64 行，它们应不同）
        assert not torch.allclose(feat_db[:64], feat_pcen[:64],
                                  atol=0.01), "PCEN 应与 dB 产生不同的 mel 频谱"

    def test_call_method(self, processor, sample_signal_1s):
        """__call__ 应等同于 process()。"""
        by_call = processor(sample_signal_1s, sr=42000)
        by_process = processor.process(sample_signal_1s, sr=42000)
        assert torch.equal(by_call, by_process)


class TestAugProcessorSignalAugs:
    """UT-05 ~ UT-08: 信号级和频谱级增强"""

    @pytest.fixture
    def aug(self):
        return AugProcessor({
            "noise": 0.7,
            "stretch": 0.5,
            "scale": 0.6,
            "pitch": 0.0,
            "specaug": 0.7,
            "flip": 0.7,
        })

    def test_add_noise_preserves_length(self, aug, sample_signal_1s):
        """UT-05: add_noise 不改变信号长度。"""
        original_len = len(sample_signal_1s)
        noisy = aug.add_noise(sample_signal_1s)
        assert len(noisy) == original_len
        # 加噪后信号应不同
        assert not np.allclose(sample_signal_1s, noisy)

    def test_time_stretch_preserves_length(self, aug, sample_signal_1s):
        """UT-06: time_stretch 输出长度等于输入长度（截断/补零后）。"""
        original_len = len(sample_signal_1s)
        stretched = aug.time_stretch(sample_signal_1s, rate_range=(1.1, 1.1))
        assert len(stretched) == original_len, \
            f"期望 {original_len}, 实际 {len(stretched)}"

    def test_spec_augment_masking(self):
        """UT-07: spec_augment 确实将部分元素置零。"""
        aug = AugProcessor({"specaug": 1.0, "flip": 0.0})
        # 创建一个非零特征
        feature = torch.ones(64, 100)
        masked = aug.spec_augment(feature,
                                  num_frame_masks=5, num_freq_masks=5,
                                  max_frame_mask=20, max_freq_mask=30)
        # 应有部分元素被置零
        assert (masked == 0).any(), "specaug 应将部分元素置零"

    def test_flip_time(self, aug):
        """UT-08: flip_time 前后，首尾时间帧互换。"""
        feature = torch.arange(20, dtype=torch.float32).unsqueeze(0).expand(3, -1)
        flipped = aug.flip_time(feature)
        # 最后一列应与第一列交换
        assert torch.equal(feature[:, 0], flipped[:, -1]), "flip 后首列应=原末列"
        assert torch.equal(feature[:, -1], flipped[:, 0]), "flip 后末列应=原首列"


class TestAugProcessorPipeline:
    """UT-09, UT-10: 增强管线行为"""

    def test_process_notrain_returns_features(self, sample_signal_1s):
        """UT-09: is_train=False 时返回特征（不应抛出 AttributeError）。"""
        aug = AugProcessor({"noise": 0.7, "specaug": 0.7})
        feat = aug.process(sample_signal_1s, sr=42000, is_train=False)
        assert isinstance(feat, torch.Tensor)
        assert feat.ndim == 2

    def test_process_empty_aug_dict(self, sample_signal_1s):
        """空 aug_pro_dict 时 process 应直接返回特征，不做增强。"""
        aug = AugProcessor({})
        feature1 = aug.process(sample_signal_1s, sr=42000, is_train=True)
        feature2 = aug.extract_features(sample_signal_1s, sr=42000)
        # 无增强时 process 应等于 extract_features
        assert torch.equal(feature1, feature2), \
            "空概率字典时 process 应等同于 extract_features"

    def test_process_is_train_true(self, sample_signal_1s):
        """is_train=True 时，有概率的增强可能被应用（应无异常）。"""
        aug = AugProcessor({"noise": 1.0, "scale": 1.0, "specaug": 1.0, "flip": 1.0})
        feat = aug.process(sample_signal_1s, sr=42000, is_train=True)
        assert isinstance(feat, torch.Tensor)
        # 增强后应不同于原始特征
        feat_raw = aug.extract_features(sample_signal_1s, sr=42000)
        assert not torch.equal(feat, feat_raw), \
            "全概率增强后应与原始特征不同"

    def test_pitch_not_forced_when_prob_zero(self, sample_signal_1s):
        """UT-10: aug_pro_dict={'pitch': 0.0} 时 pitch 不应被强制应用。"""
        # 仅设 pitch=0.0，其他信号增强也设为 0，min_aug_num=0 避免强制补充
        aug = AugProcessor({"noise": 0.0, "stretch": 0.0, "scale": 0.0, "pitch": 0.0,
                           "specaug": 0.0, "flip": 0.0})
        feat = aug.process(sample_signal_1s, sr=42000, is_train=True, min_aug_num=0)
        feat_raw = aug.extract_features(sample_signal_1s, sr=42000)
        # 因为没有信号增强被应用，频谱增强也不会被强制应用
        assert torch.equal(feat, feat_raw), \
            "所有增强概率为 0 时，process 应等同于 extract_features"

    def test_aug_pro_dict_none(self, sample_signal_1s):
        """aug_pro_dict=None 时增强完全禁用。"""
        aug = AugProcessor(aug_pro_dict=None)
        feat = aug.process(sample_signal_1s, sr=42000, is_train=True)
        feat_raw = aug.extract_features(sample_signal_1s, sr=42000)
        assert torch.equal(feat, feat_raw), \
            "aug_pro_dict=None 时 process 应等同于 extract_features"
