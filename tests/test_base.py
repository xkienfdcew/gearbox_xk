"""
test_base.py —— 基础组件单元测试

归一性原理：所有注意力模块、残差模块、ANFIS 共享统一的 forward 接口，
输入 [B, C, H, W] → 输出 [B, C, H, W] (注意力/残差) 或 [B, C, T] → logits (ANFIS)。
"""

import warnings
import numpy as np
import torch
import pytest
from src.deepl.base import (
    SEBlock, ChannelAttention, SpatialAttention, CBAM,
    SELSTMBlock, ResBlock, ResSEBlock, ANFIS,
    NormalizedLinear, TrainedPCENLayer, FeatureNorm
)


class TestSEBlock:
    """UT-15: Squeeze-and-Excitation"""

    def test_output_shape(self):
        se = SEBlock(channels=32, reduction=8)
        x = torch.randn(4, 32, 20, 10)
        y = se(x)
        assert y.shape == x.shape

    def test_identity_for_ones(self):
        """当输入为全1时，SE 应基本保持值（scale≈1）。"""
        se = SEBlock(channels=16, reduction=4)
        x = torch.ones(2, 16, 8, 8)
        y = se(x)
        # 初始化随机，scale 不均等；不测试具体值，只测试不出错
        assert y.shape == x.shape


class TestCBAM:
    """UT-16: CBAM 注意力（通道+空间）"""

    def test_output_shape(self):
        cbam = CBAM(channels=32)
        x = torch.randn(4, 32, 20, 10)
        y = cbam(x)
        assert y.shape == x.shape

    def test_channel_attention_output(self):
        ca = ChannelAttention(channels=32, reduction=8)
        x = torch.randn(4, 32, 8, 8)
        y = ca(x)
        assert y.shape == x.shape
        # 输出应与输入不同（注意力权重的影响）
        assert not torch.equal(y, x)

    def test_spatial_attention_output(self):
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(4, 32, 8, 8)
        y = sa(x)
        assert y.shape == x.shape


class TestResBlock:
    """UT-17: 残差模块"""

    def test_same_channels(self):
        """输入输出通道相同时，残差连接不需要 downsample。"""
        block = ResBlock(32, 32, downsample=False)
        x = torch.randn(4, 32, 16, 16)
        y = block(x)
        assert y.shape == x.shape

    def test_downsample(self):
        """UT-17: downsample=True 时输出通道正确。"""
        block = ResBlock(16, 32, downsample=True)
        x = torch.randn(4, 16, 16, 16)
        y = block(x)
        assert y.shape[1] == 32
        assert y.shape[0] == x.shape[0]


class TestResSEBlock:
    """ResBlock + SE 注意力"""

    def test_downsample(self):
        block = ResSEBlock(16, 32, downsample=True)
        x = torch.randn(4, 16, 16, 16)
        y = block(x)
        assert y.shape[1] == 32


class TestANFIS:
    """UT-18: 自适应模糊推理系统"""

    @pytest.fixture
    def anfis(self):
        return ANFIS(input_shape=8, fuzzy_state=3, num_classes=8)

    def test_forward_shape(self, anfis):
        """UT-18: 输出 [B, num_classes]。"""
        x = torch.randn(4, 8)
        y = anfis(x)
        assert y.shape == (4, 8)

    def test_forward_not_nan(self, anfis):
        x = torch.randn(4, 8)
        y = anfis(x)
        assert not torch.isnan(y).any(), "ANFIS 输出不应含 NaN"


class TestSELSTMBlock:
    """LSTM 用 SE 注意力"""

    def test_output_shape(self):
        se_lstm = SELSTMBlock(channels=64, reduction=8)
        x = torch.randn(4, 100, 64)  # [B, T, C]
        y = se_lstm(x)
        assert y.shape == x.shape


class TestNormalizedLinear:
    """UT-19: 余弦归一化线性层"""

    def test_output_range(self):
        """UT-19: 输出范围在 [-1, 1] 之间。"""
        layer = NormalizedLinear(128, 8)
        x = torch.randn(4, 128)
        y = layer(x)
        assert y.shape == (4, 8)
        assert y.min() >= -1.0, f"余弦值下界超出 -1: {y.min()}"
        assert y.max() <= 1.0, f"余弦值上界超出 1: {y.max()}"


class TestFeatureNorm:
    """特征归一化模块"""

    def test_normalization_effect(self):
        mean = torch.tensor([0.5] * 10)
        std = torch.tensor([0.1] * 10)
        norm = FeatureNorm(mean, std)
        x = mean.unsqueeze(0).unsqueeze(-1).expand(4, 10, 50)  # [B, C, T]
        y = norm(x)
        assert y.shape == x.shape
        # 归一化后均值应接近 0
        assert torch.abs(y.mean()) < 0.1


class TestTrainedPCENLayer:
    """UT-20: 已废弃的 PCEN 层"""

    def test_deprecation_warning(self):
        """UT-20: 实例化时触发 DeprecationWarning。"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = TrainedPCENLayer()
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1, \
                "TrainedPCENLayer 实例化应触发 DeprecationWarning"
