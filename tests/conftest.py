"""
pytest 配置与共享 fixture。

归一性原理：所有测试使用统一的随机种子、统一的设备，
确保各模块在统一的测试环境中验证。
"""

import os
import sys
import pytest
import numpy as np
import torch

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import set_seed


@pytest.fixture(scope="session", autouse=True)
def global_seed():
    """全测试会话统一随机种子（可复现性）。"""
    set_seed(42)


@pytest.fixture(scope="session")
def device():
    """统一设备选择。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_signal_1s():
    """生成 1 秒 42kHz 的正弦测试信号。"""
    sr = 42000
    t = np.linspace(0, 1, sr, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float64)


@pytest.fixture
def sample_signal_2s():
    """生成 2 秒 42kHz 的正弦测试信号。"""
    sr = 42000
    t = np.linspace(0, 2, 2 * sr, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float64)


@pytest.fixture
def sample_2d_feature():
    """随机 2D 特征 [B=4, C=190, T=83]（模拟 mel+MFCC 特征）。"""
    return torch.randn(4, 190, 83)


@pytest.fixture
def sample_logits_labels():
    """随机 logits [B=4, C=8] 和整数标签。"""
    logits = torch.randn(4, 8)
    labels = torch.randint(0, 8, (4,))
    return logits, labels


@pytest.fixture
def num_classes():
    return 8
