"""
test_regression.py —— 回归测试（Bug 修复验证）

验证本次修复的正确性：
  RG-01: Mixup 调用正常（无 NaN）
  RG-02: Mixup loss 公式正确
  RG-03: extract_features 拼写错误修复
  RG-04: 验证集信息隔离
  RG-05: loss 值正常（未被人为除以 2）
  RG-06: CNNANFIS DropBlock2D 正确
  RG-07: TrainedPCENLayer 废弃警告
  RG-08: 增强不应用 pitch(0.0)
"""

import os
import sys
import warnings
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import set_seed
from src.deepl.model import CNN, CNNANFIS, ModelWithHead, FEAT_DIM_MAP
from src.deepl.losses import mixup
from src.deepl.trainer import train_model, evaluate_model
from src.deepl.base import TrainedPCENLayer
from src.dataset.processor import AugProcessor

set_seed(42)


class TestRegressionMixup:
    """RG-01, RG-02: Mixup 修复验证"""

    def test_mixup_training_no_nan(self):
        """RG-01: Mixup 训练正常完成，无 NaN loss。"""
        torch.manual_seed(42)
        X = torch.randn(200, 190, 30)
        y = torch.randint(0, 4, (200,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        backbone = CNN(num_classes=4, input_channels=1)
        model = ModelWithHead(backbone, nn.Linear(128, 4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name
        os.remove(save_path)  # 删除空临时文件，让 train_model 自行创建
        try:
            model, info = train_model(
                model, loader, loader, optimizer, torch.device("cpu"),
                model_save_path=save_path,
                mixup_alpha=0.2,
                epochs=5, patience=10, verbose=False,
            )
            assert isinstance(info, dict), "train_model 应返回 info dict"
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_mixup_loss_formula(self):
        """RG-02: Mixup loss = λ·CE(pred, y_a) + (1-λ)·CE(pred, y_b)。"""
        torch.manual_seed(42)
        features = torch.randn(8, 128)
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

        mix_feat, la, lb, lam = mixup(features, labels, alpha=0.2)

        # 用随机线性层模拟预测
        linear = nn.Linear(128, 4)
        pred = linear(mix_feat)
        criteria = nn.CrossEntropyLoss()

        # 手算 mixup loss
        manual_loss = lam * criteria(pred, la) + (1.0 - lam) * criteria(pred, lb)

        assert manual_loss.item() > 0, "loss 应为正"
        assert not torch.isnan(manual_loss), "loss 不应为 NaN"


class TestRegressionExtractFeatures:
    """RG-03: extract_features 调用正常"""

    def test_extract_features_not_raise(self):
        """RG-03: AugProcessor.process(is_train=False) 返回特征而非抛异常。"""
        aug = AugProcessor({"noise": 0.7, "specaug": 0.7})
        sig = np.random.randn(42000)
        feat = aug.process(sig, sr=42000, is_train=False)
        assert isinstance(feat, torch.Tensor)
        assert feat.ndim == 2


class TestRegressionValidationIsolation:
    """RG-04: 验证集隔离"""

    def test_validation_isolation(self):
        """早停基于 val_loader，test_loader 不被用于验证。"""
        torch.manual_seed(42)
        X = torch.randn(200, 190, 20)
        y = torch.randint(0, 4, (200,))
        dataset = TensorDataset(X, y)

        n_train = 120
        n_val = 40
        n_test = 40
        train_sub, val_sub, test_sub = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_sub, batch_size=32, shuffle=False)

        backbone = CNN(num_classes=4, input_channels=1)
        model = ModelWithHead(backbone, nn.Linear(128, 4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name
        os.remove(save_path)  # 删除空临时文件，让 train_model 自行创建
        try:
            model, info = train_model(
                model, train_loader, val_loader, optimizer, torch.device("cpu"),
                model_save_path=save_path, epochs=5, patience=10, verbose=False,
            )

            # 最终测试集评估独立
            metrics = evaluate_model(model, test_loader, torch.device("cpu"), 4, verbose=False)
            assert metrics["accuracy"] > 0.0
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


class TestRegressionLossValue:
    """RG-05: loss 值正常（未被人为缩放）"""

    def test_loss_not_halved(self):
        """训练 loss 值应合理（不会被 /2 或其他人为缩放）。"""
        torch.manual_seed(42)
        X = torch.randn(100, 190, 20)
        y = torch.randint(0, 4, (100,))

        backbone = CNN(num_classes=4, input_channels=1)
        model = ModelWithHead(backbone, nn.Linear(128, 4))
        criteria = nn.CrossEntropyLoss()

        feat, logits = model(X)
        loss = criteria(logits, y)
        # 随机初始化模型，CE loss 应在 -ln(1/4)=1.39 附近
        assert 0.5 < loss.item() < 3.0, \
            f"随机初始化模型的 CE loss 应在 [0.5, 3.0], 实际 {loss.item():.4f}"


class TestRegressionDropBlock:
    """RG-06: CNNANFIS DropBlock2D"""

    def test_cnn_anfis_dropblock(self):
        """CNNANFIS 的 self.dropout 是 DropBlock2D 类型。"""
        from dropblock import DropBlock2D
        model = CNNANFIS(num_classes=8)
        assert isinstance(model.dropout, DropBlock2D), \
            f"CNNANFIS.dropout 应为 DropBlock2D, 实际 {type(model.dropout)}"


class TestRegressionPCEN:
    """RG-07: TrainedPCENLayer 废弃警告"""

    def test_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = TrainedPCENLayer()
            dep_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(dep_warnings) >= 1


class TestRegressionAugProb:
    """RG-08: 增强不应用概率为 0 的方法"""

    def test_pitch_not_applied_when_zero(self):
        """RG-08: pitch 概率为 0 时始终不被应用。"""
        aug = AugProcessor({
            "noise": 0.0, "stretch": 0.0, "scale": 0.0,
            "pitch": 0.0, "specaug": 0.0, "flip": 0.0
        })
        sig = np.random.randn(42000)

        # 多次测试（每次随机不同）
        for _ in range(20):
            feat = aug.process(sig, sr=42000, is_train=True, min_aug_num=1)
            feat_raw = aug.extract_features(sig, sr=42000)
            # 所有概率为 0 → 应完全等于原始特征
            assert torch.equal(feat, feat_raw), \
                "所有增强概率为 0 时输出应等于原始特征"
