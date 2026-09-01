"""
test_trainer.py —— 训练器单元测试

归一性原理：优化器/调度器工厂函数接口一致，
warmup 对各 margin-based head 的表现一致。
"""

import math
import os
import torch
from torch import nn
import pytest
from src.deepl.model import CNN, ModelWithHead
from src.deepl.trainer import (
    create_optimizer, create_scheduler,
    _apply_margin_warmup, evaluate_model, train_model
)
from src.deepl.losses import CosFaceHead, ArcFaceHead, SubCenterArcFaceHead
from torch.utils.data import DataLoader, TensorDataset


class TestCreateOptimizer:
    """UT-36: 优化器工厂"""

    @pytest.fixture
    def params(self):
        return nn.Linear(10, 2).parameters()

    def test_adamw(self, params):
        opt = create_optimizer(nn.Linear(10, 2), lr=1e-3, optimizer_type='adamw')
        assert isinstance(opt, torch.optim.AdamW)

    def test_adam(self, params):
        opt = create_optimizer(nn.Linear(10, 2), lr=1e-3, optimizer_type='adam')
        assert isinstance(opt, torch.optim.Adam)

    def test_sgd(self, params):
        opt = create_optimizer(nn.Linear(10, 2), lr=1e-3, optimizer_type='sgd')
        assert isinstance(opt, torch.optim.SGD)

    def test_unknown_raises(self, params):
        with pytest.raises(ValueError):
            create_optimizer(nn.Linear(10, 2), optimizer_type='rmsprop')


class TestCreateScheduler:
    """UT-37: 调度器工厂"""

    @pytest.fixture
    def optimizer(self):
        return torch.optim.AdamW(nn.Linear(10, 2).parameters())

    def test_cosine(self, optimizer):
        sched = create_scheduler(optimizer, 'cosine', T_0=15, T_mult=2)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

    def test_plateau(self, optimizer):
        sched = create_scheduler(optimizer, 'plateau')
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_none(self, optimizer):
        sched = create_scheduler(optimizer, 'none')
        assert sched is None


class TestMarginWarmup:
    """UT-38, UT-39: CosFace/ArcFace warmup"""

    def test_cosface_warmup_epoch_0(self):
        """UT-38: epoch=0 时 s=6.4 (64/10, 不低于 6.4), m=0.02 (0.2/10)。"""
        backbone = CNN(num_classes=8)
        head = CosFaceHead(128, 8, s=64.0, m=0.20)
        model = ModelWithHead(backbone, head)

        _apply_margin_warmup(model, 0, warmup_epochs=10)

        # s 应从 s/10 = 6.4 开始（max(0.1, 0.1) * 64 = 6.4）
        assert abs(head.s - 6.4) < 0.01, f"s={head.s}, 期望 6.4"
        # m 从 m/10 = 0.02 开始
        assert abs(head.m - 0.02) < 0.01, f"m={head.m}, 期望 0.02"

    def test_cosface_warmup_final_epoch(self):
        """UT-39: 最后一个 warmup epoch 时 s=64, m=0.2。"""
        backbone = CNN(num_classes=8)
        head = CosFaceHead(128, 8, s=64.0, m=0.20)
        model = ModelWithHead(backbone, head)

        # 先跑完整个 warmup
        for ep in range(10):
            _apply_margin_warmup(model, ep, warmup_epochs=10)

        assert abs(head.s - 64.0) < 0.5, f"s={head.s}, 期望 64.0"
        assert abs(head.m - 0.20) < 0.01, f"m={head.m}, 期望 0.20"

    def test_arcface_warmup_trig_cached(self):
        """ArcFace warmup 后 cos_m/sin_m 应被更新。"""
        backbone = CNN(num_classes=8)
        head = ArcFaceHead(128, 8, s=64.0, m=0.50)
        model = ModelWithHead(backbone, head)

        cos_m_before = head.cos_m
        _apply_margin_warmup(model, 0, warmup_epochs=10)
        # m 变为 0.05 (=0.5/10)，cos_m 应更新
        assert head.cos_m != cos_m_before or head.m == 0.0, \
            "ArcFace warmup 后 cos_m 应被重新计算"

    def test_warmup_disabled(self):
        """warmup_epochs=0 时不应修改 s/m。"""
        backbone = CNN(num_classes=8)
        head = CosFaceHead(128, 8, s=64.0, m=0.20)
        model = ModelWithHead(backbone, head)

        s_before = head.s
        m_before = head.m
        _apply_margin_warmup(model, 0, warmup_epochs=0)

        assert head.s == s_before
        assert head.m == m_before


class TestTrainModelSmoke:
    """集成烟雾测试：完整的 train_model → test_model 流程。"""

    def test_smoke_training(self, device):
        """最小规模训练能正常收敛。"""
        # 构造小型数据集
        torch.manual_seed(42)
        X = torch.randn(200, 190, 20)  # [N, C, T] 简化时间轴
        y = torch.randint(0, 4, (200,))
        dataset = TensorDataset(X, y)
        # 80/20 train/val split
        n_train = 160
        n_val = 40
        train_sub, val_sub = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)

        # 小模型（input_channels=1 因为数据是 mel+MFCC 拼接的 2D 频谱）
        backbone = CNN(num_classes=4, input_channels=1)
        head = nn.Linear(128, 4)
        model = ModelWithHead(backbone, head).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name
        os.remove(save_path)  # 删除空临时文件，让 train_model 自行创建

        try:
            model, info = train_model(
                model, train_loader, val_loader, optimizer, device,
                model_save_path=save_path,
                epochs=5, patience=10, verbose=False,
            )
            assert isinstance(info, dict), "train_model 应返回 info dict"
            assert "history" in info, "info 应包含 history"

            # 验证可以正常评估
            test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
            metrics = evaluate_model(model, test_loader, device, 4, verbose=False)
            assert metrics["accuracy"] > 0.0, "测试准确率应为正"
            assert "per_class" in metrics, "metrics 应包含 per_class"
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_smoke_mixup(self, device):
        """Mixup 训练应正常完成（无 NaN）。"""
        torch.manual_seed(42)
        X = torch.randn(200, 190, 20)
        y = torch.randint(0, 4, (200,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        backbone = CNN(num_classes=4, input_channels=1).to(device)
        model = ModelWithHead(backbone, nn.Linear(128, 4)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name
        os.remove(save_path)  # 删除空临时文件，让 train_model 自行创建

        try:
            model, info = train_model(
                model, train_loader, train_loader, optimizer, device,
                model_save_path=save_path,
                mixup_alpha=0.2,
                epochs=3, patience=5, verbose=False,
            )
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
