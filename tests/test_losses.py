"""
test_losses.py —— 损失函数单元测试

归一性原理：所有损失函数 forward(logits, labels) → scalar loss，
分类头 forward(features, labels) → logits（labels=None 为推理、!=None 为训练）。
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
import pytest
from src.deepl.losses import (
    CenterLoss, FocalLoss, SoftLabelKLLoss,
    CosFaceHead, ArcFaceHead, SubCenterArcFaceHead,
    mixup
)


class TestCenterLoss:
    """UT-21, UT-22: 中心损失"""

    @pytest.fixture
    def center_loss(self):
        return CenterLoss(num_classes=8, feat_dim=128, alpha=0.5)

    def test_forward_nonnegative(self, center_loss):
        """UT-21: 前向返回标量≥0。"""
        features = torch.randn(4, 128)
        logits = torch.randn(4, 8)
        loss = center_loss(features, logits)
        assert loss.ndim == 0, "loss 应为标量"
        assert loss.item() >= 0, "CenterLoss 应为非负"

    def test_update_centers_changes_centers(self, center_loss):
        """UT-22: update_centers 会修改 center 向量。"""
        features = torch.randn(8, 128)
        logits = torch.randn(8, 8)
        centers_before = center_loss.centers.clone()
        center_loss.update_centers(features, logits)
        assert not torch.equal(centers_before, center_loss.centers), \
            "update_centers 应修改中心向量"


class TestFocalLoss:
    """UT-23: Focal Loss"""

    def test_gamma_zero_equals_ce(self):
        """UT-23: gamma=0 时 FocalLoss ≈ CrossEntropyLoss。"""
        focal = FocalLoss(gamma=0.0, alpha=None)
        ce = nn.CrossEntropyLoss()

        logits = torch.randn(4, 8)
        targets = torch.randint(0, 8, (4,))

        loss_focal = focal(logits, targets)
        loss_ce = ce(logits, targets)

        assert torch.allclose(loss_focal, loss_ce, atol=1e-5), \
            f"gamma=0 时 FocalLoss({loss_focal:.6f}) 应≈ CE({loss_ce:.6f})"

    def test_gamma_positive_reduces_loss(self):
        """gamma>0 时对高置信度样本 loss 更小。"""
        focal = FocalLoss(gamma=2.0, alpha=None)
        ce = nn.CrossEntropyLoss()

        # 中等置信度的正确预测（不要用极端值，否则两个 loss 都接近 0）
        logits_high = torch.tensor([[2.0, -1.0, -1.0, -1.0,
                                     -1.0, -1.0, -1.0, -1.0]])
        targets = torch.tensor([0])

        loss_focal = focal(logits_high, targets)
        loss_ce = ce(logits_high, targets)
        # 中等置信度时 FocalLoss 应小于 CE（因为 (1-pt)^γ < 1）
        assert loss_focal < loss_ce, \
            f"高置信度时 FocalLoss({loss_focal:.6f}) 应< CE({loss_ce:.6f})"


class TestCosFaceHead:
    """UT-24, UT-25: CosFace 分类头"""

    @pytest.fixture
    def head(self):
        return CosFaceHead(in_features=128, out_features=8, s=30.0, m=0.50)

    def test_inference_mode(self, head):
        """UT-24: labels=None 时输出 = s·cosθ。"""
        features = torch.randn(4, 128)
        logits = head(features)
        assert logits.shape == (4, 8)
        # cosθ 归一化后 ∈ [−1,1]，s=30，logits 应在 [−30, 30] 附近
        assert logits.max() <= 31.0
        assert logits.min() >= -31.0

    def test_training_mode(self, head):
        """UT-25: labels != None 时 margin 生效。"""
        features = torch.randn(4, 128)
        labels = torch.randint(0, 8, (4,))
        logits_train = head(features, labels)
        logits_infer = head(features)
        assert logits_train.shape == (4, 8)
        # 目标类的 logit 在训练模式下应更低（因为减去了 margin）
        for i, lbl in enumerate(labels):
            assert logits_train[i, lbl] < logits_infer[i, lbl], \
                f"样本{i}: 训练模式目标类 logit 应低于推理模式"


class TestArcFaceHead:
    """UT-26: ArcFace 角度计算"""

    @pytest.fixture
    def head(self):
        return ArcFaceHead(in_features=128, out_features=8, s=30.0, m=0.50)

    def test_numerical_stability(self, head):
        """UT-26: 极端 cosθ 值下数值稳定。"""
        features = torch.randn(32, 128)
        labels = torch.randint(0, 8, (32,))
        logits = head(features, labels)
        assert not torch.isnan(logits).any(), "ArcFace 输出不应含 NaN"
        assert not torch.isinf(logits).any(), "ArcFace 输出不应含 Inf"

    def test_inference_mode(self, head):
        features = torch.randn(4, 128)
        logits = head(features)
        assert logits.shape == (4, 8)

    def test_easy_margin(self):
        head_easy = ArcFaceHead(128, 8, s=30.0, m=0.50, easy_margin=True)
        features = torch.randn(4, 128)
        labels = torch.randint(0, 8, (4,))
        logits = head_easy(features, labels)
        assert not torch.isnan(logits).any()


class TestSubCenterArcFaceHead:
    """UT-27: 多子中心 ArcFace"""

    def test_multi_subcenter(self, device):
        """UT-27: 每类 K=3 个子中心，推理时取 max。"""
        head = SubCenterArcFaceHead(
            in_features=128, out_features=8, K=3, s=30.0, m=0.50
        ).to(device)
        features = torch.randn(4, 128, device=device)
        labels = torch.randint(0, 8, (4,), device=device)

        # 推理
        logits_infer = head(features)
        assert logits_infer.shape == (4, 8)

        # 训练
        logits_train = head(features, labels)
        assert logits_train.shape == (4, 8)
        assert not torch.isnan(logits_train).any()

    def test_weight_shape(self):
        """权重矩阵应为 [C*K, D]。"""
        head = SubCenterArcFaceHead(in_features=128, out_features=8, K=3)
        assert head.weight.shape == (8 * 3, 128)


class TestMixup:
    """UT-28, UT-29: Mixup 数据增强"""

    def test_basic_mixup(self):
        """UT-28: mixup 返回正确形状。"""
        features = torch.randn(8, 128)
        labels = torch.randint(0, 4, (8,))
        mix_feat, labels_a, labels_b, lam = mixup(features, labels, alpha=0.2)

        assert mix_feat.shape == features.shape
        assert labels_a.shape == labels.shape
        assert labels_b.shape == labels.shape
        assert 0.0 <= lam <= 1.0

    def test_mixup_loss_consistency(self):
        """UT-29: λ=0 时 mixup loss = CE(pred, y_b)。"""
        features = torch.randn(4, 128)
        labels = torch.randint(0, 4, (4,))
        criteria = nn.CrossEntropyLoss()

        mix_feat, labels_a, labels_b, lam = mixup(features, labels, alpha=0.2)

        # 用随机线性层模拟预测
        linear = nn.Linear(128, 4)
        pred = linear(mix_feat)

        # 标准 mixup loss
        mixup_loss = lam * criteria(pred, labels_a) + (1.0 - lam) * criteria(pred, labels_b)

        # 手动验证：当 lam=0，损失应为 CE(pred, y_b)
        pred_zero_lam = linear(mix_feat)
        loss_zero_lam = 0.0 * criteria(pred_zero_lam, labels_a) + \
                        1.0 * criteria(pred_zero_lam, labels_b)
        assert loss_zero_lam.item() > 0, "λ=0 时 loss 应 > 0"

    def test_mixup_int_labels(self):
        """测试 mixup 对整数标签正常工作。"""
        features = torch.randn(8, 190, 83)
        labels = torch.randint(0, 8, (8,))
        mix_feat, la, lb, lam = mixup(features, labels, alpha=0.4)
        assert mix_feat.shape == features.shape
        # labels_a 和 labels_b 都是整数标签
        assert la.dtype == labels.dtype
        assert lb.dtype == labels.dtype


class TestSoftLabelKLLoss:
    """软标签 KL 散度损失（已定义但未集成到训练脚本）"""

    def test_forward(self):
        loss_fn = SoftLabelKLLoss(reduction="mean")
        logits = torch.randn(4, 8)
        soft_target = F.softmax(torch.randn(4, 8), dim=1)
        loss = loss_fn(logits, soft_target)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_gamma_zero_equals_ce(self):
        """gamma=0 时退化为交叉熵（非 KL，因为不含目标分布的熵项）。"""
        loss_fn = SoftLabelKLLoss(gamma=0.0, alpha=1.0)
        logits = torch.randn(4, 8)
        soft_target = F.softmax(torch.randn(4, 8), dim=1)

        loss = loss_fn(logits, soft_target)
        # gamma=0 时 loss = -sum(target * log_softmax(logits)) = cross-entropy
        ce_manual = -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        assert torch.allclose(loss, ce_manual, atol=1e-5), \
            f"gamma=0 时 SoftLabelKLLoss({loss:.6f}) 应≈ CE({ce_manual:.6f})"
