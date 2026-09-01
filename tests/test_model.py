"""
test_model.py —— 模型架构单元测试

归一性原理：所有 backbone 的 forward 返回特征向量 [B, D]（含 ANFIS 的 (feature, logits)），
ModelWithHead 统一包装 backbone + head。
"""

import torch
from torch import nn
import pytest
from src.deepl.model import (
    CNN, CNN2, CNNSE, CNN2SE, CNN_CBAM, CNN2_CBAM, CNNANFIS,
    LSTM, LSTMSE, LSTMANFIS,
    ResNet, ResNetSE,
    DualBranchCNN, DualBranchCNNSE,
    MODEL_REGISTRY, FEAT_DIM_MAP, get_feat_dim,
    ModelWithHead, create_model
)
from src.deepl.losses import CosFaceHead, ArcFaceHead


# ── 所有非 ANFIS 模型的列表（排除双分支，它们不接受 input_channels） ──
CNN_MODELS = [CNN, CNN2, CNNSE, CNN2SE, CNN_CBAM, CNN2_CBAM,
              ResNet, ResNetSE]
DUAL_BRANCH_MODELS = [DualBranchCNN, DualBranchCNNSE]
LSTM_MODELS = [LSTM, LSTMSE]
ANFIS_MODELS = [CNNANFIS, LSTMANFIS]

INPUT_CHANNELS = 1
INPUT_SIZE = 190  # mel(64) + mfcc(42×3)
NUM_CLASSES = 8
BATCH_SIZE = 4


# ═══════════════════════════════════════════════════════════════
#  UT-30 ~ UT-33: 各模型 forward 输出验证
# ═══════════════════════════════════════════════════════════════

class TestCNNModels:
    """UT-30: CNN 系列 forward → [B, feat_dim]"""

    @pytest.mark.parametrize("model_cls", CNN_MODELS)
    def test_forward_shape(self, model_cls):
        """各 CNN backbone 的输出应为 [B, D] 特征向量。"""
        model = model_cls(num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS)
        x = torch.randn(BATCH_SIZE, 190, 83)  # [B, C, T] 2D输入
        out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 2, f"{model.name}: 期望 2D 输出, 得到 {out.ndim}D"
        assert out.shape[0] == BATCH_SIZE

    @pytest.mark.parametrize("model_cls", CNN_MODELS)
    def test_has_name(self, model_cls):
        """所有模型应有 name 属性。"""
        model = model_cls(num_classes=NUM_CLASSES)
        assert hasattr(model, 'name'), f"{model_cls.__name__} 缺少 name 属性"
        assert isinstance(model.name, str) and len(model.name) > 0

    @pytest.mark.parametrize("model_cls", DUAL_BRANCH_MODELS)
    def test_dual_branch_shape(self, model_cls):
        """双分支模型 forward 输出 [B, 2*feat_dim]。"""
        model = model_cls(num_classes=NUM_CLASSES)
        x = torch.randn(BATCH_SIZE, 190, 83)
        out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 2, f"{model.name}: 期望 2D, 得到 {out.ndim}D"
        assert out.shape[0] == BATCH_SIZE


class TestLSTMModels:
    """UT-31: LSTM 系列 forward → [B, feat_dim]"""

    @pytest.mark.parametrize("model_cls", LSTM_MODELS)
    def test_forward_shape(self, model_cls):
        model = model_cls(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
        x = torch.randn(BATCH_SIZE, INPUT_SIZE, 83)  # [B, C, T]
        out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 2, f"{model.name}: 期望 2D, 得到 {out.ndim}D"
        assert out.shape[0] == BATCH_SIZE

    @pytest.mark.parametrize("model_cls", LSTM_MODELS)
    def test_lstm_needs_input_size(self, model_cls):
        """LSTM 模型不提供 input_size 时应报错。"""
        with pytest.raises(ValueError):
            create_model(model_cls.__name__.lower(), num_classes=8)


class TestANFISModels:
    """UT-33: ANFIS 系列 forward → (feature, logits)"""

    @pytest.mark.parametrize("model_cls", ANFIS_MODELS)
    def test_forward_returns_tuple(self, model_cls):
        """ANFIS forward 应返回 (feature, logits) 元组。"""
        if model_cls == CNNANFIS:
            model = model_cls(num_classes=NUM_CLASSES)
            x = torch.randn(BATCH_SIZE, 190, 83)
        else:
            model = model_cls(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
            x = torch.randn(BATCH_SIZE, INPUT_SIZE, 83)

        out = model(x)
        assert isinstance(out, tuple), f"{model.name}: 应返回 tuple"
        assert len(out) == 2, f"{model.name}: tuple 应有 2 个元素"
        feature, logits = out
        assert feature.ndim == 2
        assert logits.ndim == 2
        assert logits.shape[1] == NUM_CLASSES


class TestDualBranchModels:
    """UT-32: 双分支 CNN"""

    def test_dual_cnn_shape(self):
        model = DualBranchCNN(num_classes=NUM_CLASSES)
        x = torch.randn(BATCH_SIZE, 190, 83)
        out = model(x)
        assert out.shape == (BATCH_SIZE, 128), \
            f"DualBranchCNN 输出应为 (4, 128), 实际 {out.shape}"

    def test_dual_cnn_se_shape(self):
        model = DualBranchCNNSE(num_classes=NUM_CLASSES)
        x = torch.randn(BATCH_SIZE, 190, 83)
        out = model(x)
        assert out.shape == (BATCH_SIZE, 128)


# ═══════════════════════════════════════════════════════════════
#  创建函数 & 注册表
# ═══════════════════════════════════════════════════════════════

class TestModelRegistry:
    """模型注册表完整性验证"""

    def test_all_models_registered(self):
        """所有已知模型名都在注册表中。"""
        expected = {
            'cnn', 'cnn2', 'cnn_se', 'cnn2_se',
            'cnn_cbam', 'cnn2_cbam', 'cnn_anfis',
            'lstm', 'lstm_se', 'lstm_anfis',
            'resnet', 'resnet_se',
            'dual_cnn', 'dual_cnn_se',
        }
        registered = set(MODEL_REGISTRY.keys())
        missing = expected - registered
        assert not missing, f"注册表缺少: {missing}"

    def test_create_model_valid(self):
        """create_model 对每个注册的模型名都应成功。"""
        for name in MODEL_REGISTRY:
            is_lstm = MODEL_REGISTRY[name][1]
            kwargs = {'num_classes': 8}
            if is_lstm:
                kwargs['input_size'] = 190
            try:
                model = create_model(name, **kwargs)
                assert model is not None, f"create_model('{name}') 返回 None"
            except Exception as e:
                pytest.fail(f"create_model('{name}') 失败: {e}")


class TestFeatDimMap:
    """FEAT_DIM_MAP 完整性"""

    def test_all_models_have_feat_dim(self):
        """每个注册模型都应有特征维度映射。"""
        for name in MODEL_REGISTRY:
            assert name in FEAT_DIM_MAP, \
                f"'{name}' 缺少 FEAT_DIM_MAP 条目"

    def test_get_feat_dim(self, device):
        """get_feat_dim 应正确推断 CNN 的特征维度。"""
        model = CNN(num_classes=8).to(device)
        dim = get_feat_dim(model, input_shape=(1, 190, 83), device=device)
        assert dim == FEAT_DIM_MAP['cnn'], \
            f"get_feat_dim 返回 {dim}, FEAT_DIM_MAP 为 {FEAT_DIM_MAP['cnn']}"

    def test_get_feat_dim_lstm(self, device):
        """get_feat_dim 应正确推断 LSTM 的特征维度。"""
        model = LSTM(input_size=190, num_classes=8).to(device)
        dim = get_feat_dim(model, input_shape=(190, 83), device=device)
        assert dim == FEAT_DIM_MAP['lstm'], \
            f"get_feat_dim 返回 {dim}, FEAT_DIM_MAP 为 {FEAT_DIM_MAP['lstm']}"


# ═══════════════════════════════════════════════════════════════
#  UT-34, UT-35: ModelWithHead 包装器
# ═══════════════════════════════════════════════════════════════

class TestModelWithHead:
    """ModelWithHead 包装器"""

    def test_linear_head(self):
        """UT-34: ModelWithHead + Linear head。"""
        backbone = CNN(num_classes=8)
        head = nn.Linear(128, 8)
        model = ModelWithHead(backbone, head)
        x = torch.randn(4, 190, 83)
        feat, logits = model(x)
        assert feat.shape == (4, 128)
        assert logits.shape == (4, 8)

    def test_cosface_head(self):
        """UT-35: ModelWithHead + CosFaceHead —— margin 生效。"""
        backbone = CNN(num_classes=8)
        head = CosFaceHead(128, 8, s=30.0, m=0.50)
        model = ModelWithHead(backbone, head)
        x = torch.randn(4, 190, 83)
        labels = torch.tensor([0, 1, 2, 3])

        # 推理模式
        _, logi_infer = model(x)
        # 训练模式
        _, logi_train = model(x, labels)

        # 训练模式下目标类 logit 应偏低（margin 生效）
        for i, lbl in enumerate(labels):
            assert logi_train[i, lbl] < logi_infer[i, lbl], \
                f"CosFace margin 未生效: train[{i},{lbl}]={logi_train[i,lbl]:.2f} >= infer={logi_infer[i,lbl]:.2f}"

    def test_name_property(self):
        """ModelWithHead.name 应委托给 backbone.name。"""
        backbone = CNN(num_classes=8)
        model = ModelWithHead(backbone, nn.Linear(128, 8))
        assert model.name == 'cnn'
