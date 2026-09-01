import inspect

import torch
from torch import nn
from dropblock import DropBlock2D
from torch.nn.functional import leaky_relu, softplus, gelu

from src.net.cnn import CNN, CNN2, CNNSE, CNN2SE, CNN_CBAM, CNN2_CBAM, CNN_CBAMResidual, CNN2_CBAMResidual
from src.net.lstm import LSTM, LSTMSE
from src.net.resnet import ResNet, ResNetSE


# =========================
# 模型注册表
# =========================

# 模型名 → (模型类, 是否为 LSTM（需要 input_size）)
MODEL_REGISTRY = {
    'cnn':         (CNN,         False),
    'cnn2':        (CNN2,        False),
    'cnn_se':      (CNNSE,       False),
    'cnn2_se':     (CNN2SE,      False),
    'cnn_cbam':    (CNN_CBAM,    False),
    'cnn2_cbam':   (CNN2_CBAM,   False),
    'cnn_cbam_residual':   (CNN_CBAMResidual,    False),
    'cnn2_cbam_residual':  (CNN2_CBAMResidual,   False),
    'lstm':        (LSTM,        True),
    'lstm_se':     (LSTMSE,      True),
    'resnet':      (ResNet,      False),
    'resnet_se':   (ResNetSE,    False),
}

# 模型名 → 特征维度（用于 CenterLoss / CosFace 等）
# 注意：此为默认值，若修改模型超参会自动失效。
# 优先使用 get_feat_dim() 从模型实例动态推断。
FEAT_DIM_MAP = {
    'cnn': 128, 'cnn2': 128, 'cnn_se': 128, 'cnn2_se': 128,
    'cnn_cbam': 128, 'cnn2_cbam': 128, 'cnn_cbam_residual': 128, 'cnn2_cbam_residual': 128,
    'lstm': 512, 'lstm_se': 512,
    'resnet': 128, 'resnet_se': 128,
}


def get_feat_dim(model, input_shape=(190, 200), device=None):
    """通过一次 dummy forward 推断模型输出的特征维度。

    归一性原理：所有 backbone 的 forward 返回 feature [B, D] 或 (feature, logits)，
    本函数统一提取 feature 并返回其维度 D。

    Args:
        model:      backbone 实例（未包装 ModelWithHead）
        input_shape: 虚拟输入的形状。
                     - (C, T)：构造 [B, C, T]（CNN 会自动补通道维，LSTM 直接用）
                     - (1, C, T)：构造 [B, 1, C, T]（仅 CNN）
                    默认 (190, 200)
        device:     设备（None 则使用模型参数所在设备）

    Returns:
        feat_dim: 特征向量的维度
    """
    if device is None:
        device = next(model.parameters()).device
    dummy = torch.randn(2, *input_shape, device=device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
        if isinstance(out, tuple):
            feature = out[0]
        else:
            feature = out
    return feature.shape[-1]


class DualModel(nn.Module):
    """双分支模型包装：同一模型结构分别处理 mel 分支与 MFCC 分支，特征拼接后分类。

    输入 x 为拼接特征 [B, mel_channels + mfcc_channels, T]：
      - 前 mel_channels 行 → mel 分支
      - 后 mfcc_channels 行 → MFCC 分支
    """

    def __init__(self, model_cls, num_classes=8, mel_channels=64, mfcc_channels=126,
                 out_features=128, **kwargs):
        super().__init__()
        # model_cls 是类；取其实例 name 属性（实例化后才有）
        self._model_cls = model_cls
        self.name = f"dual_{model_cls.__name__}"
        self.mel_channels = mel_channels
        self.mfcc_channels = mfcc_channels

        self.model_mel = model_cls(num_classes=num_classes, out_features=out_features // 2, input_size=mel_channels, **kwargs)
        self.model_mfcc = model_cls(num_classes=num_classes, out_features=out_features // 2, input_size=mfcc_channels, **kwargs)


    def forward(self, x):
        x_mel = x[:, :self.mel_channels, :]
        x_mfcc = x[:, self.mel_channels:self.mel_channels + self.mfcc_channels, :]
    
        feat1 = self.model_mel(x_mel)
        feat2 = self.model_mfcc(x_mfcc)


        feat = torch.cat([feat1, feat2], dim=1)
        return feat


class FusionGate(nn.Module):
    """特征融合门控模块"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # 计算门控权重
        gate = torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))
        # 对输入特征进行加权融合
        fused = x * gate
        return fused


class ModelWithHead(nn.Module):
    """Backbone + 分类头的包装器。

    backbone 输出特征向量，head 映射到 logits。
    支持 nn.Linear / CosFaceHead / ArcFaceHead / SubCenterArcFaceHead，
    通过 labels 参数切换训练/推理模式。
    """

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head


    @property
    def name(self):
        return getattr(self.backbone, 'name', 'model')

    def forward(self, x, labels=None):
        features = self.backbone(x)
        from src.train.losses import CosFaceHead, ArcFaceHead, SubCenterArcFaceHead
        if labels is not None and isinstance(
            self.head, (CosFaceHead, ArcFaceHead, SubCenterArcFaceHead)
        ):
            logits = self.head(features, labels)
        else:
            logits = self.head(features)
        return features, logits


def create_model(modelname, input_size, num_classes=8, **kwargs):
    """
    根据名称创建模型实例

    参数:
        modelname: 模型名（见 MODEL_REGISTRY 的 key）
        input_size: LSTM 类模型的输入维度（非 LSTM 模型可忽略）
        num_classes: 类别数
        dual: 是否为双分支模型

    返回:
        model 实例
    """
    if modelname not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {modelname}，可选: {list(MODEL_REGISTRY.keys())}")

    model_cls, needs_input_size = MODEL_REGISTRY[modelname]

    if isinstance(input_size, (tuple, list)) and len(input_size) == 2:
        mel_channels, mfcc_channels = input_size
        return DualModel(model_cls, num_classes=num_classes, mel_channels=mel_channels, mfcc_channels=mfcc_channels, **kwargs)
    if needs_input_size:
        return model_cls(input_size=input_size, num_classes=num_classes, **kwargs)
    else:
        return model_cls(num_classes=num_classes, **kwargs)