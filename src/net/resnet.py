import torch
import warnings
from torch import nn
from dropblock import DropBlock2D
from torch.nn.functional import leaky_relu, softplus, gelu


from src.net.base import (
    ResBlock, ResSEBlock
)

class ResNet(nn.Module):
    """残差网络"""

    def __init__(self, input_channels=1, num_classes=8, base_filters=32, out_features=128, **kwargs):
        super().__init__()
        if kwargs:
            warnings.warn(f"ResNet 忽略未使用参数: {list(kwargs)}")
        self.name = 'resnet'

        self.res11 = ResBlock(input_channels, base_filters, downsample=True)
        self.res21 = ResBlock(base_filters, base_filters * 2, downsample=True)
        self.res31 = ResBlock(base_filters * 2, out_features, downsample=True)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x.unsqueeze(1) if x.ndim == 3 else x
        x = self.res11(x)
        x = self.res21(x)
        x = self.res31(x)
        x = self.gap(x)
        feature = x.view(x.size(0), -1)
        return feature


class ResNetSE(nn.Module):
    """残差网络 + SE 注意力"""

    def __init__(self, input_channels=1, num_classes=8, base_filters=32, out_features=128, **kwargs):
        super().__init__()
        if kwargs:
            warnings.warn(f"ResNetSE 忽略未使用参数: {list(kwargs)}")
        self.name = 'resnet_se'

        self.res11 = ResSEBlock(input_channels, base_filters, downsample=True)
        self.res21 = ResSEBlock(base_filters, base_filters * 2, downsample=True)
        self.res31 = ResSEBlock(base_filters * 2, out_features, downsample=True)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x.unsqueeze(1) if x.ndim == 3 else x
        x = self.res11(x)
        x = self.res21(x)
        x = self.res31(x)
        x = self.gap(x)
        feature = x.view(x.size(0), -1)
        return feature

