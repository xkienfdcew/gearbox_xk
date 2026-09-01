import torch
import warnings
from torch import nn
from dropblock import DropBlock2D
from torch.nn.functional import leaky_relu, softplus, gelu

from src.net.base import (
    SEBlock, SELSTMBlock, CBAM, CBAMResidual
)

class CNN(nn.Module):
    """基础 CNN"""

    def __init__(self, num_classes=8, base_filters=32, 
                input_channels=1 ,out_features=128,
                dropblock_p=0.15, dropblock_block_size=6, **kwargs):
        super().__init__()
        if kwargs:
            warnings.warn(f"CNN 忽略未使用参数: {list(kwargs)}")
        self.name = 'cnn'

        self.Conv1 = nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.pool1 = nn.MaxPool2d(2)

        self.Conv2 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.Conv3 = nn.Conv2d(base_filters * 2, out_features, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_features)

        self.dropout = DropBlock2D(dropblock_p, dropblock_block_size)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x.unsqueeze(1) if x.ndim == 3 else x
        x = self.pool1(self.bn1(gelu(self.Conv1(x))))
        x = self.pool2(self.bn2(gelu(self.Conv2(x))))
        x = self.bn3(gelu(self.Conv3(x)))
        x = self.dropout(x)
        x = self.gap(x)
        feature = x.view(x.size(0), -1)
        return feature


class CNN2(nn.Module):
    """双卷积 CNN（每层双 conv）"""

    def __init__(self, num_classes=8, base_filters=32, input_channels=1, out_features=128,
                 dropblock_p=0.15, dropblock_block_size=5, **kwargs):
        super().__init__()
        if kwargs:
            warnings.warn(f"CNN2 忽略未使用参数: {list(kwargs)}")
        self.name = 'cnn2'

        self.Conv11 = nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1)
        self.Conv12 = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.pool1 = nn.MaxPool2d(2)

        self.Conv21 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1)
        self.Conv22 = nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.Conv3 = nn.Conv2d(base_filters * 2, out_features, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_features)

        self.dropout = DropBlock2D(dropblock_p, dropblock_block_size)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x.unsqueeze(1) if x.ndim == 3 else x
        x = self.pool1(self.bn1(gelu(self.Conv12(gelu(self.Conv11(x))))))
        x = self.pool2(self.bn2(gelu(self.Conv22(gelu(self.Conv21(x))))))
        x = self.bn3(gelu(self.Conv3(x)))
        x = self.dropout(x)
        x = self.gap(x)
        feature = x.view(x.size(0), -1)
        return feature


class CNNSE(nn.Module):
    """CNN + SE 通道注意力"""

    def __init__(self, num_classes=8, base_filters=32, input_channels=1, out_features=128,
                 dropblock_p=0.15, dropblock_block_size=5, se_reduction=16, **kwargs):
        super().__init__()
        if kwargs:
            warnings.warn(f"CNNSE 忽略未使用参数: {list(kwargs)}")
        self.name = 'cnn_se'

        self.Conv1 = nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.se1 = SEBlock(base_filters, reduction=se_reduction)
        self.pool1 = nn.MaxPool2d(2)

        self.Conv2 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)
        self.se2 = SEBlock(base_filters * 2, reduction=se_reduction)
        self.pool2 = nn.MaxPool2d(2)

        self.Conv3 = nn.Conv2d(base_filters * 2, out_features, kernel_size=3, padding=1)
        self.se3 = SEBlock(out_features, reduction=se_reduction)
        self.bn3 = nn.BatchNorm2d(out_features)

        self.dropout = DropBlock2D(dropblock_p, dropblock_block_size)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x.unsqueeze(1) if x.ndim == 3 else x
        x = self.pool1(self.bn1(gelu(self.Conv1(x))))
        x = self.pool2(self.se2(self.bn2(gelu(self.Conv2(x)))))
        x = self.se3(self.bn3(gelu(self.Conv3(x))))
        x = self.dropout(x)
        x = self.gap(x)
        feature = x.view(x.size(0), -1)
        return feature


class CNN2SE(nn.Module):
    """双卷积 CNN + SE 通道注意力"""

    def __init__(self, num_classes=8, base_filters=32, input_channels=1, out_features=128,
                 dropblock_p=0.1, dropblock_block_size=5, se_reduction=16, **kwargs):
        super().__init__()
        if kwargs:
            warnings.warn(f"CNN2SE 忽略未使用参数: {list(kwargs)}")
        self.name = 'cnn2_se'

        self.Conv11 = nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1)
        self.Conv12 = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.se1 = SEBlock(base_filters, reduction=se_reduction)
        self.pool1 = nn.MaxPool2d(2)

        self.Conv21 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1)
        self.Conv22 = nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)
        self.se2 = SEBlock(base_filters * 2, reduction=se_reduction)
        self.pool2 = nn.MaxPool2d(2)

        self.Conv3 = nn.Conv2d(base_filters * 2, out_features, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_features)
        self.se3 = SEBlock(out_features, reduction=se_reduction)

        self.dropout = DropBlock2D(dropblock_p, dropblock_block_size)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x.unsqueeze(1) if x.ndim == 3 else x
        x = self.pool1(self.bn1(gelu(self.Conv12(gelu(self.Conv11(x))))))
        x = self.pool2(self.se2(self.bn2(gelu(self.Conv22(gelu(self.Conv21(x)))))))
        x = self.se3(self.bn3(gelu(self.Conv3(x))))
        x = self.dropout(x)
        x = self.gap(x)
        feature = x.view(x.size(0), -1)
        return feature


class CNN_CBAM(nn.Module):
    """CNN + CBAM 注意力（通道 + 空间）"""

    def __init__(self, num_classes=8, base_filters=32, input_channels=1, out_features=128,
                 dropblock_p=0.15, dropblock_block_size=5,
                 cbam_reduction=16, cbam_kernel=7, **kwargs):
        super().__init__()
        if kwargs:
            warnings.warn(f"CNN_CBAM 忽略未使用参数: {list(kwargs)}")
        self.name = 'cnn_cbam'

        self.Conv1 = nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1)
        self.cbam1 = CBAM(base_filters, reduction=cbam_reduction, kernel_size=cbam_kernel)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.pool1 = nn.MaxPool2d(2)

        self.Conv2 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1)
        self.cbam2 = CBAM(base_filters * 2, reduction=cbam_reduction, kernel_size=cbam_kernel)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.Conv3 = nn.Conv2d(base_filters * 2, out_features, kernel_size=3, padding=1)
        self.cbam3 = CBAM(out_features, reduction=cbam_reduction,
                         kernel_size=cbam_kernel)
        self.bn3 = nn.BatchNorm2d(out_features)

        self.dropout = DropBlock2D(dropblock_p, dropblock_block_size)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x.unsqueeze(1) if x.ndim == 3 else x
        x = self.pool1(self.bn1(gelu(self.Conv1(x))))
        x = self.pool2(self.cbam2(self.bn2(gelu(self.Conv2(x)))))
        x = self.cbam3(self.bn3(gelu(self.Conv3(x))))
        x = self.dropout(x)
        x = self.gap(x)
        feature = x.view(x.size(0), -1)
        return feature


class CNN2_CBAM(nn.Module):
    """双卷积 CNN + CBAM 注意力"""

    def __init__(self, num_classes=8, base_filters=32, input_channels=1, out_features=128,
                 dropblock_p=0.10, dropblock_block_size=5,
                 cbam_reduction=16, cbam_kernel=7, **kwargs):
        super().__init__()
        if kwargs:
            warnings.warn(f"CNN2_CBAM 忽略未使用参数: {list(kwargs)}")
        self.name = 'cnn2_cbam'

        self.Conv11 = nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1)
        self.Conv12 = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.cbam1 = CBAM(base_filters, reduction=cbam_reduction, kernel_size=cbam_kernel)
        self.pool1 = nn.MaxPool2d(2)

        self.Conv21 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1)
        self.Conv22 = nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)
        self.cbam2 = CBAM(base_filters * 2, reduction=cbam_reduction, kernel_size=cbam_kernel)
        self.pool2 = nn.MaxPool2d(2)

        self.Conv3 = nn.Conv2d(base_filters * 2, out_features, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_features)
        self.cbam3 = CBAM(out_features, reduction=max(cbam_reduction // 2, 2),
                         kernel_size=cbam_kernel)

        self.dropout = DropBlock2D(dropblock_p, dropblock_block_size)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x.unsqueeze(1) if x.ndim == 3 else x
        x = self.pool1(self.cbam1(self.bn1(gelu(self.Conv12(gelu(self.Conv11(x)))))))
        x = self.pool2(self.cbam2(self.bn2(gelu(self.Conv22(gelu(self.Conv21(x)))))))
        x = self.cbam3(self.bn3(gelu(self.Conv3(x))))
        x = self.dropout(x)
        x = self.gap(x)
        feature = x.view(x.size(0), -1)
        return feature


class CNN_CBAMResidual(nn.Module):
    """CNN + CBAM 注意力（通道 + 空间）"""

    def __init__(self, num_classes=8, base_filters=32, input_channels=1, out_features=128,
                 dropblock_p=0.15, dropblock_block_size=5,
                 cbam_reduction=16, cbam_kernel=7, **kwargs):
        super().__init__()
        if kwargs:
            warnings.warn(f"CNN_CBAMResidual 忽略未使用参数: {list(kwargs)}")
        self.name = 'cnn_cbam_residual'

        self.Conv1 = nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1)
        self.cbam1 = CBAMResidual(base_filters, reduction=cbam_reduction, kernel_size=cbam_kernel)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.pool1 = nn.MaxPool2d(2)

        self.Conv2 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1)
        self.cbam2 = CBAMResidual(base_filters * 2, reduction=cbam_reduction, kernel_size=cbam_kernel)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.Conv3 = nn.Conv2d(base_filters * 2, out_features, kernel_size=3, padding=1)
        self.cbam3 = CBAMResidual(out_features, reduction=max(cbam_reduction // 2, 2),
                         kernel_size=cbam_kernel)
        self.bn3 = nn.BatchNorm2d(out_features)

        self.dropout = DropBlock2D(dropblock_p, dropblock_block_size)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x.unsqueeze(1) if x.ndim == 3 else x
        x = self.pool1(self.bn1(gelu(self.Conv1(x))))
        x = self.pool2(self.bn2(gelu(self.Conv2(x))))
        x = self.cbam3(self.bn3(gelu(self.Conv3(x))))
        x = self.dropout(x)
        x = self.gap(x)
        feature = x.view(x.size(0), -1)
        return feature


class CNN2_CBAMResidual(nn.Module):
    """双卷积 CNN + CBAM 注意力"""

    def __init__(self, num_classes=8, base_filters=32, input_channels=1, out_features=128,
                 dropblock_p=0.10, dropblock_block_size=5,
                 cbam_reduction=16, cbam_kernel=7, **kwargs):
        super().__init__()
        if kwargs:
            warnings.warn(f"CNN2_CBAMResidual 忽略未使用参数: {list(kwargs)}")
        self.name = 'cnn2_cbam_residual'

        self.Conv11 = nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1)
        self.Conv12 = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.cbam1 = CBAMResidual(base_filters, reduction=cbam_reduction, kernel_size=cbam_kernel)
        self.pool1 = nn.MaxPool2d(2)

        self.Conv21 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1)
        self.Conv22 = nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)
        self.cbam2 = CBAMResidual(base_filters * 2, reduction=cbam_reduction, kernel_size=cbam_kernel)
        self.pool2 = nn.MaxPool2d(2)

        self.Conv3 = nn.Conv2d(base_filters * 2, out_features, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_features)
        self.cbam3 = CBAMResidual(out_features, reduction=max(cbam_reduction // 2, 2),
                         kernel_size=cbam_kernel)

        self.dropout = DropBlock2D(dropblock_p, dropblock_block_size)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x.unsqueeze(1) if x.ndim == 3 else x
        x = self.pool1(self.cbam1(self.bn1(gelu(self.Conv12(gelu(self.Conv11(x)))))))
        x = self.pool2(self.cbam2(self.bn2(gelu(self.Conv22(gelu(self.Conv21(x)))))))
        x = self.cbam3(self.bn3(gelu(self.Conv3(x))))
        x = self.dropout(x)
        x = self.gap(x)
        feature = x.view(x.size(0), -1)
        return feature

# class CNNANFIS(nn.Module):
#     """CNN + ANFIS 模糊推理（CNN 提取特征后由 ANFIS 分类）"""

#     def __init__(self, num_classes=8, input_channels=1,
#                  dropblock_p=0.15, dropblock_block_size=5,
#                  anfis_fuzzy_state=3):
#         super().__init__()
#         self.name = 'cnn_anfis'

#         self.Conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.pool1 = nn.MaxPool2d(2)

#         self.Conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool2 = nn.MaxPool2d(2)

#         self.Conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)

#         self.dropout = DropBlock2D(dropblock_p, dropblock_block_size)
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(128, 8)
#         self.anfis = ANFIS(input_shape=8, fuzzy_state=anfis_fuzzy_state, num_classes=num_classes)

#     def forward(self, x, labels=None):
#         x = x.unsqueeze(1) if x.ndim == 3 else x
#         x = self.pool1(self.bn1(gelu(self.Conv1(x))))
#         x = self.pool2(self.bn2(gelu(self.Conv2(x))))
#         x = self.bn3(gelu(self.Conv3(x)))
#         x = self.dropout(x)
#         x = self.gap(x)
#         feature = x.view(x.size(0), -1)
#         anfis_in = gelu(self.fc(feature))
#         logits = self.anfis(anfis_in)
#         return feature, logits

