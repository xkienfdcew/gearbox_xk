import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import leaky_relu, softplus, gelu, relu
import torchaudio


# class ANFIS(nn.Module):
#     """自适应模糊推理系统"""

#     def __init__(self, input_shape, fuzzy_state, num_classes):
#         super().__init__()
#         self.ins = input_shape
#         self.fus = fuzzy_state
#         self.rln = self.fus ** self.ins
#         self.nuc = num_classes

#         self.mu = nn.Parameter(torch.randn(self.ins, self.fus) * 0.5)
#         self.sigma_raw = nn.Parameter(torch.ones(self.ins, self.fus) * 0.5)
#         self.cons_para = nn.Parameter(torch.randn(self.rln, self.ins + 1, self.nuc) * 0.01)
#         self._build_rule_index()

#     def _build_rule_index(self):
#         ind_list = []
#         for i in range(self.ins):
#             rep = int(self.fus ** i)
#             step = int(self.rln / rep / self.fus)
#             idx = torch.repeat_interleave(torch.arange(self.fus), rep).repeat(step)
#             ind_list.append(idx)
#         self.register_buffer('rule_ind', torch.stack(ind_list, dim=0))

#     def membershipFunction(self, x):
#         x = x.unsqueeze(-1)
#         mu = self.mu.unsqueeze(0)
#         sigma = softplus(self.sigma_raw).unsqueeze(0)
#         return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

#     def ruleWeight(self, mf):
#         B = mf.shape[0]
#         weight = torch.ones(B, self.rln, device=mf.device)
#         for i in range(self.ins):
#             weight *= mf[:, i][:, self.rule_ind[i]]
#         return weight

#     def forward(self, x):
#         B = x.shape[0]
#         mf = self.membershipFunction(x)
#         weight = self.ruleWeight(mf)
#         weight_norm = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
#         weight_norm = weight_norm.unsqueeze(-1)

#         x_aug = torch.cat([x, torch.ones(B, 1, device=x.device)], dim=-1)
#         x_aug = x_aug.unsqueeze(1).unsqueeze(-1)
#         cons = self.cons_para.unsqueeze(0)
#         rule_out = torch.sum(cons * x_aug, dim=-2)
#         out = torch.sum(weight_norm * rule_out, dim=1)
#         return out


# class FeatureNorm(nn.Module):
#     """特征归一化模块（可用于数据预处理）"""

#     def __init__(self, mean, std):
#         super().__init__()
#         self.register_buffer('mean', mean.view(1, -1, 1))
#         self.register_buffer('std', std.view(1, -1, 1) + 1e-6)

#     def forward(self, x):
#         return (x - self.mean) / self.std


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力（用于 CNN）"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.global_avg_pool(x).view(B, C)
        y = relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(B, C, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    """CBAM 通道注意力 —— 同时用 AvgPool 和 MaxPool 压缩空间信息。

    与 SE 的区别：SE 只用 AvgPool，CBAM 额外加入 MaxPool 分支，
    两个分支共享同一个 MLP，最后逐元素求和 → sigmoid。

    residual=True 时为改进版：输出 x + x·tanh(gate)，gate 从 0 初始化，
    模块从恒等映射开始训练，避免乘性门控把特征压灭后无法恢复。
    """

    def __init__(self, channels, reduction=16, residual=False):
        super().__init__()
        self.residual = residual
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享 MLP
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        if residual:
            # gate = tanh(fc2(...))，fc2 置零 → gate=0 → 输出恒等于输入
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        B, C, _, _ = x.size()

        avg_y = self.avg_pool(x).view(B, C)
        max_y = self.max_pool(x).view(B, C)

        avg_out = self.fc2(relu(self.fc1(avg_y)))
        max_out = self.fc2(relu(self.fc1(max_y)))

        if self.residual:
            gate = torch.tanh(avg_out + max_out).view(B, C, 1, 1)
            return x + x * gate
        scale = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """CBAM 空间注意力 —— 沿通道轴池化后由 7×7 卷积生成空间注意力图。

    输入:  [B, C, H, W]
    输出:  [B, C, H, W]（乘了空间 mask）

    residual=True 时为改进版：输出 x + x·tanh(gate)，卷积从 0 初始化，
    模块从恒等映射开始训练。
    """

    def __init__(self, kernel_size=7, residual=False):
        super().__init__()
        self.residual = residual
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        if residual:
            # gate = tanh(conv(...))，卷积置零 → gate=0 → 输出恒等于输入
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)   # [B, 1, H, W]
        max_out = x.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
        pooled = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        if self.residual:
            gate = torch.tanh(self.conv(pooled))
            return x + x * gate
        scale = torch.sigmoid(self.conv(pooled))
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018)。

    标准顺序：先通道注意力，再空间注意力。
    通道注意力决定"哪些通道重要"，空间注意力决定"通道内哪些位置重要"。

    Args:
        channels:   输入通道数
        reduction:  通道注意力的降维比例（默认 16）
        kernel_size: 空间注意力的卷积核大小（默认 7）
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class CBAMResidual(CBAM):
    """改进版 CBAM（残差门控 + 恒等初始化）。

    与原版 CBAM 的区别：
      1. 通道/空间门控都改为加性残差形式 x + x·tanh(gate)；
      2. 门控参数从 0 初始化，模块从恒等映射开始训练，
         不会把初始特征压到 0.5^6≈1.6%，也避免训练中 mask 饱和后无法恢复；
      3. 默认 kernel=3，适合 42 帧这样的小特征图（7×7 感受野会横跨整个时间轴）。

    Args:
        channels:   输入通道数
        reduction:  通道注意力的降维比例（默认 16）
        kernel_size: 空间注意力的卷积核大小（默认 3）
    """

    def __init__(self, channels, reduction=16, kernel_size=3):
        super().__init__(channels, reduction=reduction, kernel_size=kernel_size)
        self.channel_att = ChannelAttention(channels, reduction, residual=True)
        self.spatial_att = SpatialAttention(kernel_size, residual=True)


class SELSTMBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力（用于 LSTM）"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        B, T, C = x.size()
        y = x.mean(dim=1)
        y = gelu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(B, 1, C)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    """残差模块"""

    def __init__(self, input_channels, output_channels, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels)
        ) if downsample else None

    def forward(self, x):
        fx = self.bn1(gelu(self.conv1(x)))
        fx = self.bn2(gelu(self.conv2(fx)))
        if self.downsample is not None:
            x = self.downsample(x)
        return gelu(fx + x)


class ResSEBlock(nn.Module):
    """带 SE 注意力的残差模块"""

    def __init__(self, input_channels, output_channels, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.se = SEBlock(output_channels, reduction=8)
        self.downsample = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels)
        ) if downsample else None

    def forward(self, x):
        fx = self.bn1(gelu(self.conv1(x)))
        fx = self.bn2(gelu(self.conv2(fx)))
        fx = self.se(fx)
        if self.downsample is not None:
            x = self.downsample(x)
        return gelu(fx + x)


class TrainedPCENLayer(nn.Module):
    """[已废弃] 可训练的 PCEN 层（简化版，仅 3 个全局参数 + 固定平滑器融合）

    此版本仅保留以兼容旧模型权重，新实验不应使用。
    如需 PCEN 归一化，请在 Processor 中设置 use_pcen=True 使用 librosa.pcen。

    .. warning::
        此类已废弃，将在未来版本中移除。请勿在新实验中使用。
    """

    def __init__(self):
        super().__init__()
        warnings.warn(
            "TrainedPCENLayer is deprecated. "
            "Use Processor(use_pcen=True) for librosa.pcen instead. "
            "This class is kept for compatibility with old checkpoints.",
            DeprecationWarning, stacklevel=2
        )
        self.alpha_log = nn.Parameter(torch.empty(1))
        self.delta_log = nn.Parameter(torch.empty(1))
        self.r_log = nn.Parameter(torch.empty(1))
        self.s = torch.tensor([0.015, 0.02, 0.04, 0.08])
        self.weight_s = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32))

        nn.init.normal_(self.alpha_log, mean=0, std=0.1)
        nn.init.normal_(self.delta_log, mean=0, std=0.1)
        nn.init.normal_(self.r_log, mean=0, std=0.1)

    def forward(self, x):
        alpha = torch.exp(self.alpha_log)
        delta = torch.exp(self.delta_log)
        r = torch.exp(self.r_log)

        M = [torchaudio.functional.lfilter(
            x,
            a_coeffs=torch.tensor([1.0, self.s[i].item() - 1.0], device=x.device),
            b_coeffs=torch.tensor([self.s[i].item(), 0], device=x.device),
            clamp=True
        ) for i in range(4)]
        M = torch.stack(M, dim=0)
        M = M * self.weight_s.view(4, 1, 1, 1)
        M = M.sum(dim=0)

        eps = 1e-6
        base = x / (M + eps)
        base = torch.clamp(base, min=eps)
        term = base ** alpha + delta
        term = torch.clamp(term, min=eps)
        output = term ** r - delta ** r
        return output


class NormalizedLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # 可学习的权重矩阵，等同于标准全连接层的权重
        # 形状为 [in_features, out_features]
        self.weight = nn.Parameter(torch.FloatTensor(input_size, output_size))
        # 使用正态分布初始化权重
        nn.init.xavier_normal_(self.weight)      
    
    def forward(self, x):
        # 1. L2 归一化：对输入特征和权重进行归一化，使其模长为1
        #   对一个batch内的每个输入特征进行归一化， dim=1
        input_norm = F.normalize(x, p=2, dim=1)
        #   对权重矩阵的列进行归一化， dim=0
        weight_norm = F.normalize(self.weight, p=2, dim=0)

        # 2. 计算余弦相似度：归一化特征与归一化权重的点积
        #   形状: [batch_size, output_size]
        cosine = torch.mm(input_norm, weight_norm)
        #   为防止数值问题，将余弦值截断在 [-1, 1] 区间内
        cosine = torch.clamp(cosine, -1, 1)

        return cosine
