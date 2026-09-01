import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import log_softmax
import math


class CenterLoss(nn.Module):
    """
    中心损失 (Center Loss) 实现

    参数:
        num_classes: 类别总数
        feat_dim: 特征维度
        alpha: 中心更新时的动量，通常取 0.5
    """

    def __init__(self, num_classes, feat_dim, alpha=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha

        # 初始化中心，不通过梯度下降更新
        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim), requires_grad=False
        )

    def forward(self, features, logits):
        """
        计算中心损失

        参数:
            features: (batch_size, feat_dim)
            logits: (batch_size, num_classes) 或 (batch_size,) 类别索引
        返回:
            loss: 标量损失值
        """
        batch_size = features.size(0)
        if logits.dim() > 1:
            idx = torch.argmax(logits, dim=-1)
        else:
            idx = logits
        centers_batch = self.centers[idx]  # (B, D)
        diff = features - centers_batch
        loss = 0.5 * (diff.pow(2).sum()) / batch_size
        return loss

    @torch.no_grad()
    def update_centers(self, features, logits):
        """
        根据当前 batch 更新类别中心

        参数:
            features: (batch_size, feat_dim)
            logits: (batch_size, num_classes) 或 (batch_size,) 类别索引
        """
        if logits.dim() > 1:
            labels = torch.argmax(logits, dim=-1)
        else:
            labels = logits
        unique_labels = labels.unique()
        for label in unique_labels:
            mask = labels == label
            feats_of_class = features[mask]
            center = self.centers[label]
            delta = (feats_of_class - center).sum(dim=0) / (1 + mask.sum())
            self.centers[label] = center + self.alpha * delta


class FocalLoss(nn.Module):
    """
    Focal Loss 用于处理类别不平衡

    参数:
        gamma: 聚焦参数，默认 2.0
        alpha: 类别权重，形状 (num_classes,)
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        ce_loss = -log_pt
        modulating_factor = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * modulating_factor * ce_loss
        else:
            loss = modulating_factor * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SoftLabelKLLoss(nn.Module):
    """
    适配软标签分布的 KL 散度损失，调用方式模仿 CrossEntropyLoss

    参数:
        reduction: 'mean' | 'sum' | 'none'
        dim: softmax/log_softmax 的维度
        gamma: focal 调制因子（0 则退化为标准 KL）
        alpha: 损失缩放系数

    Input:
        logits: 模型原始输出 [B, C]
        soft_target: 软标签概率分布 [B, C]，每行概率和为1
    """

    def __init__(self, reduction="mean", dim=-1, gamma=3.0, alpha=1.0):
        super().__init__()
        self.reduction = reduction
        self.dim = dim
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, soft_target):
        log_p = log_softmax(logits, dim=self.dim)
        p = torch.exp(log_p)

        one = torch.tensor(1.0, device=log_p.device, dtype=log_p.dtype)
        modulating_factor = (one - p) ** self.gamma
        loss_per_sample = -( modulating_factor * soft_target * log_p).sum(dim=self.dim)
        loss_per_sample *= self.alpha

        if self.reduction == "none":
            return loss_per_sample
        elif self.reduction == "sum":
            return torch.sum(loss_per_sample)
        elif self.reduction == "mean":
            return torch.mean(loss_per_sample)
        else:
            raise ValueError(f"不支持 reduction={self.reduction}，可选 mean/sum/none")


class CosFaceHead(nn.Module):
    """CosFace 分类头 —— 替代 nn.Linear 使用。

    训练模式 (labels 不为 None):  输出 s*(cosθ - m·one_hot)，直接送 CrossEntropyLoss
    推理模式 (labels 为 None):    输出 s*cosθ（不带 margin 的各类别概率 logits）

    Args:
        in_features:  输入特征维度
        out_features: 类别数
        s: 缩放因子，默认 30.0
        m: cosine margin，默认 0.50
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None:
            return cosine * self.s                    # 推理：纯 cosine 得分（带缩放）
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return (cosine - one_hot * self.m) * self.s   # 训练：cosine margin


class ArcFaceHead(nn.Module):
    """ArcFace 分类头 —— 替代 nn.Linear 使用。

    训练模式 (labels 不为 None):  输出 s*cos(θ + m)，直接送 CrossEntropyLoss
    推理模式 (labels 为 None):    输出 s*cosθ（不带 margin 的各类别概率 logits）

    Args:
        in_features:  输入特征维度
        out_features: 类别数
        s: 缩放因子，默认 30.0
        m: angular margin（弧度），默认 0.50
        easy_margin:  是否启用 easy margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, labels=None):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None:
            return cosine * self.s                    # 推理：纯 cosine 得分

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.s                        # 训练：angular margin


class SubCenterArcFaceHead(nn.Module):
    """Sub-center ArcFace：每个类别 K 个子中心，由最匹配子中心负责分类。

    标准 ArcFace 每类只有一个权重向量，要求类内特征高度聚集。但电机故障
    诊断中同一故障在不同工况（转速/负载）下声学表现差异大——单一中心无法
    同时覆盖多个子模态。给每类 K 个子中心后，样本只需靠近"同一工况"那一个，
    不同工况由不同子中心承接，margin 施加于最匹配的那个子中心。

    训练模式 (labels 不为 None):
        1. 计算特征与所有 C×K 个子中心的余弦相似度 → reshape [B, C, K]
        2. 每类取 max（硬选择最匹配子中心）→ [B, C]
        3. 找到目标类中最匹配的子中心，对其施加 angular margin
        4. 其余类别保持 max 后的原始 cosine 作为对比 logit

    推理模式 (labels 为 None):
        每类取 max 子中心 → s*cosθ 作为各类别 logit

    Args:
        in_features:  输入特征维度
        out_features: 类别数
        K:            每类子中心数（默认 3）
        s:            缩放因子，默认 30.0
        m:            angular margin（弧度），默认 0.50
        easy_margin:  是否启用 easy margin
    """

    def __init__(self, in_features, out_features, K=3, s=30.0, m=0.50,
                 easy_margin=False):
        super().__init__()
        self.out_features = out_features
        self.K = K
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        # 权重矩阵: [C * K, D] —— 每个类有 K 个子中心
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features * K, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

        # ArcFace 三角函数缓存
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, labels=None):
        # 所有子中心的余弦相似度: [B, C*K]
        cosine_all = F.linear(
            F.normalize(features), F.normalize(self.weight)
        )
        cosine_all = cosine_all.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # 推理：每类取 max 子中心
        if labels is None:
            cosine_2d = cosine_all.view(-1, self.out_features, self.K)
            cos_max = cosine_2d.max(dim=2)[0]           # [B, C]
            return cos_max * self.s

        # ── 训练 ──
        B = features.size(0)
        cosine_2d = cosine_all.view(B, self.out_features, self.K)  # [B, C, K]

        # 每类取 max 子中心作为该类得分: [B, C]
        cos_max_per_class, _ = cosine_2d.max(dim=2)

        # 目标类中：找到最匹配的那个子中心及其余弦值
        cos_target_all = cosine_2d[torch.arange(B), labels, :]     # [B, K]
        cos_target_max, sub_idx = cos_target_all.max(dim=1)         # [B], [B]

        # ── ArcFace margin 施加于目标类的获胜子中心 ──
        sine = torch.sqrt((1.0 - torch.pow(cos_target_max, 2)).clamp(0, 1))
        phi = cos_target_max * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cos_target_max > 0, phi, cos_target_max)
        else:
            phi = torch.where(cos_target_max > self.th, phi,
                              cos_target_max - self.mm)

        # 构造最终 logits：max-per-class 为基础，目标类替换为 margin 版本
        logits = cos_max_per_class.clone()
        logits[torch.arange(B), labels] = phi
        return logits * self.s


class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        计算 SimCLR 损失

        参数:
            features: [3N, D]，每个样本有两个增强视图
        返回:
            loss: 标量损失值
        """
        device = features.device
        batch_size = features.shape[0] // 2

        # 归一化特征向量
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 创建标签：正样本对的索引
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels, labels], dim=0)

        # # mask 去除自身相似度
        # mask = torch.eye(2 * batch_size, device=device).bool()
        # similarity_matrix.masked_fill_(mask, 1)

        # 计算交叉熵损失
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


def mixup(features, labels, exclude=[], alpha=0.2):
    """
    Mixup 数据增强（标准实现）

    对特征进行线性插值，返回两份原始标签和插值系数 λ，
    由调用方按 λ·CE(pred, y_a) + (1-λ)·CE(pred, y_b) 计算 loss。

    参数:
        features: [N, ...] 特征张量
        labels:   [N] 或 [N, C] — 1D 整数标签或 one-hot 概率分布
        alpha:    Beta 分布的参数

    返回:
        mixup_features, labels_a, labels_b, lam
    """

    exclude_idx = labels.unsqueeze(-1).eq(torch.tensor(exclude, device=labels.device)).any(dim=-1)
    features = features[~exclude_idx]
    labels = labels[~exclude_idx]

    shuffle_idx = torch.randperm(features.shape[0])
    shuffle_feature = features[shuffle_idx]
    shuffle_labels = labels[shuffle_idx]

    lam = np.random.beta(alpha, alpha)
    mixup_features = lam * features + (1.0 - lam) * shuffle_feature

    return mixup_features, labels, shuffle_labels, lam