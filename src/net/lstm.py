import torch
import warnings
from torch import nn
from dropblock import DropBlock2D
from torch.nn.functional import leaky_relu, softplus, gelu

from src.net.base import (
    SELSTMBlock
)


class LSTM(nn.Module):
    """双层双向 LSTM"""

    def __init__(self, input_size, num_classes=8, hidden_size=256, out_features=128, num_layers=1,
                 lstm_dropout=0.2, **kwargs):
        super().__init__()
        if kwargs:
            warnings.warn(f"LSTM 忽略未使用参数: {list(kwargs)}")
        self.name = 'lstm'

        # 输入归一化：每个时间步内对 190 维特征做 LayerNorm，
        # 缓解 mel(dB)/MFCC/Δ 之间尺度差异对 LSTM 的影响。
        self.norm1 = nn.LayerNorm(input_size)
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, bidirectional=True)
        self.dir1 = 2 if self.lstm1.bidirectional else 1
        self.lstm2 = nn.LSTM(hidden_size * self.dir1, out_features //  4, num_layers,
                             batch_first=True, bidirectional=True)
        self.dir2 = 2 if self.lstm2.bidirectional else 1
        self.dropout = nn.Dropout(lstm_dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        # 用所有时间步的 mean+max 池化代替“只取最后一步”，
        # 双向 LSTM 的中间帧信息不再被丢弃。
        pooled = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        feature = self.dropout(pooled)
        return feature


class LSTMSE(nn.Module):
    """LSTM + SE 通道注意力"""

    def __init__(self, input_size, num_classes=8, hidden_size=256, out_features=128, num_layers=1,
                 lstm_dropout=0.2, se_reduction=8, **kwargs):
        super().__init__()
        if kwargs:
            warnings.warn(f"LSTMSE 忽略未使用参数: {list(kwargs)}")
        self.name = 'lstm_se'

        self.norm = nn.LayerNorm(input_size)
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, bidirectional=True)
        self.dir1 = 2 if self.lstm1.bidirectional else 1
        self.se1 = SELSTMBlock(hidden_size * self.dir1, reduction=16)
        self.lstm2 = nn.LSTM(hidden_size * self.dir1, out_features, num_layers,
                             batch_first=True, bidirectional=True)
        self.dir2 = 2 if self.lstm2.bidirectional else 1
        self.se2 = SELSTMBlock(out_features * self.dir2, reduction=se_reduction)
        self.dropout = nn.Dropout(lstm_dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.se2(x)
        pooled = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        feature = self.dropout(pooled)
        return feature


# class LSTMANFIS(nn.Module):
#     """LSTM + ANFIS 模糊推理"""

#     def __init__(self, input_size, num_classes=8, hidden_size=256, num_layers=1,
#                  lstm_dropout=0.3, anfis_fuzzy_state=3):
#         super().__init__()
#         self.name = 'lstm_anfis'

#         self.norm = nn.LayerNorm(input_size)
#         self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers,
#                              batch_first=True, bidirectional=True)
#         self.dir1 = 2 if self.lstm1.bidirectional else 1
#         self.lstm2 = nn.LSTM(hidden_size * self.dir1, hidden_size // 2, num_layers,
#                              batch_first=True, bidirectional=True)
#         self.dir2 = 2 if self.lstm2.bidirectional else 1
#         self.dropout = nn.Dropout(lstm_dropout)
#         self.fc = nn.Linear((hidden_size // 2) * self.dir2 * 2, 8)
#         self.anfis = ANFIS(input_shape=8, fuzzy_state=anfis_fuzzy_state, num_classes=num_classes)

#     def forward(self, x, labels=None):
#         x = x.transpose(1, 2)
#         x = self.norm(x)
#         x, _ = self.lstm1(x)
#         x, _ = self.lstm2(x)
#         pooled = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
#         feature = self.dropout(pooled)
#         anfis_in = gelu(self.fc(feature))
#         logits = self.anfis(anfis_in)
#         return feature, logits