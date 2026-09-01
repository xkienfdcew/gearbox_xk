"""
工具函数：固定随机种子以保证实验可复现。
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_cudnn: bool = True):
    """固定 Python / NumPy / PyTorch 随机种子。

    Args:
        seed:              随机种子
        deterministic_cudnn: 是否开启 cuDNN 确定性模式（会略微降低性能）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # DataLoader worker 确定性（需配合 worker_init_fn）
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def worker_init_fn(worker_id: int):
    """DataLoader 的 worker_init_fn，确保每个 worker 使用不同但确定的种子。"""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)
