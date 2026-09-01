"""
test_utils.py —— 工具函数单元测试

归一性原理：验证 set_seed 的确定性保证。
"""

import numpy as np
from src.utils import set_seed, worker_init_fn


class TestSetSeed:
    """UT-01, UT-02: 随机种子确定性"""

    def test_reproducibility(self):
        """UT-01: 两次 set_seed(42) 后产生的随机数应一致。"""
        set_seed(42)
        a = np.random.rand()
        b = np.random.rand()

        set_seed(42)
        c = np.random.rand()
        d = np.random.rand()

        assert a == c, "第一次 rand 不匹配——set_seed 未生效"
        assert b == d, "第二次 rand 不匹配——set_seed 未生效"

    def test_worker_init_fn_no_error(self):
        """UT-02: worker_init_fn 不应抛出异常。"""
        import torch
        torch.manual_seed(42)
        worker_init_fn(0)  # 不应 raise
