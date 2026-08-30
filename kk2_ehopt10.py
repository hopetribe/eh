# -*- coding: utf-8 -*-
"""兼容入口: GCN 「金筹九转」 指标 (实现已迁移至 gcn/ 包)。

历史入口保持可用; 新代码请直接使用 gcn 包:
    from gcn.recipes.gcn_main import compute_ehopt10
"""
from gcn.core.tdx import *  # noqa: F401,F403 -- TDX 函数库 (模块级兼容)
from gcn.core import tdx  # noqa: F401
from gcn.data.sample import make_sample_data  # noqa: F401
from gcn.plot import plot_result  # noqa: F401
from gcn.recipes.gcn_main import (  # noqa: F401
    B_CHIP_LOW, VERSIONS, _load_ohlcv, compute_ehopt10,
)

if __name__ == "__main__":
    from gcn.recipes.gcn_main import _self_test
    _self_test()
