# -*- coding: utf-8 -*-
"""合成样本数据 (几何随机游走), 用于自检与演示。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample_data(n: int = 800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.015, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    span = np.abs(rng.normal(0, 0.012, n)) + 0.002
    high = np.maximum(open_, close) * (1 + span)
    low = np.minimum(open_, close) * (1 - span)
    volume = np.abs(rng.lognormal(10, 0.6, n)) * (1 + np.abs(rng.normal(0, 2, n)) * (np.abs(ret) > 0.01))
    out = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
    out.index = pd.bdate_range("2025-01-01", periods=n)
    return out
