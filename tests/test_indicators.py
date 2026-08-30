# -*- coding: utf-8 -*-
"""通用技术指标库测试 (移植自 kk2_indicators 自检 + 注册表)。"""
import numpy as np
import pandas as pd

from gcn.core import indicators
from gcn.core.registry import INDICATORS, list_indicators


def _mk(n=300, seed=42):
    rng = np.random.default_rng(seed)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    h = c * (1 + np.abs(rng.normal(0, 0.01, n)))
    l = c * (1 - np.abs(rng.normal(0, 0.01, n)))
    v = pd.Series(rng.lognormal(10, 0.5, n))
    return h, l, c, v


def test_atr_wilder_ewm_equivalence():
    h, l, c, _ = _mk()
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    assert np.allclose(indicators.atr(h, l, c, 14), tr.ewm(alpha=1 / 14, adjust=False).mean(),
                       equal_nan=True)


def test_adx_mfi_value_range():
    h, l, c, v = _mk()
    d = indicators.adx(h, l, c, 14)
    assert d["ADX"].dropna().between(0, 100).all()
    assert indicators.mfi(h, l, c, v, 14).dropna().between(0, 100).all()


def test_obv_roc_cci_bollinger():
    h, l, c, v = _mk()
    assert (indicators.obv(c, v).diff().dropna() != 0).any()
    assert np.isfinite(indicators.roc(c, 10).dropna()).all()
    assert np.isfinite(indicators.cci(h, l, c, 20).dropna()).all()
    bb = indicators.bollinger(c, 20, 2)
    assert (bb["BW"].dropna() >= 0).all()
    assert bb["PB"].dropna().between(-1, 2).all()


def test_rolling_pct_rank_range():
    _, _, c, _ = _mk()
    pr = indicators.rolling_pct_rank(c, 100)
    assert pr.dropna().between(0, 100).all()


def test_registry():
    for name in ("adx", "atr", "mfi", "obv", "roc", "cci", "bollinger"):
        assert name in INDICATORS
        assert callable(INDICATORS[name])
    assert list_indicators() == sorted(list_indicators())
