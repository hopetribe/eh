# -*- coding: utf-8 -*-
"""TDX 算子测试: 向量化实现 vs 参考逐行实现 等价性 + 基础语义。"""
import numpy as np
import pandas as pd

from gcn.core.tdx import (BACKSET, BARSLAST, BARSLASTCOUNT, CROSS, EMA, MA,
                          REF, SMA, STDP, STD, BETWEEN)


def _ref_sma(x, n, m):
    x = pd.Series(x, dtype=float)
    out = np.full(len(x), np.nan)
    last = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        out[i] = v if np.isnan(last) else (m * v + (n - m) * last) / n
        last = out[i]
    return out


def _ref_barslast(c):
    out = np.full(len(c), np.nan)
    last = -1
    for i in range(len(c)):
        if c[i]:
            last = i
        if last >= 0:
            out[i] = i - last
    return out


def _ref_barslastcount(c):
    out = np.zeros(len(c), dtype=int)
    run = 0
    for i in range(len(c)):
        run = run + 1 if c[i] else 0
        out[i] = run
    return out


def _ref_backset(c, n):
    out = np.zeros(len(c), dtype=bool)
    for i in range(len(c)):
        if c[i]:
            out[max(0, i - n + 1):i + 1] = True
    return out


def test_sma_matches_recursion():
    rng = np.random.default_rng(7)
    for n, m in ((3, 1), (12, 2), (30, 5)):
        x = pd.Series(100 + rng.normal(0, 5, 500))
        got = SMA(x, n, m).to_numpy()
        want = _ref_sma(x, n, m)
        assert np.allclose(got, want, equal_nan=True), (n, m)


def test_sma_leading_nan():
    x = pd.Series([np.nan, np.nan, 1.0, 2.0, 3.0])
    got = SMA(x, 3, 1).to_numpy()
    assert np.isnan(got[:2]).all() and np.isclose(got[2], 1.0)


def test_ma_ema_stdp_std_ref():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(0, 1, 200))
    assert np.allclose(MA(x, 10).dropna(), x.rolling(10).mean().dropna())
    assert np.allclose(EMA(x, 12).to_numpy(),
                       x.ewm(span=12, adjust=False).mean().to_numpy())
    assert np.allclose(STDP(x, 20).dropna(), x.rolling(20).std(ddof=0).dropna())
    assert np.allclose(STD(x, 20).dropna(), x.rolling(20).std(ddof=1).dropna())
    assert np.allclose(REF(x, 3).dropna(), x.shift(3).dropna())


def test_cross_between():
    up = pd.Series([1.0, 1, 2, 1, 4])
    dn = pd.Series([2.0, 1, 1, 3, 3])
    assert CROSS(up, dn).tolist() == [False, False, True, False, True]
    assert BETWEEN(pd.Series([5.0, 1, 9, 8]), 2, 8).tolist() == [True, False, False, True]


def test_barslast_vector_matches_recursion():
    rng = np.random.default_rng(3)
    for _ in range(30):
        c = rng.random(200) > 0.6
        s = pd.Series(c)
        got = BARSLAST(s).to_numpy()
        want = _ref_barslast(c)
        assert all((np.isnan(g) and np.isnan(w)) or g == w for g, w in zip(got, want))


def test_barslastcount_vector_matches_recursion():
    rng = np.random.default_rng(4)
    for _ in range(30):
        c = rng.random(200) > 0.5
        got = BARSLASTCOUNT(pd.Series(c)).to_numpy()
        assert (got == _ref_barslastcount(c)).all()


def test_backset_scalar_vector_matches_recursion():
    rng = np.random.default_rng(5)
    for n in (1, 3, 7):
        for _ in range(10):
            c = rng.random(150) > 0.7
            got = BACKSET(pd.Series(c), n).to_numpy()
            assert (got == _ref_backset(c, n)).all()


def test_backset_series_period():
    c = pd.Series([False, False, True, False])
    n = pd.Series([1, 1, 3, 1])
    # i=2 且 n=3 -> 标记 [0,1,2]
    assert BACKSET(c, n).tolist() == [True, True, True, False]
