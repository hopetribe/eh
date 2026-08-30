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


# ---------------- 富途官方指标扩展 ----------------

def test_bbi_macd_bias_identity():
    _, _, c, _ = _mk()
    b = indicators.bbi(c)
    want = (c.rolling(3).mean() + c.rolling(6).mean()
            + c.rolling(12).mean() + c.rolling(24).mean()) / 4.0
    assert np.allclose(b.dropna(), want.dropna())
    m = indicators.macd(c)
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    assert np.allclose(m["DIF"].dropna(), dif.dropna())
    assert np.allclose(m["MACD"].dropna(), ((dif - m["DEA"]) * 2).dropna())
    bi = indicators.bias(c, 6)
    ma6 = c.rolling(6).mean()
    assert np.allclose(bi.dropna(), ((c - ma6) / ma6 * 100).dropna())


def test_kdj_wr_range_and_identity():
    h, l, c, _ = _mk()
    k = indicators.kdj(h, l, c)
    assert k["K"].dropna().between(-20, 120).all()
    assert np.allclose(k["J"].dropna(), (3 * k["K"] - 2 * k["D"]).dropna())
    w = indicators.wr(h, l, c, 10)
    assert w.dropna().between(0, 100).all()


def test_sar_trend_side():
    # 定义性质: 单边趋势中 SAR 始终位于价格另一侧且不反转
    n = 100
    up_c = pd.Series(np.linspace(50.0, 100.0, n))
    s_up = indicators.sar(up_c * 1.01, up_c * 0.99)
    assert s_up.notna().all()
    assert (s_up.iloc[5:] < up_c.iloc[5:]).mean() > 0.95  # 上涨: SAR 在下方
    dn_c = pd.Series(np.linspace(100.0, 50.0, n))
    s_dn = indicators.sar(dn_c * 1.01, dn_c * 0.99)
    assert (s_dn.iloc[5:] > dn_c.iloc[5:]).mean() > 0.95  # 下跌: SAR 在上方


def test_trix_dma_mtm_psy():
    _, _, c, _ = _mk()
    assert np.isfinite(indicators.trix(c, 12).dropna()).all()
    d = indicators.dma(c)
    want = c.rolling(10).mean() - c.rolling(50).mean()
    assert np.allclose(d["DIF"].dropna(), want.dropna())
    m = indicators.mtm(c)
    assert np.allclose(m["MTMMA"].dropna(), m["MTM"].rolling(6).mean().dropna())
    p = indicators.psy(c, 12)
    assert p.dropna().between(0, 100).all()


def test_vr_emv_vpt_volume_ratio():
    h, l, c, v = _mk()
    r = indicators.vr(h, l, c, v, 26)
    assert r.dropna().between(0, 1e5).all()
    e = indicators.emv(h, l, v)
    assert np.isfinite(e["EMV"].dropna()).all()
    vpt = indicators.vpt(c, v)
    # VPT 为累计量, 与分段和一致
    seg = vpt.iloc[10] + (v * c.pct_change()).iloc[11:21].sum()
    assert np.isclose(vpt.iloc[20], seg)
    vr_ = indicators.volume_ratio(v, 5)
    want = v / v.rolling(5).mean().shift(1)
    assert np.allclose(vr_.dropna(), want.dropna())


def test_arbr_cr():
    h, l, c, v = _mk()
    import pandas as pd
    o = (h + l + c) / 3.0  # 用典型价代替开盘价做口径验证
    ab = indicators.arbr(h, l, c, o, 26)
    assert ab["AR"].dropna().between(0, 1e4).all()
    assert ab["BR"].dropna().between(0, 1e4).all()
    crv = indicators.cr(h, l, c, 26)
    assert crv.dropna().ge(0).all()
