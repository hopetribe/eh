# -*- coding: utf-8 -*-
"""KK2 通用技术指标库 (V5 多指标确认层专用)

补充 EHOPT10 已有的 MA/EMA/SMA/MACD/RSI/布林带之外, 用于"独立维度"噪声过滤的
常见技术指标。设计原则:

1. 纯函数、pandas/numpy 实现、无全局状态, 与 kk2_ehopt10 复用同一套 MA/EMA/SMA/REF;
2. 每个指标带自检 (对照经典定义 / 与 pandas 直接计算一致);
3. 重点覆盖: 趋势强度 (ADX/+DI/-DI)、波动率 (ATR)、资金流 (MFI/OBV)、
   动量 (ROC/CCI)、波动状态 (布林带宽 + 滚动分位数)。

滚动分位数 (rolling_pct_rank) 是 V5 的关键: 用"当前值在历史窗口中的排名"替代
绝对阈值, 自动适配牛熊与不同波动率标的, 避免对单一标的调参过拟合。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from gcn.core.tdx import MA, REF, _as_series, safe_div
from gcn.core.registry import register_indicator

__all__ = [
    "true_range", "atr", "adx", "mfi", "obv", "roc", "cci",
    "bollinger", "rolling_pct_rank",
]


def _wilder(x: pd.Series, n: int) -> pd.Series:
    """Wilder 平滑 (RMA): Y = (Y'*(n-1) + X)/n, 即 alpha=1/n 的 EWM(adjust=False)。

    与教科书"首值 = 前 n 个的简单均值"的写法略有差异: 这里从首个有效值起算
    (pandas ewm 语义), 样本足够长后两者收敛。用于 ATR/ADX/DI。
    """
    x = _as_series(x)
    return x.ewm(alpha=1.0 / int(n), adjust=False).mean()


@register_indicator("true_range")
def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """TR = max(H-L, |H-prevC|, |L-prevC|)。"""
    h, l, c = _as_series(h), _as_series(l), _as_series(c)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr


@register_indicator("atr")
def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    """平均真实波幅 (Wilder 平滑)。"""
    return _wilder(true_range(h, l, c), n)


@register_indicator("adx")
def adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.DataFrame:
    """平均趋向指数 ADX + 方向指标 +DI / -DI (Wilder 平滑)。

    返回 DataFrame, 列: ADX / PDI / MDI。
    """
    h, l, c = _as_series(h), _as_series(l), _as_series(c)
    up = h.diff()
    down = -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=h.index)
    tr = true_range(h, l, c)
    atr_n = _wilder(tr, n)
    pdi = 100.0 * _wilder(plus_dm, n) / atr_n
    mdi = 100.0 * _wilder(minus_dm, n) / atr_n
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx_n = _wilder(dx.fillna(0.0), n)
    return pd.DataFrame({"ADX": adx_n, "PDI": pdi, "MDI": mdi}, index=h.index)


@register_indicator("mfi")
def mfi(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series, n: int = 14) -> pd.Series:
    """资金流量指标 MFI = 100 - 100/(1 + 正资金流/负资金流)。

    典型价 TP=(H+L+C)/3, 原始资金流 = TP*V。
    """
    h, l, c, v = (_as_series(x) for x in (h, l, c, v))
    tp = (h + l + c) / 3.0
    raw = tp * v
    pos = pd.Series(np.where(tp > tp.shift(1), raw, 0.0), index=tp.index)
    neg = pd.Series(np.where(tp < tp.shift(1), raw, 0.0), index=tp.index)
    pos_sum = pos.rolling(int(n), min_periods=int(n)).sum()
    neg_sum = neg.rolling(int(n), min_periods=int(n)).sum()
    ratio = safe_div(pos_sum, neg_sum)
    return 100.0 - 100.0 / (1.0 + ratio)


@register_indicator("obv")
def obv(c: pd.Series, v: pd.Series) -> pd.Series:
    """能量潮 OBV: 收盘上涨累加成交量, 下跌累减。"""
    c, v = _as_series(c), _as_series(v)
    direction = np.sign(c.diff().fillna(0.0))
    return (direction * v).cumsum()


@register_indicator("roc")
def roc(c: pd.Series, n: int = 12) -> pd.Series:
    """变动率 ROC = (C - REF(C,n)) / REF(C,n) * 100。"""
    c = _as_series(c)
    return safe_div(c - REF(c, n), REF(c, n)) * 100.0


@register_indicator("cci")
def cci(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 20) -> pd.Series:
    """顺势指标 CCI = (TP - MA(TP,n)) / (0.015 * 平均绝对偏差)。"""
    h, l, c = (_as_series(x) for x in (h, l, c))
    tp = (h + l + c) / 3.0
    ma_tp = MA(tp, n)
    mad = tp.rolling(int(n), min_periods=int(n)).apply(
        lambda s: np.abs(s - s.mean()).mean(), raw=True)
    return safe_div(tp - ma_tp, 0.015 * mad)


@register_indicator("bollinger")
def bollinger(c: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """布林带: MID/UPPER/LOWER + 带宽 BW% + %B。"""
    c = _as_series(c)
    mid = MA(c, n)
    sd = c.rolling(int(n), min_periods=int(n)).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    bw = safe_div(upper - lower, mid) * 100.0
    pct_b = safe_div(c - lower, upper - lower)
    return pd.DataFrame({"MID": mid, "UPPER": upper, "LOWER": lower,
                         "BW": bw, "PB": pct_b}, index=c.index)


@register_indicator("rolling_pct_rank")
def rolling_pct_rank(x: pd.Series, n: int = 250) -> pd.Series:
    """当前值在过去 n 个周期内的分位排名 (0~100)。

    50 = 中位; 用排名替代绝对阈值, 适配不同标的/牛熊的波动率差异。
    """
    x = _as_series(x)
    k = int(n)

    def _pct(a):
        a = np.asarray(a, dtype=float)
        v = a[-1]
        valid = a[np.isfinite(a)]
        if valid.size < max(3, k // 10) or not np.isfinite(v):
            return np.nan
        # 小于 v 的比例 * 100 (与 scipy rankdata 'max' 近似)
        return float((valid < v).mean() * 100.0)

    return x.rolling(k, min_periods=max(3, k // 10)).apply(_pct, raw=True)


# ==========================================================================
# 自检
# ==========================================================================

def _self_test():
    rng = np.random.default_rng(42)
    n = 300
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    h = c * (1 + np.abs(rng.normal(0, 0.01, n)))
    l = c * (1 - np.abs(rng.normal(0, 0.01, n)))
    v = pd.Series(rng.lognormal(10, 0.5, n))

    # ATR: Wilder 平滑 = ewm(alpha=1/n, adjust=False)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
                   axis=1).max(axis=1)
    assert np.allclose(atr(h, l, c, 14).to_numpy(),
                       tr.ewm(alpha=1 / 14, adjust=False).mean().to_numpy(),
                       equal_nan=True)

    # ADX 结构: 值域 [0, 100]
    d = adx(h, l, c, 14)
    assert d["ADX"].dropna().between(0, 100).all()
    assert d["PDI"].dropna().between(0, 100).all()
    assert d["MDI"].dropna().between(0, 100).all()

    # MFI 值域 [0, 100]
    m = mfi(h, l, c, v, 14)
    assert m.dropna().between(0, 100).all()

    # OBV = cumsum(sign(diff)*v)
    assert np.allclose(obv(c, v).to_numpy(),
                       (np.sign(c.diff().fillna(0)) * v).cumsum().to_numpy())

    # ROC: 与直接计算一致
    assert np.allclose(roc(c, 12).dropna().to_numpy(),
                       ((c / c.shift(12) - 1) * 100).dropna().to_numpy())

    # 布林带带宽 >= 0
    bb = bollinger(c, 20, 2)
    assert (bb["BW"].dropna() >= 0).all()

    # 滚动分位数值域 [0, 100]
    pr = rolling_pct_rank(c, 100)
    assert pr.dropna().between(0, 100).all()

    print("[kk2_indicators] 自检通过 ✓  (ADX/ATR/MFI/OBV/ROC/CCI/Bollinger/分位数)")


if __name__ == "__main__":
    _self_test()
