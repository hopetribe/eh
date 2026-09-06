# -*- coding: utf-8 -*-
"""GCN 「金筹九转」 核心库 · TDX/富途公式函数库

逐个对应通达信/富途公式内置函数 (MA/EMA/SMA/HHV/LLV/STDP/STD/REF/CROSS/
BARSLAST/BARSLASTCOUNT/BACKSET/COUNT/IF/BETWEEN/ISLASTBAR ...), 是所有
指标配方 (recipes) 的原子算子层。

语义约定
--------
- 无效值: 窗口不足或除零产生 NaN, 参与比较运算时视为 False (与 TDX 一致);
- SMA(X,N,M): Y=(M*X+(N-M)*Y')/N —— 已向量化为 ewm(alpha=M/N), 与逐行
  递归实现逐值等价 (见 tests/test_tdx.py 的等价性断言);
- BARSLAST/BARSLASTCOUNT/BACKSET 均已向量化, BACKSET 的周期参数为序列时
  自动回退逐行实现。

新增基础算子: 在本模块实现纯函数并加入 __all__ 即可被所有配方使用。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _common_index(*values) -> pd.Index:
    """Return a stable outer index for element-wise operands."""
    indexes = [value.index for value in values if isinstance(value, pd.Series)]
    if not indexes:
        return pd.RangeIndex(1)
    index = indexes[0]
    for other in indexes[1:]:
        if not index.equals(other):
            index = index.union(other, sort=False)
    return index


def _as_series(x, index=None) -> pd.Series:
    """转为 float Series (标量自动广播)。"""
    if isinstance(x, pd.Series):
        if index is not None and not x.index.equals(index):
            x = x.reindex(index)
        return x.astype(float)
    if index is None:
        index = pd.RangeIndex(1)
    return pd.Series(float(x), index=index)


def _as_bool(x, index=None) -> pd.Series:
    """转为 bool Series; NaN 视为 False (对应 TDX 无效值参与逻辑运算)。"""
    if isinstance(x, pd.Series):
        if index is not None and not x.index.equals(index):
            x = x.reindex(index)
        values = [False if pd.isna(value) else bool(value) for value in x.to_numpy()]
        return pd.Series(values, index=x.index, dtype=bool)
    value = False if pd.isna(x) else bool(x)
    if index is None:
        index = pd.RangeIndex(1)
    return pd.Series(value, index=index, dtype=bool)


def safe_div(a, b) -> pd.Series:
    """除法: 分母为 0 时返回 NaN (对应 TDX 运算无效值), 而不是 inf。"""
    index = _common_index(a, b)
    a = _as_series(a, index)
    b = _as_series(b, index)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    return r.replace([np.inf, -np.inf], np.nan)


def MA(x, n) -> pd.Series:
    """简单移动平均, 窗口不足 N 时为无效值 (与 TDX 一致)。"""
    x = _as_series(x)
    return x.rolling(int(math.floor(n)), min_periods=int(math.floor(n))).mean()


def EMA(x, n) -> pd.Series:
    """指数移动平均；从首个有效值起算，内部 NaN 原位保留且不推进递归。"""
    x = _as_series(x)
    k = int(n)
    if k <= 0:
        raise ValueError("EMA period must be positive")
    alpha = 2.0 / (k + 1.0)
    values = x.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    last = np.nan
    for i, value in enumerate(values):
        if np.isnan(value):
            continue
        last = value if np.isnan(last) else alpha * value + (1.0 - alpha) * last
        out[i] = last
    return pd.Series(out, index=x.index)


def SMA(x, n, m) -> pd.Series:
    """TDX 平滑移动平均: Y = (X*M + Y'*(N-M)) / N, 首个有效值处 Y=X。"""
    x = _as_series(x)
    a = x.to_numpy(dtype=float)
    y = np.full(len(a), np.nan)
    n, m = float(n), float(m)
    last = np.nan
    for i in range(len(a)):
        xv = a[i]
        if np.isnan(xv):
            continue  # 无效值跳过, 递归从首个有效值开始
        if np.isnan(last):
            y[i] = xv
        else:
            y[i] = (m * xv + (n - m) * last) / n
        last = y[i]
    return pd.Series(y, index=x.index)


def HHV(x, n) -> pd.Series:
    """N 周期内最高值 (窗口不足为无效值)。"""
    x = _as_series(x)
    k = int(math.floor(n))
    if k <= 0:  # TDX: N=0 表示从头开始
        return x.expanding().max()
    return x.rolling(k, min_periods=k).max()


def LLV(x, n) -> pd.Series:
    """N 周期内最低值 (窗口不足为无效值)。"""
    x = _as_series(x)
    k = int(math.floor(n))
    if k <= 0:
        return x.expanding().min()
    return x.rolling(k, min_periods=k).min()


def STDP(x, n) -> pd.Series:
    """总体标准差 (ddof=0), 对应 TDX STDP。"""
    x = _as_series(x)
    k = int(math.floor(n))
    return x.rolling(k, min_periods=k).std(ddof=0)


def STD(x, n) -> pd.Series:
    """样本标准差 (ddof=1), 对应 TDX STD。"""
    x = _as_series(x)
    k = int(math.floor(n))
    return x.rolling(k, min_periods=k).std(ddof=1)


def REF(x, n) -> pd.Series:
    """向前引用: N 个周期前的值。"""
    return _as_series(x).shift(int(n))


def MAXA(a, b) -> pd.Series:
    """逐元素最大值 (对应 TDX MAX); 任一入参无效则结果无效。"""
    index = _common_index(a, b)
    a = _as_series(a, index)
    b = _as_series(b, index)
    pair = pd.concat([a, b], axis=1)
    return pair.max(axis=1).where(pair.notna().all(axis=1))


def ABS_(x) -> pd.Series:
    return _as_series(x).abs()


def POW_(x, p) -> pd.Series:
    index = _common_index(x, p)
    return _as_series(x, index).pow(_as_series(p, index))


def IF(cond, a, b) -> pd.Series:
    """条件赋值: cond 为真取 a, 否则取 b (无效条件视为假)。"""
    index = _common_index(cond, a, b)
    cond = _as_bool(cond, index)
    a = _as_series(a, index)
    b = _as_series(b, index)
    return pd.Series(np.where(cond.to_numpy(), a.to_numpy(), b.to_numpy()), index=index)


def CROSS(a, b) -> pd.Series:
    """上穿: A>B 且 前一周期 A<=B。"""
    index = _common_index(a, b)
    a = _as_series(a, index)
    b = _as_series(b, index)
    prev_a, prev_b = a.shift(1), b.shift(1)
    return (a > b) & (prev_a <= prev_b)


def BETWEEN(a, b, c) -> pd.Series:
    """A 介于 B 和 C 之间 (闭区间, 不分大小端)。"""
    index = _common_index(a, b, c)
    a = _as_series(a, index)
    b = _as_series(b, index)
    c = _as_series(c, index)
    return ((a >= b) & (a <= c)) | ((a >= c) & (a <= b))


def BARSLAST(cond) -> pd.Series:
    """上一次条件成立到当前的周期数; 当期成立为 0; 从未成立为 NaN。

    向量化: np.maximum.accumulate 定位最近一次成立位置 (tests 有等价断言)。
    """
    b = _as_bool(cond)
    c = b.to_numpy()
    idx = np.arange(len(c))
    last_true = np.where(c, idx, -1)
    np.maximum.accumulate(last_true, out=last_true)
    out = (idx - last_true).astype(float)
    out[last_true < 0] = np.nan
    return pd.Series(out, index=b.index)


def BARSLASTCOUNT(cond) -> pd.Series:
    """连续满足条件的周期数 (当期不满足则为 0)。

    向量化: 最近一次"不满足"位置 + 前向填充 (tests 有等价断言)。
    """
    b = _as_bool(cond)
    c = b.to_numpy()
    idx = np.arange(len(c))
    last_false = np.where(~c, idx, -1)
    np.maximum.accumulate(last_false, out=last_false)
    out = np.where(c, idx - np.maximum(last_false, -1), 0)
    return pd.Series(out.astype(np.int64), index=b.index)


def COUNT(cond, n) -> pd.Series:
    """最近 N 周期内条件成立的次数 (起始不足 N 按已有数据统计)。"""
    b = _as_bool(cond).astype(float)
    k = int(n)
    if k <= 0:  # TDX: N=0 表示从第一根有效数据累计
        return b.expanding(min_periods=1).sum()
    return b.rolling(k, min_periods=1).sum()


def BACKSET(cond, n) -> pd.Series:
    """条件满足时, 将向前 N 个周期(含当期)置为 1。

    标量 N 向量化 (前视滑动窗口计数); N 为序列时回退逐行实现
    (EHOPT10 九转标注的 BACKSET(..., NINE2_UP_COUNT) 用到序列周期)。
    """
    b = _as_bool(cond)
    c = b.to_numpy()
    if isinstance(n, pd.Series):  # 变量周期: 逐行回退
        n_arr = n.to_numpy()
        out = np.zeros(len(c), dtype=bool)
        for i in range(len(c)):
            if c[i]:
                k = max(int(n_arr[i]), 1)
                out[max(0, i - k + 1): i + 1] = True
        return pd.Series(out, index=b.index)
    k = max(int(n), 1)
    cum = np.cumsum(c)
    j = np.arange(len(c))
    right = np.minimum(j + k - 1, len(c) - 1)
    win_true = cum[right] - np.where(j == 0, 0, cum[j - 1])
    return pd.Series(win_true > 0, index=b.index)


def ISLASTBAR(index) -> pd.Series:
    """最后一根 K 线为 True。"""
    out = np.zeros(len(index), dtype=bool)
    if len(out):
        out[-1] = True
    return pd.Series(out, index=index)


def AND(a, b):
    index = _common_index(a, b)
    return _as_bool(a, index) & _as_bool(b, index)


def OR_(a, b):
    index = _common_index(a, b)
    return _as_bool(a, index) | _as_bool(b, index)


def NOT(a):
    return ~_as_bool(a)

__all__ = [
    "_as_series", "_as_bool", "safe_div", "MA", "EMA", "SMA", "HHV", "LLV",
    "STDP", "STD", "REF", "MAXA", "ABS_", "POW_", "IF", "CROSS", "BETWEEN",
    "BARSLAST", "BARSLASTCOUNT", "COUNT", "BACKSET", "ISLASTBAR",
    "AND", "OR_", "NOT",
]
