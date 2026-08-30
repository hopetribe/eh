# -*- coding: utf-8 -*-
"""
KK2 富途主图指标 EHOPT10 —— 纯 Python (pandas/numpy) 实现
对应公式文件: kk2-futu-main-indicator-ehopt10.txt

指标结构
--------
1. 主图布林带:  MID/UPPER/LOWER = MA(CLOSE,SD) ± WIDTH*STDP(CLOSE,SD)
2. 获利筹 (盈利筹码) 综合摆动指标
3. B/S 评分系统 (B_SCORE / S_SCORE) 与各类买卖信号:
   绝反(绝地反弹)、B_BASE_BULL、B_STAGE_SIGNAL、B_BEAR_RECOVER、B_CRASH_RECOVER、
   S_BLOWOFF、S_BEAR_RALLY、S_CROSS、S_DELAY
4. NINE2 九转系统: N 周期涨跌比较 + 量能确认 + MACD(6,13,5) + 布林带宽过滤,
   输出 ★买/★卖 文本信号, 以及九转 1-9 数字标注 (EHOPT10 新增:
   1-8 仅在"完成 9 序列"或"最后一根K线上进行中的 5-8 序列"时标注)

参数 (见富途参数表)
------------------
SD     : 20  (min 2,   max 1000) 布林带周期
WIDTH  : 2   (min 0,   max 1000) 布林带宽度倍数
N      : 4   (min 0,   max 1000) 九转比较前第 N 根K线收盘价
OFFSET : 15  (min 0,   max 1000) 九转数字标注的偏移距离(仅影响绘图位置)

用法
----
    from kk2_ehopt10 import compute_ehopt10
    res = compute_ehopt10(df, SD=20, WIDTH=2, N=4, OFFSET=15)
    # df: DataFrame, 需包含列 open/high/low/close/volume (大小写不敏感),
    #     按时间升序排列。返回的 DataFrame 索引与输入一致。

绘图(可选, 需要 matplotlib):
    python3 kk2_ehopt10.py            # 运行自检 + 合成数据示例 + 输出PNG
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# B_SCORE_1 的获利筹低位门槛 (原公式 8; 放宽可提升低点覆盖率, 见回测诊断)
B_CHIP_LOW = 8.0

__all__ = ["compute_ehopt10", "VERSIONS", "plot_result", "make_sample_data"]

# 指标版本 (供 UI 切换对比):
#   v3 = 参考指标版 (git tag v3.0.0): 绝反=原始信号, 3% 反包, 无去重
#   v4 = 回测驱动的质量优化 (当前默认):
#        - 反包K线阈值 3% -> 5% (11 标的 52 次触发诊断: 3~4% 弱反弹是主要噪音)
#        - 同标的 10 日去重 (密集重复触发降噪)
#        - 10 年验证: 52 -> 29 次触发, 20日 +8.74% -> +15.52%, 胜率 61.5% -> 75.9%
#   共同口径: ★买/★卖已弃用 (回测证据: 无预测力甚至反向); 绝反不加趋势过滤
#   (10 年验证: 熊市触发 +4.57% 优于牛市 -5.02%, 趋势过滤全部有害)
VERSIONS = ("v3", "v4")


# ==========================================================================
# 一、TDX / 富途公式函数库 (逐个对应公式中用到的内置函数)
# ==========================================================================

def _as_series(x, index=None) -> pd.Series:
    """转为 float Series (标量自动广播)。"""
    if isinstance(x, pd.Series):
        if index is not None and not x.index.equals(index):
            x = x.reindex(index)
        return x.astype(float)
    if index is None:
        raise ValueError("标量入参需要提供 index")
    return pd.Series(float(x), index=index)


def _as_bool(x) -> pd.Series:
    """转为 bool Series; NaN 视为 False (对应 TDX 无效值参与逻辑运算)。"""
    if isinstance(x, pd.Series):
        return x.fillna(False).astype(bool)
    return pd.Series(bool(x), index=None)


def safe_div(a, b) -> pd.Series:
    """除法: 分母为 0 时返回 NaN (对应 TDX 运算无效值), 而不是 inf。"""
    a = _as_series(a) if not isinstance(a, pd.Series) else a.astype(float)
    b = _as_series(b, a.index) if not isinstance(b, pd.Series) else b.astype(float).reindex(a.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    return r.replace([np.inf, -np.inf], np.nan)


def MA(x, n) -> pd.Series:
    """简单移动平均, 窗口不足 N 时为无效值 (与 TDX 一致)。"""
    x = _as_series(x)
    return x.rolling(int(math.floor(n)), min_periods=int(math.floor(n))).mean()


def EMA(x, n) -> pd.Series:
    """指数移动平均: Y = (2*X + (N-1)*Y') / (N+1), 从首个有效值起算。"""
    x = _as_series(x)
    return x.ewm(span=int(n), adjust=False).mean()


def SMA(x, n, m) -> pd.Series:
    """TDX 平滑移动平均: Y = (X*M + Y'*(N-M)) / N, 首个有效值处 Y=X。"""
    a = _as_series(x).to_numpy(dtype=float)
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
    return pd.Series(y, index=_as_series(x).index)


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
    x = _as_series(x) if isinstance(x, pd.Series) else _as_series(x, _REF_DEFAULT_INDEX)
    return x.shift(int(n))


# REF 在纯标量场景下的占位索引 (实际公式中均为序列运算, 不会用到)
_REF_DEFAULT_INDEX = pd.RangeIndex(0)


def MAXA(a, b) -> pd.Series:
    """逐元素最大值 (对应 TDX MAX); 任一入参无效则结果无效。"""
    a = a if isinstance(a, pd.Series) else _as_series(a)
    b = b if isinstance(b, pd.Series) else _as_series(b, a.index)
    pair = pd.concat([a, b], axis=1)
    return pair.max(axis=1).where(pair.notna().all(axis=1))


def ABS_(x) -> pd.Series:
    return x.abs() if isinstance(x, pd.Series) else abs(x)


def POW_(x, p) -> pd.Series:
    x = x if isinstance(x, pd.Series) else _as_series(x)
    return x.pow(float(p))


def IF(cond, a, b) -> pd.Series:
    """条件赋值: cond 为真取 a, 否则取 b (无效条件视为假)。"""
    cond = _as_bool(cond)
    idx = cond.index
    a_arr = a.to_numpy(dtype=float) if isinstance(a, pd.Series) else np.full(len(idx), float(a))
    b_arr = b.to_numpy(dtype=float) if isinstance(b, pd.Series) else np.full(len(idx), float(b))
    if isinstance(a, pd.Series) and not a.index.equals(idx):
        a_arr = a.reindex(idx).to_numpy(dtype=float)
    if isinstance(b, pd.Series) and not b.index.equals(idx):
        b_arr = b.reindex(idx).to_numpy(dtype=float)
    return pd.Series(np.where(cond.to_numpy(), a_arr, b_arr), index=idx)


def CROSS(a, b) -> pd.Series:
    """上穿: A>B 且 前一周期 A<=B。"""
    a = a if isinstance(a, pd.Series) else _as_series(a)
    b = b if isinstance(b, pd.Series) else _as_series(b, a.index)
    prev_a, prev_b = a.shift(1), b.shift(1)
    return (a > b) & (prev_a <= prev_b)


def BETWEEN(a, b, c) -> pd.Series:
    """A 介于 B 和 C 之间 (闭区间, 不分大小端)。"""
    a = a if isinstance(a, pd.Series) else _as_series(a)
    b = b if isinstance(b, pd.Series) else _as_series(b, a.index)
    c = c if isinstance(c, pd.Series) else _as_series(c, a.index)
    return ((a >= b) & (a <= c)) | ((a >= c) & (a <= b))


def BARSLAST(cond) -> pd.Series:
    """上一次条件成立到当前的周期数; 当期成立为 0; 从未成立为 NaN。"""
    c = _as_bool(cond).to_numpy()
    out = np.full(len(c), np.nan)
    last_true = -1
    for i in range(len(c)):
        if c[i]:
            last_true = i
        if last_true >= 0:
            out[i] = i - last_true
    return pd.Series(out, index=_as_bool(cond).index)


def BARSLASTCOUNT(cond) -> pd.Series:
    """连续满足条件的周期数 (当期不满足则为 0)。"""
    b = _as_bool(cond)
    c = b.to_numpy()
    out = np.zeros(len(c), dtype=int)
    run = 0
    for i in range(len(c)):
        run = run + 1 if c[i] else 0
        out[i] = run
    return pd.Series(out, index=b.index)


def COUNT(cond, n) -> pd.Series:
    """最近 N 周期内条件成立的次数 (起始不足 N 按已有数据统计)。"""
    b = _as_bool(cond).astype(float)
    return b.rolling(int(n), min_periods=1).sum()


def BACKSET(cond, n) -> pd.Series:
    """条件满足时, 将向前 N 个周期(含当期)置为 1。

    与 TDX 不同处仅在于: 这里支持 N 为序列 (EHOPT10 公式里
    BACKSET(NINE2_UP_LIVE>0, NINE2_UP_COUNT) 的周期数是变量)。
    """
    b = _as_bool(cond)
    c = b.to_numpy()
    n_arr = n.to_numpy() if isinstance(n, pd.Series) else None
    out = np.zeros(len(c), dtype=bool)
    for i in range(len(c)):
        if c[i]:
            k = int(n_arr[i]) if n_arr is not None else int(n)
            k = max(k, 1)
            out[max(0, i - k + 1): i + 1] = True
    return pd.Series(out, index=b.index)


def ISLASTBAR(index) -> pd.Series:
    """最后一根 K 线为 True。"""
    out = np.zeros(len(index), dtype=bool)
    if len(out):
        out[-1] = True
    return pd.Series(out, index=index)


def AND(a, b):
    return _as_bool(a) & _as_bool(b)


def OR_(a, b):
    return _as_bool(a) | _as_bool(b)


def NOT(a):
    return ~_as_bool(a)


# ==========================================================================
# 二、EHOPT10 指标主体
# ==========================================================================

def _load_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """列名归一化为 open/high/low/close/volume (大小写不敏感)。"""
    m = {c.lower(): c for c in df.columns}
    need = ["open", "high", "low", "close", "volume"]
    missing = [k for k in need if k not in m]
    if missing:
        raise ValueError(f"输入数据缺少列: {missing} (需要 open/high/low/close/volume)")
    out = pd.DataFrame(
        {k: df[m[k]].astype(float) for k in need},
        index=df.index,
    )
    return out


def compute_ehopt10(df: pd.DataFrame,
                    SD: int = 20,
                    WIDTH: float = 2,
                    N: int = 4,
                    OFFSET: int = 15,
                    version: str = "v4") -> pd.DataFrame:
    """计算 KK2 EHOPT10 主图指标。

    参数与富途指标参数表一致:
      SD=20, WIDTH=2, N=4, OFFSET=15 (默认值见参数表截图)
      version: "v4" 优化版 (默认, 绝反 5%反包+10日去重) | "v3" 参考指标版

    返回 DataFrame, 主要输出列:
      DIS/MID/UPPER/LOWER          主图布林带
      获利筹                       盈利筹码摆动指标 (V2), V3 为其 SMA 平滑
      DIF/DEA/MACD/RSI1            常用衍生指标 (供信号使用)
      绝反                         绝地反弹信号 (20/0)
      B_SCORE/S_SCORE              买卖评分
      B_CONDITION/S_CONDITION ...  各条件
      B_SIGNAL/S_SIGNAL            图标信号 (DRAWICON 7/8)
      NINE2_BUY_SIGNAL/...         九转 ★买/★卖 (DRAWTEXT)
      NINE2_UP_COUNT/NINE2_UP_LABEL/NINE2_UP_9      上九转计数与标注
      NINE2_DOWN_COUNT/NINE2_DOWN_LABEL/NINE2_DOWN_9 下九转计数与标注
      ICON_JUEFAN                  DRAWICON(绝反...,LOW,34)
    """
    data = _load_ohlcv(df)
    OPEN, HIGH, LOW, CLOSE, VOLA = (data[k] for k in ["open", "high", "low", "close", "volume"])
    idx = data.index

    # ---------- 主图布林带 ----------
    # DIS:=STDP(CLOSE,SD);
    # MID:MA(CLOSE,SD);
    # UPPER:MID+WIDTH*DIS;  LOWER:MID-WIDTH*DIS;
    DIS = STDP(CLOSE, SD)
    MID = MA(CLOSE, SD)
    UPPER = MID + WIDTH * DIS
    LOWER = MID - WIDTH * DIS

    # ---------- 获利筹综合摆动 ----------
    VAR1 = safe_div(CLOSE - LLV(LOW, 13), HHV(HIGH, 13) - LLV(LOW, 13)) * 100
    VAR2 = safe_div(CLOSE - LLV(LOW, 9), HHV(HIGH, 9) - LLV(LOW, 9)) * 100
    VAR3 = (SMA(VAR2, 3, 1) - 18) * 1.55
    VAR4 = SMA(VAR3, 3, 1)
    VAR5 = safe_div(CLOSE - LLV(LOW, 100), HHV(HIGH, 100) - LLV(LOW, 100)) * 100
    VAR6 = SMA(VAR5, 10, 1)
    VAR7 = (EMA(SMA(VAR6, 8, 1), 34) - 25) * 2.6
    VAR8 = POW_(MA(CLOSE, 5), 2) + MA(CLOSE, 5)
    VAR9 = POW_(MA(LOW, 5), 2) + MA(LOW, 5)
    VARA = POW_(MA(HIGH, 5), 2) + MA(HIGH, 5)
    VARB = (safe_div(VAR8 - LLV(VAR9, 64), HHV(VARA, 64) - LLV(VAR9, 64)) * 150 + 65 - 10) / 2
    VARC = SMA(VARB, 3, 1) * 1.5 - 46
    VARD = SMA(VARC, 3, 1)
    VARE = 3 * VARC - 2 * VARD
    VARF = (VAR1 + VAR2 + VAR5) / 3
    VAR10 = ((VAR3 + VAR3 + VAR6 + VARC) / 4 - 15) * 1.67
    VAR11 = ((VAR4 + VAR4 + VAR7 + VARD) / 4 - 15) * 1.67
    VAR12 = ((VAR1 + VAR5 * 2 + VARF + VAR10 + VAR11 + VARB + VARC + VARD + VARE) / 10 - 15) * 1.67
    VAR13 = (HHV(HIGH, 5) - CLOSE) / (HHV(HIGH, 5) - LLV(LOW, 5)) * (-1) + 0.9
    VAR14 = (HHV(HIGH, 10) - CLOSE) / (HHV(HIGH, 10) - LLV(LOW, 10)) * (-1) + 0.92
    VAR15 = (HHV(HIGH, 15) - CLOSE) / (HHV(HIGH, 15) - LLV(LOW, 15)) * (-1) + 0.93
    VAR16 = (HHV(HIGH, 55) - CLOSE) / (HHV(HIGH, 55) - LLV(LOW, 55)) * (-1) + 0.94
    VAR17 = (HHV(HIGH, 89) - CLOSE) / (HHV(HIGH, 89) - LLV(LOW, 89)) * (-1) + 0.95
    VAR18 = (HHV(HIGH, 120) - CLOSE) / (HHV(HIGH, 120) - LLV(LOW, 120)) * (-1) + 0.91
    VAR19 = (HHV(HIGH, 180) - CLOSE) / (HHV(HIGH, 180) - LLV(LOW, 180)) * (-1) + 0.96
    VAR1A = ((VAR13 * 8 + VAR14 * 8 + VAR15 * 8 + VAR16 + VAR17 + VAR18 + VAR19) / 28 - 0.1) * 185
    VAR1B = SMA(VAR1A, 3, 1)
    VAR1C = (SMA(VAR1B, 8, 1) - 8) * 1.18
    # 获利筹:=(VAR3+VAR6+VAR1B*2)/4;
    profit_chip = (VAR3 + VAR6 + VAR1B * 2) / 4

    V2 = profit_chip
    V3 = SMA(profit_chip, 3, 1)

    # ---------- MACD (12,26,9) ----------
    SHORT, LONG, M = 12, 26, 9
    DIF = EMA(CLOSE, SHORT) - EMA(CLOSE, LONG)
    DEA = EMA(DIF, M)
    MACD = (DIF - DEA) * 2

    # ---------- RSI (6,12,24) ----------
    P1, P2, P3 = 6, 12, 24
    LC = REF(CLOSE, 1)
    TEMP1 = MAXA(CLOSE - LC, 0)  # MAX(CLOSE-LC,0)
    TEMP2 = (CLOSE - LC).abs()
    RSI1 = safe_div(SMA(TEMP1, P1, 1), SMA(TEMP2, P1, 1)) * 100
    RSI2 = safe_div(SMA(TEMP1, P2, 1), SMA(TEMP2, P2, 1)) * 100
    RSI3 = safe_div(SMA(TEMP1, P3, 1), SMA(TEMP2, P3, 1)) * 100

    # ---------- 量能条件 ----------
    VOL1 = MA(VOLA, 1)
    VOL2 = MA(REF(VOLA, 1), 3)
    VOL3 = MA(VOLA, 2)
    VOL4 = MA(REF(VOLA, 2), 3)
    VOL5 = MA(VOLA, 3)
    VOL6 = MA(REF(VOLA, 3), 3)

    VOLC_B = (VOL1 > VOL2 * 1.60) | (VOL3 > VOL4 * 1.60) | (VOL5 > VOL6 * 1.60)
    VOLC_S = (VOL1 > VOL2 * 1.10) | (VOL3 > VOL4 * 1.10) | (VOL5 > VOL6 * 1.10)

    VARA1 = safe_div(CLOSE - LLV(LOW, 60.13547854), HHV(HIGH, 60.13547854) - LLV(LOW, 60)) * 80
    VARB1 = SMA(VARA1, 7, 1)
    VARC1 = SMA(VARB1, 5, 1)

    # ---------- 绝地反弹 (版本化) ----------
    # v3: 参考指标原始信号 (3% 反包, 无去重)
    # v4: 反包阈值 5% + 同标的 10 日去重 (噪音治理, 见模块头注释)
    VAR2N = LLV(LOW, 3) <= LLV(LOW, 60)
    jf_thr = 1.03 if version == "v3" else 1.05
    VAR3N = (CLOSE > OPEN) & ((safe_div(CLOSE, OPEN) > jf_thr) | (CLOSE > jf_thr * REF(CLOSE, 1)))
    JF_RAW = (VAR2N & VAR3N & VOLC_B).fillna(False)  # 未去重原始触发 (供 B_BEAR_SETUP)
    if version == "v4":
        jf_gap_prev = REF(BARSLAST(JF_RAW), 1)
        # 当期成立且距上次触发 >= 10 日 (首次触发放行)。
        # BARSLAST 在触发当日为 0, 须取前一日值: 前一日距上次 >=9 即今日间隔 >=10。
        ICON_JUEFAN = JF_RAW & (jf_gap_prev.isna() | (jf_gap_prev >= 9))
    else:
        ICON_JUEFAN = JF_RAW  # DRAWICON(绝反, LOW, 34) 原样
    juefan = IF(ICON_JUEFAN, 20, 0)
    FAST_CRASH = CLOSE < REF(CLOSE, 10) * 0.85
    # 量能 1.2x 宽松版绝反 (事件研究对照列)
    VOLC_B_LOOSE = (VOL1 > VOL2 * 1.20) | (VOL3 > VOL4 * 1.20) | (VOL5 > VOL6 * 1.20)
    juefan_loose = IF(VAR2N & VAR3N & VOLC_B_LOOSE, 20, 0)

    # ---------- B 评分 ----------
    B_SCORE_1 = IF((REF(profit_chip, 1) < B_CHIP_LOW) & (profit_chip > REF(profit_chip, 1)), 1, 0)
    B_SCORE_2 = IF((profit_chip > 30) & (profit_chip < 80)
                   & (CLOSE > 1.02 * REF(CLOSE, 1))
                   & (VOLA > MA(VOLA, 10) * 2.0) & (VOLA < MA(VOLA, 100) * 5.0), 1, 0)
    B_SCORE_3 = IF(CROSS(V2, V3), 1, 0) + IF(V2 > REF(V2, 1), 1, 0)
    B_SCORE_4 = IF((REF(MACD, 1) < REF(MACD, 2)) & (MACD > REF(MACD, 1)), 1, 0) \
        + IF((REF(CLOSE, 1) < REF(LOWER, 1)) & (CLOSE > 0.2 * MID + 0.8 * LOWER), 1, 0)
    B_SCORE_5 = IF(VOLC_B, 1, 0)

    B_SCORE = B_SCORE_1 + B_SCORE_2 + B_SCORE_3 + B_SCORE_4 + B_SCORE_5

    # B_CONDITION:= B_SCORE>=5 AND (B1 OR B2) OR B_SCORE>=4 AND (B1 OR B2) AND (B5 OR REF(获利筹,1)<-5);
    # (AND 优先级高于 OR)
    B_CONDITION = ((B_SCORE >= 5) & ((B_SCORE_1 == 1) | (B_SCORE_2 == 1))) | \
                  ((B_SCORE >= 4) & ((B_SCORE_1 == 1) | (B_SCORE_2 == 1))
                   & ((B_SCORE_5 == 1) | (REF(profit_chip, 1) < -5)))

    # ---------- S 评分 ----------
    S_SCORE_1 = IF(HHV(profit_chip, 3) > 0.96 * HHV(REF(profit_chip, 20), 150), 1, 0) \
        + IF(HHV(profit_chip, 2) > 85, 1, 0) \
        + IF(HHV(profit_chip, 2) > 100, 1, 0)
    S_SCORE_2 = IF(CROSS(V3, V2), 1, 0) + IF(V2 < REF(V2, 1), 1, 0) \
        + IF((REF(MACD, 1) > REF(MACD, 2)) & (MACD < REF(MACD, 1)), 1, 0)
    S_SCORE_3 = IF((LLV(REF(MACD, 1), 15) > 0) & (MACD < 0), 1, 0) \
        + IF((REF(CLOSE, 1) > REF(UPPER, 1)) & (CLOSE > 0.2 * MID + 0.8 * UPPER), 1, 0) \
        + IF(VOLC_S, 1, 0)
    S_SCORE_4 = IF(CROSS(RSI1, 85), 1, 0)

    S_SCORE = S_SCORE_1 + S_SCORE_2 + S_SCORE_3 + S_SCORE_4

    S_CONDITION = (S_SCORE >= 7) | ((S_SCORE >= 6) & (SMA(S_SCORE, 3, 1) >= 5.0))
    DOWN_LONG = (profit_chip > 70) & (MA(profit_chip, 100) > 15) & (MA(profit_chip, 100) < 50)
    S_CONDITION_DOWN_LONG = DOWN_LONG & (S_SCORE >= 5)
    UP_LAST = (MA(profit_chip, 100) > 65) & (HHV(profit_chip, 3) < HHV(profit_chip, 30) * 0.9)
    S_CONDITION_UP_LAST = UP_LAST & (S_SCORE >= 5)

    # ---------- 阶段底部信号 (B_STAGE) ----------
    DOWN_SEQ = BARSLASTCOUNT(CLOSE < REF(CLOSE, 4))
    STAGE_LOW = (LLV(LOW, 10) <= LLV(LOW, 60) * 1.04) \
        & (LLV(LOW, 10) <= HHV(HIGH, 40) * 0.86) \
        & (LLV(LOW, 10) <= MA(CLOSE, 200) * 1.20)
    STAGE_EXHAUST = (HHV(REF(DOWN_SEQ, 1), 15) >= 8) | \
                    ((HHV(REF(DOWN_SEQ, 1), 15) >= 6) & (LLV(LOW, 10) <= REF(LLV(LOW, 60), 20) * 0.90))
    STAGE_TREND = (CLOSE > MA(CLOSE, 200)) & (MA(CLOSE, 60) > REF(MA(CLOSE, 60), 10) * 0.95) \
        & (MID > REF(MID, 10) * 0.93)
    STAGE_CONFIRM = (CLOSE > MA(CLOSE, 10)) & (CLOSE > REF(HHV(HIGH, 3), 1)) \
        & (RSI1 > 50) & (MACD > REF(MACD, 1))
    B_STAGE_RAW = STAGE_LOW & STAGE_EXHAUST & STAGE_TREND & STAGE_CONFIRM
    B_STAGE_SIGNAL = B_STAGE_RAW & (COUNT(B_STAGE_RAW, 25) == 1) & (~B_CONDITION) & (juefan == 0)

    # ---------- 综合 B 信号 ----------
    B_BASE_BULL = B_CONDITION & (CLOSE >= MA(CLOSE, 200)) & (~FAST_CRASH)
    B_BEAR_SETUP = B_CONDITION | JF_RAW  # 沿用未去重的原始触发
    B_BEAR_RECOVER = (CLOSE < MA(CLOSE, 200)) & (COUNT(REF(B_BEAR_SETUP, 1), 12) > 0) \
        & CROSS(CLOSE, MID) & (MA(CLOSE, 5) > REF(MA(CLOSE, 5), 2)) & (RSI1 > 50) & (MACD > REF(MACD, 2))
    CRASH_SETUP = (LLV(LOW, 3) <= LLV(LOW, 120) * 1.02) & (LOW < MA(CLOSE, 200) * 0.75) \
        & (VOLA > MA(VOLA, 20) * 1.80)
    B_CRASH_RECOVER = (COUNT(REF(CRASH_SETUP, 1), 15) > 0) & (CLOSE > MA(CLOSE, 5)) \
        & (CLOSE > REF(HHV(HIGH, 3), 1)) & (RSI1 > 45) & (MACD > REF(MACD, 1))
    B_ALL_RAW = B_BASE_BULL | B_STAGE_SIGNAL | B_BEAR_RECOVER | B_CRASH_RECOVER
    # DRAWICON(B_SIGNAL,LOW,7)
    B_SIGNAL = B_ALL_RAW & (COUNT(B_ALL_RAW, 20) == 1)

    # ---------- 综合 S 信号 ----------
    MA5 = MA(CLOSE, 5)
    MAJOR_TOP = (HHV(HIGH, 20) >= HHV(HIGH, 80) * 0.98) & (HHV(HIGH, 20) >= LLV(LOW, 60) * 1.30) \
        & (HHV(RSI1, 20) > 78) & (HHV(profit_chip, 20) > 80)
    S_CROSS = MAJOR_TOP & CROSS(MID, MA5) & (MID < REF(MID, 3)) & (MACD < 0)
    S_DELAY = MAJOR_TOP & (MA5 < MID) & (MID < REF(MID, 2)) & (COUNT(CROSS(MID, MA5), 8) > 0) & (MACD < 0)
    S_BEAR_RALLY = (CLOSE < MA(CLOSE, 200)) & (HHV(HIGH, 15) >= LLV(LOW, 60) * 1.35) \
        & (REF(HIGH, 1) >= REF(MA(CLOSE, 60), 1)) & (REF(HIGH, 1) >= HHV(REF(HIGH, 1), 20)) \
        & (REF(RSI1, 1) > 70) & (CLOSE < REF(LOW, 1)) & (MACD < REF(MACD, 1))
    S_BLOWOFF = (CLOSE > MA(CLOSE, 200)) & (REF(HIGH, 1) >= HHV(REF(HIGH, 1), 20)) \
        & (REF(CLOSE, 1) >= REF(CLOSE, 20) * 1.25) & (REF(RSI1, 1) > 70) & (CLOSE < REF(CLOSE, 1) * 0.92)
    S_RAW = S_BLOWOFF | S_BEAR_RALLY | S_CROSS | S_DELAY
    # DRAWICON(S_SIGNAL,S_POSITION,8);  S_POSITION:=1.008*HIGH
    S_SIGNAL = S_RAW & (COUNT(S_RAW, 40) == 1)
    S_POSITION = 1.008 * HIGH

    # ==================== NINE2：九转+成交量+MACD+布林带过滤 ====================
    # N 为比较前第 N 根 K 线收盘价 (参数表默认 4)
    NINE2_VOL_DAYS = 3
    NINE2_BOLL_PERIOD = 14
    NINE2_BOLL_WIDTH = 1.8
    NINE2_WIDTH_THRESHOLD = 0.3

    NINE2_COND_UP = CLOSE > REF(CLOSE, N)
    NINE2_COND_DOWN = CLOSE < REF(CLOSE, N)
    NINE2_COUNT_UP_RAW = BARSLAST(~NINE2_COND_UP)
    NINE2_COUNT_UP_DAYS = IF(NINE2_COUNT_UP_RAW == 0, 0, NINE2_COUNT_UP_RAW)
    NINE2_SIGNAL_UP = NINE2_COUNT_UP_DAYS >= 6

    NINE2_COUNT_DOWN_RAW = BARSLAST(~NINE2_COND_DOWN)
    NINE2_COUNT_DOWN_DAYS = IF(NINE2_COUNT_DOWN_RAW == 0, 0, NINE2_COUNT_DOWN_RAW)
    NINE2_SIGNAL_DOWN = NINE2_COUNT_DOWN_DAYS >= 6

    NINE2_VOL_MA = MA(VOLA, NINE2_VOL_DAYS)
    NINE2_VOL_CONFIRM_UP = VOLA > REF(NINE2_VOL_MA, 1) * 1.1
    NINE2_VOL_CONFIRM_DOWN = VOLA < REF(NINE2_VOL_MA, 1) * 0.9

    NINE2_DIF = EMA(CLOSE, 6) - EMA(CLOSE, 13)
    NINE2_DEA = EMA(NINE2_DIF, 5)
    NINE2_MACD_CROSS_UP = CROSS(NINE2_DIF, NINE2_DEA)
    NINE2_MACD_CROSS_DOWN = CROSS(NINE2_DEA, NINE2_DIF)

    NINE2_MID = MA(CLOSE, NINE2_BOLL_PERIOD)
    NINE2_UPPER = NINE2_MID + NINE2_BOLL_WIDTH * STD(CLOSE, NINE2_BOLL_PERIOD)
    NINE2_LOWER = NINE2_MID - NINE2_BOLL_WIDTH * STD(CLOSE, NINE2_BOLL_PERIOD)
    NINE2_WIDTH_VALUE = safe_div(NINE2_UPPER - NINE2_LOWER, NINE2_MID) * 100
    NINE2_FILTER_TREND = NINE2_WIDTH_VALUE > NINE2_WIDTH_THRESHOLD

    # DRAWTEXT(...,'★买',COLORYELLOW) / (…,'★卖',COLORMAGENTA)
    NINE2_BUY_SIGNAL = NINE2_SIGNAL_DOWN & NINE2_VOL_CONFIRM_DOWN & NINE2_MACD_CROSS_UP & NINE2_FILTER_TREND
    NINE2_SELL_SIGNAL = NINE2_SIGNAL_UP & NINE2_VOL_CONFIRM_UP & NINE2_MACD_CROSS_DOWN & NINE2_FILTER_TREND

    # ==================== NINE2 九转 1-9 标注（按 NINE 筛选, EHOPT10）====================
    # 仅标注: ① 已完成 9 序列的 1-9  (BACKSET(NINE2_UP_NINE>0,9))
    #         ② 最后一根K线上进行中的 5-8 序列 (BACKSET(...LIVE..., NINE2_UP_COUNT))
    NINE2_UP_COUNT = BARSLASTCOUNT(NINE2_COND_UP)
    NINE2_UP_NINE = NINE2_UP_COUNT == 9
    NINE2_UP_LIVE = ISLASTBAR(idx) & BETWEEN(NINE2_UP_COUNT, 5, 8)
    NINE2_UP_LABEL = (BACKSET(NINE2_UP_NINE, 9) | BACKSET(NINE2_UP_LIVE, NINE2_UP_COUNT)) \
        * NINE2_UP_COUNT
    # DRAWNUMBER(0<LABEL<9, H, LABEL, OFFSET) COLORFF00FF;  DRAWNUMBER(COUNT=9, H, 9) COLORGREEN
    NINE2_UP_LABEL_SHOWN = (NINE2_UP_LABEL > 0) & (NINE2_UP_LABEL < 9)
    NINE2_UP_9_SHOWN = NINE2_UP_COUNT == 9

    NINE2_DOWN_COUNT = BARSLASTCOUNT(NINE2_COND_DOWN)
    NINE2_DOWN_NINE = NINE2_DOWN_COUNT == 9
    NINE2_DOWN_LIVE = ISLASTBAR(idx) & BETWEEN(NINE2_DOWN_COUNT, 5, 8)
    NINE2_DOWN_LABEL = (BACKSET(NINE2_DOWN_NINE, 9) | BACKSET(NINE2_DOWN_LIVE, NINE2_DOWN_COUNT)) \
        * NINE2_DOWN_COUNT
    # DRAWNUMBER(0<LABEL<9, L, LABEL, -OFFSET) COLORGREEN;  DRAWNUMBER(COUNT=9, L, 9) COLORFF00FF
    NINE2_DOWN_LABEL_SHOWN = (NINE2_DOWN_LABEL > 0) & (NINE2_DOWN_LABEL < 9)
    NINE2_DOWN_9_SHOWN = NINE2_DOWN_COUNT == 9

    # ==========================================================================
    # 输出
    # ==========================================================================
    out = pd.DataFrame(index=idx)
    out["OPEN"], out["HIGH"], out["LOW"], out["CLOSE"], out["VOLUME"] = OPEN, HIGH, LOW, CLOSE, VOLA

    # 主图布林带 (富途颜色: MID #FFAEC9, UPPER #FFC90E, LOWER #0CAEE6)
    out["DIS"], out["MID"], out["UPPER"], out["LOWER"] = DIS, MID, UPPER, LOWER

    # 获利筹等衍生
    out["获利筹"] = profit_chip
    out["V3"] = V3
    out["DIF"], out["DEA"], out["MACD"] = DIF, DEA, MACD
    out["RSI1"] = RSI1

    # 信号
    out["绝反"] = juefan
    out["ICON_JUEFAN"] = ICON_JUEFAN          # DRAWICON(绝反, LOW, 34)
    out["JUEFAN_LOOSE"] = juefan_loose != 0
    out["B_SCORE"], out["S_SCORE"] = B_SCORE, S_SCORE
    out["B_CONDITION"], out["S_CONDITION"] = B_CONDITION, S_CONDITION
    out["S_CONDITION_DOWN_LONG"] = S_CONDITION_DOWN_LONG
    out["S_CONDITION_UP_LAST"] = S_CONDITION_UP_LAST
    out["B_STAGE_SIGNAL"] = B_STAGE_SIGNAL
    out["B_SIGNAL"] = B_SIGNAL                # DRAWICON(B_SIGNAL,LOW,7)
    out["S_SIGNAL"] = S_SIGNAL                # DRAWICON(S_SIGNAL,S_POSITION,8)
    out["S_POSITION"] = S_POSITION
    out["NINE2_BUY_SIGNAL"] = NINE2_BUY_SIGNAL   # DRAWTEXT ★买
    out["NINE2_SELL_SIGNAL"] = NINE2_SELL_SIGNAL  # DRAWTEXT ★卖

    # 九转标注
    out["NINE2_UP_COUNT"] = NINE2_UP_COUNT
    out["NINE2_UP_LABEL"] = np.where(NINE2_UP_LABEL_SHOWN, NINE2_UP_LABEL, 0)
    out["NINE2_UP_9"] = NINE2_UP_9_SHOWN
    out["NINE2_DOWN_COUNT"] = NINE2_DOWN_COUNT
    out["NINE2_DOWN_LABEL"] = np.where(NINE2_DOWN_LABEL_SHOWN, NINE2_DOWN_LABEL, 0)
    out["NINE2_DOWN_9"] = NINE2_DOWN_9_SHOWN

    return out


# ==========================================================================
# 三、绘图 (可选, 需要 matplotlib)
# ==========================================================================

def plot_result(res: pd.DataFrame, OFFSET: int = 15, title: str = "KK2 EHOPT10",
                save_path: str | None = None, show: bool = False):
    """将指标画成图: 布林带 + 收盘价 + 全部信号标注。

    OFFSET 与富途参数一致, 作为九转数字距高低点的偏移(pt)。
    """
    import matplotlib
    if not show and save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 中文字体回退 (★买/★卖/绝反 标注), 找不到则维持默认
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for fname in ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS",
                  "Microsoft YaHei", "SimHei"]:
        if fname in available:
            plt.rcParams["font.family"] = fname
            break
    plt.rcParams["axes.unicode_minus"] = False

    x = np.arange(len(res))
    fig, ax = plt.subplots(figsize=(16, 9))

    ax.plot(x, res["CLOSE"], color="black", lw=1.0, label="CLOSE")
    ax.plot(x, res["MID"], color="#FFAEC9", lw=1.0, label="MID")       # COLORFFAEC9
    ax.plot(x, res["UPPER"], color="#FFC90E", lw=1.0, label="UPPER")   # COLORFFC90E
    ax.plot(x, res["LOWER"], color="#0CAEE6", lw=1.0, label="LOWER")   # COLOR0CAEE6

    def _idx(cond):
        return np.asarray(res.index)[_as_bool(cond).to_numpy()]

    # DRAWICON 7/8 (B/S 信号) 与 34 (绝反)
    ax.scatter(x[_as_bool(res["B_SIGNAL"])], res.loc[_as_bool(res["B_SIGNAL"]), "LOW"] * 0.99,
               marker="^", s=90, color="red", zorder=5, label="B_SIGNAL")
    ax.scatter(x[_as_bool(res["S_SIGNAL"])], res.loc[_as_bool(res["S_SIGNAL"]), "S_POSITION"],
               marker="v", s=90, color="green", zorder=5, label="S_SIGNAL")
    ax.scatter(x[_as_bool(res["ICON_JUEFAN"])], res.loc[_as_bool(res["ICON_JUEFAN"]), "LOW"] * 0.985,
               marker="D", s=45, color="orange", zorder=5, label="绝反")

    # DRAWTEXT ★买/★卖
    for cond, price, txt, color in [
        (res["NINE2_BUY_SIGNAL"], res["LOW"] * 0.98, "★买", "yellow"),
        (res["NINE2_SELL_SIGNAL"], res["HIGH"] * 1.02, "★卖", "magenta"),
    ]:
        sel = _as_bool(cond)
        for i, p in zip(x[sel.to_numpy()], price[sel]):
            ax.annotate(txt, (i, p), ha="center", va="bottom", color=color, fontsize=10)

    # 九转数字标注: 上方 1-8 品红 / 9 绿; 下方 1-8 绿 / 9 品红
    for col, price, side, color in [
        ("NINE2_UP_LABEL", res["HIGH"], 1, "#FF00FF"),
        ("NINE2_UP_9", res["HIGH"], 1, "green"),
        ("NINE2_DOWN_LABEL", res["LOW"], -1, "green"),
        ("NINE2_DOWN_9", res["LOW"], -1, "#FF00FF"),
    ]:
        v = res[col]
        is_nine = v.dtype == bool
        sel = v.to_numpy() if is_nine else (v > 0).to_numpy()
        for i, val in zip(x[sel], v[sel]):
            txt = "9" if is_nine else str(int(val))
            ax.annotate(txt, (i, price.iloc[i]), xytext=(0, OFFSET * side),
                        textcoords="offset points", ha="center", color=color, fontsize=8)

    ax.set_title(title)
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ==========================================================================
# 四、自检与示例
# ==========================================================================

def make_sample_data(n: int = 800, seed: int = 7) -> pd.DataFrame:
    """生成合成 OHLCV 数据 (几何随机游走), 用于自检与演示。"""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.015, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    span = np.abs(rng.normal(0, 0.012, n)) + 0.002
    high = np.maximum(open_, close) * (1 + span)
    low = np.minimum(open_, close) * (1 - span)
    volume = np.abs(rng.lognormal(10, 0.6, n)) * (1 + np.abs(rng.normal(0, 2, n)) * (np.abs(ret) > 0.01))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
    df.index = pd.bdate_range("2025-01-01", periods=n)
    return df


def _self_test():
    """单元自检: 验证基础函数与全量指标的不变量。"""
    # --- 基础函数 ---
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert np.allclose(EMA(s, 3).to_numpy(),
                       pd.Series([1, 2, 3, 4, 5.]).ewm(span=3, adjust=False).mean().to_numpy())

    y = SMA(pd.Series([1.0, 2, 3, 4, 5]), 3, 1)
    # Y = (X*1 + Y'*2)/3:  1, 4/3, 17/9, 70/27, 275/81
    expect = [1.0, 4 / 3, 17 / 9, 70 / 27, 275 / 81]
    assert np.allclose(y.to_numpy(), expect, atol=1e-10), y.tolist()

    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=100))
    assert np.allclose(STDP(x, 10).dropna(), x.rolling(10).std(ddof=0).dropna())
    assert np.allclose(STD(x, 10).dropna(), x.rolling(10).std(ddof=1).dropna())
    assert np.allclose(MA(x, 5).dropna(), x.rolling(5).mean().dropna())

    cr = pd.Series([1.0, 1, 2, 1, 4])
    cb = pd.Series([2.0, 1, 1, 3, 3])
    assert CROSS(cr, cb).tolist() == [False, False, True, False, True]

    blc = BARSLASTCOUNT(pd.Series([True, True, False, True, True, True]))
    assert blc.tolist() == [1, 2, 0, 1, 2, 3]

    bl = BARSLAST(pd.Series([False, True, False, False, True, False]))
    blv = bl.to_numpy()
    assert np.isnan(blv[0]) and blv[1:5].tolist() == [0.0, 1.0, 2.0, 0.0]

    bs = BACKSET(pd.Series([False, False, True, False, False]), 2)
    assert bs.tolist() == [False, True, True, False, False]
    bs2 = BACKSET(pd.Series([False, False, False, True, False]),
                  pd.Series([1, 1, 1, 3, 1]))  # 变量周期
    assert bs2.tolist() == [False, True, True, True, False]

    assert BETWEEN(pd.Series([5.0, 1, 9, 8]), 2, 8).tolist() == [True, False, False, True]
    assert BETWEEN(pd.Series([5.0, 9, 1]), 8, 2).tolist() == [True, False, False]  # 边界倒置等价 [2,8]

    # --- 全量指标不变量 ---
    df = make_sample_data(900)
    res = compute_ehopt10(df, SD=20, WIDTH=2, N=4, OFFSET=15)

    close = res["CLOSE"]
    assert np.allclose(res["MID"].dropna(), close.rolling(20).mean().dropna())
    disp = close.rolling(20).std(ddof=0)
    assert np.allclose(res["UPPER"].dropna(), (res["MID"] + 2 * disp).dropna())
    assert np.allclose(res["LOWER"].dropna(), (res["MID"] - 2 * disp).dropna())

    # 9 计数与标注一致性
    up = res["NINE2_UP_COUNT"]
    # COUNT 与 COND_UP 直接复算
    cond_up = close > close.shift(4)
    blc = BARSLASTCOUNT(cond_up)
    assert (up.to_numpy() == blc.to_numpy()).all()
    # '9' 标注 <=> 计数恰为 9
    assert (res["NINE2_UP_9"].to_numpy() == (up == 9).to_numpy()).all()
    assert (res["NINE2_DOWN_9"].to_numpy() == (res["NINE2_DOWN_COUNT"] == 9).to_numpy()).all()
    # 1-8 标注必须落在某个"完成9序列"窗口或最后一根K线的进行中序列内
    lbl = res["NINE2_UP_LABEL"]
    ok_mask = np.zeros(len(lbl), dtype=bool)
    for i in np.where(up.to_numpy() == 9)[0]:
        ok_mask[i - 8: i + 1] = True
    last_cnt = int(up.iloc[-1])
    if 5 <= last_cnt <= 8:
        ok_mask[len(lbl) - last_cnt:] = True
    assert ((lbl.to_numpy() > 0) <= ok_mask).all(), "1-8 标注出现于不应标注的位置"
    # 标注值 == 当根计数
    assert (lbl.to_numpy()[lbl.to_numpy() > 0] == up.to_numpy()[lbl.to_numpy() > 0]).all()

    # 信号频率约束 (去重逻辑)
    assert COUNT(res["B_SIGNAL"], 20).max() <= 1
    assert COUNT(res["S_SIGNAL"], 40).max() <= 1

    # --- NINE2 专项: 末根K线进行中 5-8 序列的标注 + 完成9序列标注 ---
    closes = np.concatenate([np.arange(1, 101, dtype=float), np.arange(105, 95, -1)])
    sdf = pd.DataFrame({"open": closes, "high": closes + 1, "low": closes - 1,
                        "close": closes, "volume": np.full(len(closes), 1000.0)})
    r2 = compute_ehopt10(sdf, SD=20, WIDTH=2, N=4, OFFSET=15)
    # 末尾连续 6 根下跌比较成立 -> 最后一根K线计数 6 (在 5-8 之间) -> 标注 1..6
    assert int(r2["NINE2_DOWN_COUNT"].iloc[-1]) == 6
    assert r2["NINE2_DOWN_LABEL"].iloc[-6:].tolist() == [1, 2, 3, 4, 5, 6]
    assert int(r2["NINE2_DOWN_9"].sum()) == 0
    # 上升段恰有一次完成 9 序列: 1-8 各标注一次 (和=36), '9' 标注一次
    assert int(r2["NINE2_UP_9"].sum()) == 1
    assert int(r2["NINE2_UP_LABEL"].sum()) == sum(range(1, 9))

    # 信号列均为 bool
    for col in ["B_SIGNAL", "S_SIGNAL", "NINE2_BUY_SIGNAL", "NINE2_SELL_SIGNAL",
                "ICON_JUEFAN", "B_CONDITION", "S_CONDITION"]:
        assert res[col].dtype == bool, col

    # 参数变化生效 (截取共同尾部比较, 预热期长度不同)
    res2 = compute_ehopt10(df, SD=60, WIDTH=1.5, N=2, OFFSET=9)
    assert not np.allclose(res["MID"].iloc[-300:], res2["MID"].iloc[-300:])

    # --- 版本不变量: v3 无去重 (触发数 >= v4); v4 相邻触发间隔 >= 10 根 ---
    r3 = compute_ehopt10(df, version="v3")
    assert r3["ICON_JUEFAN"].sum() >= res["ICON_JUEFAN"].sum()
    idx4 = np.where(res["ICON_JUEFAN"].to_numpy())[0]
    assert all(b - a >= 10 for a, b in zip(idx4, idx4[1:])), "绝反触发间隔 <10"
    cond2 = res2["CLOSE"] > res2["CLOSE"].shift(2)
    assert (res2["NINE2_UP_COUNT"].to_numpy() == BARSLASTCOUNT(cond2).to_numpy()).all()

    print("[self-test] 全部通过 ✓")


if __name__ == "__main__":
    _self_test()

    df = make_sample_data(900)
    res = compute_ehopt10(df, SD=20, WIDTH=2, N=4, OFFSET=15)

    last = res.iloc[-1]
    print(f"\n样本数据 {len(df)} 根K线, 参数 SD=20 WIDTH=2 N=4 OFFSET=15")
    print(f"最新: CLOSE={last['CLOSE']:.2f} MID={last['MID']:.2f} "
          f"UPPER={last['UPPER']:.2f} LOWER={last['LOWER']:.2f}")
    print(f"获利筹={last['获利筹']:.2f} B_SCORE={int(last['B_SCORE'])} S_SCORE={int(last['S_SCORE'])} "
          f"九转 上{int(last['NINE2_UP_COUNT'])}/下{int(last['NINE2_DOWN_COUNT'])}")

    sig_cols = {
        "B_SIGNAL": "B买(DRAWICON7)", "S_SIGNAL": "S卖(DRAWICON8)",
        "NINE2_BUY_SIGNAL": "★买", "NINE2_SELL_SIGNAL": "★卖",
        "ICON_JUEFAN": "绝反ICON34",
    }
    print("\n最近 60 根K线内的信号:")
    recent = res.tail(60)
    hits = []
    for col, name in sig_cols.items():
        for t in recent.index[recent[col].fillna(False)]:
            hits.append((t, name))
    for t, name in sorted(hits):
        print(f"  {t}  {name}")
    if not hits:
        print("  (无)")

    print("\n最近 10 根K线九转标注 (UP_LABEL=上方品红数字, DOWN_LABEL=下方绿色数字):")
    print(res[["CLOSE", "NINE2_UP_COUNT", "NINE2_UP_LABEL", "NINE2_UP_9",
               "NINE2_DOWN_COUNT", "NINE2_DOWN_LABEL", "NINE2_DOWN_9"]].tail(10).to_string())

    try:
        plot_result(res, OFFSET=15, save_path="kk2_ehopt10_sample.png")
        print("\n示例图已保存: kk2_ehopt10_sample.png")
    except ImportError:
        print("\n(未安装 matplotlib, 跳过绘图; pip install matplotlib 后可运行 plot_result)")
