# -*- coding: utf-8 -*-
"""GCN 「金筹九转」 主图指标配方 (EHOPT10 的工程化实现)

配方阶段 (自上而下依赖顺序, 修改时保持顺序):
  1. 布林带主框架        DIS/MID/UPPER/LOWER
  2. 获利筹综合摆动      VAR1..VAR1C -> 获利筹/V3   (指标核心)
  3. MACD / RSI          DIF/DEA/MACD/RSI1-3
  4. 量能条件            VOL1..VOL6/VOLC_B/VOLC_S
  5. 绝地反弹 (版本化)   v3=原始信号 / v4=5%反包+10日去重
  6. B/S 评分系统        B_SCORE/S_SCORE 及其条件
  7. 阶段信号            B_STAGE/B_BEAR_RECOVER/CRASH -> B_SIGNAL
  8. 顶部信号            MAJOR_TOP -> S_SIGNAL
  9. NINE2 九转系统      九转计数/★买/★卖 (★已弃用, 仅观测)

版本:
  v3 = 参考指标原始信号 (3%反包, 无去重)   [git tag v3.0.0]
  v4 = 5%反包 + 10日去重 (当前默认)        [git tag v4.0.0]
  共同口径: ★买/★卖已弃用; 绝反不加趋势过滤 (熊市触发优于牛市)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gcn.core.tdx import (
    BARSLAST, BARSLASTCOUNT, BETWEEN, COUNT, CROSS, EMA, HHV, IF,
    LLV, MA, MAXA, POW_, REF, SMA, STDP, STD, safe_div,
)

# B_SCORE_1 的获利筹低位门槛 (原公式 8; 放宽可提升低点覆盖率, 见回测诊断)
B_CHIP_LOW = 8.0

VERSIONS = ("v3", "v4")

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
    if version not in VERSIONS:
        raise ValueError(f"未知配方版本: {version!r}，仅支持 {', '.join(VERSIONS)}")
    if isinstance(N, (bool, np.bool_)) or not np.isfinite(N) or int(N) != N or int(N) <= 0:
        raise ValueError("N 必须是正整数")
    N = int(N)

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

    # ==================== NINE2 九转 1-9 标注（因果显示）====================
    # 每根 K 线只显示当时已经形成的计数；不通过 BACKSET 回填历史，避免图表后见信息。
    NINE2_UP_COUNT = BARSLASTCOUNT(NINE2_COND_UP)
    NINE2_UP_LABEL = IF(BETWEEN(NINE2_UP_COUNT, 1, 8), NINE2_UP_COUNT, 0)
    # DRAWNUMBER(0<LABEL<9, H, LABEL, OFFSET) COLORFF00FF;  DRAWNUMBER(COUNT=9, H, 9) COLORGREEN
    NINE2_UP_LABEL_SHOWN = (NINE2_UP_LABEL > 0) & (NINE2_UP_LABEL < 9)
    NINE2_UP_9_SHOWN = NINE2_UP_COUNT == 9

    NINE2_DOWN_COUNT = BARSLASTCOUNT(NINE2_COND_DOWN)
    NINE2_DOWN_LABEL = IF(BETWEEN(NINE2_DOWN_COUNT, 1, 8), NINE2_DOWN_COUNT, 0)
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


def _self_test():
    """配方不变量自检 (python3 -m gcn.recipes.gcn_main 运行)。"""
    from gcn.data.sample import make_sample_data
    df = make_sample_data(600)
    res = compute_ehopt10(df)

    # 黄金基准等价 (重构前后逐值一致)
    import pickle
    from pathlib import Path
    gp = Path(__file__).resolve().parents[2] / "tests" / "golden_v4.pkl"
    if gp.exists():
        with open(gp, "rb") as f:
            blob = pickle.load(f)
        g = blob["golden"]["v4"]
        for col in g.columns:
            a, b = g[col].to_numpy(dtype=float), res[col].to_numpy(dtype=float)
            assert np.allclose(a, b, equal_nan=True), f"黄金基准不一致: {col}"

    # 版本不变量: v3 无去重 (触发数 >= v4); v4 相邻触发间隔 >= 10 根
    r3 = compute_ehopt10(df, version="v3")
    assert r3["ICON_JUEFAN"].sum() >= res["ICON_JUEFAN"].sum()
    idx4 = np.where(res["ICON_JUEFAN"].to_numpy())[0]
    assert all(b - a >= 10 for a, b in zip(idx4, idx4[1:])), "绝反触发间隔 <10"

    # 参数变化生效
    res2 = compute_ehopt10(df, SD=60, WIDTH=1.5, N=2, OFFSET=9)
    assert not np.allclose(res["MID"].iloc[-300:], res2["MID"].iloc[-300:])
    print("[gcn_main] 配方自检通过 ✓ (含黄金基准等价)")


if __name__ == "__main__":
    _self_test()
