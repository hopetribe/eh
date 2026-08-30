# -*- coding: utf-8 -*-
"""GCN 回测引擎: 事件研究 (信号预测力) + 策略回测 (次日开盘成交)。

方法论与稳定性设计
------------------
1. 事件研究: 信号 T 日收盘确认 -> T+1 开盘入场, T+h 收盘离场的毛收益,
   与全样本基线对比 (胜率/均值/t 值), 度量信号本身的预测力;
2. 策略回测: 只做多全进全出, 双边计入成本, 支持最长持有与跟踪止损 (trail);
3. slice_years: 指标在全量历史计算后按年切片, 保证回测窗口预热不丢失。

局限: 指标参数为历史调优 (样本内评估); 未建模滑点; 建议配合
kk2_v5_report 式的消融/跨标的稳定性检验使用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# 默认回测/关注标的池 (与 UI 默认关注列表一致)
DEFAULT_SYMBOLS = ("TQQQ", "MSFT", "NFLX", "YINN", "SNOW", "TSLA",
                   "MRNA", "NVDA", "TEM", "GOOGL", "SIVE", "AAOI")

SIGNAL_LABELS = [
    ("B_SIGNAL", "B买(DRAWICON7)"),
    ("NINE2_BUY_SIGNAL", "★买(九转买,已弃用)"),
    ("ICON_JUEFAN", "绝反(ICON34,参考指标版)"),
    ("JUEFAN_LOOSE", "绝反宽松量能(对照)"),
    ("S_SIGNAL", "S卖(DRAWICON8)"),
    ("NINE2_SELL_SIGNAL", "★卖(九转卖,已弃用)"),
    ("B_CONDITION", "B条件(原始)"),
    ("S_CONDITION", "S条件(原始)"),
    ("B_STAGE_SIGNAL", "阶段底(B_STAGE)"),
]

HORIZONS = (3, 5, 10, 20)

# 策略预设: ★买/★卖 九转信号已弃用; S卖 保留, 另提供 S条件 评分离场
PRESETS = [
    {"name": "B买 → S卖", "entry": ["B_SIGNAL"], "exit": ["S_SIGNAL"]},
    {"name": "B买+绝反 → S卖", "entry": ["B_SIGNAL", "ICON_JUEFAN"], "exit": ["S_SIGNAL"]},
    {"name": "绝反 → S卖", "entry": ["ICON_JUEFAN"], "exit": ["S_SIGNAL"]},
    {"name": "B买 → S条件", "entry": ["B_SIGNAL"], "exit": ["S_CONDITION"]},
    {"name": "B买+绝反 → S条件", "entry": ["B_SIGNAL", "ICON_JUEFAN"],
     "exit": ["S_CONDITION"]},
    {"name": "绝反 → S条件", "entry": ["ICON_JUEFAN"], "exit": ["S_CONDITION"]},
]


# ==========================================================================
# 事件研究: 信号预测力
# ==========================================================================

def _forward_returns(res: pd.DataFrame, horizon: int) -> pd.Series:
    """T 日信号 -> T+1 开盘入场, T+horizon 收盘离场的毛收益 (索引对齐到信号日 T)。"""
    open_next = res["OPEN"].shift(-1)
    close_h = res["CLOSE"].shift(-horizon)
    return close_h / open_next - 1.0


def _stats_of(r: pd.Series) -> dict:
    if len(r) == 0:
        return {"n": 0, "win": None, "mean": None, "med": None, "t": None}
    mean = float(r.mean())
    sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    t = mean / (sd / np.sqrt(len(r))) if np.isfinite(sd) and sd > 0 else None
    return {"n": int(len(r)), "win": round(float((r > 0).mean()) * 100, 1),
            "mean": round(mean * 100, 2), "med": round(float(r.median()) * 100, 2),
            "t": round(float(t), 2) if t is not None else None}


def event_study(res: pd.DataFrame, horizons=HORIZONS) -> list:
    n = len(res)
    split = int(n * 0.6)
    fr = {h: _forward_returns(res, h) for h in horizons}
    base = {h: _stats_of(fr[h].dropna()) for h in horizons}

    rows = []
    for col, label in SIGNAL_LABELS:
        if col not in res.columns:
            continue
        sig = res[col].fillna(False).astype(bool)
        row = {"signal": col, "label": label, "count": int(sig.sum()), "horizons": {}}
        for h in horizons:
            st = _stats_of(fr[h][sig].dropna())
            st["base_win"] = base[h]["win"]
            st["base_mean"] = base[h]["mean"]
            st["excess"] = round(st["mean"] - base[h]["mean"], 2) if st["mean"] is not None else None
            row["horizons"][str(h)] = st
        # 分段一致性 (5 日远期)
        row["split5"] = {
            "in_sample": _stats_of(fr[5].iloc[:split][sig.iloc[:split]].dropna()),
            "out_sample": _stats_of(fr[5].iloc[split:][sig.iloc[split:]].dropna()),
        }
        rows.append(row)
    rows.append({"signal": "_BASE", "label": "基线(任意日同口径)", "count": int(n),
                 "horizons": {str(h): base[h] for h in horizons}, "split5": None})
    return rows


# ==========================================================================
# 策略回测: 只做多, 信号收盘确认 -> 次日开盘成交
# ==========================================================================

def _one_strategy(res: pd.DataFrame, entry_cols, exit_cols,
                  cost: float, max_hold, trail: float | None = None) -> dict:
    """单策略模拟。trail: 跟踪止损比例 (如 0.15 = 从入场后最高收盘回撤 15% 离场),
    与信号离场可叠加 (先到先出)。"""
    o = res["OPEN"].to_numpy(dtype=float)
    c = res["CLOSE"].to_numpy(dtype=float)
    entry = res[list(entry_cols)].fillna(False).astype(bool).any(axis=1).to_numpy()
    exitc = res[list(exit_cols)].fillna(False).astype(bool).any(axis=1).to_numpy()
    n = len(res)

    cash, shares = 1.0, 0.0
    equity = np.full(n, np.nan)
    pend_buy = pend_sell = False
    entry_i, basis = -1, 1.0
    hi_since_entry = np.nan
    trades = []

    for t in range(n):
        # 1) 执行昨日收盘生成的挂单 (今日开盘价)
        if shares > 0 and pend_sell:
            proceeds = shares * o[t] * (1 - cost)
            trades.append({"i": entry_i, "j": t, "ret": float(proceeds / basis - 1),
                           "hold": int(t - entry_i)})
            cash, shares = proceeds, 0.0
        elif shares == 0 and pend_buy and np.isfinite(o[t]):
            basis = cash
            shares = cash * (1 - cost) / o[t]
            cash = 0.0
            entry_i = t
            hi_since_entry = c[t]
        pend_buy = pend_sell = False
        # 2) 当日收盘评估信号 -> 挂次日开盘单
        if shares > 0:
            hi_since_entry = max(hi_since_entry, c[t])
            trail_hit = trail is not None and c[t] <= hi_since_entry * (1 - trail)
            if exitc[t] or trail_hit or (max_hold is not None and t - entry_i >= max_hold):
                pend_sell = True
        elif entry[t]:
            pend_buy = True
        # 3) 收盘盯市
        equity[t] = cash + shares * c[t]

    return {"equity": equity, "trades": trades}


def _perf(equity: np.ndarray, trades: list) -> dict:
    n = len(equity)
    daily = np.diff(equity) / equity[:-1] if n > 1 else np.array([0.0])
    sd = float(daily.std(ddof=1)) if n > 2 else 0.0
    sharpe = float(daily.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else None
    peak = np.maximum.accumulate(equity)
    mdd = float((1 - equity / peak).max())
    rets = np.array([t["ret"] for t in trades]) if trades else np.array([])
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    return {
        "total": round(float(equity[-1] - 1) * 100, 2),
        "cagr": round((float(equity[-1]) ** (TRADING_DAYS / n) - 1) * 100, 2) if n else None,
        "mdd": round(mdd * 100, 2),
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "trades": len(trades),
        "win": round(float((rets > 0).mean()) * 100, 1) if len(rets) else None,
        "avg": round(float(rets.mean()) * 100, 2) if len(rets) else None,
        "pf": round(float(wins.sum() / abs(losses.sum())), 2) if len(wins) and len(losses) else None,
        "avg_hold": round(float(np.mean([t["hold"] for t in trades])), 1) if trades else None,
    }


def _buy_hold(res: pd.DataFrame, cost: float) -> dict:
    o = float(res["OPEN"].iloc[0])
    equity = res["CLOSE"].to_numpy(dtype=float) * (1 - cost) / o
    return {"equity": equity, "trades": []}


def slice_years(res: pd.DataFrame, years: float):
    """按年截取指标结果 (指标已在全量历史计算, 切片只影响回测区间, 预热不丢失)。"""
    if not years:
        return res
    start = res.index[-1] - pd.DateOffset(years=float(years))
    return res.loc[res.index >= start]


def run_backtest(res: pd.DataFrame, cost: float = 0.001, max_hold=None,
                 presets=None) -> dict:
    """完整回测报告 (纯 Python 数据, 便于 CLI 打印或 JSON 序列化)。"""
    presets = presets or PRESETS
    # 入场/离场列不存在的预设自动跳过 (自定义/扩展预设引用可选列时不再抛错)
    cols = set(res.columns)
    presets = [p for p in presets
               if all(c in cols for c in list(p["entry"]) + list(p["exit"]))]
    strategies = []
    for p in presets:
        bt = _one_strategy(res, p["entry"], p["exit"], cost, max_hold,
                           trail=p.get("trail"))
        row = {"name": p["name"], **_perf(bt["equity"], bt["trades"]),
               "exposure": round(_exposure(bt, res), 3)}
        strategies.append(row)
        p["_equity"] = bt["equity"]

    bh = _buy_hold(res, cost)
    bh_perf = _perf(bh["equity"], [])
    strategies.append({"name": "基准: 买入持有", **bh_perf, "exposure": 1.0})

    return {
        "events": event_study(res),
        "strategies": strategies,
        "equity": {**{p["name"]: [round(float(x), 6) for x in p["_equity"]] for p in presets},
                   "基准: 买入持有": [round(float(x), 6) for x in bh["equity"]]},
    }


def _exposure(bt: dict, res: pd.DataFrame) -> float:
    """持仓时间占比: 用 trade 区间还原。"""
    held = np.zeros(len(res), dtype=bool)
    for t in bt["trades"]:
        held[t["i"]: t["j"] + 1] = True
    return float(held.mean())
