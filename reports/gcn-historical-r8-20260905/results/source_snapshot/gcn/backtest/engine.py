# -*- coding: utf-8 -*-
"""GCN 回测引擎: 事件研究 (信号预测力) + 策略回测 (次日开盘成交)。

方法论与稳定性设计
------------------
1. 事件研究: 信号当根收盘确认 -> 下一根 K 线开盘入场, 第 h 根收盘离场的毛收益,
   与全样本基线对比 (胜率/均值/t 值), 度量信号本身的预测力;
2. 策略回测: 只做多全进全出, 双边计入成本, 支持最长持有与跟踪止损 (trail);
3. slice_years: 指标在全量历史计算后按年切片, 保证回测窗口预热不丢失。

局限: 指标参数为历史调优 (样本内评估); 未建模滑点; 建议配合
kk2_v5_report 式的消融/跨标的稳定性检验使用。
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
import pandas as pd

TRADING_DAYS = 252

TIMEFRAMES = {
    "1d": {"period_label": "日", "periods_per_year": 252},
    "1wk": {"period_label": "周", "periods_per_year": 52},
    # 采用每年 252 个交易日、每交易日 6.5 小时的统一市场近似。
    "60m": {"period_label": "小时", "periods_per_year": 1638},
    "15m": {"period_label": "15分钟", "periods_per_year": 6552},
    "5m": {"period_label": "5分钟", "periods_per_year": 19656},
}

# 默认回测/关注标的池 (与 UI 默认关注列表一致)
DEFAULT_SYMBOLS = ("TQQQ", "MSFT", "NFLX", "YINN", "SNOW", "TSLA",
                   "MRNA", "NVDA", "TEM", "GOOGL", "AAOI")

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
    ("B_STAGE_ENTRY_SIGNAL", "阶段确认(B_STAGE_EXP)"),
    ("B_SETUP", "v5 B买Setup"),
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
    {"name": "阶段Setup → S卖", "entry": ["B_STAGE_SETUP"], "exit": ["S_SIGNAL"]},
    {"name": "阶段确认 → S卖", "entry": ["B_STAGE_ENTRY_SIGNAL"], "exit": ["S_SIGNAL"]},
]

V5_RECOMMENDED_PRESET = {
    "name": "v5推荐: B确认+绝反 → S卖 + 20%止损",
    "entry": ["B_SIGNAL", "ICON_JUEFAN"],
    "exit": ["S_SIGNAL"],
    "trail": 0.20,
}


def presets_for_version(version: str):
    """返回版本对应策略集；v5 首项为审计验证后的正式推荐。"""
    if version == "v5":
        return [V5_RECOMMENDED_PRESET, *PRESETS]
    return PRESETS


# ==========================================================================
# 事件研究: 信号预测力
# ==========================================================================

def _forward_returns(res: pd.DataFrame, horizon: int) -> pd.Series:
    """信号当根收盘确认 -> 下一根开盘入场, 第 horizon 根收盘离场的毛收益。"""
    open_next = res["OPEN"].shift(-1)
    close_h = res["CLOSE"].shift(-horizon)
    return close_h / open_next - 1.0


def _stats_of(r: pd.Series, benchmark_mean: float | None = None,
              max_lag: int = 0) -> dict:
    """收益统计；有 benchmark 时，t/p 检验相对基线的超额收益。

    t 使用 Bartlett 核 Newey-West 标准误，降低重叠远期窗口导致的
    自相关低估。p 为渐近双侧值，随后由 event_study 统一做 BH 校正。
    """
    r = r.dropna().astype(float)
    if len(r) == 0:
        return {"n": 0, "win": None, "mean": None, "med": None,
                "t": None, "p": None, "q": None}
    mean = float(r.mean())
    t = p = None
    if benchmark_mean is not None and len(r) > 1:
        excess = r.to_numpy(dtype=float) - float(benchmark_mean)
        centered = excess - excess.mean()
        n = len(excess)
        lag = min(max(int(max_lag), 0), n - 1)
        long_var = float(np.dot(centered, centered) / n)
        for k in range(1, lag + 1):
            gamma = float(np.dot(centered[k:], centered[:-k]) / n)
            long_var += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
        if np.isfinite(long_var) and long_var > 0:
            t = float(excess.mean() / math.sqrt(long_var / n))
            p = float(math.erfc(abs(t) / math.sqrt(2.0)))
    return {"n": int(len(r)), "win": round(float((r > 0).mean()) * 100, 1),
            "mean": round(mean * 100, 2), "med": round(float(r.median()) * 100, 2),
            "t": round(t, 2) if t is not None else None,
            "p": round(p, 6) if p is not None else None, "q": None}


def _bh_adjust(stats: list[dict]):
    """对一组统计结果原地写入 Benjamini-Hochberg q 值。"""
    ranked = sorted(((float(st["p"]), st) for st in stats if st.get("p") is not None),
                    key=lambda x: x[0])
    m = len(ranked)
    next_q = 1.0
    for rank in range(m, 0, -1):
        p, st = ranked[rank - 1]
        next_q = min(next_q, p * m / rank)
        st["q"] = round(next_q, 6)


def event_study(res: pd.DataFrame, horizons=HORIZONS) -> list:
    horizons = tuple(dict.fromkeys(int(h) for h in horizons))
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("horizons 必须包含正整数")
    n = len(res)
    split = int(n * 0.6)
    fr = {h: _forward_returns(res, h) for h in horizons}
    base = {h: _stats_of(fr[h].dropna()) for h in horizons}
    split_h = 5 if 5 in horizons else horizons[0]

    def relative_stats(sample: pd.Series, baseline: pd.Series, horizon: int) -> dict:
        baseline = baseline.dropna()
        base_stats = _stats_of(baseline)
        base_mean = float(baseline.mean()) if len(baseline) else None
        stats = _stats_of(sample, benchmark_mean=base_mean, max_lag=horizon - 1)
        stats["base_win"] = base_stats["win"]
        stats["base_mean"] = base_stats["mean"]
        stats["excess"] = (round(stats["mean"] - base_stats["mean"], 2)
                           if stats["mean"] is not None and base_stats["mean"] is not None
                           else None)
        return stats

    rows = []
    tested_stats = []
    for col, label in SIGNAL_LABELS:
        if col not in res.columns:
            continue
        sig = res[col].fillna(False).astype(bool)
        row = {"signal": col, "label": label, "count": int(sig.sum()), "horizons": {}}
        for h in horizons:
            st = relative_stats(fr[h][sig], fr[h], h)
            row["horizons"][str(h)] = st
            tested_stats.append(st)
        # 切点前的信号必须在切点前完成远期窗口，避免标签归属泄漏。
        in_end = max(split - split_h, 0)
        in_base = fr[split_h].iloc[:in_end].dropna()
        out_base = fr[split_h].iloc[split:].dropna()
        split_stats = {
            "horizon": split_h,
            "in_sample": relative_stats(
                fr[split_h].iloc[:in_end][sig.iloc[:in_end]],
                in_base, split_h,
            ),
            "out_sample": relative_stats(
                fr[split_h].iloc[split:][sig.iloc[split:]],
                out_base, split_h,
            ),
        }
        row["split"] = split_stats
        row["split5"] = split_stats if split_h == 5 else None
        rows.append(row)
    _bh_adjust(tested_stats)
    rows.append({"signal": "_BASE", "label": "基线(任意日同口径)", "count": int(n),
                 "horizons": {str(h): base[h] for h in horizons},
                 "split": None, "split5": None})
    return rows


# ==========================================================================
# 策略回测: 只做多, 信号收盘确认 -> 次日开盘成交
# ==========================================================================

def _one_strategy(res: pd.DataFrame, entry_cols, exit_cols,
                  cost: float, max_hold, trail: float | None = None,
                  hard_stop: float | None = None,
                  profit_keep: float | None = None,
                  terminal_policy: str = "liquidate",
                  entry_hard_stop_col: str | None = None,
                  entry_max_hold_col: str | None = None,
                  entry_exit_cols: tuple[str, str] | None = None) -> dict:
    """单策略模拟。trail: 跟踪止损比例 (如 0.15 = 从入场后最高收盘回撤 15% 离场),
    hard_stop: 相对入场开盘价的初始止损比例。profit_keep: 浮盈达到trail比例后，
    保留净保本线上方的峰值利润比例。三者均在收盘确认，下一根开盘成交，
    可与信号离场叠加 (先到先出)。terminal_policy="mark" 时尾部持仓只盯市，
    不生成强制平仓交易。entry_hard_stop_col为信号日的可选风险比例列；
    非空值覆盖该笔入场hard_stop，NaN沿用全局值，持仓后锁定不再改变。
    entry_max_hold_col同样以信号日取值覆盖该笔max_hold，买入当天计第1根。
    entry_exit_cols=(入场启用列, 额外退出列)，启用标记亦锁定在信号日。"""
    if terminal_policy not in {"liquidate", "mark"}:
        raise ValueError("terminal_policy 必须是 liquidate 或 mark")
    if profit_keep is not None:
        if trail is None:
            raise ValueError("profit_keep 只能与 trail 一起使用")
        if (isinstance(trail, (bool, np.bool_))
                or not isinstance(trail, Real)
                or not np.isfinite(trail)
                or not 0 < trail < 1):
            raise ValueError("启用 profit_keep 时 trail 必须是 (0, 1) 内的有限数")
        trail = float(trail)
        if (isinstance(profit_keep, (bool, np.bool_))
                or not isinstance(profit_keep, Real)
                or not np.isfinite(profit_keep)
                or not 0 < profit_keep < 1):
            raise ValueError("profit_keep 必须是 (0, 1) 内的有限数")
        profit_keep = float(profit_keep)
    if hard_stop is not None:
        if (isinstance(hard_stop, (bool, np.bool_))
                or not isinstance(hard_stop, Real)
                or not np.isfinite(hard_stop)
                or not 0 < hard_stop < 1):
            raise ValueError("hard_stop 必须是 (0, 1) 内的有限数")
        hard_stop = float(hard_stop)
    entry_stops = np.full(len(res), np.nan)
    if entry_hard_stop_col is not None:
        for pos, value in enumerate(res[entry_hard_stop_col]):
            if pd.isna(value):
                continue
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or not 0 < value < 1):
                raise ValueError("entry_hard_stop_col值必须是NaN或(0, 1)内的有限数")
            entry_stops[pos] = float(value)
    entry_holds = np.full(len(res), np.nan)
    if entry_max_hold_col is not None:
        for pos, value in enumerate(res[entry_max_hold_col]):
            if pd.isna(value):
                continue
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 1 or int(value) != value):
                raise ValueError("entry_max_hold_col值必须是NaN或正整数")
            entry_holds[pos] = int(value)
    o = res["OPEN"].to_numpy(dtype=float)
    c = res["CLOSE"].to_numpy(dtype=float)
    if (not np.isfinite(o).all() or not np.isfinite(c).all()
            or (o <= 0).any() or (c <= 0).any()):
        raise ValueError("策略回测要求 OPEN/CLOSE 全部为有限正数")
    entry = res[list(entry_cols)].fillna(False).astype(bool).any(axis=1).to_numpy()
    exitc = res[list(exit_cols)].fillna(False).astype(bool).any(axis=1).to_numpy()
    n = len(res)
    entry_extra_enabled = np.zeros(n, dtype=bool)
    extra_exit = np.zeros(n, dtype=bool)
    if entry_exit_cols is not None:
        if (not isinstance(entry_exit_cols, tuple) or len(entry_exit_cols) != 2
                or any(not isinstance(col, str) or col not in res.columns
                       for col in entry_exit_cols)):
            raise ValueError("entry_exit_cols必须是两个已存在列名组成的tuple")
        enabled_col, exit_col = entry_exit_cols
        entry_extra_enabled = res[enabled_col].fillna(False).astype(bool).to_numpy()
        extra_exit = res[exit_col].fillna(False).astype(bool).to_numpy()

    cash, shares = 1.0, 0.0
    equity = np.full(n, np.nan)
    held = np.zeros(n, dtype=bool)
    pend_buy = pend_sell = False
    pend_sell_reason = None
    entry_i, basis = -1, 1.0
    hi_since_entry = entry_open = np.nan
    active_hard_stop = hard_stop
    active_max_hold = max_hold
    active_extra_exit = False
    profit_armed = False
    profit_floor = np.nan
    trades = []

    for t in range(n):
        # 1) 执行昨日收盘生成的挂单 (今日开盘价)
        if shares > 0 and pend_sell:
            proceeds = shares * o[t] * (1 - cost)
            trades.append({"i": entry_i, "j": t, "ret": float(proceeds / basis - 1),
                           "pnl": float(proceeds - basis), "hold": int(t - entry_i),
                           "exit_reason": pend_sell_reason or "signal"})
            cash, shares = proceeds, 0.0
            entry_open = np.nan
            profit_armed = False
            profit_floor = np.nan
        elif shares == 0 and pend_buy and np.isfinite(o[t]):
            basis = cash
            shares = cash * (1 - cost) / o[t]
            cash = 0.0
            entry_i = t
            entry_open = o[t]
            active_hard_stop = (float(entry_stops[t - 1])
                                if np.isfinite(entry_stops[t - 1]) else hard_stop)
            active_max_hold = (int(entry_holds[t - 1])
                               if np.isfinite(entry_holds[t - 1]) else max_hold)
            active_extra_exit = bool(entry_extra_enabled[t - 1])
            hi_since_entry = c[t]
            profit_armed = False
            profit_floor = np.nan
        pend_buy = pend_sell = False
        pend_sell_reason = None
        # 2) 当日收盘评估信号 -> 挂次日开盘单
        if shares > 0:
            hi_since_entry = max(hi_since_entry, c[t])
            hard_stop_hit = (active_hard_stop is not None
                             and c[t] <= entry_open * (1 - active_hard_stop))
            entry_exit_hit = active_extra_exit and extra_exit[t]
            trail_hit = trail is not None and c[t] <= hi_since_entry * (1 - trail)
            profit_lock_hit = False
            if profit_keep is not None:
                break_even = entry_open / (1 - cost) ** 2
                trail_floor = hi_since_entry * (1 - trail)
                profit_floor = break_even + profit_keep * (hi_since_entry - break_even)
                profit_armed = (profit_armed
                                or hi_since_entry >= entry_open * (1 + trail))
                profit_lock_hit = (
                    profit_armed
                    and profit_floor > trail_floor
                    and not trail_hit
                    and c[t] <= profit_floor
                )
            if (exitc[t] or hard_stop_hit or entry_exit_hit or profit_lock_hit or trail_hit
                    or (active_max_hold is not None and t - entry_i + 1 >= active_max_hold)):
                pend_sell = True
                pend_sell_reason = ("signal" if exitc[t] else
                                    "hard_stop" if hard_stop_hit else
                                    "entry_signal" if entry_exit_hit else
                                    "profit_lock" if profit_lock_hit else
                                    "trail" if trail_hit else "max_hold")
        elif entry[t]:
            pend_buy = True
        # 3) 收盘盯市
        held[t] = shares > 0
        equity[t] = cash + shares * c[t]

    # 末端仍持仓时按最后收盘价强制平仓，确保交易、收益与双边费用口径一致。
    terminal_liquidated = False
    if n and shares > 0 and terminal_policy == "liquidate":
        proceeds = shares * c[-1] * (1 - cost)
        trades.append({"i": entry_i, "j": n, "ret": float(proceeds / basis - 1),
                       "pnl": float(proceeds - basis), "hold": int(n - entry_i),
                       "exit_reason": "terminal"})
        equity[-1] = proceeds
        terminal_liquidated = True

    position_open = bool(shares > 0 and not terminal_liquidated)
    pending_buy_active = bool(not position_open and not terminal_liquidated and pend_buy)
    if position_open:
        terminal_status = "pending_exit" if pend_sell else "open"
    else:
        terminal_status = "pending_entry" if pending_buy_active else "flat"
    state = {
        "status": terminal_status,
        "position": "open" if position_open else "flat",
        "entry_i": int(entry_i) if position_open else None,
        "entry_open": float(entry_open) if position_open else None,
        "highest_close": float(hi_since_entry) if position_open else None,
        "pending_buy": pending_buy_active,
        "pending_sell_reason": (pend_sell_reason
                                if position_open and pend_sell else None),
        "profit_armed": bool(position_open and profit_armed),
        "profit_floor": (float(profit_floor)
                         if position_open and profit_armed else None),
        "mark_equity": float(equity[-1]) if n else None,
    }
    return {"equity": equity, "trades": trades, "held": held, "state": state}


def _perf(equity: np.ndarray, trades: list,
          periods_per_year: int = TRADING_DAYS) -> dict:
    n = len(equity)
    if not n:
        raise ValueError("回测数据为空")
    curve = np.concatenate(([1.0], np.asarray(equity, dtype=float)))
    period_rets = np.diff(curve) / curve[:-1]
    sd = float(period_rets.std(ddof=1)) if len(period_rets) > 1 else 0.0
    sharpe = float(period_rets.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else None
    peak = np.maximum.accumulate(curve)
    mdd = float((1 - curve / peak).max())
    rets = np.array([t["ret"] for t in trades]) if trades else np.array([])
    pnls = np.array([t.get("pnl", t["ret"]) for t in trades]) if trades else np.array([])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    terminal = float(equity[-1])
    return {
        "total": round((terminal - 1) * 100, 2),
        "cagr": round((terminal ** (periods_per_year / n) - 1) * 100, 2)
                if terminal > 0 else None,
        "mdd": round(mdd * 100, 2),
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "trades": len(trades),
        "win": round(float((rets > 0).mean()) * 100, 1) if len(rets) else None,
        "avg": round(float(rets.mean()) * 100, 2) if len(rets) else None,
        "pf": round(float(wins.sum() / abs(losses.sum())), 2) if len(wins) and len(losses) else None,
        "avg_hold": round(float(np.mean([t["hold"] for t in trades])), 1) if trades else None,
    }


def _buy_hold(res: pd.DataFrame, cost: float) -> dict:
    if res.empty:
        raise ValueError("回测数据为空")
    o = float(res["OPEN"].iloc[0])
    equity = res["CLOSE"].to_numpy(dtype=float) * (1 - cost) / o
    equity[-1] *= 1 - cost
    return {"equity": equity, "trades": [], "held": np.ones(len(res), dtype=bool)}


def slice_years(res: pd.DataFrame, years: float, interval: str = "1d"):
    """按年截取指标结果 (指标已在全量历史计算, 切片只影响回测区间, 预热不丢失)。"""
    if years is None:
        return res
    try:
        years = float(years)
    except (TypeError, ValueError) as exc:
        raise ValueError("years 必须是正有限数") from exc
    if not np.isfinite(years) or years <= 0:
        raise ValueError("years 必须是正有限数")
    interval = str(interval).strip().lower()
    if interval not in TIMEFRAMES:
        raise ValueError(f"未知K线周期: {interval}")
    if res.empty:
        return res
    if isinstance(res.index, pd.DatetimeIndex):
        if years.is_integer():
            start = res.index[-1] - pd.DateOffset(years=int(years))
        else:
            start = res.index[-1] - pd.Timedelta(days=365.2425 * years)
        return res.loc[res.index >= start]
    periods = max(1, int(math.ceil(years * TIMEFRAMES[interval]["periods_per_year"])))
    return res.iloc[-periods:]


def run_backtest(res: pd.DataFrame, cost: float = 0.001, max_hold=None,
                 presets=None, interval: str = "1d") -> dict:
    """完整回测报告 (纯 Python 数据, 便于 CLI 打印或 JSON 序列化)。"""
    if res.empty:
        raise ValueError("回测数据为空")
    if not np.isfinite(cost) or not 0 <= cost < 1:
        raise ValueError("cost 必须是 [0, 1) 内的有限数")
    if max_hold is not None and (isinstance(max_hold, bool) or int(max_hold) != max_hold
                                 or int(max_hold) <= 0):
        raise ValueError("max_hold 必须是正整数")
    interval = str(interval).strip().lower()
    if interval not in TIMEFRAMES:
        raise ValueError(f"未知K线周期: {interval}")
    timeframe = {"interval": interval, **TIMEFRAMES[interval]}
    periods_per_year = timeframe["periods_per_year"]
    presets = PRESETS if presets is None else presets
    # 入场/离场列不存在的预设自动跳过 (自定义/扩展预设引用可选列时不再抛错)
    cols = set(res.columns)
    presets = [p for p in presets
               if all(c in cols for c in list(p["entry"]) + list(p["exit"]))]
    strategies = []
    curves = {}
    for p in presets:
        bt = _one_strategy(res, p["entry"], p["exit"], cost, max_hold,
                           trail=p.get("trail"), hard_stop=p.get("hard_stop"),
                           profit_keep=p.get("profit_keep"))
        row = {"name": p["name"], **_perf(bt["equity"], bt["trades"], periods_per_year),
               "exposure": round(_exposure(bt, res), 3)}
        strategies.append(row)
        curves[p["name"]] = [round(float(x), 6) for x in bt["equity"]]

    bh = _buy_hold(res, cost)
    bh_perf = _perf(bh["equity"], [], periods_per_year)
    strategies.append({"name": "基准: 买入持有", **bh_perf, "exposure": 1.0})

    return {
        "events": event_study(res),
        "strategies": strategies,
        "equity": {**curves,
                   "基准: 买入持有": [round(float(x), 6) for x in bh["equity"]]},
        "timeframe": timeframe,
        "event_methodology": {
            "return": "signal-bar close -> next-bar open entry -> horizon-bar close exit",
            "t_stat": "excess return vs same-horizon baseline, Newey-West HAC",
            "multiple_testing": "Benjamini-Hochberg q-value",
            "split": "60/40; crossing forward windows excluded",
        },
    }


def _exposure(bt: dict, res: pd.DataFrame) -> float:
    """持仓时间占比: 用 trade 区间还原。"""
    if "held" in bt:
        held = np.asarray(bt["held"], dtype=bool)
        return float(held.mean()) if len(held) else 0.0
    held = np.zeros(len(res), dtype=bool)
    for t in bt["trades"]:
        held[t["i"]: t["j"]] = True
    return float(held.mean())
