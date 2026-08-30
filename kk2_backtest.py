#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KK2 EHOPT10 指标回测模块 (科学回测)

方法论
------
1. 事件研究法 (信号预测力): 信号在 T 日收盘确认, 以 T+1 开盘价为入场参考价,
   统计持有到 T+h 收盘的毛收益, 与"全样本任意一天同口径持有"的基线对比。
   输出 样本数/胜率/平均收益/t值, 不受仓位管理干扰, 度量信号本身的预测力。
2. 策略回测 (只做多, 全进全出): T 日收盘出信号 -> T+1 开盘价成交 (杜绝未来函数),
   双边计入交易成本; 输出 总收益/CAGR/最大回撤/夏普/交易数/胜率/盈亏比/平均持仓,
   并与买入持有基准对比。
3. 分段一致性检验: 样本前 60% vs 后 40%, 对比 5 日远期收益的均值与胜率,
   作为信号过拟合的哨兵。

局限声明 (解读回测结果时必须注意)
----------------------------------
- 指标参数 (SD/WIDTH/N 及各阈值) 为历史手工调优, 对本段样本属于"样本内"评估;
- 未做 walk-forward 参数滚动验证;
- 长牛 ETF (如 TQQQ) 买入持有基线很强, 跑赢基准对择时策略天然苛刻;
- 撮合为理想市价单, 未建模滑点随流动性的变化 (可用成本参数近似)。

CLI 用法:
    python3 kk2_backtest.py            # 默认对 TQQQ + QQQ 日K 各跑一次并打印报告
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from kk2_ehopt10 import compute_ehopt10

TRADING_DAYS = 252

# 参与事件研究的信号列 (与富途指标输出一一对应)
SIGNAL_LABELS = [
    ("B_SIGNAL", "B买(DRAWICON7)"),
    ("NINE2_BUY_SIGNAL", "★买(九转买)"),
    ("ICON_JUEFAN", "绝反(ICON34)"),
    ("S_SIGNAL", "S卖(DRAWICON8)"),
    ("NINE2_SELL_SIGNAL", "★卖(九转卖)"),
    ("B_CONDITION", "B条件(原始)"),
    ("S_CONDITION", "S条件(原始)"),
    ("B_STAGE_SIGNAL", "阶段底(B_STAGE)"),
]

HORIZONS = (3, 5, 10, 20)

# 预设策略: 入场信号 -> 离场信号
PRESETS = [
    {"name": "B买 → S卖", "entry": ["B_SIGNAL"], "exit": ["S_SIGNAL"]},
    {"name": "★买 → ★卖", "entry": ["NINE2_BUY_SIGNAL"], "exit": ["NINE2_SELL_SIGNAL"]},
    {"name": "B买+绝反 → S卖", "entry": ["B_SIGNAL", "ICON_JUEFAN"], "exit": ["S_SIGNAL"]},
    {"name": "绝反 → S卖", "entry": ["ICON_JUEFAN"], "exit": ["S_SIGNAL"]},
    {"name": "B买+绝反 → S卖+★卖", "entry": ["B_SIGNAL", "ICON_JUEFAN"],
     "exit": ["S_SIGNAL", "NINE2_SELL_SIGNAL"]},
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
                  cost: float, max_hold) -> dict:
    o = res["OPEN"].to_numpy(dtype=float)
    c = res["CLOSE"].to_numpy(dtype=float)
    entry = res[list(entry_cols)].fillna(False).astype(bool).any(axis=1).to_numpy()
    exitc = res[list(exit_cols)].fillna(False).astype(bool).any(axis=1).to_numpy()
    n = len(res)

    cash, shares = 1.0, 0.0
    equity = np.full(n, np.nan)
    pend_buy = pend_sell = False
    entry_i, basis = -1, 1.0
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
        pend_buy = pend_sell = False
        # 2) 当日收盘评估信号 -> 挂次日开盘单
        if shares > 0:
            if exitc[t] or (max_hold is not None and t - entry_i >= max_hold):
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


def run_backtest(res: pd.DataFrame, cost: float = 0.001, max_hold=None) -> dict:
    """完整回测报告 (纯 Python 数据, 便于 CLI 打印或 JSON 序列化)。"""
    strategies = []
    for p in PRESETS:
        bt = _one_strategy(res, p["entry"], p["exit"], cost, max_hold)
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
        "equity": {**{p["name"]: [round(float(x), 6) for x in p["_equity"]] for p in PRESETS},
                   "基准: 买入持有": [round(float(x), 6) for x in bh["equity"]]},
    }


def _exposure(bt: dict, res: pd.DataFrame) -> float:
    """持仓时间占比: 用 trade 区间还原。"""
    held = np.zeros(len(res), dtype=bool)
    for t in bt["trades"]:
        held[t["i"]: t["j"] + 1] = True
    return float(held.mean())


# ==========================================================================
# CLI: 对真实标的跑一次完整回测并打印报告
# ==========================================================================

def _fmt_row_events(row) -> str:
    def cell(st):
        if not st or not st.get("n"):
            return " " * 22
        t = f"t={st['t']:>5.2f}" if st.get("t") is not None else "t=   --"
        return f"{st['win']:>5.1f}% {st['mean']:>+6.2f}% {t:>9}"
    cells = [f"{row['label']:<16}", f"{row['count']:>4}"]
    for h in ("3", "5", "10", "20"):
        cells.append(cell(row["horizons"].get(h)))
    return "  ".join(cells)


def print_report(symbol: str, params: dict, cost: float, report: dict, n: int):
    line = "=" * 108
    print(f"\n{line}\n{symbol}  参数 {params}  单边成本 {cost * 100:.2f}%  样本 {n} 根日K\n{line}")

    print("\n【信号预测力 · 事件研究】(T+1开盘入场, 毛收益; 单元格 = 胜率 / 平均收益 / t值; 基线为任意日同口径)")
    header = f"{'信号':<16}{'样本':>5}" + "".join(f"{'| ' + h + '日':>24}" for h in ("3", "5", "10", "20"))
    print("  " + header)
    print("  " + "-" * len(header))
    for row in report["events"]:
        print("  " + _fmt_row_events(row))

    print("\n【5日远期 · 分段一致性】(前60% vs 后40% —— 检验信号是否随时间衰减)")
    print(f"  {'信号':<16}{'前60% 胜率/均值':>24}{'后40% 胜率/均值':>24}")
    for row in report["events"]:
        if not row.get("split5"):
            continue
        i, o = row["split5"]["in_sample"], row["split5"]["out_sample"]

        def f(st):
            return f"{st['win']}% / {st['mean']:+.2f}%" if st and st.get("n") else "  --  "
        print(f"  {row['label']:<16}{f(i):>24}{f(o):>24}")

    print("\n【策略回测 · 只做多, 次日开盘成交】")
    print(f"  {'策略':<20}{'总收益':>10}{'CAGR':>9}{'回撤':>8}{'夏普':>7}"
          f"{'交易':>5}{'胜率':>7}{'均笔':>8}{'盈亏比':>7}{'均持仓':>7}{'仓位':>6}")
    for s in report["strategies"]:
        f = lambda v, pat, suf="": (format(v, pat) + suf) if v is not None else "--"
        print(f"  {s['name']:<20}{f(s['total'], '+9.1f', '%'):>10}{f(s['cagr'], '8.1f', '%'):>9}"
              f"{f(s['mdd'], '7.1f', '%'):>8}{f(s['sharpe'], '7.2f'):>7}{s['trades']:>5}"
              f"{f(s['win'], '6.1f', '%'):>7}{f(s['avg'], '7.2f', '%'):>8}"
              f"{f(s['pf'], '6.2f'):>7}{f(s['avg_hold'], '6.1f'):>7}"
              f"{s['exposure'] * 100:>5.0f}%")


def main():
    import argparse
    from kk2_ehopt10_ui import DEFAULT_COUNT, clamp_params, df_from_rows, fetch_quote

    ap = argparse.ArgumentParser(description="KK2 EHOPT10 回测")
    ap.add_argument("--symbols", default="TQQQ,QQQ", help="逗号分隔, 默认 TQQQ,QQQ")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--cost", type=float, default=0.001, help="单边成本, 默认 0.001")
    ap.add_argument("--max-hold", type=int, default=None)
    ap.add_argument("--sd", type=int, default=20)
    ap.add_argument("--width", type=float, default=2)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    params = clamp_params({"SD": args.sd, "WIDTH": args.width, "N": args.n, "OFFSET": 15})
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        try:
            q = fetch_quote(sym, args.interval, DEFAULT_COUNT)
            df = df_from_rows(q["rows"])
        except Exception as exc:  # noqa: BLE001
            print(f"[跳过] {sym}: {exc}")
            continue
        res = compute_ehopt10(df, **params)
        report = run_backtest(res, cost=args.cost, max_hold=args.max_hold)
        print_report(f"{q['symbol']} ({df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d})",
                     params, args.cost, report, len(df))


if __name__ == "__main__":
    main()
