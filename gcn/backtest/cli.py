# -*- coding: utf-8 -*-
"""GCN 回测命令行入口与报告打印。"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gcn.backtest.engine import _one_strategy, _exposure, _perf, run_backtest

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


DEFAULT_SYMBOLS = ("TQQQ", "MSFT", "NFLX", "YINN", "SNOW", "TSLA",
                   "MRNA", "NVDA", "TEM", "GOOGL", "SIVE", "AAOI")


def main():
    import argparse
    from kk2_ehopt10_ui import DEFAULT_COUNT, clamp_params, df_from_rows, fetch_quote

    ap = argparse.ArgumentParser(description="KK2 EHOPT10 回测")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                    help="逗号分隔, 默认关注列表 12 只标的")
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
        print_sensitivity(df, params, args.cost, args.max_hold)


def print_sensitivity(df: pd.DataFrame, params: dict, cost: float, max_hold):
    """SD 参数敏感性扫描 (检验推荐组合是否依赖特定参数, 过拟合哨兵)。"""
    print("\n【SD 敏感性扫描 · 推荐组合 B买+绝反 → S条件】")
    print(f"  {'SD':>5}{'总收益':>10}{'CAGR':>9}{'回撤':>8}{'夏普':>7}{'交易':>5}{'胜率':>7}{'仓位':>6}")
    for sd in (14, 20, 30, 50):
        p = dict(params)
        p["SD"] = sd
        res = compute_ehopt10(df, version="v3", **p)
        bt = _one_strategy(res, ["B_SIGNAL", "ICON_JUEFAN"], ["S_CONDITION"], cost, max_hold)
        st = _perf(bt["equity"], bt["trades"])
        sharpe = f"{st['sharpe']:.2f}" if st["sharpe"] is not None else "--"
        win = f"{st['win']:.1f}%" if st["win"] is not None else "--"
        print(f"  {sd:>5}{st['total']:>+9.1f}%{st['cagr']:>8.1f}%{st['mdd']:>7.1f}%"
              f"{sharpe:>7}{st['trades']:>5}{win:>7}{_exposure(bt, res) * 100:>5.0f}%")


if __name__ == "__main__":
    main()