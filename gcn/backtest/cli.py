# -*- coding: utf-8 -*-
"""GCN 回测命令行入口与报告打印。"""
from __future__ import annotations

import argparse
import math

import pandas as pd

from gcn.backtest.engine import (
    DEFAULT_SYMBOLS, TIMEFRAMES, _one_strategy, _exposure, _perf, run_backtest,
)
from gcn.data.service import DEFAULT_COUNT, df_from_rows, fetch_quote
from gcn.recipes.gcn_main import VERSIONS, compute_ehopt10


def _horizon_label(horizon, period_label: str) -> str:
    return (f"{horizon}{period_label}" if period_label in {"日", "周", "小时"}
            else f"{horizon}×{period_label}")


def _fmt_row_events(row, horizons) -> str:
    def cell(st):
        if not st or not st.get("n"):
            return " " * 34
        if row.get("signal") == "_BASE":
            return f"胜{st['win']:>5.1f}% 均值{st['mean']:>+6.2f}%"
        excess = (f"{st['excess']:+.2f}%" if st.get("excess") is not None else "--")
        t = f"{st['t']:.2f}" if st.get("t") is not None else "--"
        q = f"{st['q']:.4f}" if st.get("q") is not None else "--"
        return f"胜{st['win']:>5.1f}% 超额{excess:>7} t={t:>5} q={q:>6}"
    cells = [f"{row['label']:<16}", f"{row['count']:>4}"]
    for h in horizons:
        cells.append(cell(row["horizons"].get(h)))
    return "  ".join(cells)


def print_report(symbol: str, params: dict, cost: float, report: dict, n: int):
    timeframe = report.get("timeframe", {"period_label": "日"})
    unit = timeframe["period_label"]
    horizon_keys = list(report["events"][-1]["horizons"])
    line = "=" * 108
    print(f"\n{line}\n{symbol}  参数 {params}  单边成本 {cost * 100:.2f}%  "
          f"样本 {n} 根{unit}K\n{line}")

    print("\n【信号预测力 · 事件研究】(下一根K线开盘入场, 毛收益; "
          "t=相对同期基线的HAC统计量, q=多重检验校正)")
    header = f"{'信号':<16}{'样本':>5}" + "".join(
        f"{'| ' + _horizon_label(h, unit):>36}" for h in horizon_keys)
    print("  " + header)
    print("  " + "-" * len(header))
    for row in report["events"]:
        print("  " + _fmt_row_events(row, horizon_keys))

    split_h = next((r["split"]["horizon"] for r in report["events"] if r.get("split")), 5)
    print(f"\n【{_horizon_label(split_h, unit)}远期 · 分段一致性】"
          "(前60% vs 后40%，跨切点窗口已排除)")
    print(f"  {'信号':<16}{'前60% 胜率/超额':>24}{'后40% 胜率/超额':>24}")
    for row in report["events"]:
        if not row.get("split"):
            continue
        i, o = row["split"]["in_sample"], row["split"]["out_sample"]

        def f(st):
            return (f"{st['win']}% / {st['excess']:+.2f}%"
                    if st and st.get("n") and st.get("excess") is not None else "  --  ")
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

def _finite_range(name: str, lo: float, hi: float):
    def parse(raw):
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"{name} 必须是数字") from exc
        if not math.isfinite(value) or not lo <= value <= hi:
            raise argparse.ArgumentTypeError(f"{name} 必须在 [{lo}, {hi}] 内")
        return value
    return parse


def _positive_int(raw):
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("max-hold 必须是正整数") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("max-hold 必须是正整数")
    return value


def _bounded_int(name: str, lo: int, hi: int):
    def parse(raw):
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"{name} 必须是整数") from exc
        if not lo <= value <= hi:
            raise argparse.ArgumentTypeError(f"{name} 必须在 [{lo}, {hi}] 内")
        return value
    return parse


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="KK2 EHOPT10 回测")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                    help="逗号分隔, 默认关注列表 12 只标的")
    ap.add_argument("--interval", choices=tuple(TIMEFRAMES), default="1d")
    ap.add_argument("--version", choices=VERSIONS, default="v4")
    ap.add_argument("--cost", type=_finite_range("cost", 0.0, 0.05), default=0.001,
                    help="单边成本, 范围 0~0.05, 默认 0.001")
    ap.add_argument("--max-hold", type=_positive_int, default=None)
    ap.add_argument("--sd", type=_bounded_int("sd", 2, 120), default=20)
    ap.add_argument("--width", type=_finite_range("width", 0.0, 100.0), default=2)
    ap.add_argument("--n", type=_bounded_int("n", 1, 1000), default=4)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    params = {"SD": args.sd, "WIDTH": args.width, "N": args.n, "OFFSET": 15}
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        try:
            q = fetch_quote(sym, args.interval, DEFAULT_COUNT)
            df = df_from_rows(q["rows"])
        except Exception as exc:  # noqa: BLE001
            print(f"[跳过] {sym}: {exc}")
            continue
        res = compute_ehopt10(df, **params, version=args.version)
        report = run_backtest(res, cost=args.cost, max_hold=args.max_hold,
                              interval=args.interval)
        print_report(f"{q['symbol']} ({df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d})",
                     params, args.cost, report, len(df))
        print_sensitivity(df, params, args.cost, args.max_hold,
                          version=args.version, interval=args.interval)


def print_sensitivity(df: pd.DataFrame, params: dict, cost: float, max_hold,
                      version: str = "v4", interval: str = "1d"):
    """SD 参数敏感性扫描 (检验推荐组合是否依赖特定参数, 过拟合哨兵)。"""
    print("\n【SD 敏感性扫描 · 推荐组合 B买+绝反 → S条件】")
    print(f"  {'SD':>5}{'总收益':>10}{'CAGR':>9}{'回撤':>8}{'夏普':>7}{'交易':>5}{'胜率':>7}{'仓位':>6}")
    for sd in (14, 20, 30, 50):
        p = dict(params)
        p["SD"] = sd
        res = compute_ehopt10(df, version=version, **p)
        bt = _one_strategy(res, ["B_SIGNAL", "ICON_JUEFAN"], ["S_CONDITION"], cost, max_hold)
        st = _perf(bt["equity"], bt["trades"], TIMEFRAMES[interval]["periods_per_year"])
        sharpe = f"{st['sharpe']:.2f}" if st["sharpe"] is not None else "--"
        win = f"{st['win']:.1f}%" if st["win"] is not None else "--"
        print(f"  {sd:>5}{st['total']:>+9.1f}%{st['cagr']:>8.1f}%{st['mdd']:>7.1f}%"
              f"{sharpe:>7}{st['trades']:>5}{win:>7}{_exposure(bt, res) * 100:>5.0f}%")


if __name__ == "__main__":
    main()
