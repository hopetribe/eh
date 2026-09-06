"""r22：原v5真实持仓日内、隔夜、费用与相邻交易链诊断。"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r16 import audit_trades
from gcn.backtest.signal_research_r14 import COMPONENTS
from gcn.backtest.historical_research import CORE, SNAPSHOT_SHA, load_snapshot
from gcn.recipes.gcn_main import compute_ehopt10

WINDOWS = (("training", "2021-08-27", "2024-08-26"),
           ("validation", "2024-08-27", "2025-08-26"),
           ("recent", "2025-08-27", "2026-08-26"),
           ("full", "2021-08-27", "2026-08-26"))


EXTRA_SCHEMA = {
    **{col: "string" for col in ("entry_kind", "last_held_close_date",
                                 "worst_overnight_date", "worst_intraday_date", "previous_trade_id",
                                 "previous_exit_date", "previous_exit_reason", "previous_entry_kind")},
    **{col: "float64" for col in ("entry_gap_pct", "overnight_factor", "intraday_factor", "cost_factor",
                                  "overnight_log_pct", "intraday_log_pct", "cost_log_pct",
                                  "reconstructed_return_pct", "last_held_close", "last_close_net_pct",
                                  "exit_overnight_pct", "exit_gap_impact_pp", "trail_reference",
                                  "worst_overnight_pct", "worst_intraday_pct", "pair_return_pct", "chain_return_pct")},
    "close_below_trail": "boolean",
    "flat_bars": "Int64",
}
PATH_SCHEMA = {
    **{col: "string" for col in ("symbol", "trade_id", "date", "observation")},
    **{col: "float64" for col in ("open", "close", "overnight_factor", "intraday_factor",
                                  "running_overnight_factor", "running_intraday_factor", "running_gross_factor")},
}


def audit_frame(symbol: str, frame: pd.DataFrame, start: pd.Timestamp,
                end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """不改变原订单；路径字段按观察时点因果，交易级极值仅作事后归因。"""
    frame = frame.loc[:end]
    base = frame.loc[start:end]
    original, result = audit_trades(symbol, frame, start, end)
    extras, paths = [], []
    previous = None
    previous_j = None
    chain = 1.
    for row, order in zip(original.to_dict("records"), result["trades"]):
        i, j = order["i"], order["j"]
        terminal = row["exit_reason"] == "terminal"
        overnight = intraday = 1.
        segments = []
        for t in range(i, j + int(not terminal)):
            at_open = t == j
            op = float(base.OPEN.iloc[t])
            night = op / float(base.CLOSE.iloc[t-1]) if t > i else np.nan
            close = np.nan if at_open else float(base.CLOSE.iloc[t])
            day = close / op if not at_open else np.nan
            overnight *= night if np.isfinite(night) else 1.
            intraday *= day if np.isfinite(day) else 1.
            segments.append({"symbol": symbol, "trade_id": row["trade_id"],
                             "date": base.index[t].date().isoformat(), "observation": "open" if at_open else "close",
                             "open": op, "close": close, "overnight_factor": night, "intraday_factor": day,
                             "running_overnight_factor": overnight, "running_intraday_factor": intraday,
                             "running_gross_factor": overnight * intraday})
        paths.extend(segments)
        last_close = float(base.CLOSE.iloc[j-1])
        close_net = (last_close / row["entry_open"] * .999**2 - 1) * 100
        extra = {"entry_kind": "B" if row["entry_b"] else "JF",
                 "entry_gap_pct": (row["entry_open"] / float(base.CLOSE.iloc[i-1]) - 1) * 100,
                 "overnight_factor": overnight, "intraday_factor": intraday, "cost_factor": .999**2,
                 "overnight_log_pct": np.log(overnight)*100, "intraday_log_pct": np.log(intraday)*100,
                 "cost_log_pct": 2*np.log(.999)*100,
                 "reconstructed_return_pct": (overnight * intraday * .999**2 - 1) * 100,
                 "last_held_close_date": base.index[j-1].date().isoformat(), "last_held_close": last_close,
                 "last_close_net_pct": close_net,
                 "exit_overnight_pct": np.nan if terminal else (row["exit_price"] / last_close - 1)*100,
                 "exit_gap_impact_pp": 0. if terminal else row["return_pct"] - close_net,
                 "trail_reference": row["peak_close"] * .8,
                 "close_below_trail": last_close <= row["peak_close"] * .8}
        for kind in ("overnight", "intraday"):
            observed = [s for s in segments if np.isfinite(s[kind + "_factor"])]
            worst = min(observed, key=lambda s: s[kind + "_factor"]) if observed else None
            extra["worst_" + kind + "_pct"] = (worst[kind + "_factor"]-1)*100 if worst else np.nan
            extra["worst_" + kind + "_date"] = worst["date"] if worst else None
        if not np.isclose(extra["reconstructed_return_pct"], row["return_pct"], rtol=1e-11, atol=1e-10):
            raise ValueError(f"{symbol}: 日内/隔夜/费用未还原原净收益")
        chain *= 1 + row["return_pct"] / 100
        extra["chain_return_pct"] = (chain-1)*100
        if previous is not None:
            extra.update(previous_trade_id=previous["trade_id"], previous_exit_date=previous["exit_date"],
                         previous_exit_reason=previous["exit_reason"],
                         previous_entry_kind="B" if previous["entry_b"] else "JF", flat_bars=i-previous_j,
                         pair_return_pct=((1+previous["return_pct"]/100)*(1+row["return_pct"]/100)-1)*100)
        extras.append(extra)
        previous, previous_j = row, j
    trades = pd.concat([original, pd.DataFrame(extras, columns=EXTRA_SCHEMA).astype(EXTRA_SCHEMA)], axis=1)
    path_table = pd.DataFrame(paths, columns=PATH_SCHEMA).astype(PATH_SCHEMA)
    expected = sum(order["hold"] + int(order["exit_reason"] != "terminal") for order in result["trades"])
    if len(path_table) != expected:
        raise ValueError(f"{symbol}: 原持仓观察数量不一致")
    if not np.isclose(chain, result["equity"][-1], rtol=1e-11, atol=1e-12):
        raise ValueError(f"{symbol}: 实际交易链未还原期末资金")
    check = {"symbol": symbol, "trades": len(trades), "path_rows": len(path_table),
             "reconciled": True, "chain_reconciled": True,
             "b_signals": int(base.B_SIGNAL.sum()), "jf_signals": int(base.ICON_JUEFAN.sum()),
             "entry_signals": int((base.B_SIGNAL | base.ICON_JUEFAN).sum()), "s_signals": int(base.S_SIGNAL.sum()),
             "b_trades": int(trades.entry_kind.eq("B").sum()), "jf_trades": int(trades.entry_kind.eq("JF").sum()),
             "signal_exits": int(trades.exit_reason.eq("signal").sum())}
    return trades, path_table, check


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """固定分层的描述统计；来源多标签重叠，均值/中位数不是组合收益。"""
    rows = []
    def add(scope, group_by, group, subset):
        n = len(subset)
        rows.append({"scope": scope, "group_by": group_by, "group": group, "trades": n,
                     "symbols": int(subset.symbol.nunique()), "wins": int(subset.return_pct.gt(0).sum()),
                     "win_rate_pct": subset.return_pct.gt(0).mean()*100 if n else np.nan,
                     "mean_return_pct": subset.return_pct.mean(), "median_return_pct": subset.return_pct.median(),
                     "worst_return_pct": subset.return_pct.min(),
                     "median_overnight_log_pct": subset.overnight_log_pct.median(),
                     "median_intraday_log_pct": subset.intraday_log_pct.median(),
                     "negative_overnight_trades": int(subset.overnight_factor.lt(1).sum()),
                     "negative_intraday_trades": int(subset.intraday_factor.lt(1).sum()),
                     "exit_gap_worsened_trades": int(subset.exit_overnight_pct.lt(0).sum()),
                     "last_close_win_to_loss": int((subset.last_close_net_pct.gt(0) & subset.return_pct.le(0)).sum()),
                     "median_exit_gap_impact_pp": subset.exit_gap_impact_pp.median(),
                     "worst_overnight_pct": subset.worst_overnight_pct.min(),
                     "worst_intraday_pct": subset.worst_intraday_pct.min(),
                     "reentries": int(subset.previous_trade_id.notna().sum())})
    for scope, subset in (("all", trades), ("B", trades[trades.entry_kind.eq("B")]),
                           ("JF", trades[trades.entry_kind.eq("JF")])):
        add(scope, "all", "all", subset)
        for reason in ("signal", "trail", "terminal"):
            add(scope, "exit_reason", reason, subset[subset.exit_reason.eq(reason)])
        for group, mask in (("win", subset.return_pct.gt(0)), ("nonpositive", subset.return_pct.le(0))):
            add(scope, "outcome", group, subset[mask])
        for value in (False, True):
            add(scope, "profit_to_loss", str(value).lower(), subset[subset.profit_to_loss.eq(value)])
        for symbol in sorted(trades.symbol.unique()):
            add(scope, "symbol", symbol, subset[subset.symbol.eq(symbol)])
    b = trades[trades.entry_kind.eq("B")]
    for col in COMPONENTS:
        add("B", "source", col, b[b[col]])
    add("B", "source", "multiple", b[b[list(COMPONENTS)].sum(axis=1).gt(1)])
    return pd.DataFrame(rows).astype({col: "string" for col in ("scope", "group_by", "group")})


def run_diagnostic(snapshot: Path, output: Path, *, window: str = "training") -> dict:
    """每次只冻结一个协议窗口；先训练机制，再披露其他固定窗口。"""
    selected = next((list(spec) for spec in WINDOWS if spec[0] == window), None)
    if selected is None:
        raise ValueError("仅允许协议固定窗口")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("诊断目录非空，请使用新的输出目录")
    root = Path(__file__).resolve().parents[2]
    names = ("gcn/backtest/signal_research_r22.py", "gcn/backtest/signal_research_r16.py",
             "gcn/backtest/signal_research_r14.py", "gcn/backtest/historical_research.py",
             "gcn/backtest/signal_audit.py", "gcn/backtest/engine.py", "gcn/recipes/gcn_main.py",
             "gcn/core/tdx.py", "gcn/core/indicators.py")
    sources = {name: (root / name).read_bytes() for name in names}
    protocol_path = root / "reports/gcn-historical-r22-20260905/protocol.md"
    protocol = protocol_path.read_bytes()
    frames, quality = load_snapshot(snapshot)
    start, end = pd.Timestamp(selected[1]), pd.Timestamp(selected[2])
    trade_parts, path_parts, checks = [], [], []
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:end], version="v5", diagnostics=True)
        trades, paths, check = audit_frame(symbol, frame, start, end)
        trade_parts.append(trades)
        path_parts.append(paths)
        checks.append(check)
    trades = pd.concat(trade_parts, ignore_index=True)
    paths = pd.concat(path_parts, ignore_index=True)
    tables = {"trades": trades, "paths": paths, "summary": summarize_trades(trades),
              "reconciliation": pd.DataFrame(checks)}
    decision = {"research_version": "gcn-historical-r22", "stage": "diagnostic_only",
                "recommended": "v5", "production_changed": False, "core": CORE, "window": selected,
                "trades": "original v5, next OPEN, trail 0.20, cost 0.001 each side",
                "window_policy": "independent flat start, terminal liquidation; overlapping windows are not independent",
                "attribution": "held intraday factor * held overnight factor * 0.999^2 reproduces actual net return",
                "paths": "entry-day gap excluded; real OPEN exit includes only overnight factor, no later CLOSE",
                "terminal": "last CLOSE boundary liquidation, no fictitious next OPEN gap",
                "chain": "actual adjacent trades; flat_bars counts fully flat bars; overlapping pairs are not independent",
                "statistics": "post-hoc descriptive factors and extrema, not entry filters or portfolio returns"}
    for name, raw in sources.items():
        if (root / name).read_bytes() != raw:
            raise ValueError(f"计算期间源码变化：{name}")
    if protocol_path.read_bytes() != protocol:
        raise ValueError("计算期间协议变化")
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.insert(0, "window", window)
        table.to_csv(output / (name + ".csv"), index=False)
    (output / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    (output / "protocol.md").write_bytes(protocol)
    for name, raw in sources.items():
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    manifest = {"research_version": "gcn-historical-r22", "window": selected,
                "parent_manifest_sha256": SNAPSHOT_SHA, "source_quality": quality,
                "protocol_sha256": hashlib.sha256(protocol).hexdigest(),
                "algorithm_sources": {name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()},
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "outputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(output.iterdir()) if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return decision


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", choices=[spec[0] for spec in WINDOWS], default="training")
    args = parser.parse_args()
    print(json.dumps(run_diagnostic(args.snapshot, args.output, window=args.window), indent=2, ensure_ascii=False))
