"""r16原生S多标签来源与退出质量诊断；不生成新策略。"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.engine import _one_strategy
from gcn.backtest.historical_research import CORE, SNAPSHOT_SHA, load_snapshot
from gcn.backtest.signal_research_r14 import COMPONENTS, WINDOWS, trace_setups, _regime
from gcn.backtest.signal_audit import _forward_path
from gcn.core.tdx import COUNT
from gcn.recipes.gcn_main import compute_ehopt10

SOURCES = ("S_BLOWOFF", "S_BEAR_RALLY", "S_CROSS", "S_DELAY")
TRADE_COLUMNS = ("symbol", "trade_id", "entry_date", "entry_signal_date", "entry_open", "entry_b", "entry_jf",
                 "setup_date", *COMPONENTS, "exit_date", "exit_signal_date", "exit_reason", "exit_price",
                 "return_pct", "hold_bars", "peak_close", "peak_close_date", "peak_close_gain_pct",
                 "exit_gross_pct", "giveback_pp", "peak_to_exit_drawdown_pct", "profit_to_loss",
                 *SOURCES, "exit_raw_s", "exit_s_suppressed", "raw_s_count", "emitted_s_count", "suppressed_s_count")


def audit_trades(symbol: str, frame: pd.DataFrame, start: pd.Timestamp,
                 end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """原v5独立窗口成交；峰值只含实际持有的收盘，期末强平另行标记。"""
    frame = frame.loc[:end]
    base = frame.loc[start:end]
    if len(base) < 2:
        raise ValueError("退出审计至少需要两根K线")
    result = _one_strategy(base, ["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"],
                           .001, None, trail=.20, terminal_policy="liquidate")
    trace = trace_setups(frame)
    entries = {row["resolution_date"]: row for row in trace[trace.status.eq("confirmed")].to_dict("records")}
    rows = []
    for trade in result["trades"]:
        i, j = trade["i"], trade["j"]
        terminal = trade["exit_reason"] == "terminal"
        entry_date = base.index[i].date().isoformat()
        entry_signal_date = base.index[i - 1].date().isoformat()
        entry_b, entry_jf = bool(base.B_SIGNAL.iloc[i - 1]), bool(base.ICON_JUEFAN.iloc[i - 1])
        source = entries[entry_signal_date] if entry_b else {}
        held = base.iloc[i:j]
        peak_date = held.CLOSE.idxmax()
        peak = float(held.CLOSE.loc[peak_date])
        entry_open = float(base.OPEN.iloc[i])
        exit_price = float(base.CLOSE.iloc[-1] if terminal else base.OPEN.iloc[j])
        peak_gain, exit_gross = (peak / entry_open - 1) * 100, (exit_price / entry_open - 1) * 100
        signal_exit = trade["exit_reason"] == "signal"
        if signal_exit and not base.S_SIGNAL.iloc[j - 1]:
            raise ValueError("S实际退出与前一根信号不一致")
        rows.append({"symbol": symbol, "trade_id": symbol + ":" + entry_date,
                     "entry_date": entry_date, "entry_signal_date": entry_signal_date,
                     "entry_open": entry_open, "entry_b": entry_b, "entry_jf": entry_jf,
                     "setup_date": source.get("setup_date"), **{col: source.get(col, False) for col in COMPONENTS},
                     "exit_date": base.index[min(j, len(base) - 1)].date().isoformat(),
                     "exit_signal_date": None if terminal else base.index[j - 1].date().isoformat(),
                     "exit_reason": trade["exit_reason"], "exit_price": exit_price,
                     "return_pct": trade["ret"] * 100, "hold_bars": trade["hold"],
                     "peak_close": peak, "peak_close_date": peak_date.date().isoformat(),
                     "peak_close_gain_pct": peak_gain, "exit_gross_pct": exit_gross,
                     "giveback_pp": peak_gain - exit_gross,
                     "peak_to_exit_drawdown_pct": (1 - exit_price / peak) * 100,
                     "profit_to_loss": bool(peak > entry_open and trade["ret"] <= 0),
                     **{col: bool(base[col].iloc[j - 1]) if signal_exit else False for col in SOURCES},
                     "exit_raw_s": None if terminal else bool(base.S_RAW.iloc[j - 1]),
                     "exit_s_suppressed": None if terminal else bool(base.S_RAW.iloc[j - 1] and not base.S_SIGNAL.iloc[j - 1]),
                     "raw_s_count": int(held.S_RAW.sum()), "emitted_s_count": int(held.S_SIGNAL.sum()),
                     "suppressed_s_count": int((held.S_RAW & ~held.S_SIGNAL).sum())})
    trades = pd.DataFrame(rows, columns=TRADE_COLUMNS).astype({
        **{col: "string" for col in TRADE_COLUMNS},
        **{col: "boolean" for col in ("entry_b", "entry_jf", *COMPONENTS, "profit_to_loss",
                                      *SOURCES, "exit_raw_s", "exit_s_suppressed")},
        **{col: "float64" for col in ("entry_open", "exit_price", "return_pct", "peak_close",
                                      "peak_close_gain_pct", "exit_gross_pct", "giveback_pp",
                                      "peak_to_exit_drawdown_pct")},
        **{col: "int64" for col in ("hold_bars", "raw_s_count", "emitted_s_count", "suppressed_s_count")}})
    return trades, result


def audit_frame(symbol: str, frame: pd.DataFrame, start: pd.Timestamp,
                end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """每个raw S保留发出/抑制/执行状态；未成熟标签不借用窗口后的行情。"""
    frame = frame.loc[:end]
    if not frame[list(SOURCES)].any(axis=1).equals(frame.S_RAW):
        raise ValueError(f"{symbol}: S组件并集不一致")
    if not (frame.S_RAW & COUNT(frame.S_RAW, 40).eq(1)).equals(frame.S_SIGNAL):
        raise ValueError(f"{symbol}: S去重标记不一致")
    base = frame.loc[start:end]
    trades, result = audit_trades(symbol, frame, start, end)
    owners = np.full(len(base), None, dtype=object)
    executed = {}
    for raw, row in zip(result["trades"], trades.to_dict("records")):
        owners[raw["i"]:raw["j"]] = row["trade_id"]
        if raw["exit_reason"] == "signal":
            executed[raw["j"] - 1] = row["trade_id"]
    ma200 = frame.CLOSE.rolling(200, min_periods=200).mean()
    rows = []
    for pos in np.flatnonzero(frame.S_RAW.to_numpy()):
        date = frame.index[pos]
        if not start <= date <= end:
            continue
        local = base.index.get_loc(date)
        held, emitted = bool(result["held"][local]), bool(frame.S_SIGNAL.iloc[pos])
        if not emitted:
            status = "suppressed_held" if held else "suppressed_flat"
        elif not held:
            status = "ignored_flat"
        elif local == len(base) - 1:
            status = "pending_at_cutoff"
        elif executed.get(local) == owners[local]:
            status = "executed"
        else:
            raise ValueError(f"{symbol}: 持仓中S与实际退出不一致")
        path = _forward_path(frame, pos, 20, outcome_end=end)
        complete = bool(np.isfinite(path["return"]))
        rows.append({"symbol": symbol, "date": date.date().isoformat(), "emitted": emitted,
                     "held": held, "status": status, "trade_id": owners[local],
                     "regime": _regime(frame.CLOSE.iloc[pos], ma200.iloc[pos]),
                     **{col: bool(frame[col].iloc[pos]) for col in SOURCES},
                     "outcome_complete": complete,
                     "outcome_date": frame.index[pos + 20].date().isoformat() if complete else None,
                     "ret20_pct": path["return"] * 100, "mfe20_pct": path["mfe"] * 100,
                     "mae20_pct": path["mae"] * 100, "win": path["return"] < 0 if complete else None,
                     "interference": bool(path["return"] > 0 and path["mfe"] >= .10) if complete else None})
    columns = ("symbol", "date", "emitted", "held", "status", "trade_id", "regime", *SOURCES,
               "outcome_complete", "outcome_date", "ret20_pct", "mfe20_pct", "mae20_pct", "win", "interference")
    events = pd.DataFrame(rows, columns=columns).astype({
        **{col: "string" for col in columns},
        **{col: "boolean" for col in ("emitted", "held", *SOURCES, "outcome_complete", "win", "interference")},
        **{col: "float64" for col in ("ret20_pct", "mfe20_pct", "mae20_pct")}})
    check = {"symbol": symbol, "raw": len(events), "emitted": int(base.S_SIGNAL.sum()),
             "suppressed": int((base.S_RAW & ~base.S_SIGNAL).sum()),
             "executed": int(events.status.eq("executed").sum()),
             "ignored_flat": int(events.status.eq("ignored_flat").sum()),
             "pending_at_cutoff": int(events.status.eq("pending_at_cutoff").sum()),
             "trades": len(trades), "signal_exits": int(trades.exit_reason.eq("signal").sum()),
             "reconciled": True}
    if check["executed"] != check["signal_exits"]:
        raise ValueError(f"{symbol}: 退出事件数量不一致")
    return events, trades, check


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    """发出/执行/抑制分开统计；重合组件不构成独立样本。"""
    rows = []

    def add(scope, group_by, group, subset):
        complete = subset[subset.outcome_complete.eq(True)]
        n = len(complete)
        rows.append({"scope": scope, "group_by": group_by, "group": group,
                     "events": len(subset), "complete": n, "incomplete": len(subset) - n,
                     "symbols": int(complete.symbol.nunique()), "wins": int(complete.win.sum()),
                     "interference": int(complete.interference.sum()),
                     "win_rate_pct": float(complete.win.mean() * 100) if n else None,
                     "interference_rate_pct": float(complete.interference.mean() * 100) if n else None,
                     "mean_ret20_pct": complete.ret20_pct.mean(),
                     "median_ret20_pct": complete.ret20_pct.median(),
                     "mean_mfe20_pct": complete.mfe20_pct.mean(),
                     "mean_mae20_pct": complete.mae20_pct.mean()})

    scopes = {"raw": events, "emitted": events[events.emitted.eq(True)],
              "suppressed": events[events.emitted.eq(False)]}
    for status in ("executed", "ignored_flat", "suppressed_held", "suppressed_flat", "pending_at_cutoff"):
        scopes[status] = events[events.status.eq(status)]
    for scope, selected in scopes.items():
        add(scope, "all", "all", selected)
        for component in SOURCES:
            add(scope, "component", component, selected[selected[component].eq(True)])
        for regime in ("bull", "bear", "unknown"):
            add(scope, "regime", regime, selected[selected.regime.eq(regime)])
        for symbol in sorted(events.symbol.unique()):
            add(scope, "symbol", symbol, selected[selected.symbol.eq(symbol)])
        combinations = pd.Series(["+".join(col for col in SOURCES if row[col])
                                  for _, row in selected.iterrows()], index=selected.index, dtype="string")
        for combination in sorted(combinations.unique()):
            add(scope, "component_set", combination, selected[combinations.eq(combination)])
    return pd.DataFrame(rows)


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """实际交易净收益与收盘峰值回吐；不以20根标签替代成交绩效。"""
    rows = []

    def add(group_by, group, subset):
        n = len(subset)
        rows.append({"group_by": group_by, "group": group, "trades": n,
                     "symbols": int(subset.symbol.nunique()), "wins": int(subset.return_pct.gt(0).sum()),
                     "win_rate_pct": float(subset.return_pct.gt(0).mean() * 100) if n else None,
                     "mean_return_pct": subset.return_pct.mean(), "median_return_pct": subset.return_pct.median(),
                     "worst_return_pct": subset.return_pct.min(), "profit_to_loss": int(subset.profit_to_loss.sum()),
                     "mean_peak_close_gain_pct": subset.peak_close_gain_pct.mean(),
                     "mean_giveback_pp": subset.giveback_pp.mean(), "median_giveback_pp": subset.giveback_pp.median(),
                     "max_giveback_pp": subset.giveback_pp.max(),
                     **{col: int(subset[col].sum()) for col in ("raw_s_count", "emitted_s_count", "suppressed_s_count")}})

    add("all", "all", trades)
    for reason in ("signal", "trail", "terminal"):
        add("exit_reason", reason, trades[trades.exit_reason.eq(reason)])
    for label, b, jf in (("B", True, False), ("JF", False, True), ("B+JF", True, True)):
        add("entry_kind", label, trades[trades.entry_b.eq(b) & trades.entry_jf.eq(jf)])
    for component in COMPONENTS:
        add("entry_component", component, trades[trades[component].eq(True)])
    for component in SOURCES:
        add("exit_s_component", component, trades[trades[component].eq(True)])
    for symbol in sorted(trades.symbol.unique()):
        add("symbol", symbol, trades[trades.symbol.eq(symbol)])
    return pd.DataFrame(rows)


def run_diagnostic(snapshot: Path, output: Path) -> dict:
    """固定四窗口，仅冻结原生事件和实际成交；不生成候选或晋升。"""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("诊断目录非空，请使用新的输出目录")
    root = Path(__file__).resolve().parents[2]
    paths = ("gcn/backtest/signal_research_r16.py", "gcn/backtest/signal_research_r14.py",
             "gcn/backtest/historical_research.py", "gcn/backtest/signal_audit.py",
             "gcn/backtest/engine.py", "gcn/recipes/gcn_main.py",
             "gcn/core/tdx.py", "gcn/core/indicators.py")
    sources = {name: (root / name).read_bytes() for name in paths}
    protocol_path = root / "reports/gcn-historical-r16-20260905/protocol.md"
    protocol = protocol_path.read_bytes()
    frames, quality = load_snapshot(snapshot)
    tables = {name: [] for name in ("events", "trades", "event_summary", "trade_summary", "reconciliation")}
    for window, first, last in WINDOWS:
        start, end = pd.Timestamp(first), pd.Timestamp(last)
        window_events, window_trades = [], []
        for symbol in CORE:
            raw = frames[symbol].loc[:end]
            frame = compute_ehopt10(raw, version="v5", diagnostics=True)
            default = compute_ehopt10(raw, version="v5")
            if not frame[default.columns].equals(default):
                raise ValueError(f"{symbol}: 诊断开关改变默认输出")
            events, trades, check = audit_frame(symbol, frame, start, end)
            window_events.append(events)
            window_trades.append(trades)
            tables["reconciliation"].append(pd.DataFrame([{"window": window, **check}]))
        events = pd.concat(window_events, ignore_index=True)
        trades = pd.concat(window_trades, ignore_index=True)
        for name, table in (("events", events), ("trades", trades),
                            ("event_summary", summarize_events(events)),
                            ("trade_summary", summarize_trades(trades))):
            table.insert(0, "window", window)
            tables[name].append(table)
    decision = {"research_version": "gcn-historical-r16", "stage": "diagnostic_only",
                "recommended": "v5", "production_changed": False, "core": CORE, "windows": WINDOWS,
                "component_groups_overlap": True,
                "event_outcome": "next OPEN to twentieth CLOSE, complete within window; gross returns",
                "trades": "original v5, next OPEN, trail 0.20, cost 0.001 each side",
                "window_policy": "independent flat start, terminal liquidation; not continuous performance",
                "peak": "highest CLOSE while actually held; excludes real OPEN exit day's CLOSE",
                "giveback_pp": "peak CLOSE gain minus exit gross gain, both relative to entry OPEN"}
    for name, raw in sources.items():
        if (root / name).read_bytes() != raw:
            raise ValueError(f"计算期间源码变化：{name}")
    if protocol_path.read_bytes() != protocol:
        raise ValueError("计算期间协议变化")
    output.mkdir(parents=True, exist_ok=True)
    for name, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_csv(output / (name + ".csv"), index=False)
    (output / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    (output / "protocol.md").write_bytes(protocol)
    for name, raw in sources.items():
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    manifest = {"research_version": "gcn-historical-r16", "parent_manifest_sha256": SNAPSHOT_SHA,
                "source_quality": quality, "protocol_sha256": hashlib.sha256(protocol).hexdigest(),
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
    args = parser.parse_args()
    print(json.dumps(run_diagnostic(args.snapshot, args.output), indent=2, ensure_ascii=False))
