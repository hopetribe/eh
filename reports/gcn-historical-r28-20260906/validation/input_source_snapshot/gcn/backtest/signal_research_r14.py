"""r14：原生v5 Setup因果溯源；仅诊断，不生成新交易策略。"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.historical_research import CORE, SNAPSHOT_SHA, load_snapshot
from gcn.backtest.signal_audit import _forward_path
from gcn.recipes.gcn_main import compute_ehopt10

COMPONENTS = ("B_BASE_BULL", "B_STAGE_COMPONENT", "B_BEAR_RECOVER", "B_CRASH_RECOVER")
TRACE_COLUMNS = ("setup_i", "setup_date", "setup_high", "setup_regime", *COMPONENTS,
                 "status", "resolution_i", "resolution_date", "resolution_regime", "wait_bars")
WINDOWS = (("training", "2021-08-27", "2024-08-26"),
           ("validation", "2024-08-27", "2025-08-26"),
           ("recent", "2025-08-27", "2026-08-27"),
           ("full", "2021-08-27", "2026-08-27"))


def _regime(close: float, ma200: float) -> str:
    return "unknown" if not np.isfinite(ma200) else "bull" if close >= ma200 else "bear"


def trace_setups(frame: pd.DataFrame) -> pd.DataFrame:
    """逐根独立还原五根确认状态，组件保留Setup当根完整多标签集合。

    pending行只包含截至最后一根已知的等待天数；已决行在追加未来数据后不变。
    调用方负责用配方输出逐日核对确认及过期日期，不依赖未来收益生成状态。
    """
    ma200 = frame.CLOSE.rolling(200, min_periods=200).mean()
    rows, pending = [], None

    def resolve(status: str, pos: int) -> None:
        nonlocal pending
        pending.update(status=status, resolution_i=pos,
                       resolution_date=frame.index[pos].date().isoformat(),
                       resolution_regime=_regime(frame.CLOSE.iloc[pos], ma200.iloc[pos]),
                       wait_bars=pos - pending["setup_i"])
        pending = None

    for pos in range(len(frame)):
        if frame.B_SETUP.iloc[pos]:
            if pending is not None:
                resolve("replaced", pos)
            pending = {"setup_i": pos, "setup_date": frame.index[pos].date().isoformat(),
                       "setup_high": float(frame.HIGH.iloc[pos]),
                       "setup_regime": _regime(frame.CLOSE.iloc[pos], ma200.iloc[pos]),
                       **{col: bool(frame[col].iloc[pos]) for col in COMPONENTS},
                       "status": "pending", "resolution_i": None, "resolution_date": None,
                       "resolution_regime": None, "wait_bars": 0}
            rows.append(pending)
            continue
        if pending is None:
            continue
        pending["wait_bars"] = pos - pending["setup_i"]
        close, mid = frame.CLOSE.iloc[pos], frame.MID.iloc[pos]
        if (np.isfinite(pending["setup_high"]) and np.isfinite(close) and np.isfinite(mid)
                and close > pending["setup_high"] and close > mid):
            resolve("confirmed", pos)
        elif pending["wait_bars"] == 5:
            resolve("expired", pos)
    return pd.DataFrame(rows, columns=TRACE_COLUMNS).astype({
        "setup_i": "int64", "resolution_i": "Int64", "wait_bars": "int64",
        "setup_high": "float64", **{col: bool for col in COMPONENTS},
        **{col: "string" for col in ("setup_date", "setup_regime", "status",
                                     "resolution_date", "resolution_regime")},
    })


def diagnose_frame(symbol: str, frame: pd.DataFrame, start: pd.Timestamp,
                   end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """核对全前缀状态后，只报告窗口内事件，保留跨起点Setup的真实来源。"""
    frame = frame.loc[:end]
    trace = trace_setups(frame)
    for status, column in (("confirmed", "B_ENTRY_SIGNAL"), ("expired", "B_SETUP_EXPIRED")):
        replay = pd.Series(False, index=frame.index)
        replay.iloc[trace.loc[trace.status.eq(status), "resolution_i"].astype(int).tolist()] = True
        if not replay.equals(frame[column]):
            raise ValueError(f"{symbol}: 确认/过期追踪与{column}不一致")
    if not frame.B_SIGNAL.equals(frame.B_ENTRY_SIGNAL):
        raise ValueError(f"{symbol}: B买与确认标记不一致")
    ma200 = frame.CLOSE.rolling(200, min_periods=200).mean()
    sources = {int(row["resolution_i"]): row for row in
               trace[trace.status.eq("confirmed")].to_dict("records")}
    rows = []
    for signal, column in (("b", "B_SIGNAL"), ("jf", "ICON_JUEFAN"), ("s", "S_SIGNAL")):
        for pos in np.flatnonzero(frame[column].to_numpy()):
            if not start <= frame.index[pos] <= end:
                continue
            source = sources[pos] if signal == "b" else {}
            path = _forward_path(frame, pos, 20, outcome_end=end)
            complete = bool(np.isfinite(path["return"]))
            rows.append({"symbol": symbol, "date": frame.index[pos].date().isoformat(),
                         "signal": signal, "regime": _regime(frame.CLOSE.iloc[pos], ma200.iloc[pos]),
                         "setup_date": source.get("setup_date"),
                         "setup_regime": source.get("setup_regime"),
                         "wait_bars": source.get("wait_bars"),
                         **{col: source.get(col, False) for col in COMPONENTS},
                         "outcome_complete": complete,
                         "outcome_date": frame.index[pos + 20].date().isoformat() if complete else None,
                         "ret20_pct": path["return"] * 100, "mfe20_pct": path["mfe"] * 100,
                         "mae20_pct": path["mae"] * 100,
                         "win": (path["return"] < 0 if signal == "s" else path["return"] > 0) if complete else None,
                         "interference": ((path["return"] > 0 and path["mfe"] >= .10) if signal == "s"
                                          else (path["return"] < 0 and path["mae"] <= -.08)) if complete else None})
    columns = ("symbol", "date", "signal", "regime", "setup_date", "setup_regime", "wait_bars",
               *COMPONENTS, "outcome_complete", "outcome_date", "ret20_pct", "mfe20_pct",
               "mae20_pct", "win", "interference")
    events = pd.DataFrame(rows, columns=columns).sort_values(["date", "signal"]).reset_index(drop=True)
    first = start.date().isoformat()
    states = trace[trace.setup_date.ge(first) | trace.resolution_date.ge(first).fillna(False)
                   | trace.status.eq("pending")].reset_index(drop=True)
    states.insert(0, "symbol", symbol)
    window = frame.loc[start:end]
    check = {"symbol": symbol, "start": first, "end": end.date().isoformat(),
             "prefix_bars": len(frame), "all_setups": len(trace),
             "all_confirmed": int(trace.status.eq("confirmed").sum()),
             "all_expired": int(trace.status.eq("expired").sum()),
             "confirmed": int(window.B_ENTRY_SIGNAL.sum()), "expired": int(window.B_SETUP_EXPIRED.sum()),
             "reconciled": True}
    return events, states, check


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    """完整20根结果作为唯一胜率分母；组件分组可重叠，组合标签则互斥。"""
    rows = []

    def add(signal: str, group_by: str, group: str, subset: pd.DataFrame) -> None:
        complete = subset[subset.outcome_complete.eq(True)]
        n = len(complete)
        rows.append({"signal": signal, "group_by": group_by, "group": group,
                     "events": len(subset), "complete": n, "incomplete": len(subset) - n,
                     "symbols": int(complete.symbol.nunique()), "wins": int(complete.win.sum()),
                     "interference": int(complete.interference.sum()),
                     "win_rate_pct": float(complete.win.mean() * 100) if n else None,
                     "interference_rate_pct": float(complete.interference.mean() * 100) if n else None,
                     "mean_ret20_pct": complete.ret20_pct.mean(),
                     "median_ret20_pct": complete.ret20_pct.median(),
                     "mean_mfe20_pct": complete.mfe20_pct.mean(),
                     "mean_mae20_pct": complete.mae20_pct.mean()})

    for signal in ("b", "jf", "s"):
        selected = events[events.signal.eq(signal)]
        add(signal, "all", "all", selected)
        for regime in ("bull", "bear", "unknown"):
            add(signal, "regime", regime, selected[selected.regime.eq(regime)])
        for symbol in sorted(events.symbol.unique()):
            add(signal, "symbol", symbol, selected[selected.symbol.eq(symbol)])
    b = events[events.signal.eq("b")]
    for col in COMPONENTS:
        add("b", "component", col, b[b[col].eq(True)])
    for wait in range(1, 6):
        add("b", "wait_bars", str(wait), b[b.wait_bars.eq(wait)])
    for regime in ("bull", "bear", "unknown"):
        add("b", "setup_regime", regime, b[b.setup_regime.eq(regime)])
        for confirmed in ("bull", "bear", "unknown"):
            add("b", "regime_pair", regime + "->" + confirmed,
                b[b.setup_regime.eq(regime) & b.regime.eq(confirmed)])
    combinations = pd.Series(["+".join(col for col in COMPONENTS if row[col])
                              for _, row in b.iterrows()], index=b.index, dtype="string")
    for combination in sorted(combinations.unique()):
        add("b", "component_set", combination, b[combinations.eq(combination)])
    return pd.DataFrame(rows)


def run_diagnostic(snapshot: Path, output: Path) -> dict:
    """冻结输入、协议、源码和四窗口诊断；本阶段没有候选选型或晋升。"""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("诊断目录非空，请使用新的输出目录")
    root = Path(__file__).resolve().parents[2]
    paths = ("gcn/backtest/signal_research_r14.py", "gcn/backtest/historical_research.py",
             "gcn/backtest/signal_audit.py", "gcn/backtest/engine.py", "gcn/recipes/gcn_main.py",
             "gcn/core/tdx.py", "gcn/core/indicators.py")
    sources = {name: (root / name).read_bytes() for name in paths}
    protocol_path = root / "reports/gcn-historical-r14-20260905/protocol.md"
    protocol = protocol_path.read_bytes()
    frames, quality = load_snapshot(snapshot)
    tables = {name: [] for name in ("events", "states", "summary", "reconciliation")}
    for window, first, last in WINDOWS:
        start, end = pd.Timestamp(first), pd.Timestamp(last)
        window_events = []
        for symbol in CORE:
            raw = frames[symbol].loc[:end]
            frame = compute_ehopt10(raw, version="v5", diagnostics=True)
            default = compute_ehopt10(raw, version="v5")
            if not frame[default.columns].equals(default):
                raise ValueError(f"{symbol}: 诊断开关改变默认输出")
            if not frame[list(COMPONENTS)].any(axis=1).equals(frame.B_ALL_RAW):
                raise ValueError(f"{symbol}: 组件并集与B_ALL_RAW不一致")
            events, states, check = diagnose_frame(symbol, frame, start, end)
            window_events.append(events)
            states.insert(0, "window", window)
            tables["states"].append(states)
            tables["reconciliation"].append(pd.DataFrame([{"window": window, **check}]))
        events = pd.concat(window_events, ignore_index=True)
        summary = summarize_events(events)
        for name, table in (("events", events), ("summary", summary)):
            table.insert(0, "window", window)
            tables[name].append(table)
    decision = {"research_version": "gcn-historical-r14", "stage": "diagnostic_only",
                "recommended": "v5", "production_changed": False, "core": CORE,
                "windows": WINDOWS, "component_groups_overlap": True,
                "outcome": "next OPEN to twentieth CLOSE, complete within window; gross returns"}
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
    manifest = {"research_version": "gcn-historical-r14", "parent_manifest_sha256": SNAPSHOT_SHA,
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
