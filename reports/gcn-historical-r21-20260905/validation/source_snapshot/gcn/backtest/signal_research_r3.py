"""历史r3：只给新增P来源入场附加初始风险退出，保留原v5。"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r2 import additional_signals, candidate_failures
from gcn.backtest.signal_audit import missed_turn_table
from gcn.backtest.historical_research import load_snapshot, CORE, SNAPSHOT_SHA, evaluate_rule, event_quality
from gcn.recipes.gcn_main import compute_ehopt10

STOPS = {"v5": None, "P": None, "P-stop5": .05, "P-stop8": .08, "P-stop12": .12}


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pullback = additional_signals(frame)["PULLBACK_SIGNAL"]
    original = frame["B_SIGNAL"] | frame["ICON_JUEFAN"]
    result = {}
    for rule, stop in STOPS.items():
        signals = frame[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]].copy()
        signals["ENTRY_STOP"] = np.nan
        if rule != "v5":
            signals["B_SIGNAL"] |= pullback
        if stop is not None:
            signals.loc[pullback & ~original, "ENTRY_STOP"] = stop
        result[rule] = signals
    return result


def executed_turns(prepared: dict, result: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """仅用于事后覆盖诊断：按同一实际成交路径和持仓判断可行动性。"""
    bundles = {}
    for symbol, bundle in prepared.items():
        frame = bundle["frame"].loc[:end].copy()
        frame[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]] = False
        for trade in result["trades"]:
            if trade["symbol"] != symbol:
                continue
            i = frame.index.get_loc(pd.Timestamp(trade["entry_date"]))
            frame.loc[frame.index[i - 1], "B_SIGNAL"] = True
            if trade["exit_reason"] != "terminal":
                j = frame.index.get_loc(pd.Timestamp(trade["exit_date"]))
                frame.loc[frame.index[j - 1], "S_SIGNAL"] = True
        bundles[symbol] = {"v4": frame}
    turns = missed_turn_table(bundles, start, end)
    turns["actionable"] = [
        not bool(result["positions"].loc[pd.Timestamp(r.date), r.symbol]) if r.kind == "buy"
        else bool(result["positions"].loc[pd.Timestamp(r.date), r.symbol])
        for r in turns.itertuples()
    ]
    return turns


def choose_training(rows: list[dict]) -> str | None:
    base = next(r for r in rows if r["rule"] == "v5")
    eligible = [r for r in rows if STOPS.get(r["rule"]) is not None
                and not candidate_failures(r, base) and np.isfinite(r["calmar"])]
    eligible.sort(key=lambda r: (-r["calmar"], list(STOPS).index(r["rule"])))
    return eligible[0]["rule"] if eligible else None


def run_training(snapshot: Path, output: Path) -> dict:
    if (output / "manifest.json").exists():
        raise FileExistsError("研究阶段已固化，请使用新的输出目录")
    root = Path(__file__).resolve().parents[2]
    source_paths = ("gcn/backtest/signal_research_r3.py", "gcn/backtest/signal_research_r2.py",
                    "gcn/backtest/historical_research.py", "gcn/backtest/signal_audit.py",
                    "gcn/backtest/engine.py", "gcn/recipes/gcn_main.py", "gcn/core/tdx.py")
    sources = {p: (root / p).read_bytes() for p in source_paths}
    protocol = (root / "reports/gcn-historical-r3-20260905/protocol.md").read_bytes()
    frames, quality = load_snapshot(snapshot)
    start, end = pd.Timestamp("2021-08-27"), pd.Timestamp("2024-08-26")
    prepared = {}
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:end], version="v5")
        prepared[symbol] = {"frame": frame, "rules": candidate_signals(frame)}
    rows, trades, events, turns = [], [], [], []
    for rule in STOPS:
        perf = evaluate_rule(prepared, rule, start, end, entry_hard_stop_col="ENTRY_STOP", include_positions=True)
        event = event_quality(prepared, rule, start, end)
        missed = executed_turns(prepared, perf, start, end)
        row = {"rule": rule, **perf["stats"], **event["stats"]}
        for kind in ("buy", "sell"):
            actionable = missed[(missed["kind"] == kind) & missed["actionable"]]
            row[kind + "_turns"] = len(actionable)
            row[kind + "_covered"] = int(actionable["covered"].sum())
        equity = (1 + perf["returns"].mean(axis=1)).cumprod()
        trough = (1 - equity / equity.cummax()).idxmax()
        row["drawdown_peak"] = str(equity.loc[:trough].idxmax().date())
        row["drawdown_trough"] = str(trough.date())
        row["hard_stop_exits"] = sum(t["exit_reason"] == "hard_stop" for t in perf["trades"])
        rows.append(row)
        for trade in perf["trades"]:
            bundle = prepared[trade["symbol"]]
            frame = bundle["frame"]
            signal_pos = frame.index.get_loc(pd.Timestamp(trade["entry_date"])) - 1
            original = frame["B_SIGNAL"].iloc[signal_pos] or frame["ICON_JUEFAN"].iloc[signal_pos]
            risk = bundle["rules"][rule]["ENTRY_STOP"].iloc[signal_pos]
            trades.append({"rule": rule, **trade, "entry_origin": "v5" if original else "additional",
                           "entry_stop_pct": float(risk * 100)})
        events.extend({"rule": rule, **e} for e in event["events"])
        missed.insert(0, "rule", rule)
        turns.append(missed)
    selected = choose_training(rows)
    decision = {"research_version": "gcn-historical-r3", "selected": selected,
                "validation_status": "pending" if selected else "not_run_no_eligible_candidate",
                "training_start": str(start.date()), "training_end": str(end.date()),
                "recommended": "v5", "production_changed": False,
                "failures": {r["rule"]: candidate_failures(r, rows[0]) for r in rows if STOPS[r["rule"]] is not None}}
    for path, raw in sources.items():
        if (root / path).read_bytes() != raw:
            raise ValueError(f"计算期间源码变化：{path}")
    output.mkdir(parents=True, exist_ok=True)
    for filename, table in {"training.csv": pd.DataFrame(rows), "trades.csv": pd.DataFrame(trades),
                            "events.csv": pd.DataFrame(events), "missed_turns.csv": pd.concat(turns, ignore_index=True)}.items():
        table.to_csv(output / filename, index=False)
    (output / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    (output / "protocol.md").write_bytes(protocol)
    for filename, raw in sources.items():
        destination = output / "source_snapshot" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    manifest = {"research_version": "gcn-historical-r3", "parent_manifest_sha256": SNAPSHOT_SHA,
                "source_quality": quality, "protocol_sha256": hashlib.sha256(protocol).hexdigest(),
                "algorithm_sources": {p: hashlib.sha256(raw).hexdigest() for p, raw in sources.items()},
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return decision


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.snapshot, args.output), indent=2, ensure_ascii=False))
