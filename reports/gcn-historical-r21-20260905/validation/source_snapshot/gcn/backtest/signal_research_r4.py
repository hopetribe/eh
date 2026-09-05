"""历史r4：新增P入场的期限与MID失效退出，默认v5不变。"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r3 import candidate_signals as baseline_signals
from gcn.backtest.signal_research_r3 import executed_turns
from gcn.backtest.signal_research_r2 import candidate_failures
from gcn.backtest.historical_research import load_snapshot, CORE, SNAPSHOT_SHA, evaluate_rule, event_quality
from gcn.recipes.gcn_main import compute_ehopt10

CHALLENGERS = ("P-stop5-hold10", "P-stop5-hold20", "P-stop5-hold40", "P-stop5-mid2")
RULES = ("v5", "P-stop5") + CHALLENGERS
HOLD_DAYS = dict(zip(CHALLENGERS[:3], (10, 20, 40)))


def choose_training(rows: list[dict]) -> str | None:
    return _choose_training(rows, CHALLENGERS)


def _choose_training(rows: list[dict], challengers: tuple[str, ...], *, failure_checker=None) -> str | None:
    base = next(r for r in rows if r["rule"] == "v5")
    checker = failure_checker or candidate_failures
    eligible = [r for r in rows if r["rule"] in challengers
                and not checker(r, base) and np.isfinite(r["calmar"])]
    eligible.sort(key=lambda r: (-r["calmar"], challengers.index(r["rule"])))
    return eligible[0]["rule"] if eligible else None


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    baseline = baseline_signals(frame)
    below_mid = frame["CLOSE"] < frame["MID"]
    mid_exit = below_mid & below_mid.shift(1, fill_value=False)
    result = {}
    for rule in RULES:
        signals = baseline["v5" if rule == "v5" else "P-stop5"].copy()
        additional = signals["ENTRY_STOP"].notna()
        signals["ENTRY_LIMIT"] = np.nan
        signals["USE_EXTRA"] = False
        signals["EXTRA_EXIT"] = mid_exit
        if rule in HOLD_DAYS:
            signals.loc[additional, "ENTRY_LIMIT"] = HOLD_DAYS[rule]
        elif rule == "P-stop5-mid2":
            signals["USE_EXTRA"] = additional
        result[rule] = signals
    return result


def run_training(snapshot: Path, output: Path) -> dict:
    return _run_training(snapshot, output, research_version="gcn-historical-r4",
                         protocol_relative="reports/gcn-historical-r4-20260905/protocol.md",
                         candidate_builder=candidate_signals, challengers=CHALLENGERS)


def _run_training(snapshot: Path, output: Path, *, research_version: str,
                  protocol_relative: str, candidate_builder, challengers: tuple[str, ...],
                  extra_sources: tuple[str, ...] = (),
                  controls: tuple[str, ...] = ("v5", "P-stop5"),
                  profit_keeps: dict[str, float] | None = None,
                  entry_profit_enabled_col: str | None = None,
                  entry_floor_col: str | None = None,
                  entry_floor_confirm_bars: int = 1,
                  entry_breakeven_base_col: str | None = None,
                  failure_checker=None) -> dict:
    """增量研究共用执行与归档口径，候选、对照和协议由各研究显式指定。"""
    if (output / "manifest.json").exists():
        raise FileExistsError("研究阶段已固化，请使用新的输出目录")
    root = Path(__file__).resolve().parents[2]
    source_paths = ("gcn/backtest/signal_research_r4.py", "gcn/backtest/signal_research_r3.py",
                    "gcn/backtest/signal_research_r2.py", "gcn/backtest/historical_research.py",
                    "gcn/backtest/signal_audit.py", "gcn/backtest/engine.py",
                    "gcn/recipes/gcn_main.py", "gcn/core/tdx.py", "gcn/core/indicators.py") + extra_sources
    sources = {p: (root / p).read_bytes() for p in source_paths}
    protocol_path = root / protocol_relative
    protocol = protocol_path.read_bytes()
    frames, quality = load_snapshot(snapshot)
    start, end = pd.Timestamp("2021-08-27"), pd.Timestamp("2024-08-26")
    prepared = {}
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:end], version="v5")
        prepared[symbol] = {"frame": frame, "rules": candidate_builder(frame)}
    rows, trades, events, turns = [], [], [], []
    for rule in controls + challengers:
        perf = evaluate_rule(prepared, rule, start, end, entry_hard_stop_col="ENTRY_STOP",
                             entry_max_hold_col="ENTRY_LIMIT", entry_exit_cols=("USE_EXTRA", "EXTRA_EXIT"),
                             profit_keep=(profit_keeps or {}).get(rule),
                             entry_profit_enabled_col=entry_profit_enabled_col,
                             entry_floor_col=entry_floor_col,
                             entry_floor_confirm_bars=entry_floor_confirm_bars,
                             entry_breakeven_base_col=entry_breakeven_base_col,
                             include_positions=True)
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
        for reason in ("hard_stop", "max_hold", "entry_signal") + (("entry_floor",) if entry_floor_col else ()):
            row[reason + "_exits"] = sum(t["exit_reason"] == reason for t in perf["trades"])
        if entry_breakeven_base_col is not None:
            row["breakeven_exits"] = sum(t["exit_reason"] == "breakeven" for t in perf["trades"])
        rows.append(row)
        for trade in perf["trades"]:
            bundle = prepared[trade["symbol"]]
            frame = bundle["frame"]
            signal_pos = frame.index.get_loc(pd.Timestamp(trade["entry_date"])) - 1
            original = frame["B_SIGNAL"].iloc[signal_pos] or frame["ICON_JUEFAN"].iloc[signal_pos]
            settings = bundle["rules"][rule].iloc[signal_pos]
            trades.append({"rule": rule, **trade, "entry_origin": "v5" if original else "additional",
                           "entry_stop_pct": float(settings["ENTRY_STOP"] * 100),
                           "entry_limit": float(settings["ENTRY_LIMIT"]),
                           "use_extra_exit": bool(settings["USE_EXTRA"])})
            if entry_profit_enabled_col is not None:
                value = settings[entry_profit_enabled_col]
                trades[-1]["entry_profit_enabled"] = False if pd.isna(value) else bool(value)
            if entry_floor_col is not None:
                trades[-1].update(entry_floor_price=float(settings[entry_floor_col]),
                                  entry_signal_date=frame.index[signal_pos].date().isoformat(),
                                  entry_b=bool(frame.B_SIGNAL.iloc[signal_pos]),
                                  entry_jf=bool(frame.ICON_JUEFAN.iloc[signal_pos]))
            if entry_floor_confirm_bars != 1:
                trades[-1]["entry_floor_confirm_bars"] = int(entry_floor_confirm_bars)
            if entry_breakeven_base_col is not None:
                trades[-1].update(entry_signal_date=frame.index[signal_pos].date().isoformat(),
                                  entry_b=bool(frame.B_SIGNAL.iloc[signal_pos]),
                                  entry_jf=bool(frame.ICON_JUEFAN.iloc[signal_pos]))
        events.extend({"rule": rule, **e} for e in event["events"])
        missed.insert(0, "rule", rule)
        turns.append(missed)
    checker = failure_checker or candidate_failures
    selected = _choose_training(rows, challengers, failure_checker=checker)
    decision = {"research_version": research_version, "selected": selected,
                "validation_status": "pending" if selected else "not_run_no_eligible_candidate",
                "training_start": str(start.date()), "training_end": str(end.date()),
                "recommended": "v5", "production_changed": False,
                "failures": {r["rule"]: checker(r, rows[0]) for r in rows if r["rule"] in challengers}}
    for path, raw in sources.items():
        if (root / path).read_bytes() != raw:
            raise ValueError(f"计算期间源码变化：{path}")
    if protocol_path.read_bytes() != protocol:
        raise ValueError("计算期间协议变化")
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
    manifest = {"research_version": research_version, "parent_manifest_sha256": SNAPSHOT_SHA,
                "source_quality": quality, "protocol_sha256": hashlib.sha256(protocol).hexdigest(),
                "algorithm_sources": {p: hashlib.sha256(raw).hexdigest() for p, raw in sources.items()},
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(output.iterdir()) if p.is_file()}}
    if profit_keeps is not None:
        manifest["profit_keeps"] = profit_keeps
    if entry_profit_enabled_col is not None:
        manifest["entry_profit_enabled_col"] = entry_profit_enabled_col
    if entry_floor_col is not None:
        manifest["entry_floor_col"] = entry_floor_col
    if entry_floor_confirm_bars != 1:
        manifest["entry_floor_confirm_bars"] = int(entry_floor_confirm_bars)
    if entry_breakeven_base_col is not None:
        manifest.update(entry_breakeven_base_col=entry_breakeven_base_col,
                        breakeven_arm_r=1., breakeven_reference_cost=.001)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return decision


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.snapshot, args.output), indent=2, ensure_ascii=False))
