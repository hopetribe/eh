"""历史r2：超跌恢复、趋势回调与短周期风险提示，独立于正式v5。"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.historical_research import (
    accepted_cooldown, training_failures, load_snapshot, CORE, SNAPSHOT_SHA,
    evaluate_rule, event_quality,
)
from gcn.backtest.signal_audit import missed_turn_table, Candidate
from gcn.recipes.gcn_main import compute_ehopt10

RULES = ("v5", "R", "P", "RP", "E", "RE", "PE", "RPE")


def additional_signals(frame: pd.DataFrame) -> pd.DataFrame:
    close, high, low = frame["CLOSE"], frame["HIGH"], frame["LOW"]
    ma5 = close.rolling(5).mean()
    setup = ((low <= low.rolling(20).min() * 1.03)
             & (low <= high.rolling(20).max() * 0.85)).fillna(False)
    recent = setup.shift(1, fill_value=False).rolling(10).sum() > 0
    recovery = (recent & (close > high.shift(1).rolling(3).max()) & (close > ma5)
                & (frame["MACD"] > frame["MACD"].shift(1)) & (frame["RSI1"] > 40))
    ma60 = close.rolling(60).mean()
    pullback = ((close >= close.rolling(200).mean()) & (ma60 >= ma60.shift(10))
                & (close > frame["MID"]) & (close.shift(1) <= frame["MID"].shift(1))
                & (frame["MACD"] > frame["MACD"].shift(1)) & (frame["RSI1"] > 50))
    early_sell = ((close.rolling(5).max() >= low.rolling(60).min() * 1.20)
                  & (close < low.shift(1).rolling(3).min()) & (close < ma5)
                  & (frame["MACD"] < frame["MACD"].shift(1)))
    return pd.DataFrame({"RECOVERY_SETUP": setup,
                         "RECOVERY_SIGNAL": accepted_cooldown(recovery, 20),
                         "PULLBACK_SIGNAL": accepted_cooldown(pullback, 20),
                         "EARLY_S_SIGNAL": accepted_cooldown(early_sell, 20)}, index=frame.index)


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    extra = additional_signals(frame)
    original = frame[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]]
    result = {}
    for rule in RULES:
        signals = original.copy()
        if "R" in rule:
            signals["B_SIGNAL"] |= extra["RECOVERY_SIGNAL"]
        if "P" in rule:
            signals["B_SIGNAL"] |= extra["PULLBACK_SIGNAL"]
        if "E" in rule:
            signals["S_SIGNAL"] |= extra["EARLY_S_SIGNAL"]
        result[rule] = signals
    return result


def candidate_failures(row: dict, base: dict) -> list[str]:
    failures = training_failures(row, base)
    if "E" in row["rule"]:
        for key, passed in (("s_win", row["s_win"] >= base["s_win"]),
                            ("s_interference", row["s_interference"] <= base["s_interference"])):
            if not passed:
                failures.append(key)
    if ("R" in row["rule"] or "P" in row["rule"]) and row["buy_covered"] < base["buy_covered"]:
        failures.append("buy_covered")
    return failures


def choose_training(rows: list[dict]) -> str | None:
    base = next(r for r in rows if r["rule"] == "v5")
    eligible = [r for r in rows if r["rule"] != "v5" and not candidate_failures(r, base)
                and np.isfinite(r["calmar"])]
    eligible.sort(key=lambda r: (-r["calmar"], RULES.index(r["rule"])))
    return eligible[0]["rule"] if eligible else None


def run_training(snapshot: Path, output: Path) -> dict:
    """固定训练阶段；没有合格者时不计算任何挑战者验证收益。"""
    if (output / "manifest.json").exists():
        raise FileExistsError("研究阶段已固化，请使用新的输出目录")
    root = Path(__file__).resolve().parents[2]
    source_paths = ("gcn/backtest/signal_research_r2.py", "gcn/backtest/historical_research.py",
                    "gcn/backtest/signal_audit.py", "gcn/backtest/engine.py",
                    "gcn/recipes/gcn_main.py", "gcn/core/tdx.py")
    sources = {p: (root / p).read_bytes() for p in source_paths}
    protocol = (root / "reports/gcn-historical-r2-20260905/protocol.md").read_bytes()
    frames, quality = load_snapshot(snapshot)
    start, end = pd.Timestamp("2021-08-27"), pd.Timestamp("2024-08-26")
    prepared = {}
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:end], version="v5")
        prepared[symbol] = {"frame": frame, "rules": candidate_signals(frame)}
    rows, trades, events, turns = [], [], [], []
    for rule in RULES:
        perf = evaluate_rule(prepared, rule, start, end)
        event = event_quality(prepared, rule, start, end)
        bundles = {}
        for symbol, bundle in prepared.items():
            signals = bundle["rules"][rule]
            bundles[symbol] = {"v4": bundle["frame"],
                               "entries": {"entry": signals["B_SIGNAL"] | signals["ICON_JUEFAN"]},
                               "exits": {"exit": signals["S_SIGNAL"]}}
        missed = missed_turn_table(bundles, start, end, candidate=Candidate("entry", "exit", .20, None))
        row = {"rule": rule, **perf["stats"], **event["stats"]}
        for kind in ("buy", "sell"):
            actionable = missed[(missed["kind"] == kind) & missed["actionable"]]
            row[kind + "_turns"] = len(actionable)
            row[kind + "_covered"] = int(actionable["covered"].sum())
        rows.append(row)
        for trade in perf["trades"]:
            frame = prepared[trade["symbol"]]["frame"]
            signal_pos = frame.index.get_loc(pd.Timestamp(trade["entry_date"])) - 1
            original = frame["B_SIGNAL"].iloc[signal_pos] or frame["ICON_JUEFAN"].iloc[signal_pos]
            trades.append({"rule": rule, **trade, "entry_origin": "v5" if original else "additional"})
        events.extend({"rule": rule, **e} for e in event["events"])
        missed.insert(0, "rule", rule)
        turns.append(missed)
    selected = choose_training(rows)
    decision = {"research_version": "gcn-historical-r2", "selected": selected,
                "validation_status": "pending" if selected else "not_run_no_eligible_candidate",
                "training_start": str(start.date()), "training_end": str(end.date()),
                "recommended": "v5", "production_changed": False,
                "failures": {r["rule"]: candidate_failures(r, rows[0]) for r in rows[1:]}}
    for path, raw in sources.items():
        if (root / path).read_bytes() != raw:
            raise ValueError(f"计算期间源码变化：{path}")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {"training.csv": pd.DataFrame(rows), "trades.csv": pd.DataFrame(trades),
                 "events.csv": pd.DataFrame(events), "missed_turns.csv": pd.concat(turns, ignore_index=True)}
    for filename, table in artifacts.items():
        table.to_csv(output / filename, index=False)
    (output / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    (output / "protocol.md").write_bytes(protocol)
    for filename, raw in sources.items():
        destination = output / "source_snapshot" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    manifest = {"research_version": "gcn-historical-r2", "parent_manifest_sha256": SNAPSHOT_SHA,
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
