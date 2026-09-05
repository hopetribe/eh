"""r7固定时间验证：只评估训练选出的一个候选，不允许换第二名。"""
from __future__ import annotations

import hashlib
import json
import platform
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r2 import candidate_failures
from gcn.backtest.signal_research_r3 import executed_turns
from gcn.backtest.signal_research_r4 import _choose_training
from gcn.backtest.signal_research_r7 import candidate_signals, CHALLENGERS, RULES
from gcn.backtest.historical_research import load_snapshot, CORE, SNAPSHOT_SHA, evaluate_rule, event_quality
from gcn.recipes.gcn_main import compute_ehopt10


def validation_failures(row: dict, base: dict) -> list[str]:
    failures = candidate_failures(row, base)
    for key in ("win", "sharpe"):
        if row[key] is None or not row[key] >= base[key]:
            failures.append(key)
    if not (row["entry_win"] > base["entry_win"] or row["buy_covered"] > base["buy_covered"]):
        failures.append("no_entry_improvement")
    return failures


def run_validation(snapshot: Path, training: Path, output: Path) -> dict:
    return _run_validation(snapshot, training, output, research_version="gcn-historical-r7",
                           protocol_relative="reports/gcn-historical-r7-20260905/protocol.md",
                           candidate_builder=candidate_signals, rules=RULES, challengers=CHALLENGERS,
                           validator_source="gcn/backtest/signal_research_r7_validation.py")


def _run_validation(snapshot: Path, training: Path, output: Path, *, research_version: str,
                    protocol_relative: str, candidate_builder, rules: tuple[str, ...],
                    challengers: tuple[str, ...], validator_source: str,
                    profit_keeps: dict[str, float] | None = None,
                    entry_profit_enabled_col: str | None = None,
                    entry_floor_col: str | None = None,
                    entry_floor_confirm_bars: int = 1,
                    entry_breakeven_base_col: str | None = None,
                    extra_validator_sources: tuple[str, ...] = (),
                    training_failure_checker=None, validation_failure_checker=None) -> dict:
    """同版训练/验证与参数绑定；各研究显式指定候选，不放宽原门槛。"""
    if (output / "manifest.json").exists():
        raise FileExistsError("验证阶段已固化，请使用新的输出目录")
    root = Path(__file__).resolve().parents[2]
    training_raw = (training / "manifest.json").read_bytes()
    training_manifest = json.loads(training_raw)
    if (training_manifest["research_version"] != research_version
            or training_manifest["parent_manifest_sha256"] != SNAPSHOT_SHA):
        raise ValueError("训练研究版本或输入摘要不匹配")
    for name, digest in training_manifest["outputs"].items():
        if Path(name).name != name or hashlib.sha256((training / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"训练工件摘要不匹配：{name}")
    rows = pd.read_csv(training / "training.csv").to_dict("records")
    selected = _choose_training(rows, challengers, failure_checker=training_failure_checker)
    decision_before = json.loads((training / "decision.json").read_text())
    if ([r["rule"] for r in rows] != list(rules) or selected is None
            or selected != decision_before["selected"]):
        raise ValueError("训练未选出可验证的唯一候选")
    if (training_manifest.get("profit_keeps") != profit_keeps
            or training_manifest.get("entry_profit_enabled_col") != entry_profit_enabled_col):
        raise ValueError("训练盈利保护配置不匹配")
    recorded_confirm = training_manifest.get("entry_floor_confirm_bars", 1)
    if (training_manifest.get("entry_floor_col") != entry_floor_col
            or type(recorded_confirm) is not int or recorded_confirm < 1
            or isinstance(entry_floor_confirm_bars, (bool, np.bool_))
            or not isinstance(entry_floor_confirm_bars, Integral)
            or recorded_confirm != entry_floor_confirm_bars):
        raise ValueError("训练底部失效配置不匹配")
    breakeven_config = {"entry_breakeven_base_col": entry_breakeven_base_col,
                        "breakeven_arm_r": 1.0, "breakeven_reference_cost": .001}
    if entry_breakeven_base_col is None:
        if any(training_manifest.get(key) is not None for key in breakeven_config):
            raise ValueError("训练净保本配置不匹配")
    elif (not isinstance(entry_breakeven_base_col, str) or not entry_breakeven_base_col
          or any(training_manifest.get(key) != value for key, value in breakeven_config.items())
          or any(type(training_manifest.get(key)) not in (int, float)
                 for key in ("breakeven_arm_r", "breakeven_reference_cost"))):
        raise ValueError("训练净保本配置不匹配")
    sources = {}
    for name, digest in training_manifest["algorithm_sources"].items():
        if not name.startswith("gcn/") or ".." in Path(name).parts or not name.endswith(".py"):
            raise ValueError("训练源码路径无效")
        raw = (root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"训练后算法源码变化：{name}")
        sources[name] = raw
    for own_source in ("gcn/backtest/signal_research_r7_validation.py", validator_source) + extra_validator_sources:
        sources[own_source] = (root / own_source).read_bytes()
    protocol = (root / protocol_relative).read_bytes()
    if hashlib.sha256(protocol).hexdigest() != training_manifest["protocol_sha256"]:
        raise ValueError("训练后协议变化")
    environment = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    if environment != training_manifest["environment"]:
        raise ValueError("验证运行环境与训练不一致")
    frames, quality = load_snapshot(snapshot)
    start, end = pd.Timestamp("2024-08-27"), pd.Timestamp("2025-08-26")
    prepared = {}
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:end], version="v5")
        prepared[symbol] = {"frame": frame, "rules": candidate_builder(frame)}
    comparisons, trades, events, turns = [], [], [], []
    for rule in ("v5", selected):
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
        comparisons.append(row)
        for trade in perf["trades"]:
            frame = prepared[trade["symbol"]]["frame"]
            pos = frame.index.get_loc(pd.Timestamp(trade["entry_date"])) - 1
            original = frame["B_SIGNAL"].iloc[pos] or frame["ICON_JUEFAN"].iloc[pos]
            trades.append({"rule": rule, **trade, "entry_origin": "v5" if original else "additional"})
            if entry_profit_enabled_col is not None:
                value = prepared[trade["symbol"]]["rules"][rule][entry_profit_enabled_col].iloc[pos]
                trades[-1]["entry_profit_enabled"] = False if pd.isna(value) else bool(value)
            if entry_floor_col is not None:
                settings = prepared[trade["symbol"]]["rules"][rule].iloc[pos]
                trades[-1].update(entry_floor_price=float(settings[entry_floor_col]),
                                  entry_signal_date=frame.index[pos].date().isoformat(),
                                  entry_b=bool(frame.B_SIGNAL.iloc[pos]),
                                  entry_jf=bool(frame.ICON_JUEFAN.iloc[pos]))
            if entry_floor_confirm_bars != 1:
                trades[-1]["entry_floor_confirm_bars"] = int(entry_floor_confirm_bars)
            if entry_breakeven_base_col is not None:
                trades[-1].update(entry_signal_date=frame.index[pos].date().isoformat(),
                                  entry_b=bool(frame.B_SIGNAL.iloc[pos]),
                                  entry_jf=bool(frame.ICON_JUEFAN.iloc[pos]))
        events.extend({"rule": rule, **e} for e in event["events"])
        missed.insert(0, "rule", rule)
        turns.append(missed)
    checker = validation_failure_checker or validation_failures
    failures = checker(comparisons[1], comparisons[0])
    decision = {"research_version": research_version, "selected": selected,
                "status": "rejected_keep_v5" if failures else "passed_validation_pending_stress",
                "validation_start": str(start.date()), "validation_end": str(end.date()),
                "failures": failures, "recommended": "v5", "production_changed": False}
    for name, raw in sources.items():
        if (root / name).read_bytes() != raw:
            raise ValueError(f"计算期间源码变化：{name}")
    if (training / "manifest.json").read_bytes() != training_raw:
        raise ValueError("计算期间训练manifest变化")
    output.mkdir(parents=True, exist_ok=True)
    for name, table in {"comparisons.csv": pd.DataFrame(comparisons), "trades.csv": pd.DataFrame(trades),
                        "events.csv": pd.DataFrame(events), "missed_turns.csv": pd.concat(turns, ignore_index=True)}.items():
        table.to_csv(output / name, index=False)
    (output / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    (output / "protocol.md").write_bytes(protocol)
    for name, raw in sources.items():
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    manifest = {"research_version": research_version, "stage": "validation",
                "training_manifest_sha256": hashlib.sha256(training_raw).hexdigest(),
                "parent_manifest_sha256": SNAPSHOT_SHA, "protocol_sha256": hashlib.sha256(protocol).hexdigest(),
                "source_quality": quality, "environment": environment,
                "algorithm_sources": {name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()},
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
        manifest.update(breakeven_config)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return decision


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--training", type=Path, default=Path("reports/gcn-historical-r7-20260905/results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_validation(args.snapshot, args.training, args.output), indent=2, ensure_ascii=False))
