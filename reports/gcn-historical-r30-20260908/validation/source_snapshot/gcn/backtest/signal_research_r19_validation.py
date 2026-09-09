"""r19固定验证：绑定训练价格/确认长度，以冻结退出改善门槛判定。"""
from __future__ import annotations

import json
from pathlib import Path

from gcn.backtest.signal_research_r17 import candidate_failures
from gcn.backtest.signal_research_r19 import candidate_signals, CHALLENGERS, RULES, CONFIRM_BARS
from gcn.backtest.signal_research_r7_validation import _run_validation


def validation_failures(row: dict, base: dict) -> list[str]:
    failures = candidate_failures(row, base)
    for key in ("win", "sharpe"):
        if row[key] is None or not row[key] >= base[key]:
            failures.append(key)
    if not (row["mdd"] <= base["mdd"] * .95
            or (row["win"] is not None and row["win"] > base["win"])):
        failures.append("no_material_improvement")
    return failures


def run_validation(snapshot: Path, training: Path, output: Path) -> dict:
    return _run_validation(snapshot, training, output, research_version="gcn-historical-r19",
                           protocol_relative="reports/gcn-historical-r19-20260905/protocol.md",
                           candidate_builder=candidate_signals, rules=RULES, challengers=CHALLENGERS,
                           validator_source="gcn/backtest/signal_research_r19_validation.py",
                           entry_floor_col="ENTRY_FLOOR", entry_floor_confirm_bars=CONFIRM_BARS,
                           training_failure_checker=candidate_failures,
                           validation_failure_checker=validation_failures)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--training", type=Path, default=Path("reports/gcn-historical-r19-20260905/results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_validation(args.snapshot, args.training, args.output), indent=2, ensure_ascii=False))
