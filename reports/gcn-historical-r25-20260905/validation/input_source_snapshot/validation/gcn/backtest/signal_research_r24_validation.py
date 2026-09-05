"""r24固定验证：绑定训练共同压力标记与自然边界，沿用原退出改善门槛。"""
from __future__ import annotations

import json
from pathlib import Path

from gcn.backtest.signal_research_r17 import candidate_failures
from gcn.backtest.signal_research_r19_validation import validation_failures
from gcn.backtest.signal_research_r24 import candidate_signals, CHALLENGERS, RULES
from gcn.backtest.signal_research_r7_validation import _run_validation


def run_validation(snapshot: Path, training: Path, output: Path) -> dict:
    return _run_validation(snapshot, training, output, research_version="gcn-historical-r24",
                           protocol_relative="reports/gcn-historical-r24-20260905/protocol.md",
                           candidate_builder=candidate_signals, rules=RULES, challengers=CHALLENGERS,
                           validator_source="gcn/backtest/signal_research_r24_validation.py",
                           entry_joint_pressure_col="ENTRY_JOINT_PRESSURE",
                           extra_validator_sources=("gcn/backtest/signal_research_r19_validation.py",),
                           training_failure_checker=candidate_failures, validation_failure_checker=validation_failures)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--training", type=Path, default=Path("reports/gcn-historical-r24-20260905/results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_validation(args.snapshot, args.training, args.output), indent=2, ensure_ascii=False))
