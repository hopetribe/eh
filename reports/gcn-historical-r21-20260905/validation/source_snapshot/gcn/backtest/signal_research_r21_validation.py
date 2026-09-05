"""r21固定验证：绑定1R成熟后保本配置，复用已冻结退出门槛。"""
from __future__ import annotations

import json
from pathlib import Path

from gcn.backtest.signal_research_r17 import candidate_failures
from gcn.backtest.signal_research_r19_validation import validation_failures
from gcn.backtest.signal_research_r21 import candidate_signals, RULES, CHALLENGERS
from gcn.backtest.signal_research_r7_validation import _run_validation


def run_validation(snapshot: Path, training: Path, output: Path) -> dict:
    return _run_validation(snapshot, training, output, research_version="gcn-historical-r21",
                           protocol_relative="reports/gcn-historical-r21-20260905/protocol.md",
                           candidate_builder=candidate_signals, rules=RULES, challengers=CHALLENGERS,
                           validator_source="gcn/backtest/signal_research_r21_validation.py",
                           extra_validator_sources=("gcn/backtest/signal_research_r19_validation.py",),
                           entry_breakeven_base_col="ENTRY_BE_BASE",
                           training_failure_checker=candidate_failures,
                           validation_failure_checker=validation_failures)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--training", type=Path, default=Path("reports/gcn-historical-r21-20260905/results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_validation(args.snapshot, args.training, args.output), indent=2, ensure_ascii=False))
