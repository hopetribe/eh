"""r12固定验证：仅重放训练通过的量能确认候选。"""
from __future__ import annotations

import json
from pathlib import Path

from gcn.backtest.signal_research_r12 import candidate_signals, CHALLENGERS, RULES
from gcn.backtest.signal_research_r7_validation import _run_validation


def run_validation(snapshot: Path, training: Path, output: Path) -> dict:
    return _run_validation(snapshot, training, output, research_version="gcn-historical-r12",
                           protocol_relative="reports/gcn-historical-r12-20260905/protocol.md",
                           candidate_builder=candidate_signals, rules=RULES, challengers=CHALLENGERS,
                           validator_source="gcn/backtest/signal_research_r12_validation.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--training", type=Path, default=Path("reports/gcn-historical-r12-20260905/results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_validation(args.snapshot, args.training, args.output), indent=2, ensure_ascii=False))
