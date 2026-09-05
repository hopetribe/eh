"""历史r8：固定P-confirm5与既有profit50盈利保护的唯一组合。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gcn.backtest.signal_research_r7 import candidate_signals as baseline_signals
from gcn.backtest.signal_research_r4 import _run_training

CONTROLS = ("v5", "profit50", "P-confirm5")
CHALLENGERS = ("P-confirm5-profit50",)
RULES = CONTROLS + CHALLENGERS
PROFIT_KEEPS = {"profit50": .5, "P-confirm5-profit50": .5}


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    baseline = baseline_signals(frame)
    return {name: baseline["v5" if name in ("v5", "profit50") else "P-confirm5"].copy()
            for name in RULES}


def run_training(snapshot: Path, output: Path) -> dict:
    return _run_training(snapshot, output, research_version="gcn-historical-r8",
                         protocol_relative="reports/gcn-historical-r8-20260905/protocol.md",
                         candidate_builder=candidate_signals, challengers=CHALLENGERS,
                         controls=CONTROLS, profit_keeps=PROFIT_KEEPS,
                         extra_sources=("gcn/backtest/signal_research_r5.py",
                                        "gcn/backtest/signal_research_r6.py",
                                        "gcn/backtest/signal_research_r7.py",
                                        "gcn/backtest/signal_research_r8.py"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.snapshot, args.output), indent=2, ensure_ascii=False))
