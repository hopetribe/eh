"""历史r9：固定盈利保护仅适用于原v5入场来源。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gcn.backtest.signal_research_r8 import candidate_signals as baseline_signals
from gcn.backtest.signal_research_r4 import _run_training

CONTROLS = ("v5", "P-confirm5", "P-confirm5-profit50")
CHALLENGERS = ("P-confirm5-v5profit50",)
RULES = CONTROLS + CHALLENGERS
PROFIT_KEEPS = {"P-confirm5-profit50": .5, "P-confirm5-v5profit50": .5}


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    baseline = baseline_signals(frame)
    result = {}
    for name in RULES:
        signals = baseline[name if name in CONTROLS else "P-confirm5"].copy()
        signals["ENTRY_PROFIT_ENABLED"] = ((frame["B_SIGNAL"] | frame["ICON_JUEFAN"])
                                           if name in CHALLENGERS else name == "P-confirm5-profit50")
        result[name] = signals
    return result


def run_training(snapshot: Path, output: Path) -> dict:
    return _run_training(snapshot, output, research_version="gcn-historical-r9",
                         protocol_relative="reports/gcn-historical-r9-20260905/protocol.md",
                         candidate_builder=candidate_signals, challengers=CHALLENGERS,
                         controls=CONTROLS, profit_keeps=PROFIT_KEEPS,
                         entry_profit_enabled_col="ENTRY_PROFIT_ENABLED",
                         extra_sources=("gcn/backtest/signal_research_r5.py",
                                        "gcn/backtest/signal_research_r6.py",
                                        "gcn/backtest/signal_research_r7.py",
                                        "gcn/backtest/signal_research_r8.py",
                                        "gcn/backtest/signal_research_r9.py"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.snapshot, args.output), indent=2, ensure_ascii=False))
