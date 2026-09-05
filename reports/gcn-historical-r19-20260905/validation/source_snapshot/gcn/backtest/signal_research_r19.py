"""r19：纯绝反固定底部需连续两根收盘失守才下一OPEN退出。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gcn.backtest.signal_research_r17 import candidate_failures
from gcn.backtest.signal_research_r18 import candidate_signals as prior_signals
from gcn.backtest.signal_research_r18 import CHALLENGERS as PRIOR_CHALLENGERS
from gcn.backtest.signal_research_r4 import _run_training

CONTROLS = ("v5",)
CHALLENGERS = ("JF-base-low-confirm2",)
RULES = CONTROLS + CHALLENGERS
CONFIRM_BARS = 2


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    previous = prior_signals(frame)
    return {"v5": previous["v5"], CHALLENGERS[0]: previous[PRIOR_CHALLENGERS[0]]}


def run_training(snapshot: Path, output: Path) -> dict:
    return _run_training(snapshot, output, research_version="gcn-historical-r19",
                         protocol_relative="reports/gcn-historical-r19-20260905/protocol.md",
                         candidate_builder=candidate_signals, challengers=CHALLENGERS, controls=CONTROLS,
                         failure_checker=candidate_failures, entry_floor_col="ENTRY_FLOOR",
                         entry_floor_confirm_bars=CONFIRM_BARS,
                         extra_sources=("gcn/backtest/signal_research_r17.py",
                                        "gcn/backtest/signal_research_r18.py",
                                        "gcn/backtest/signal_research_r19.py"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.snapshot, args.output), indent=2, ensure_ascii=False))
