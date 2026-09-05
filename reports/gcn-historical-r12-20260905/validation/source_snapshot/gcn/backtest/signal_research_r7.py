"""历史r7：趋势回调信号的3/5根突破确认。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gcn.backtest.signal_research_r6 import candidate_signals as baseline_signals
from gcn.backtest.signal_research_r4 import _run_training
from gcn.recipes.gcn_main import _stage_confirmation

CONTROLS = ("v5", "P-mid5-hold20-stop5")
WINDOWS = {"P-confirm3": 3, "P-confirm5": 5}
CHALLENGERS = tuple(WINDOWS)
RULES = CONTROLS + CHALLENGERS


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    baseline = baseline_signals(frame)
    setup = baseline[CONTROLS[1]]["ENTRY_STOP"].notna()
    original = frame["B_SIGNAL"] | frame["ICON_JUEFAN"]
    result = {name: baseline[name].copy() for name in CONTROLS}
    for rule, window in WINDOWS.items():
        confirmed, _ = _stage_confirmation(setup, frame["HIGH"], frame["CLOSE"], frame["MID"], window)
        signals = baseline["v5"].copy()
        signals["B_SIGNAL"] |= confirmed
        additional = confirmed & ~original
        signals.loc[additional, "ENTRY_STOP"] = .05
        signals.loc[additional, "ENTRY_LIMIT"] = 20.
        result[rule] = signals
    return result


def run_training(snapshot: Path, output: Path) -> dict:
    return _run_training(snapshot, output, research_version="gcn-historical-r7",
                         protocol_relative="reports/gcn-historical-r7-20260905/protocol.md",
                         candidate_builder=candidate_signals, challengers=CHALLENGERS,
                         controls=CONTROLS, extra_sources=("gcn/backtest/signal_research_r5.py",
                                                          "gcn/backtest/signal_research_r6.py",
                                                          "gcn/backtest/signal_research_r7.py"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.snapshot, args.output), indent=2, ensure_ascii=False))
