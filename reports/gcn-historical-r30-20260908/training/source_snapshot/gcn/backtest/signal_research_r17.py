"""r17：原生绝反入场后的固定信号LOW失效退出。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.historical_research import training_failures
from gcn.backtest.signal_research_r4 import _run_training

CONTROLS = ("v5",)
CHALLENGERS = ("JF-low-invalidation",)
RULES = CONTROLS + CHALLENGERS


def candidate_failures(row: dict, base: dict) -> list[str]:
    failures = training_failures(row, base)
    if not row["buy_covered"] >= base["buy_covered"]:
        failures.append("buy_covered")
    return failures


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    original = frame[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]].copy()
    original["ENTRY_STOP"] = original["ENTRY_LIMIT"] = original["ENTRY_FLOOR"] = np.nan
    original["USE_EXTRA"] = original["EXTRA_EXIT"] = False
    candidate = original.copy()
    candidate["ENTRY_FLOOR"] = frame.LOW.where(frame.ICON_JUEFAN & ~frame.B_SIGNAL).astype(float)
    return {"v5": original, CHALLENGERS[0]: candidate}


def run_training(snapshot: Path, output: Path) -> dict:
    return _run_training(snapshot, output, research_version="gcn-historical-r17",
                         protocol_relative="reports/gcn-historical-r17-20260905/protocol.md",
                         candidate_builder=candidate_signals, challengers=CHALLENGERS, controls=CONTROLS,
                         failure_checker=candidate_failures, entry_floor_col="ENTRY_FLOOR",
                         extra_sources=("gcn/backtest/signal_research_r17.py",))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.snapshot, args.output), indent=2, ensure_ascii=False))
