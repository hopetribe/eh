"""r18：仅将纯绝反入场的固定失效价改为原生三根底部LOW。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gcn.backtest.signal_research_r17 import candidate_failures
from gcn.backtest.signal_research_r17 import candidate_signals as prior_signals
from gcn.backtest.signal_research_r4 import _run_training
from gcn.core.tdx import LLV

CONTROLS = ("v5",)
CHALLENGERS = ("JF-base-low-invalidation",)
RULES = CONTROLS + CHALLENGERS


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    original = prior_signals(frame)["v5"]
    candidate = original.copy()
    candidate["ENTRY_FLOOR"] = LLV(frame.LOW, 3).where(frame.ICON_JUEFAN & ~frame.B_SIGNAL)
    return {"v5": original, CHALLENGERS[0]: candidate}


def run_training(snapshot: Path, output: Path) -> dict:
    return _run_training(snapshot, output, research_version="gcn-historical-r18",
                         protocol_relative="reports/gcn-historical-r18-20260905/protocol.md",
                         candidate_builder=candidate_signals, challengers=CHALLENGERS, controls=CONTROLS,
                         failure_checker=candidate_failures, entry_floor_col="ENTRY_FLOOR",
                         extra_sources=("gcn/backtest/signal_research_r17.py",
                                        "gcn/backtest/signal_research_r18.py"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.snapshot, args.output), indent=2, ensure_ascii=False))
