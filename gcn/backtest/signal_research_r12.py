"""历史r12：新增回调确认日要求成交量达到此前20根均量。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r7 import candidate_signals as baseline_signals
from gcn.backtest.signal_research_r4 import _run_training
from gcn.core.indicators import volume_ratio

CONTROLS = ("v5", "P-confirm5")
CHALLENGERS = ("P-confirm5-volume20",)
RULES = CONTROLS + CHALLENGERS


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    baseline = baseline_signals(frame)
    result = {name: baseline[name].copy() for name in CONTROLS}
    signals = baseline["P-confirm5"].copy()
    ratio = volume_ratio(frame["VOLUME"], 20)
    rejected = signals["ENTRY_STOP"].notna() & ~(np.isfinite(ratio) & ratio.ge(1))
    signals.loc[rejected, "B_SIGNAL"] = False
    signals.loc[rejected, ["ENTRY_STOP", "ENTRY_LIMIT"]] = np.nan
    result[CHALLENGERS[0]] = signals
    return result


def run_training(snapshot: Path, output: Path) -> dict:
    return _run_training(snapshot, output, research_version="gcn-historical-r12",
                         protocol_relative="reports/gcn-historical-r12-20260905/protocol.md",
                         candidate_builder=candidate_signals, challengers=CHALLENGERS,
                         controls=CONTROLS, extra_sources=("gcn/backtest/signal_research_r5.py",
                                                          "gcn/backtest/signal_research_r6.py",
                                                          "gcn/backtest/signal_research_r7.py",
                                                          "gcn/backtest/signal_research_r12.py"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.snapshot, args.output), indent=2, ensure_ascii=False))
