"""历史r5：仅过滤新增P入场的短/长期趋势斜率。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r3 import candidate_signals as baseline_signals
from gcn.backtest.signal_research_r4 import _run_training

CHALLENGERS = ("P-mid5", "P-long20", "P-dual")
RULES = ("v5", "P-stop5") + CHALLENGERS


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    baseline = baseline_signals(frame)
    mid_up = frame["MID"] > frame["MID"].shift(5)
    ma200 = frame["CLOSE"].rolling(200, min_periods=200).mean()
    long_up = ma200 > ma200.shift(20)
    filters = {"P-mid5": mid_up, "P-long20": long_up, "P-dual": mid_up & long_up}
    result = {}
    for rule in RULES:
        signals = baseline["v5" if rule == "v5" else "P-stop5"].copy()
        if rule in filters:
            accepted = signals["ENTRY_STOP"].notna() & filters[rule]
            signals["B_SIGNAL"] = frame["B_SIGNAL"] | accepted
            signals.loc[~accepted, "ENTRY_STOP"] = np.nan
        signals["ENTRY_LIMIT"] = np.nan
        signals["USE_EXTRA"] = False
        signals["EXTRA_EXIT"] = False
        result[rule] = signals
    return result


def run_training(snapshot: Path, output: Path) -> dict:
    return _run_training(snapshot, output, research_version="gcn-historical-r5",
                         protocol_relative="reports/gcn-historical-r5-20260905/protocol.md",
                         candidate_builder=candidate_signals, challengers=CHALLENGERS,
                         extra_sources=("gcn/backtest/signal_research_r5.py",))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.snapshot, args.output), indent=2, ensure_ascii=False))
