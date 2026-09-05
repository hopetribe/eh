"""历史r6：P-mid5的20根持有周期与初始风险规则。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r5 import candidate_signals as baseline_signals
from gcn.backtest.signal_research_r4 import _run_training

CONTROLS = ("v5", "P-mid5")
CHALLENGERS = ("P-mid5-hold20-stop5", "P-mid5-hold20-trail20")
RULES = CONTROLS + CHALLENGERS


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    baseline = baseline_signals(frame)
    result = {}
    for rule in RULES:
        signals = baseline["v5" if rule == "v5" else "P-mid5"].copy()
        if rule in CHALLENGERS:
            additional = signals["ENTRY_STOP"].notna()
            signals.loc[additional, "ENTRY_LIMIT"] = 20.
            if rule.endswith("trail20"):
                signals["ENTRY_STOP"] = np.nan
        result[rule] = signals
    return result


def run_training(snapshot: Path, output: Path) -> dict:
    return _run_training(snapshot, output, research_version="gcn-historical-r6",
                         protocol_relative="reports/gcn-historical-r6-20260905/protocol.md",
                         candidate_builder=candidate_signals, challengers=CHALLENGERS,
                         controls=CONTROLS, extra_sources=("gcn/backtest/signal_research_r5.py",
                                                          "gcn/backtest/signal_research_r6.py"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.snapshot, args.output), indent=2, ensure_ascii=False))
