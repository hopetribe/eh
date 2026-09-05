"""历史r11：仅新增确认回调入场使用有界Wilder ATR初始风险。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r7 import candidate_signals as baseline_signals
from gcn.backtest.signal_research_r4 import _run_training
from gcn.core.indicators import atr

CONTROLS = ("v5", "P-confirm5")
CHALLENGERS = ("P-confirm5-atr2",)
RULES = CONTROLS + CHALLENGERS


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    baseline = baseline_signals(frame)
    result = {name: baseline[name].copy() for name in CONTROLS}
    signals = baseline["P-confirm5"].copy()
    additional = signals["ENTRY_STOP"].notna()
    volatility = atr(frame["HIGH"], frame["LOW"], frame["CLOSE"], 14)
    if not (np.isfinite(volatility.loc[additional]) & volatility.loc[additional].gt(0)).all():
        raise ValueError("新增P信号日ATR必须为有限正值")
    risk = (2 * volatility / frame["CLOSE"]).clip(.05, .12)
    signals.loc[additional, "ENTRY_STOP"] = risk.loc[additional]
    result[CHALLENGERS[0]] = signals
    return result


def run_training(snapshot: Path, output: Path) -> dict:
    return _run_training(snapshot, output, research_version="gcn-historical-r11",
                         protocol_relative="reports/gcn-historical-r11-20260905/protocol.md",
                         candidate_builder=candidate_signals, challengers=CHALLENGERS,
                         controls=CONTROLS, extra_sources=("gcn/backtest/signal_research_r5.py",
                                                          "gcn/backtest/signal_research_r6.py",
                                                          "gcn/backtest/signal_research_r7.py",
                                                          "gcn/backtest/signal_research_r11.py"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.snapshot, args.output), indent=2, ensure_ascii=False))
