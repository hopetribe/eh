"""r32：持仓MACD零轴下穿且净亏的补充退出候选。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gcn.backtest.signal_research_r17 import candidate_failures
from gcn.backtest.signal_research_r17 import candidate_signals as prior_signals
from gcn.backtest.signal_research_r4 import _run_training


CONTROLS = ("v5",)
CHALLENGERS = ("held-macd-loss-exit",)
RULES = CONTROLS + CHALLENGERS


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    original = prior_signals(frame)["v5"]
    original["MACD"] = frame["MACD"]
    original["ENTRY_MACD_LOSS"] = False
    candidate = original.copy()
    candidate["ENTRY_MACD_LOSS"] = (
        original["B_SIGNAL"].fillna(False).astype(bool)
        | original["ICON_JUEFAN"].fillna(False).astype(bool)
    )
    return {"v5": original, CHALLENGERS[0]: candidate}


def run_training(snapshot: Path, output: Path) -> dict:
    return _run_training(
        snapshot,
        output,
        research_version="gcn-historical-r32",
        protocol_relative="reports/gcn-historical-r32-20260908/protocol.md",
        candidate_builder=candidate_signals,
        challengers=CHALLENGERS,
        controls=CONTROLS,
        failure_checker=candidate_failures,
        entry_macd_loss_col="ENTRY_MACD_LOSS",
        extra_sources=(
            "gcn/backtest/signal_research_r17.py",
            "gcn/backtest/signal_research_r32.py",
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("reports/signal-audit-v5-review-20260904"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.snapshot, args.output), indent=2, ensure_ascii=False))
