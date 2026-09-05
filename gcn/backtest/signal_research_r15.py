"""r15：仅消融Setup为纯崩溃恢复的原生B确认。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r14 import COMPONENTS, trace_setups
from gcn.backtest.historical_research import training_failures
from gcn.backtest.signal_research_r4 import _run_training
from gcn.recipes.gcn_main import compute_ehopt10

CONTROLS = ("v5",)
CHALLENGERS = ("B-exclude-crash-only",)
RULES = CONTROLS + CHALLENGERS


def candidate_failures(row: dict, base: dict) -> list[str]:
    """r15协议显式要求买点覆盖，不能依赖旧候选的R/P命名。"""
    failures = training_failures(row, base)
    if row["buy_covered"] < base["buy_covered"]:
        failures.append("buy_covered")
    return failures


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    diagnostic = compute_ehopt10(frame[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]],
                                 version="v5", diagnostics=True)
    signal_columns = ["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]
    if not diagnostic[signal_columns].equals(frame[signal_columns]):
        raise ValueError("来源消融只接受同一完整前史计算的规范v5信号")
    trace = trace_setups(diagnostic)
    original = frame[signal_columns].copy()
    original["ENTRY_STOP"] = original["ENTRY_LIMIT"] = np.nan
    original["USE_EXTRA"] = original["EXTRA_EXIT"] = False
    candidate = original.copy()
    rejected = trace.status.eq("confirmed") & trace.B_CRASH_RECOVER & ~trace[list(COMPONENTS[:-1])].any(axis=1)
    positions = trace.loc[rejected, "resolution_i"].astype(int).tolist()
    candidate.iloc[positions, candidate.columns.get_loc("B_SIGNAL")] = False
    return {"v5": original, CHALLENGERS[0]: candidate}


def run_training(snapshot: Path, output: Path) -> dict:
    return _run_training(snapshot, output, research_version="gcn-historical-r15",
                         protocol_relative="reports/gcn-historical-r15-20260905/protocol.md",
                         candidate_builder=candidate_signals, challengers=CHALLENGERS, controls=CONTROLS,
                         failure_checker=candidate_failures,
                         extra_sources=("gcn/backtest/signal_research_r14.py", "gcn/backtest/signal_research_r15.py"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.snapshot, args.output), indent=2, ensure_ascii=False))
