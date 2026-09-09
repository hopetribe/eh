"""r31：过滤确认日OPEN仍位于MID下方的原生B信号。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r14 import COMPONENTS
from gcn.backtest.signal_research_r17 import candidate_failures
from gcn.backtest.signal_research_r17 import candidate_signals as prior_signals
from gcn.backtest.signal_research_r4 import _run_training
from gcn.recipes.gcn_main import compute_ehopt10


CONTROLS = ("v5",)
CHALLENGERS = ("B-confirm-open-mid-filter",)
RULES = CONTROLS + CHALLENGERS


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    original = prior_signals(frame)["v5"]
    candidate = original.copy()
    finite_positive = (
        np.isfinite(frame["OPEN"].to_numpy())
        & np.isfinite(frame["MID"].to_numpy())
        & frame["OPEN"].gt(0).to_numpy()
        & frame["MID"].gt(0).to_numpy()
    )
    weak_confirmation = pd.Series(
        finite_positive & frame["OPEN"].lt(frame["MID"]).to_numpy(),
        index=original.index,
    )
    candidate["B_SIGNAL"] = (
        original["B_SIGNAL"].fillna(False).astype(bool) & ~weak_confirmation
    )
    return {"v5": original, CHALLENGERS[0]: candidate}


def training_screen(r28_training: Path) -> pd.DataFrame:
    """复算冻结训练订单的信号日OPEN/MID关系，不生成候选绩效。"""
    trades = pd.read_csv(r28_training / "trades.csv")
    rows = []
    for symbol, own in trades.groupby("symbol", sort=True):
        price_path = r28_training / "parent_snapshot/input_snapshot" / f"{symbol}_1d.csv"
        raw = pd.read_csv(price_path, parse_dates=["date"]).set_index("date")
        frame = compute_ehopt10(raw, version="v5", diagnostics=True)
        for trade in own.itertuples():
            signal = frame.loc[pd.Timestamp(trade.confirmation_date)]
            if not (
                np.isfinite([signal.OPEN, signal.MID]).all()
                and signal.OPEN > 0
                and signal.MID > 0
                and signal.OPEN < signal.MID
            ):
                continue
            sources = [name for name in COMPONENTS if bool(getattr(trade, name))]
            rows.append(
                {
                    "symbol": symbol,
                    "trade_id": trade.trade_id,
                    "setup_date": trade.setup_date,
                    "confirmation_date": trade.confirmation_date,
                    "entry_date": trade.entry_date,
                    "confirmation_open": float(signal.OPEN),
                    "confirmation_mid": float(signal.MID),
                    "confirmation_close": float(signal.CLOSE),
                    "setup_high": float(trade.setup_high),
                    "return_pct": float(trade.return_pct),
                    "trade_win": bool(trade.trade_win),
                    "source_component": "+".join(sources) if sources else "none",
                }
            )
    return pd.DataFrame(rows).sort_values(["symbol", "confirmation_date"]).reset_index(drop=True)


def run_training(snapshot: Path, output: Path) -> dict:
    return _run_training(
        snapshot,
        output,
        research_version="gcn-historical-r31",
        protocol_relative="reports/gcn-historical-r31-20260908/protocol.md",
        candidate_builder=candidate_signals,
        challengers=CHALLENGERS,
        controls=CONTROLS,
        failure_checker=candidate_failures,
        extra_sources=(
            "gcn/backtest/signal_research_r17.py",
            "gcn/backtest/signal_research_r31.py",
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
