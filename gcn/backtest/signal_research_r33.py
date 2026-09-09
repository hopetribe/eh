"""r33：原S卖缺口的自然机制训练筛查。"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.historical_research import CORE, SNAPSHOT_SHA, load_snapshot
from gcn.recipes.gcn_main import compute_ehopt10


REFERENCE_COST = 0.001
TRAINING_START = pd.Timestamp("2021-08-27")
TRAINING_END = pd.Timestamp("2024-08-26")
R32_TRAINING_MANIFEST_SHA = (
    "56c87d45a356301be01e319093cb0ca281e0dc523b14210d5a8e08500f17c96c"
)
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/gcn-historical-r33-20260908/protocol.md"
PROTOCOL_SHA = "c7c94e933aadc3a2f9c901b1ba81d4fc2fcb661fe6dff656aef64be56936425d"


def natural_mechanisms(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """返回固定的既有配方边界，不生成新参数。"""
    close = frame["CLOSE"]
    mid = frame["MID"]
    lower = frame["LOWER"]
    upper = frame["UPPER"]
    chip = frame["获利筹"]
    v3 = frame["V3"]
    macd = frame["MACD"]
    rsi = frame["RSI1"]
    ma5 = close.rolling(5, min_periods=5).mean()
    signals = {
        "close-cross-mid": (close.shift(1) >= mid.shift(1)) & (close < mid),
        "close-cross-lower": (close.shift(1) >= lower.shift(1)) & (close < lower),
        "ma5-cross-mid": (ma5.shift(1) >= mid.shift(1)) & (ma5 < mid),
        "s-condition": frame["S_CONDITION"],
        "s-condition-union": (
            frame["S_CONDITION"]
            | frame["S_CONDITION_DOWN_LONG"]
            | frame["S_CONDITION_UP_LAST"]
        ),
        "nine2-sell": frame["NINE2_SELL_SIGNAL"],
        "nine2-up9": frame["NINE2_UP_9"],
        "chip-rollover": (v3.shift(1) <= chip.shift(1)) & (v3 > chip),
        "chip-rollover-after85": (
            (v3.shift(1) <= chip.shift(1))
            & (v3 > chip)
            & (chip.rolling(2, min_periods=2).max() > 85)
        ),
        "macd-turn-down": (macd.shift(1) > macd.shift(2)) & (macd < macd.shift(1)),
        "upper-rejection-score": (
            (close.shift(1) > upper.shift(1))
            & (close > 0.2 * mid + 0.8 * upper)
        ),
        "rsi-cross85": (rsi.shift(1) <= 85) & (rsi > 85),
    }
    return {name: signal.fillna(False).astype(bool) for name, signal in signals.items()}


def first_held_trigger(
    frame: pd.DataFrame,
    trade: Mapping[str, object],
    signal: pd.Series,
    mechanism: str,
) -> dict[str, object] | None:
    """返回原持仓边界内首次触发及真实次OPEN局部反事实。"""
    entry_date = pd.Timestamp(str(trade["entry_date"]))
    exit_date = pd.Timestamp(str(trade["exit_date"]))
    entry_i = frame.index.get_loc(entry_date)
    exit_i = frame.index.get_loc(exit_date)
    stop_i = exit_i - 1 if trade["exit_reason"] != "terminal" else exit_i
    aligned = signal.reindex(frame.index).fillna(False).astype(bool)
    hits = [i for i in range(entry_i, stop_i) if aligned.iloc[i]]
    if not hits:
        return None
    trigger_i = hits[0]
    entry_open = float(frame["OPEN"].iloc[entry_i])
    exit_open = float(frame["OPEN"].iloc[trigger_i + 1])
    if not np.isfinite([entry_open, exit_open]).all() or entry_open <= 0:
        raise ValueError("entry/exit OPEN必须有限且entry OPEN为正数")
    counterfactual = (exit_open / entry_open * (1 - REFERENCE_COST) ** 2 - 1) * 100
    original = float(trade["return_pct"])
    return {
        "mechanism": mechanism,
        "symbol": str(trade["symbol"]),
        "trade_id": str(trade["trade_id"]),
        "entry_date": entry_date.date().isoformat(),
        "trigger_date": frame.index[trigger_i].date().isoformat(),
        "counterfactual_exit_date": frame.index[trigger_i + 1].date().isoformat(),
        "original_return_pct": original,
        "counterfactual_return_pct": float(counterfactual),
        "local_delta_pp": float(counterfactual - original),
        "original_winner": bool(original > 0),
        "entry_b": bool(trade["entry_b"]),
        "entry_jf": bool(trade["entry_jf"]),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_r32_trades(training: Path) -> tuple[pd.DataFrame, set[str]]:
    """验证r32归档后返回原v5订单及候选MACD退出订单键。"""
    manifest_path = training / "manifest.json"
    if _sha256(manifest_path) != R32_TRAINING_MANIFEST_SHA:
        raise ValueError("r32训练manifest摘要不匹配")
    manifest = json.loads(manifest_path.read_bytes())
    for relative, expected in manifest["outputs"].items():
        if _sha256(training / relative) != expected:
            raise ValueError(f"r32训练输出摘要不匹配: {relative}")

    trades = pd.read_csv(training / "trades.csv")
    identity = trades["symbol"].astype(str) + ":" + trades["entry_date"].astype(str)
    trades = trades.assign(trade_id=identity)
    originals = trades[trades["rule"].eq("v5")].copy()
    if len(originals) != 50 or not originals["trade_id"].is_unique:
        raise ValueError("r32原v5订单集合不符合冻结协议")
    r32_exit_keys = set(
        trades.loc[
            trades["rule"].eq("held-macd-loss-exit")
            & trades["exit_reason"].eq("macd_loss"),
            "trade_id",
        ]
    )
    return originals, r32_exit_keys


def training_screen(snapshot: Path, r32_training: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在冻结训练窗内筛查自然S卖机制；不读取验证窗表现。"""
    trades, r32_exit_keys = _verified_r32_trades(r32_training)
    frames, _ = load_snapshot(snapshot)
    rows: list[dict[str, object]] = []
    event_counts: dict[str, int] = {}
    mechanism_order: list[str] | None = None

    for symbol in CORE:
        frame = compute_ehopt10(
            frames[symbol].loc[:TRAINING_END], version="v5", diagnostics=True
        )
        mechanisms = natural_mechanisms(frame)
        if mechanism_order is None:
            mechanism_order = list(mechanisms)
        for name, signal in mechanisms.items():
            event_counts[name] = event_counts.get(name, 0) + int(
                signal.loc[TRAINING_START:TRAINING_END].sum()
            )

        symbol_trades = trades[trades["symbol"].eq(symbol)]
        for trade in symbol_trades.to_dict("records"):
            for name, signal in mechanisms.items():
                event = first_held_trigger(frame, trade, signal, name)
                if event is not None:
                    rows.append(event)

            entry_open = float(frame.loc[pd.Timestamp(str(trade["entry_date"])), "OPEN"])
            loss_signal = mechanisms["ma5-cross-mid"] & (
                frame["CLOSE"] < entry_open / (1 - REFERENCE_COST) ** 2
            )
            event = first_held_trigger(
                frame, trade, loss_signal, "ma5-cross-mid-loss"
            )
            if event is not None:
                rows.append(event)

    assert mechanism_order is not None
    mechanism_order.append("ma5-cross-mid-loss")
    events = pd.DataFrame(rows)
    events["r32_overlap"] = events["trade_id"].isin(r32_exit_keys)
    rank = {name: i for i, name in enumerate(mechanism_order)}
    events = (
        events.assign(_rank=events["mechanism"].map(rank))
        .sort_values(["_rank", "symbol", "entry_date"], kind="stable")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )

    summary_rows: list[dict[str, object]] = []
    for name in mechanism_order:
        selected = events[events["mechanism"].eq(name)]
        held = len(selected)
        winners = int(selected["original_winner"].sum())
        losses = held - winners
        overlap = int(selected["r32_overlap"].sum())
        if name == "ma5-cross-mid-loss":
            reason = "near_synonym_of_rejected_r32"
        elif held < 8:
            reason = "insufficient_held_sample"
        elif winners >= held / 2:
            reason = "winner_heavy"
        elif float(selected["local_delta_pp"].sum()) <= 0:
            reason = "negative_local_delta"
        else:
            reason = "weak_gain_with_material_winner_harm"
        summary_rows.append(
            {
                "mechanism": name,
                "emitted_events": event_counts.get(name, int(held)),
                "held_triggers": held,
                "symbols": int(selected["symbol"].nunique()),
                "original_losses": losses,
                "original_winners": winners,
                "entry_b": int(selected["entry_b"].sum()),
                "entry_jf": int(selected["entry_jf"].sum()),
                "local_delta_pp": float(selected["local_delta_pp"].sum()),
                "median_local_delta_pp": float(selected["local_delta_pp"].median()),
                "r32_overlap_trades": overlap,
                "r32_overlap_pct": 100 * overlap / held if held else 0.0,
                "status": "rejected",
                "reason": reason,
            }
        )
    return events, pd.DataFrame(summary_rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_training_screen(
    snapshot: Path, r32_training: Path, output: Path
) -> dict[str, object]:
    """运行并归档r33训练机制筛查；此阶段不注册策略候选。"""
    snapshot = Path(snapshot)
    r32_training = Path(r32_training)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output目录必须为空")
    if _sha256(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("r33协议摘要不匹配")

    events, summary = training_screen(snapshot, r32_training)
    decision = {
        "research_version": "gcn-historical-r33",
        "stage": "training_mechanism_screen",
        "status": "no_candidate",
        "candidate": None,
        "reason": "no_independent_s_exit_mechanism",
        "candidate_backtest_run": False,
        "validation_run": False,
        "production_changed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output / "events.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(output / "decision.json", decision)
    shutil.copyfile(PROTOCOL, output / "protocol.md")

    r32_manifest = json.loads((r32_training / "manifest.json").read_bytes())
    names = ("events.csv", "summary.csv", "decision.json", "protocol.md")
    manifest = {
        "research_version": "gcn-historical-r33",
        "stage": "training_mechanism_screen",
        "window": ["training", "2021-08-27", "2024-08-26"],
        "parent_manifest_sha256": SNAPSHOT_SHA,
        "r32_training_manifest_sha256": R32_TRAINING_MANIFEST_SHA,
        "r32_input_outputs": r32_manifest["outputs"],
        "protocol_sha256": PROTOCOL_SHA,
        "algorithm_sources": {
            "gcn/backtest/signal_research_r33.py": _sha256(Path(__file__)),
            "gcn/backtest/historical_research.py": _sha256(
                ROOT / "gcn/backtest/historical_research.py"
            ),
            "gcn/recipes/gcn_main.py": _sha256(ROOT / "gcn/recipes/gcn_main.py"),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "outputs": {name: _sha256(output / name) for name in names},
    }
    _write_json(output / "manifest.json", manifest)
    return {
        "events": events,
        "summary": summary,
        "decision": decision,
        "manifest": manifest,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=ROOT / "reports/signal-audit-v5-review-20260904",
    )
    parser.add_argument(
        "--r32-training",
        type=Path,
        default=ROOT / "reports/gcn-historical-r32-20260908/training",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_training_screen(args.snapshot, args.r32_training, args.output)
    print(json.dumps(result["decision"], indent=2, ensure_ascii=False))
