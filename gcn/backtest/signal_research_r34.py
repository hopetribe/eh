"""r34：过期Setup后的空仓再准入自然时钟训练筛查。"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.historical_research import CORE, SNAPSHOT_SHA, load_snapshot
from gcn.recipes.gcn_main import compute_ehopt10


TRAINING_END = pd.Timestamp("2024-08-26")
REFERENCE_COST = 0.001
R27_TRAINING_MANIFEST_SHA = (
    "201c115c61db4f610ed41893be8ba2a9ba2763a939b90089708c7d57d45c4956"
)
MECHANISMS = (
    "expired-both-level",
    "setup-high-reclaim",
    "mid-reclaim",
    "upper-breakout",
    "macd-zero-reclaim",
    "dif-dea-reclaim",
    "rsi50-reclaim",
    "suppressed-raw-b",
)
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/gcn-historical-r34-20260908/protocol.md"
PROTOCOL_SHA = "f11ba7a5f3a02f583f6b305a6d2ca40671f3deb850e33408591a13fbd1b23aea"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_r27_tables(training: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = training / "manifest.json"
    if _sha256(manifest_path) != R27_TRAINING_MANIFEST_SHA:
        raise ValueError("r27训练manifest摘要不匹配")
    manifest = json.loads(manifest_path.read_bytes())
    for relative, expected in manifest["outputs"].items():
        if _sha256(training / relative) != expected:
            raise ValueError(f"r27训练输出摘要不匹配: {relative}")
    episodes = pd.read_csv(training / "episodes.csv", float_precision="round_trip")
    observations = pd.read_csv(
        training / "observations.csv", float_precision="round_trip"
    )
    if len(episodes) != 29 or episodes["episode_id"].duplicated().any():
        raise ValueError("r27训练过期队列不符合冻结协议")
    if len(observations) != 2891 or observations.duplicated(
        ["episode_id", "date"]
    ).any():
        raise ValueError("r27训练观察路径不符合冻结协议")
    return episodes, observations


def _mechanisms(
    frame: pd.DataFrame, observations: pd.DataFrame, setup_high: float
) -> dict[str, pd.Series]:
    close = frame["CLOSE"]
    mid = frame["MID"]
    upper = frame["UPPER"]
    macd = frame["MACD"]
    dif = frame["DIF"]
    dea = frame["DEA"]
    rsi = frame["RSI1"]
    raw = {
        "expired-both-level": (close > setup_high) & (close > mid),
        "setup-high-reclaim": (close.shift(1) <= setup_high) & (close > setup_high),
        "mid-reclaim": (close.shift(1) <= mid.shift(1)) & (close > mid),
        "upper-breakout": (close.shift(1) <= upper.shift(1)) & (close > upper),
        "macd-zero-reclaim": (macd.shift(1) <= 0) & (macd > 0),
        "dif-dea-reclaim": (dif.shift(1) <= dea.shift(1)) & (dif > dea),
        "rsi50-reclaim": (rsi.shift(1) <= 50) & (rsi > 50),
        "suppressed-raw-b": (
            observations["B_ALL_RAW"].astype(bool)
            & observations["raw_b_suppressed"].astype(bool)
        ),
    }
    return {name: signal.fillna(False).astype(bool) for name, signal in raw.items()}


def training_screen(snapshot: Path, r27_training: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按冻结队列筛查自然再准入时钟，不调用交易模拟器。"""
    episodes, observations = _verified_r27_tables(Path(r27_training))
    frames, _ = load_snapshot(Path(snapshot))
    rows: list[dict[str, object]] = []

    for symbol in CORE:
        frame = compute_ehopt10(
            frames[symbol].loc[:TRAINING_END], version="v5", diagnostics=True
        )
        for episode in episodes[episodes["symbol"].eq(symbol)].to_dict("records"):
            obs = observations[
                observations["episode_id"].eq(episode["episode_id"])
            ].copy()
            dates = pd.DatetimeIndex(pd.to_datetime(obs["date"], errors="raise"))
            obs.index = dates
            local = frame.reindex(dates)
            if local[["OPEN", "LOW", "CLOSE"]].isna().any().any():
                raise ValueError(f"{episode['episode_id']}: 行情观察日期不完整")
            eligible = obs["position"].eq("flat")
            eligible &= ~obs["B_SIGNAL"].astype(bool)
            eligible &= ~obs["ICON_JUEFAN"].astype(bool)
            eligible &= ~obs["pending_buy"].astype(bool)
            signals = _mechanisms(local, obs, float(episode["setup_high"]))
            for name in MECHANISMS:
                hits = np.flatnonzero((signals[name] & eligible).to_numpy())
                if not len(hits):
                    continue
                trigger = dates[int(hits[0])]
                trigger_i = frame.index.get_loc(trigger)
                mature = bool(trigger_i + 20 < len(frame))
                entry_open = (
                    float(frame["OPEN"].iloc[trigger_i + 1])
                    if trigger_i + 1 < len(frame)
                    else np.nan
                )
                if mature:
                    return_pct = (
                        float(frame["CLOSE"].iloc[trigger_i + 20])
                        / entry_open
                        * (1 - REFERENCE_COST) ** 2
                        - 1
                    ) * 100
                    mae_pct = (
                        float(frame["LOW"].iloc[trigger_i + 1 : trigger_i + 21].min())
                        / entry_open
                        - 1
                    ) * 100
                else:
                    return_pct = np.nan
                    mae_pct = np.nan
                rows.append(
                    {
                        "mechanism": name,
                        "symbol": symbol,
                        "episode_id": str(episode["episode_id"]),
                        "expiry_date": str(episode["expiry_date"]),
                        "trigger_date": trigger.date().isoformat(),
                        "reference_open_date": (
                            frame.index[trigger_i + 1].date().isoformat()
                            if trigger_i + 1 < len(frame)
                            else None
                        ),
                        "mature": mature,
                        "return20_net_pct": float(return_pct),
                        "mae20_pct": float(mae_pct),
                        "win": bool(return_pct > 0) if mature else None,
                        "interference": (
                            bool(return_pct < 0 and mae_pct <= -8) if mature else None
                        ),
                    }
                )

    rank = {name: i for i, name in enumerate(MECHANISMS)}
    events = pd.DataFrame(rows)
    events = (
        events.assign(_rank=events["mechanism"].map(rank))
        .sort_values(["_rank", "symbol", "episode_id"], kind="stable")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )
    control = set(
        events.loc[events["mechanism"].eq("expired-both-level"), "episode_id"]
    )
    summary_rows: list[dict[str, object]] = []
    for name in MECHANISMS:
        selected = events[events["mechanism"].eq(name)]
        complete = selected[selected["mature"]]
        wins = int(complete["win"].sum())
        interference = int(complete["interference"].sum())
        leave_one = []
        for symbol in complete["symbol"].unique():
            other = complete[complete["symbol"].ne(symbol)]
            if len(other):
                leave_one.append(100 * float(other["win"].mean()))
        keys = set(selected["episode_id"])
        overlap_pct = 100 * len(keys & control) / len(keys) if keys else 0.0
        mature_count = len(complete)
        win_pct = 100 * wins / mature_count if mature_count else 0.0
        interference_pct = (
            100 * interference / mature_count if mature_count else 0.0
        )
        median = float(complete["return20_net_pct"].median())
        loo_min = min(leave_one) if leave_one else 0.0
        failures = []
        if name == "expired-both-level":
            failures.append("control_only")
        if mature_count < 8 or int(complete["symbol"].nunique()) < 5:
            failures.append("insufficient_sample")
        if win_pct < 60:
            failures.append("low_win_rate")
        if interference_pct > 30:
            failures.append("high_interference")
        if not median > 0:
            failures.append("nonpositive_median")
        if loo_min < 50:
            failures.append("fragile_leave_one_symbol")
        if name != "expired-both-level" and overlap_pct > 75:
            failures.append("synonymous_r27_recovery")
        summary_rows.append(
            {
                "mechanism": name,
                "triggered_episodes": len(selected),
                "mature_events": mature_count,
                "symbols": int(complete["symbol"].nunique()),
                "wins": wins,
                "win_rate_pct": win_pct,
                "interference": interference,
                "interference_rate_pct": interference_pct,
                "median_net_pct": median,
                "mean_net_pct": float(complete["return20_net_pct"].mean()),
                "loo_min_win_pct": loo_min,
                "control_overlap_pct": overlap_pct,
                "status": "rejected" if failures else "passed",
                "failures": ",".join(failures),
            }
        )
    return events, pd.DataFrame(summary_rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_training_screen(
    snapshot: Path, r27_training: Path, output: Path
) -> dict[str, object]:
    """运行并归档r34训练筛查；本入口不模拟候选订单。"""
    snapshot = Path(snapshot)
    r27_training = Path(r27_training)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output目录必须为空")
    if _sha256(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("r34协议摘要不匹配")
    events, summary = training_screen(snapshot, r27_training)
    passed = summary[summary["status"].eq("passed")].sort_values(
        "median_net_pct", ascending=False
    )
    candidate = str(passed.iloc[0]["mechanism"]) if len(passed) else None
    decision = {
        "research_version": "gcn-historical-r34",
        "stage": "training_reentry_clock_screen",
        "status": "candidate_selected" if candidate else "no_candidate",
        "candidate": candidate,
        "reason": (
            "training_gate_passed"
            if candidate
            else "no_independent_reentry_clock_passed"
        ),
        "candidate_backtest_run": False,
        "validation_run": False,
        "production_changed": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output / "events.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(output / "decision.json", decision)
    shutil.copyfile(PROTOCOL, output / "protocol.md")
    r27_manifest = json.loads((r27_training / "manifest.json").read_bytes())
    names = ("events.csv", "summary.csv", "decision.json", "protocol.md")
    manifest = {
        "research_version": "gcn-historical-r34",
        "stage": "training_reentry_clock_screen",
        "window": ["training", "2021-08-27", "2024-08-26"],
        "parent_manifest_sha256": SNAPSHOT_SHA,
        "r27_training_manifest_sha256": R27_TRAINING_MANIFEST_SHA,
        "r27_input_outputs": r27_manifest["outputs"],
        "protocol_sha256": PROTOCOL_SHA,
        "algorithm_sources": {
            "gcn/backtest/signal_research_r34.py": _sha256(Path(__file__)),
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
        "--r27-training",
        type=Path,
        default=ROOT / "reports/gcn-historical-r27-20260905/training",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_training_screen(args.snapshot, args.r27_training, args.output)
    print(json.dumps(result["decision"], indent=2, ensure_ascii=False))
