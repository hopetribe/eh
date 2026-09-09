"""r29 no-effect certificate for the frozen B entry-day invalidation rule."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
R28_MANIFEST_SHA = "ee1d7eeb171f30a75287baf60ca10953b65a03e714879651d0bc577bdeb6fb0a"
R22_MANIFEST_SHA = "abd706a2a5eeee260c5cde656980d95fb79c0760c3033dbfed24333cf0db28ef"
PROTOCOL_SHA = "95c8064ae1d7cec71d4deb534eeeb07d4804d79f49025c523a876a75955dffb0"


CHECK_COLUMNS = [
    "symbol",
    "trade_id",
    "entry_date",
    "entry_close",
    "setup_high",
    "entry_mid",
    "candidate_trigger",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + chr(10)
    path.write_text(text, encoding="utf-8")


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def certify_no_effect(
    episodes: pd.DataFrame,
    observations: pd.DataFrame,
    orders: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    """Certify strict path equivalence when the candidate never triggers."""
    _require_columns(
        orders,
        {"symbol", "trade_id", "entry_date", "entry_b", "entry_jf"},
        "orders",
    )
    _require_columns(
        episodes,
        {"symbol", "trade_id", "entry_date", "failure_on_entry"},
        "episodes",
    )
    _require_columns(
        observations,
        {
            "symbol", "trade_id", "entry_date", "date", "entry_today",
            "CLOSE", "setup_high", "MID", "failure_on_entry",
            "first_failure_today",
        },
        "observations",
    )

    b_orders = orders.loc[orders["entry_b"].eq(True)].copy()
    b_episodes = episodes.loc[episodes["trade_id"].isin(b_orders["trade_id"])].copy()
    if b_orders["trade_id"].duplicated().any():
        raise ValueError("B trade_id must be unique")
    if b_episodes["trade_id"].duplicated().any():
        raise ValueError("B trade_id must be unique")
    if set(b_episodes["trade_id"]) != set(b_orders["trade_id"]):
        raise ValueError("B episode coverage mismatch")

    entry_rows = observations.loc[observations["entry_today"].eq(True)].copy()
    entry_rows = entry_rows.loc[entry_rows["trade_id"].isin(b_orders["trade_id"])]
    if entry_rows["trade_id"].duplicated().any():
        raise ValueError("B entry observation coverage mismatch")
    if set(entry_rows["trade_id"]) != set(b_orders["trade_id"]):
        raise ValueError("B entry observation coverage mismatch")

    joined = b_orders[["symbol", "trade_id", "entry_date"]].merge(
        b_episodes[["trade_id", "failure_on_entry"]],
        on="trade_id",
        validate="one_to_one",
    )
    joined = joined.merge(
        entry_rows[[
            "symbol", "trade_id", "entry_date", "date", "CLOSE",
            "setup_high", "MID", "failure_on_entry", "first_failure_today",
        ]],
        on="trade_id",
        validate="one_to_one",
        suffixes=("_order", "_observation"),
    )
    if not joined["symbol_order"].eq(joined["symbol_observation"]).all():
        raise ValueError("B symbol mismatch")
    if not joined["entry_date_order"].eq(joined["entry_date_observation"]).all():
        raise ValueError("B entry date mismatch")
    if not joined["date"].eq(joined["entry_date_order"]).all():
        raise ValueError("entry observation date mismatch")

    finite = np.isfinite(joined[["CLOSE", "setup_high", "MID"]].astype(float)).all(axis=1)
    trigger = finite & joined["CLOSE"].le(joined["setup_high"])
    trigger &= joined["CLOSE"].le(joined["MID"])
    recorded = joined["failure_on_entry_order"].eq(True)
    recorded &= joined["failure_on_entry_observation"].eq(True)
    recorded &= joined["first_failure_today"].eq(True)
    if not trigger.eq(recorded).all():
        raise ValueError("trigger state mismatch")

    checks = pd.DataFrame({
        "symbol": joined["symbol_order"].astype("string"),
        "trade_id": joined["trade_id"].astype("string"),
        "entry_date": joined["entry_date_order"].astype("string"),
        "entry_close": joined["CLOSE"].astype(float),
        "setup_high": joined["setup_high"].astype(float),
        "entry_mid": joined["MID"].astype(float),
        "candidate_trigger": trigger.astype(bool),
    })
    if checks["candidate_trigger"].any():
        raise ValueError("candidate trigger present; no-effect proof unavailable")

    proof = {
        "original_trades": int(len(orders)),
        "original_b_trades": int(len(b_orders)),
        "original_jf_trades": int(orders["entry_jf"].eq(True).sum()),
        "checked_b_entries": int(len(checks)),
        "candidate_triggers": 0,
        "same_initial_state": True,
        "transition_delta_absent": True,
        "orders_identical_by_induction": True,
        "positions_identical_by_induction": True,
        "equity_identical_by_induction": True,
        "validation_material_improvement": False,
    }
    return checks, proof


def run_validation_certificate(output_dir: str | Path, *, root: str | Path = ROOT) -> dict[str, object]:
    root = Path(root)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")

    r28 = root / "reports/gcn-historical-r28-20260906/validation"
    r22 = root / "reports/gcn-historical-r22-20260905/validation"
    protocol = root / "reports/gcn-historical-r29-20260906/protocol.md"
    fixed = {
        "r28_validation_manifest": (r28 / "manifest.json", R28_MANIFEST_SHA),
        "r22_validation_manifest": (r22 / "manifest.json", R22_MANIFEST_SHA),
        "protocol": (protocol, PROTOCOL_SHA),
    }
    for name, (path, expected) in fixed.items():
        if _sha256(path) != expected:
            raise ValueError(f"{name} digest mismatch")

    episodes_path = r28 / "episodes.csv"
    observations_path = r28 / "observations.csv"
    orders_path = r22 / "trades.csv"
    r28_manifest = json.loads((r28 / "manifest.json").read_bytes())
    r22_manifest = json.loads((r22 / "manifest.json").read_bytes())
    expected_files = {
        episodes_path: r28_manifest["outputs"]["episodes.csv"],
        observations_path: r28_manifest["outputs"]["observations.csv"],
        orders_path: r22_manifest["outputs"]["trades.csv"],
    }
    for path, expected in expected_files.items():
        if _sha256(path) != expected:
            raise ValueError(f"input file digest mismatch: {path.name}")

    episodes = pd.read_csv(episodes_path, float_precision="round_trip")
    observations = pd.read_csv(observations_path, float_precision="round_trip")
    orders = pd.read_csv(orders_path, float_precision="round_trip")
    checks, proof = certify_no_effect(episodes, observations, orders)
    proof.update({
        "window": ["validation", "2024-08-27", "2025-08-26"],
        "candidate": "B-entry-day-invalidation",
        "control": "v5",
        "induction_basis": "same flat start and identical inputs",
        "induction_step": "only candidate delta has zero triggers",
    })
    decision = {
        "research_version": "gcn-historical-r29",
        "stage": "preflight_impossibility_certificate",
        "candidate": "B-entry-day-invalidation",
        "control": "v5",
        "status": "rejected",
        "reason": "validation_no_material_improvement",
        "production_changed": False,
        "candidate_backtest_run": False,
        "training_gate_evaluated": False,
        "validation_gate_impossible_by_equivalence": True,
        "net_win_rate_strictly_improved": False,
        "mdd_reduced_at_least_five_percent": False,
    }

    output.mkdir(parents=True, exist_ok=True)
    checks.to_csv(output / "entry_checks.csv", index=False)
    _write_json(output / "proof.json", proof)
    _write_json(output / "decision.json", decision)
    shutil.copyfile(protocol, output / "protocol.md")

    output_names = ("entry_checks.csv", "proof.json", "decision.json", "protocol.md")
    manifest = {
        "research_version": "gcn-historical-r29",
        "window": ["validation", "2024-08-27", "2025-08-26"],
        "candidate": "B-entry-day-invalidation",
        "control": "v5",
        "inputs": {name: expected for name, (_, expected) in fixed.items()},
        "input_files": {
            str(path.relative_to(root)): _sha256(path)
            for path in expected_files
        },
        "algorithm_sources": {
            "gcn/backtest/signal_research_r29.py": _sha256(Path(__file__)),
            "gcn/backtest/signal_research_r28.py": _sha256(root / "gcn/backtest/signal_research_r28.py"),
            "tests/test_signal_research_r29.py": _sha256(root / "tests/test_signal_research_r29.py"),
        },
        "outputs": {name: _sha256(output / name) for name in output_names},
    }
    _write_json(output / "manifest.json", manifest)
    return {"entry_checks": checks, "proof": proof, "decision": decision, "manifest": manifest}
