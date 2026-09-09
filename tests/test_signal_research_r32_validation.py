"""r32固定验证：只运行冻结的持仓MACD净亏退出候选。"""

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_r32_validator_binds_frozen_candidate_marker_and_gates(monkeypatch):
    from gcn.backtest import signal_research_r32_validation as validation
    from gcn.backtest.signal_research_r17 import candidate_failures
    from gcn.backtest.signal_research_r19_validation import validation_failures
    from gcn.backtest.signal_research_r32 import candidate_signals, CHALLENGERS, RULES

    captured = {}

    def inspect(*args, **kwargs):
        captured.update(kwargs)
        return {"checked": True}

    monkeypatch.setattr(validation, "_run_validation", inspect)
    assert validation.run_validation(Path("snapshot"), Path("training"), Path("output")) == {
        "checked": True
    }
    assert captured["research_version"] == "gcn-historical-r32"
    assert captured["protocol_relative"] == "reports/gcn-historical-r32-20260908/protocol.md"
    assert captured["candidate_builder"] is candidate_signals
    assert captured["rules"] == RULES
    assert captured["challengers"] == CHALLENGERS
    assert captured["entry_macd_loss_col"] == "ENTRY_MACD_LOSS"
    assert captured["training_failure_checker"] is candidate_failures
    assert captured["validation_failure_checker"] is validation_failures
    assert captured["validator_source"] == "gcn/backtest/signal_research_r32_validation.py"


def test_r32_fixed_validation_runs_frozen_candidate_and_carries_macd_audit(tmp_path):
    from gcn.backtest.signal_research_r32 import CHALLENGERS
    from gcn.backtest.signal_research_r32_validation import run_validation

    decision = run_validation(
        ROOT / "reports/signal-audit-v5-review-20260904",
        ROOT / "reports/gcn-historical-r32-20260908/training",
        tmp_path,
    )
    rows = pd.read_csv(tmp_path / "comparisons.csv").set_index("rule")
    trades = pd.read_csv(tmp_path / "trades.csv")

    assert list(rows.index) == ["v5", CHALLENGERS[0]]
    assert decision["selected"] == CHALLENGERS[0]
    assert decision["status"] in {"rejected_keep_v5", "passed_validation_pending_stress"}
    assert "macd_loss_enabled" in trades.columns
    exits = trades[
        trades.rule.eq(CHALLENGERS[0]) & trades.exit_reason.eq("macd_loss")
    ]
    assert exits.macd_loss_trigger_date.notna().all()


def test_r32_validation_rejects_training_macd_loss_configuration_tampering(tmp_path):
    from gcn.backtest.signal_research_r32_validation import run_validation

    training = tmp_path / "training"
    shutil.copytree(ROOT / "reports/gcn-historical-r32-20260908/training", training)
    manifest_path = training / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["macd_loss_zero_threshold"] = 0.01
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    with pytest.raises(ValueError, match="MACD净亏退出配置"):
        run_validation(
            ROOT / "reports/signal-audit-v5-review-20260904",
            training,
            tmp_path / "validation",
        )


def test_r32_formal_validation_reproduces_byte_for_byte_and_keeps_v5(tmp_path):
    from gcn.backtest.signal_research_r32 import CHALLENGERS
    from gcn.backtest.signal_research_r32_validation import run_validation

    reproduced = tmp_path / "validation"
    run_validation(
        ROOT / "reports/signal-audit-v5-review-20260904",
        ROOT / "reports/gcn-historical-r32-20260908/training",
        reproduced,
    )
    formal = ROOT / "reports/gcn-historical-r32-20260908/validation"
    manifest = json.loads((formal / "manifest.json").read_text())
    for name in list(manifest["outputs"]) + ["manifest.json"]:
        assert (reproduced / name).read_bytes() == (formal / name).read_bytes()

    decision = json.loads((formal / "decision.json").read_text())
    assert decision["status"] == "rejected_keep_v5"
    assert decision["failures"] == ["cagr", "win", "sharpe"]
    rows = pd.read_csv(formal / "comparisons.csv").set_index("rule")
    candidate = CHALLENGERS[0]
    assert rows.loc[candidate, "cagr"] < rows.loc["v5", "cagr"]
    assert rows.loc[candidate, "mdd"] < rows.loc["v5", "mdd"]
    assert rows.loc[candidate, "sharpe"] < rows.loc["v5", "sharpe"]
    assert np.isclose(rows.loc[candidate, "win"], 31.57894736842105)
