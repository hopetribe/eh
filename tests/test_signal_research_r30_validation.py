"""r30固定验证：绑定首日双分量规则和既定退出门槛。"""
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_r30_validator_reuses_frozen_exit_gates_and_binds_candidate_marker(monkeypatch):
    from gcn.backtest import signal_research_r30_validation as validation
    from gcn.backtest.signal_research_r17 import candidate_failures
    from gcn.backtest.signal_research_r19_validation import validation_failures
    from gcn.backtest.signal_research_r30 import candidate_signals, RULES, CHALLENGERS

    captured = {}

    def inspect(*args, **kwargs):
        captured.update(kwargs)
        return {"checked": True}

    monkeypatch.setattr(validation, "_run_validation", inspect)
    assert validation.run_validation(Path("snapshot"), Path("training"), Path("output")) == {
        "checked": True
    }
    assert captured["research_version"] == "gcn-historical-r30"
    assert captured["protocol_relative"] == "reports/gcn-historical-r30-20260908/protocol.md"
    assert captured["candidate_builder"] is candidate_signals
    assert captured["rules"] == RULES and captured["challengers"] == CHALLENGERS
    assert captured["entry_two_leg_rejection_col"] == "ENTRY_TWO_LEG_REJECTION"
    assert captured["training_failure_checker"] is candidate_failures
    assert captured["validation_failure_checker"] is validation_failures
    assert captured["validator_source"] == "gcn/backtest/signal_research_r30_validation.py"


def test_r30_fixed_validation_runs_only_the_frozen_candidate_and_records_audit(tmp_path):
    from gcn.backtest.signal_research_r30 import CHALLENGERS
    from gcn.backtest.signal_research_r30_validation import run_validation

    decision = run_validation(
        ROOT / "reports/signal-audit-v5-review-20260904",
        ROOT / "reports/gcn-historical-r30-20260908/training",
        tmp_path,
    )
    rows = pd.read_csv(tmp_path / "comparisons.csv").set_index("rule")
    trades = pd.read_csv(tmp_path / "trades.csv")

    assert list(rows.index) == ["v5", CHALLENGERS[0]]
    assert decision["selected"] == CHALLENGERS[0]
    assert decision["status"] in {"rejected_keep_v5", "passed_validation_pending_stress"}
    assert "two_leg_enabled" in trades.columns


def test_r30_formal_validation_reproduces_byte_for_byte_and_rejects_nflx_winner_cut(tmp_path):
    from gcn.backtest.signal_research_r30 import CHALLENGERS
    from gcn.backtest.signal_research_r30_validation import run_validation

    reproduced = tmp_path / "validation"
    run_validation(ROOT / "reports/signal-audit-v5-review-20260904",
                   ROOT / "reports/gcn-historical-r30-20260908/training", reproduced)
    formal = ROOT / "reports/gcn-historical-r30-20260908/validation"
    manifest = json.loads((formal / "manifest.json").read_text())
    for name in list(manifest["outputs"]) + ["manifest.json"]:
        assert (reproduced / name).read_bytes() == (formal / name).read_bytes()

    decision = json.loads((formal / "decision.json").read_text())
    assert decision["status"] == "rejected_keep_v5"
    assert decision["failures"] == ["win", "sharpe", "no_material_improvement"]
    rows = pd.read_csv(formal / "comparisons.csv").set_index("rule")
    assert rows.loc[CHALLENGERS[0], "cagr"] < rows.loc["v5", "cagr"]
    assert np.isclose(rows.loc[CHALLENGERS[0], "mdd"], rows.loc["v5", "mdd"])
    trades = pd.read_csv(formal / "trades.csv")
    exit_row = trades[(trades.rule.eq(CHALLENGERS[0]))
                      & trades.exit_reason.eq("two_leg_rejection")].iloc[0]
    base_row = trades[(trades.rule.eq("v5")) & trades.symbol.eq(exit_row.symbol)
                      & trades.entry_date.eq(exit_row.entry_date)].iloc[0]
    assert exit_row.symbol == "NFLX" and exit_row.entry_date == "2024-10-22"
    assert np.isclose(exit_row.return_pct, -0.518108647863369)
    assert np.isclose(base_row.return_pct, 16.47453362647122)


def test_r30_validation_rejects_training_two_leg_configuration_tampering(tmp_path):
    from gcn.backtest.signal_research_r30_validation import run_validation

    training = tmp_path / "training"
    shutil.copytree(ROOT / "reports/gcn-historical-r30-20260908/training", training)
    manifest_path = training / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["two_leg_gap_threshold"] = 1.01
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    with pytest.raises(ValueError, match="双分量首日拒绝配置"):
        run_validation(ROOT / "reports/signal-audit-v5-review-20260904",
                       training, tmp_path / "validation")
