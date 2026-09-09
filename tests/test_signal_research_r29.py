import pandas as pd
import pytest
import json


def test_no_effect_certificate_requires_complete_zero_trigger_b_entries():
    from gcn.backtest.signal_research_r29 import certify_no_effect

    orders = pd.DataFrame([
        {"symbol": "AAA", "trade_id": "AAA:2025-03-25",
         "entry_date": "2025-03-25", "entry_b": True, "entry_jf": False},
        {"symbol": "BBB", "trade_id": "BBB:2025-04-08",
         "entry_date": "2025-04-08", "entry_b": False, "entry_jf": True},
    ])
    episodes = pd.DataFrame([
        {"symbol": "AAA", "trade_id": "AAA:2025-03-25",
         "entry_date": "2025-03-25", "failure_on_entry": False},
    ])
    observations = pd.DataFrame([
        {"symbol": "AAA", "trade_id": "AAA:2025-03-25",
         "entry_date": "2025-03-25", "date": "2025-03-25",
         "entry_today": True, "CLOSE": 395.0, "setup_high": 391.0,
         "MID": 390.0, "failure_on_entry": False,
         "first_failure_today": False},
    ])

    checks, proof = certify_no_effect(episodes, observations, orders)
    assert checks["candidate_trigger"].tolist() == [False]
    assert proof["original_trades"] == 2
    assert proof["checked_b_entries"] == 1
    assert proof["candidate_triggers"] == 0
    assert proof["validation_material_improvement"] is False

    with pytest.raises(ValueError, match="trigger present"):
        certify_no_effect(
            episodes.assign(failure_on_entry=True),
            observations.assign(CLOSE=389.0, failure_on_entry=True,
                                first_failure_today=True),
            orders,
        )


def test_validation_window_has_zero_trigger_and_strict_equivalence():
    from pathlib import Path
    from gcn.backtest.signal_research_r29 import certify_no_effect

    root = Path(__file__).resolve().parents[1]
    r28 = root / "reports/gcn-historical-r28-20260906/validation"
    r22 = root / "reports/gcn-historical-r22-20260905/validation"
    episodes = pd.read_csv(r28 / "episodes.csv", float_precision="round_trip")
    observations = pd.read_csv(r28 / "observations.csv", float_precision="round_trip")
    orders = pd.read_csv(r22 / "trades.csv", float_precision="round_trip")
    checks, proof = certify_no_effect(episodes, observations, orders)

    assert len(orders) == 17
    assert len(checks) == 9
    assert int(checks["candidate_trigger"].sum()) == 0
    assert proof["original_jf_trades"] == 8
    assert proof["orders_identical_by_induction"]
    assert proof["equity_identical_by_induction"]


def test_run_validation_certificate_writes_reproducible_rejection(tmp_path):
    from gcn.backtest.signal_research_r29 import run_validation_certificate

    output = tmp_path / "r29"
    result = run_validation_certificate(output)
    assert result["decision"]["status"] == "rejected"
    assert result["decision"]["reason"] == "validation_no_material_improvement"
    assert result["proof"]["checked_b_entries"] == 9
    assert result["manifest"]["inputs"]["r28_validation_manifest"]
    assert set(result["manifest"]["outputs"]) == {
        "entry_checks.csv", "proof.json", "decision.json", "protocol.md"
    }
    with pytest.raises(ValueError, match="empty"):
        run_validation_certificate(output)


def test_validation_certificate_json_files_are_parseable(tmp_path):
    from gcn.backtest.signal_research_r29 import run_validation_certificate

    output = tmp_path / "json"
    run_validation_certificate(output)
    assert json.loads((output / "proof.json").read_text())["candidate_triggers"] == 0
    assert json.loads((output / "decision.json").read_text())["status"] == "rejected"
    assert json.loads((output / "manifest.json").read_text())["research_version"] == "gcn-historical-r29"


def test_certificate_does_not_claim_training_pass_or_candidate_backtest(tmp_path):
    from gcn.backtest.signal_research_r29 import run_validation_certificate

    decision = run_validation_certificate(tmp_path / "scope")["decision"]
    assert decision["stage"] == "preflight_impossibility_certificate"
    assert decision["candidate_backtest_run"] is False
    assert decision["training_gate_evaluated"] is False
    assert decision["validation_gate_impossible_by_equivalence"] is True


def test_formal_validation_archive_reproduces_byte_for_byte(tmp_path):
    from pathlib import Path
    from gcn.backtest.signal_research_r29 import run_validation_certificate

    output = tmp_path / "reproduced"
    result = run_validation_certificate(output)
    formal = Path(__file__).resolve().parents[1] / "reports/gcn-historical-r29-20260906/validation"
    names = list(result["manifest"]["outputs"]) + ["manifest.json"]
    for name in names:
        assert (output / name).read_bytes() == (formal / name).read_bytes()
