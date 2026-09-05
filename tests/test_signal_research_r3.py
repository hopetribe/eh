import numpy as np
import pandas as pd


def test_research_evaluation_applies_entry_risk_and_keeps_cost_stress_orders_fixed():
    from gcn.backtest.historical_research import evaluate_rule

    idx = pd.date_range("2024-01-01", periods=5)
    frame = pd.DataFrame({"OPEN": [100, 100, 91, 95, 100],
                          "CLOSE": [100, 94, 91, 95, 100]}, index=idx)
    signals = pd.DataFrame({"B_SIGNAL": [True, False, False, False, False],
                            "ICON_JUEFAN": False, "S_SIGNAL": False,
                            "ENTRY_STOP": [.05, np.nan, np.nan, np.nan, np.nan]}, index=idx)
    prepared = {"AAA": {"frame": frame, "rules": {"P-stop5": signals}}}
    for cost in (.001, .0025):
        result = evaluate_rule(prepared, "P-stop5", idx[0], idx[-1], cost=cost,
                               entry_hard_stop_col="ENTRY_STOP", include_positions=True)
        assert result["trades"][0]["exit_date"] == "2024-01-03"
        assert result["trades"][0]["exit_reason"] == "hard_stop"
        assert result["positions"]["AAA"].tolist() == [False, True, False, False, False]
        assert np.isclose(result["stats"]["total"], (.91 * (1 - cost)**2 - 1) * 100)


def test_r3_entry_risk_uses_v5_priority_for_same_day_signals(monkeypatch):
    from gcn.backtest import signal_research_r3 as research

    frame = pd.DataFrame({"B_SIGNAL": [True, False, False],
                          "ICON_JUEFAN": [False, False, True], "S_SIGNAL": False})
    monkeypatch.setattr(research, "additional_signals", lambda f: pd.DataFrame(
        {"PULLBACK_SIGNAL": True}, index=f.index))
    candidates = research.candidate_signals(frame)
    assert list(candidates) == ["v5", "P", "P-stop5", "P-stop8", "P-stop12"]
    assert candidates["P-stop8"].B_SIGNAL.tolist() == [True, True, True]
    assert np.isnan(candidates["P-stop8"].ENTRY_STOP.iloc[0])
    assert candidates["P-stop8"].ENTRY_STOP.iloc[1] == .08
    assert np.isnan(candidates["P-stop8"].ENTRY_STOP.iloc[2])
    assert candidates["v5"].ENTRY_STOP.isna().all()
    assert candidates["P"].ENTRY_STOP.isna().all()


def test_r3_execution_coverage_matches_legacy_audit_for_unchanged_v5():
    from pathlib import Path
    from gcn.backtest.historical_research import load_snapshot, CORE, evaluate_rule
    from gcn.backtest.signal_audit import Candidate, missed_turn_table
    from gcn.backtest.signal_research_r3 import candidate_signals, executed_turns
    from gcn.recipes.gcn_main import compute_ehopt10

    root = Path(__file__).resolve().parents[1]
    frames, _ = load_snapshot(root / "reports/signal-audit-v5-review-20260904")
    start, end = pd.Timestamp("2021-08-27"), pd.Timestamp("2024-08-26")
    prepared, legacy = {}, {}
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:end], version="v5")
        prepared[symbol] = {"frame": frame, "rules": candidate_signals(frame)}
        legacy[symbol] = {"v4": frame, "entries": {"entry": frame.B_SIGNAL | frame.ICON_JUEFAN},
                          "exits": {"exit": frame.S_SIGNAL}}
    result = evaluate_rule(prepared, "v5", start, end, entry_hard_stop_col="ENTRY_STOP", include_positions=True)
    expected = missed_turn_table(legacy, start, end, candidate=Candidate("entry", "exit", .20, None))
    actual = executed_turns(prepared, result, start, end)
    pd.testing.assert_frame_equal(actual, expected)


def test_r3_selector_never_promotes_the_rejected_plain_p_control():
    from gcn.backtest.signal_research_r3 import choose_training

    base = {"rule": "v5", "trades": 50, "entry_events": 54, "entry_win": 50.,
            "entry_interference": 48., "cagr": 8.72, "mdd": 15., "calmar": .58,
            "s_win": 57., "s_interference": 37., "buy_covered": 11}
    control = {**base, "rule": "P", "calmar": 10.}
    failed = {**base, "rule": "P-stop5", "mdd": 22., "calmar": 2.}
    passing = {**base, "rule": "P-stop8", "calmar": .7}
    assert choose_training([base, control, failed]) is None
    assert choose_training([base, control, failed, passing]) == "P-stop8"


def test_r3_frozen_run_rejects_risk_without_applying_new_stops_to_v5(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from gcn.backtest.signal_research_r3 import run_training

    root = Path(__file__).resolve().parents[1]
    decision = run_training(root / "reports/signal-audit-v5-review-20260904", tmp_path)
    assert decision["selected"] is None
    assert decision["validation_status"] == "not_run_no_eligible_candidate"
    assert all(decision["failures"][r] == ["mdd"] for r in ("P-stop5", "P-stop8", "P-stop12"))
    trades = pd.read_csv(tmp_path / "trades.csv")
    stopped = trades[trades.exit_reason == "hard_stop"]
    assert len(stopped) == 59
    assert stopped.entry_origin.eq("additional").all()
    assert trades.query("entry_origin=='v5'").entry_stop_pct.isna().all()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for filename, digest in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest() == digest
