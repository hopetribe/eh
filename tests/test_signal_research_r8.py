import numpy as np
import pandas as pd


def test_r8_training_keeps_component_controls_and_applies_combined_protection(tmp_path):
    import json
    from pathlib import Path
    from gcn.backtest.signal_research_r8 import candidate_signals, run_training, RULES, CHALLENGERS
    from gcn.backtest.signal_research_r2 import candidate_failures
    from gcn.data.sample import make_sample_data
    from gcn.recipes.gcn_main import compute_ehopt10

    data = make_sample_data(900, seed=17)
    signals = candidate_signals(compute_ehopt10(data, version="v5"))
    pd.testing.assert_frame_equal(signals["profit50"], signals["v5"])
    pd.testing.assert_frame_equal(signals[CHALLENGERS[0]], signals["P-confirm5"])
    root = Path(__file__).resolve().parents[1]
    decision = run_training(root / "reports/signal-audit-v5-review-20260904", tmp_path)
    rows = pd.read_csv(tmp_path / "training.csv").to_dict("records")
    assert [row["rule"] for row in rows] == list(RULES)
    assert np.allclose([rows[0]["win"], rows[1]["win"]], [48., 58.], rtol=0, atol=1e-12)
    assert rows[2]["entry_events"] == rows[3]["entry_events"] == 82
    assert decision["selected"] == (None if candidate_failures(rows[3], rows[0]) else CHALLENGERS[0])
    trades = pd.read_csv(tmp_path / "trades.csv")
    combo = trades[trades.rule.eq(CHALLENGERS[0])]
    assert combo.exit_reason.eq("profit_lock").any()
    assert combo.query("entry_origin=='v5'").entry_stop_pct.isna().all()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["profit_keeps"][CHALLENGERS[0]] == .5


def test_explicit_profit_profile_combines_with_entry_risk_and_fixed_cost_path():
    from gcn.backtest.historical_research import evaluate_rule

    idx = pd.date_range("2024-01-01", periods=7)
    frame = pd.DataFrame({"OPEN": [100., 100., 130., 120., 110., 110., 100.],
                          "CLOSE": [100., 130., 120., 110., 110., 110., 100.]}, index=idx)
    signals = pd.DataFrame({"B_SIGNAL": [True, False, False, False, False, False, False],
                            "ICON_JUEFAN": False, "S_SIGNAL": False,
                            "ENTRY_STOP": .05, "ENTRY_LIMIT": 20.}, index=idx)
    prepared = {"AAA": {"frame": frame, "rules": {"combined": signals}}}
    for cost in (.001, .0025):
        result = evaluate_rule(prepared, "combined", idx[0], idx[-1], cost,
                               entry_hard_stop_col="ENTRY_STOP", entry_max_hold_col="ENTRY_LIMIT",
                               profit_keep=.5)
        trade = result["trades"][0]
        assert trade["exit_reason"] == "profit_lock" and trade["exit_date"] == "2024-01-05"
        assert np.isclose(trade["return_pct"], (1.1 * (1 - cost)**2 - 1) * 100)
    original = evaluate_rule(prepared, "combined", idx[0], idx[-1], entry_hard_stop_col="ENTRY_STOP",
                              entry_max_hold_col="ENTRY_LIMIT")
    assert original["trades"][0]["exit_reason"] == "terminal"
