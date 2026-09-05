import numpy as np
import pandas as pd


def test_research_evaluator_passes_entry_duration_and_conditional_exit():
    from gcn.backtest.historical_research import evaluate_rule

    idx = pd.date_range("2024-01-01", periods=5)
    frame = pd.DataFrame({"OPEN": 100., "CLOSE": 100.}, index=idx)
    for conditional in (False, True):
        signals = pd.DataFrame({"B_SIGNAL": [True, False, False, False, False],
                                "ICON_JUEFAN": False, "S_SIGNAL": False,
                                "ENTRY_STOP": np.nan, "ENTRY_LIMIT": np.nan if conditional else 2.,
                                "USE_EXTRA": conditional,
                                "EXTRA_EXIT": [False, False, True, False, False]}, index=idx)
        prepared = {"AAA": {"frame": frame, "rules": {"test": signals}}}
        for cost in (.001, .0025):
            result = evaluate_rule(prepared, "test", idx[0], idx[-1], cost,
                                   entry_hard_stop_col="ENTRY_STOP", entry_max_hold_col="ENTRY_LIMIT",
                                   entry_exit_cols=("USE_EXTRA", "EXTRA_EXIT"), include_positions=True)
            trade = result["trades"][0]
            assert trade["exit_date"] == "2024-01-04"
            assert trade["exit_reason"] == ("entry_signal" if conditional else "max_hold")
            assert result["positions"]["AAA"].tolist() == [False, True, True, False, False]
            assert np.isclose(trade["return_pct"], ((1 - cost)**2 - 1) * 100)


def test_r4_rules_only_assign_duration_or_mid_exit_to_additional_entries():
    from gcn.backtest.signal_research_r4 import candidate_signals, RULES
    from gcn.data.sample import make_sample_data
    from gcn.recipes.gcn_main import compute_ehopt10

    data = make_sample_data(900, seed=17)
    frame = compute_ehopt10(data, version="v5")
    candidates = candidate_signals(frame)
    assert list(candidates) == list(RULES) == [
        "v5", "P-stop5", "P-stop5-hold10", "P-stop5-hold20", "P-stop5-hold40", "P-stop5-mid2"]
    original = frame.B_SIGNAL | frame.ICON_JUEFAN
    for rule, signals in candidates.items():
        assert signals.loc[original, "ENTRY_LIMIT"].isna().all()
        assert not signals.loc[original, "USE_EXTRA"].any()
        if "hold" in rule:
            assert set(signals.ENTRY_LIMIT.dropna()) == {float(rule.split("hold")[1])}
            assert not signals.USE_EXTRA.any()
        elif rule.endswith("mid2"):
            assert signals.USE_EXTRA.equals(signals.ENTRY_STOP.notna())
        for cutoff in (250, 600, 899):
            prefix = candidate_signals(compute_ehopt10(data.iloc[:cutoff], version="v5"))[rule]
            pd.testing.assert_frame_equal(prefix, signals.iloc[:cutoff])
