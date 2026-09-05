import numpy as np
import pandas as pd


def test_recovery_waits_for_prior_high_breakout_and_does_not_repeat():
    from gcn.backtest.signal_research_r2 import additional_signals

    frame = pd.DataFrame({"CLOSE": 100., "HIGH": 101., "LOW": 99.,
                          "MID": 100., "MACD": 1., "RSI1": 60.}, index=pd.RangeIndex(65))
    frame.loc[40:43, ["CLOSE", "HIGH", "LOW", "MACD", "RSI1"]] = [
        [81, 83, 80, -3, 20], [83, 84, 82, -2, 30],
        [86, 87, 84, -1, 45], [89, 90, 87, 0, 60]]
    result = additional_signals(frame)
    assert result["RECOVERY_SETUP"].iloc[40]
    assert np.flatnonzero(result["RECOVERY_SIGNAL"]).tolist() == [43]
    pd.testing.assert_frame_equal(additional_signals(frame.iloc[:44]), result.iloc[:44])


def test_pullback_requires_a_mid_cross_in_a_warmed_long_term_trend():
    from gcn.backtest.signal_research_r2 import additional_signals

    frame = pd.DataFrame({"CLOSE": 100., "HIGH": 103., "LOW": 98.,
                          "MID": 100., "MACD": 0., "RSI1": 60.}, index=pd.RangeIndex(250))
    for pos in (180, 220, 222):
        frame.loc[pos, "CLOSE"] = 99.
        frame.loc[pos + 1, ["CLOSE", "MACD"]] = [102., 1.]
    result = additional_signals(frame)
    assert np.flatnonzero(result["PULLBACK_SIGNAL"]).tolist() == [221]


def test_early_sell_requires_prior_rise_and_breaks_prior_lows_not_current_low():
    from gcn.backtest.signal_research_r2 import additional_signals

    frame = pd.DataFrame({"CLOSE": 100., "HIGH": 101., "LOW": 99.,
                          "MID": 100., "MACD": 1., "RSI1": 60.}, index=pd.RangeIndex(66))
    frame.loc[60:65, ["CLOSE", "HIGH", "LOW", "MACD"]] = [
        [120, 121, 119, 1], [124, 125, 122, 1], [125, 126, 123, 1],
        [126, 127, 124, 1], [120, 122, 119, 0], [118, 119, 117, -1]]
    result = additional_signals(frame)
    assert np.flatnonzero(result["EARLY_S_SIGNAL"]).tolist() == [64]
    frame.loc[:59, ["CLOSE", "HIGH", "LOW"]] = [115, 116, 114]
    assert not additional_signals(frame)["EARLY_S_SIGNAL"].any()


def test_candidate_composition_preserves_v5_juefan_and_causal_prefixes():
    from gcn.backtest.signal_research_r2 import candidate_signals, RULES
    from gcn.data.sample import make_sample_data
    from gcn.recipes.gcn_main import compute_ehopt10

    data = make_sample_data(900, seed=17)
    base = compute_ehopt10(data, version="v5")
    candidates = candidate_signals(base)
    assert list(candidates) == list(RULES) == ["v5", "R", "P", "RP", "E", "RE", "PE", "RPE"]
    for name, frame in candidates.items():
        assert frame.ICON_JUEFAN.equals(base.ICON_JUEFAN)
        assert (frame.B_SIGNAL | base.B_SIGNAL).equals(frame.B_SIGNAL)
        assert (frame.S_SIGNAL | base.S_SIGNAL).equals(frame.S_SIGNAL)
        for cutoff in (250, 600, 899):
            prefix = candidate_signals(compute_ehopt10(data.iloc[:cutoff], version="v5"))[name]
            pd.testing.assert_frame_equal(prefix, frame.iloc[:cutoff])


def test_r2_selection_rejects_lower_sell_precision_and_lost_buy_coverage():
    from gcn.backtest.signal_research_r2 import choose_training

    base = {"rule": "v5", "trades": 50, "entry_events": 54, "entry_win": 50.,
            "entry_interference": 48., "cagr": 8.72, "mdd": 15., "calmar": .58,
            "s_win": 57., "s_interference": 37., "buy_covered": 12}
    bad_sell = {**base, "rule": "E", "s_win": 52., "calmar": 3.}
    bad_cover = {**base, "rule": "R", "buy_covered": 11, "calmar": 4.}
    good = {**base, "rule": "P", "calmar": .7}
    assert choose_training([base, bad_sell, bad_cover]) is None
    assert choose_training([base, bad_sell, bad_cover, good]) == "P"


def test_frozen_r2_training_rejects_candidates_without_scoring_validation(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from gcn.backtest.signal_research_r2 import run_training

    root = Path(__file__).resolve().parents[1]
    decision = run_training(root / "reports/signal-audit-v5-review-20260904", tmp_path)
    assert decision["selected"] is None
    assert decision["validation_status"] == "not_run_no_eligible_candidate"
    rows = pd.read_csv(tmp_path / "training.csv")
    assert len(rows) == 8 and rows.query("rule == 'P'").iloc[0]["entry_events"] == 121
    events = pd.read_csv(tmp_path / "events.csv")
    assert events.outcome_date.max() <= "2024-08-26"
    assert "mdd" in decision["failures"]["P"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for filename, digest in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest() == digest
