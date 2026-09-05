import numpy as np
import pandas as pd


def test_r11_uses_wilder_signal_day_risk_bounds_and_preserves_source_and_prefix(monkeypatch):
    from gcn.backtest import signal_research_r11 as research
    from gcn.core.indicators import atr

    width = np.r_[np.ones(10), np.full(20, 4.), np.full(20, 15.)]
    frame = pd.DataFrame({"CLOSE": 100., "HIGH": 100. + width, "LOW": 100. - width,
                          "B_SIGNAL": False, "ICON_JUEFAN": False, "S_SIGNAL": False})
    frame.loc[30, "B_SIGNAL"] = True
    frame.loc[42, "ICON_JUEFAN"] = True
    frame.loc[45, "S_SIGNAL"] = True

    def baseline(f):
        original = f[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]].copy()
        original["ENTRY_STOP"] = original["ENTRY_LIMIT"] = np.nan
        original["USE_EXTRA"] = original["EXTRA_EXIT"] = False
        extra = original.copy()
        extra.loc[extra.index.intersection([0, 15, 49]),
                  ["B_SIGNAL", "ENTRY_STOP", "ENTRY_LIMIT"]] = [True, .05, 20.]
        return {"v5": original, "P-confirm5": extra}

    monkeypatch.setattr(research, "baseline_signals", baseline)
    signals = research.candidate_signals(frame)
    assert tuple(signals) == ("v5", "P-confirm5", "P-confirm5-atr2")
    candidate = signals["P-confirm5-atr2"]
    reference = baseline(frame)
    for rule in research.CONTROLS:
        pd.testing.assert_frame_equal(signals[rule], reference[rule])
    pd.testing.assert_frame_equal(candidate.drop(columns="ENTRY_STOP"),
                                  reference["P-confirm5"].drop(columns="ENTRY_STOP"))
    expected = (2 * atr(frame.HIGH, frame.LOW, frame.CLOSE, 14) / frame.CLOSE).clip(.05, .12)
    assert expected.iloc[0] == .05 and .05 < expected.iloc[15] < .12 and expected.iloc[49] == .12
    assert np.allclose(candidate.ENTRY_STOP.iloc[[0, 15, 49]], expected.iloc[[0, 15, 49]])
    assert candidate.ENTRY_STOP.drop(index=[0, 15, 49]).isna().all()
    assert candidate.ENTRY_LIMIT.iloc[[30, 42]].isna().all()
    for cutoff in (1, 18, 31, 43, 50):
        prefix = research.candidate_signals(frame.iloc[:cutoff])["P-confirm5-atr2"]
        pd.testing.assert_frame_equal(prefix, candidate.iloc[:cutoff])


def test_r11_rejects_invalid_atr_on_additional_entries_without_silent_clipping(monkeypatch):
    import pytest
    from gcn.backtest import signal_research_r11 as research

    frame = pd.DataFrame({"HIGH": 101., "LOW": 99., "CLOSE": 100.}, index=range(3))
    original = pd.DataFrame({"B_SIGNAL": [True, False, False], "ICON_JUEFAN": False,
                             "S_SIGNAL": False, "ENTRY_STOP": np.nan,
                             "ENTRY_LIMIT": np.nan, "USE_EXTRA": False, "EXTRA_EXIT": False})
    extra = original.copy()
    extra.loc[1, ["B_SIGNAL", "ENTRY_STOP", "ENTRY_LIMIT"]] = [True, .05, 20.]
    monkeypatch.setattr(research, "baseline_signals", lambda f: {"v5": original, "P-confirm5": extra})
    for value in (0., -1., np.nan, np.inf, -np.inf):
        monkeypatch.setattr(research, "atr", lambda *args: pd.Series([np.nan, value, np.nan]))
        with pytest.raises(ValueError, match="ATR"):
            research.candidate_signals(frame)
    monkeypatch.setattr(research, "atr", lambda *args: pd.Series([np.nan, 4., np.nan]))
    candidate = research.candidate_signals(frame)["P-confirm5-atr2"]
    assert candidate.ENTRY_STOP.iloc[1] == .08
    assert candidate.ENTRY_STOP.iloc[[0, 2]].isna().all()
    assert extra.ENTRY_STOP.iloc[1] == .05


def test_r11_training_freezes_signal_day_atr_risk_and_only_selects_the_new_candidate(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import pytest
    from gcn.backtest.signal_research_r11 import run_training, RULES, CHALLENGERS
    from gcn.backtest.signal_research_r2 import candidate_failures
    from gcn.backtest.historical_research import load_snapshot
    from gcn.core.indicators import atr

    root = Path(__file__).resolve().parents[1]
    snapshot = root / "reports/signal-audit-v5-review-20260904"
    decision = run_training(snapshot, tmp_path)
    rows = pd.read_csv(tmp_path / "training.csv").to_dict("records")
    assert [r["rule"] for r in rows] == list(RULES)
    assert rows[0]["trades"] == 50 and rows[1]["trades"] == 69
    failures = candidate_failures(rows[2], rows[0])
    assert decision["selected"] == (None if failures else CHALLENGERS[0])
    assert decision["failures"] == {CHALLENGERS[0]: failures}
    assert decision["recommended"] == "v5" and not decision["production_changed"]
    events = pd.read_csv(tmp_path / "events.csv")
    assert events.outcome_date.max() <= "2024-08-26"
    old_events = events[events.rule.eq("P-confirm5")].drop(columns="rule").reset_index(drop=True)
    new_events = events[events.rule.eq(CHALLENGERS[0])].drop(columns="rule").reset_index(drop=True)
    pd.testing.assert_frame_equal(old_events, new_events)
    trades = pd.read_csv(tmp_path / "trades.csv")
    challenger = trades[trades.rule.eq(CHALLENGERS[0])]
    extra = challenger[challenger.entry_origin.eq("additional")]
    original = challenger[challenger.entry_origin.eq("v5")]
    assert len(extra) > 0 and extra.entry_stop_pct.between(5, 12).all()
    assert extra.entry_limit.eq(20).all() and extra.hold_days.le(20).all()
    assert original.entry_stop_pct.isna().all() and original.entry_limit.isna().all()
    assert not trades.exit_reason.eq("profit_lock").any()
    frames, quality = load_snapshot(snapshot)
    for trade in extra.itertuples():
        frame = frames[trade.symbol].loc[:"2024-08-26"]
        pos = frame.index.get_loc(pd.Timestamp(trade.entry_date)) - 1
        expected = (2 * atr(frame.high, frame.low, frame.close, 14) / frame.close).clip(.05, .12)
        assert np.isclose(trade.entry_stop_pct, expected.iloc[pos] * 100, atol=1e-12, rtol=0)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["research_version"] == "gcn-historical-r11"
    assert manifest["source_quality"] == quality
    assert "profit_keeps" not in manifest and "entry_profit_enabled_col" not in manifest
    assert "gcn/backtest/signal_research_r11.py" in manifest["algorithm_sources"]
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((tmp_path / "source_snapshot" / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        run_training(snapshot, tmp_path)
