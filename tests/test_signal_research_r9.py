import numpy as np
import pandas as pd


def test_r9_training_records_locked_profit_scope_and_only_selects_new_rule(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import pytest
    from gcn.backtest.signal_research_r9 import run_training, RULES, CHALLENGERS, PROFIT_KEEPS
    from gcn.backtest.signal_research_r2 import candidate_failures

    root = Path(__file__).resolve().parents[1]
    decision = run_training(root / "reports/signal-audit-v5-review-20260904", tmp_path)
    rows = pd.read_csv(tmp_path / "training.csv").to_dict("records")
    assert [row["rule"] for row in rows] == list(RULES)
    assert rows[0]["trades"] == 50
    failures = candidate_failures(rows[-1], rows[0])
    assert decision["selected"] == (None if failures else CHALLENGERS[0])
    assert decision["failures"] == {CHALLENGERS[0]: failures}
    trades = pd.read_csv(tmp_path / "trades.csv")
    candidate = trades[trades.rule.eq(CHALLENGERS[0])]
    assert candidate.entry_profit_enabled.equals(candidate.entry_origin.eq("v5"))
    extra = candidate[candidate.entry_origin.eq("additional")]
    assert len(extra) and not extra.exit_reason.eq("profit_lock").any()
    assert extra.entry_stop_pct.eq(5).all() and extra.entry_limit.eq(20).all()
    assert extra.hold_days.le(20).all()
    assert candidate[candidate.entry_origin.eq("v5")].entry_limit.isna().all()
    events = pd.read_csv(tmp_path / "events.csv")
    assert events.outcome_date.max() <= "2024-08-26"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["profit_keeps"] == PROFIT_KEEPS
    assert manifest["entry_profit_enabled_col"] == "ENTRY_PROFIT_ENABLED"
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((tmp_path / "source_snapshot" / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        run_training(root / "reports/signal-audit-v5-review-20260904", tmp_path)


def test_r9_changes_only_entry_profit_scope_and_preserves_v5_priority(monkeypatch):
    from gcn.backtest import signal_research_r9 as research

    frame = pd.DataFrame({"B_SIGNAL": False, "ICON_JUEFAN": False, "S_SIGNAL": False}, index=range(26))
    frame.loc[5, "B_SIGNAL"] = True
    frame.loc[21, "ICON_JUEFAN"] = True
    frame.loc[25, "S_SIGNAL"] = True

    def baseline(f):
        original = f.copy()
        original["ENTRY_STOP"] = original["ENTRY_LIMIT"] = np.nan
        original["USE_EXTRA"] = original["EXTRA_EXIT"] = False
        extra = original.copy()
        added = extra.index.intersection([4])
        extra.loc[added, ["B_SIGNAL", "ENTRY_STOP", "ENTRY_LIMIT"]] = [True, .05, 20.]
        extra.loc[extra.index.intersection([21]), "B_SIGNAL"] = True
        return {"v5": original, "P-confirm5": extra, "P-confirm5-profit50": extra.copy()}

    monkeypatch.setattr(research, "baseline_signals", baseline)
    candidates = research.candidate_signals(frame)
    assert list(candidates) == ["v5", "P-confirm5", "P-confirm5-profit50", "P-confirm5-v5profit50"]
    reference = baseline(frame)
    for name, signals in candidates.items():
        prior = reference[name if name in research.CONTROLS else "P-confirm5"]
        pd.testing.assert_frame_equal(signals.drop(columns="ENTRY_PROFIT_ENABLED"), prior)
        if name in research.CHALLENGERS:
            assert signals.ENTRY_PROFIT_ENABLED.equals(frame.B_SIGNAL | frame.ICON_JUEFAN)
            assert signals.ENTRY_PROFIT_ENABLED.iloc[21] and not signals.ENTRY_PROFIT_ENABLED.iloc[4]
            assert signals.ENTRY_STOP.iloc[[5, 21]].isna().all()
        else:
            assert signals.ENTRY_PROFIT_ENABLED.eq(name == "P-confirm5-profit50").all()
        for cutoff in (3, 6, 22, 26):
            pd.testing.assert_frame_equal(research.candidate_signals(frame.iloc[:cutoff])[name], signals.iloc[:cutoff])


def test_entry_profit_profile_replays_source_flag_and_fixed_cost_orders():
    from gcn.backtest.historical_research import evaluate_rule

    idx = pd.date_range("2024-01-01", periods=7)
    frame = pd.DataFrame({"OPEN": [100., 100., 130., 120., 110., 110., 100.],
                          "CLOSE": [100., 130., 120., 110., 110., 110., 100.]}, index=idx)
    for enabled in (True, False, None, pd.NA, np.nan):
        signals = pd.DataFrame({"B_SIGNAL": [True, False, False, False, False, False, False],
                                "ICON_JUEFAN": False, "S_SIGNAL": False,
                                "ENTRY_STOP": .05, "ENTRY_LIMIT": 20.}, index=idx)
        signals["PROTECT"] = pd.Series([enabled, True, True, True, True, True, True], index=idx, dtype=object)
        prepared = {"AAA": {"frame": frame, "rules": {"source-profit": signals}}}
        for cost in (.001, .0025):
            result = evaluate_rule(prepared, "source-profit", idx[0], idx[-1], cost,
                                   entry_hard_stop_col="ENTRY_STOP", entry_max_hold_col="ENTRY_LIMIT",
                                   profit_keep=.5, entry_profit_enabled_col="PROTECT")
            trade = result["trades"][0]
            assert trade["exit_reason"] == ("profit_lock" if enabled is True else "terminal")
            assert trade["exit_date"] == ("2024-01-05" if enabled is True else "2024-01-07")
            price_ratio = 1.1 if enabled is True else 1.
            assert np.isclose(trade["return_pct"], (price_ratio * (1 - cost)**2 - 1) * 100)
