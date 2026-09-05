import numpy as np
import pandas as pd


def test_r12_filters_only_original_confirmation_day_with_prior_volume_and_source_priority(monkeypatch):
    from gcn.backtest import signal_research_r12 as research
    from gcn.core.indicators import volume_ratio

    frame = pd.DataFrame({"VOLUME": 100., "B_SIGNAL": False, "ICON_JUEFAN": False,
                          "S_SIGNAL": False}, index=range(55))
    frame.loc[[21, 22, 25, 30], "VOLUME"] = [50., 200., 0., 200.]
    frame.loc[19, "B_SIGNAL"] = True
    frame.loc[25, "ICON_JUEFAN"] = True
    frame.loc[45, "S_SIGNAL"] = True

    def baseline(f):
        original = f[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]].copy()
        original["ENTRY_STOP"] = original["ENTRY_LIMIT"] = np.nan
        original["USE_EXTRA"] = original["EXTRA_EXIT"] = False
        extra = original.copy()
        confirmed = extra.index.intersection([0, 20, 21, 25, 30])
        extra.loc[confirmed, "B_SIGNAL"] = True
        additional = extra.index.isin(confirmed) & ~(f.B_SIGNAL | f.ICON_JUEFAN)
        extra.loc[additional, ["ENTRY_STOP", "ENTRY_LIMIT"]] = [.05, 20.]
        return {"v5": original, "P-confirm5": extra}

    monkeypatch.setattr(research, "baseline_signals", baseline)
    signals = research.candidate_signals(frame)
    assert tuple(signals) == ("v5", "P-confirm5", "P-confirm5-volume20")
    candidate = signals["P-confirm5-volume20"]
    reference = baseline(frame)
    for rule in research.CONTROLS:
        pd.testing.assert_frame_equal(signals[rule], reference[rule])
    ratios = volume_ratio(frame.VOLUME, 20)
    assert ratios.iloc[20] == 1. and ratios.iloc[21] == .5
    assert candidate.index[candidate.B_SIGNAL].tolist() == [19, 20, 25, 30]
    assert candidate.ENTRY_STOP.iloc[[20, 30]].eq(.05).all()
    assert candidate.ENTRY_LIMIT.iloc[[20, 30]].eq(20).all()
    assert candidate.ENTRY_STOP.drop(index=[20, 30]).isna().all()
    assert candidate.ENTRY_LIMIT.drop(index=[20, 30]).isna().all()
    assert candidate.ICON_JUEFAN.equals(frame.ICON_JUEFAN) and candidate.S_SIGNAL.equals(frame.S_SIGNAL)
    for cutoff in (1, 20, 23, 31, 55):
        prefix = research.candidate_signals(frame.iloc[:cutoff])["P-confirm5-volume20"]
        pd.testing.assert_frame_equal(prefix, candidate.iloc[:cutoff])


def test_r12_rejects_nonfinite_or_nonpositive_ratios_without_disabling_original_v5(monkeypatch):
    from gcn.backtest import signal_research_r12 as research

    frame = pd.DataFrame({"VOLUME": 100.}, index=range(7))
    original = pd.DataFrame({"B_SIGNAL": [True] + [False] * 6, "ICON_JUEFAN": False,
                             "S_SIGNAL": False, "ENTRY_STOP": np.nan, "ENTRY_LIMIT": np.nan,
                             "USE_EXTRA": False, "EXTRA_EXIT": False})
    extra = original.copy()
    extra.loc[1:, ["B_SIGNAL", "ENTRY_STOP", "ENTRY_LIMIT"]] = [True, .05, 20.]
    monkeypatch.setattr(research, "baseline_signals", lambda f: {"v5": original, "P-confirm5": extra})
    monkeypatch.setattr(research, "volume_ratio", lambda *args: pd.Series([np.nan, np.inf, -np.inf, 0, -1, 1, .999]))
    candidate = research.candidate_signals(frame)["P-confirm5-volume20"]
    assert candidate.index[candidate.B_SIGNAL].tolist() == [0, 5]
    assert candidate.ENTRY_STOP.drop(index=5).isna().all()
    assert candidate.ENTRY_LIMIT.drop(index=5).isna().all()
    assert extra.B_SIGNAL.all() and extra.ENTRY_STOP.iloc[1:].eq(.05).all()


def test_r12_training_executes_filtered_signals_and_freezes_the_unique_candidate(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import pytest
    from gcn.backtest.signal_research_r12 import run_training, RULES, CHALLENGERS
    from gcn.backtest.signal_research_r2 import candidate_failures
    from gcn.backtest.historical_research import load_snapshot
    from gcn.core.indicators import volume_ratio

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
    frames, quality = load_snapshot(snapshot)
    ratios = {s: volume_ratio(f.loc[:"2024-08-26", "volume"], 20) for s, f in frames.items()}
    events = pd.read_csv(tmp_path / "events.csv")
    assert events.outcome_date.max() <= "2024-08-26"
    entries = events[events.signal.eq("entry")]
    keys = set(entries[entries.rule.eq("v5")][["symbol", "date"]].itertuples(index=False, name=None))
    candidate = entries[entries.rule.eq(CHALLENGERS[0])]
    extras = candidate[[tuple(v) not in keys for v in candidate[["symbol", "date"]].itertuples(index=False, name=None)]]
    assert len(extras) == 13
    assert all(ratios[t.symbol].loc[t.date] >= 1 for t in extras.itertuples())
    assert keys.issubset(set(candidate[["symbol", "date"]].itertuples(index=False, name=None)))
    trades = pd.read_csv(tmp_path / "trades.csv")
    selected = trades[trades.rule.eq(CHALLENGERS[0])]
    additional = selected[selected.entry_origin.eq("additional")]
    original = selected[selected.entry_origin.eq("v5")]
    assert len(additional) > 0 and additional.entry_stop_pct.eq(5).all()
    assert additional.entry_limit.eq(20).all() and additional.hold_days.le(20).all()
    assert original.entry_stop_pct.isna().all() and original.entry_limit.isna().all()
    for trade in additional.itertuples():
        pos = ratios[trade.symbol].index.get_loc(pd.Timestamp(trade.entry_date)) - 1
        assert ratios[trade.symbol].iloc[pos] >= 1
    assert not trades.exit_reason.eq("profit_lock").any()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["research_version"] == "gcn-historical-r12" and manifest["source_quality"] == quality
    assert "profit_keeps" not in manifest and "entry_profit_enabled_col" not in manifest
    assert "gcn/backtest/signal_research_r12.py" in manifest["algorithm_sources"]
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((tmp_path / "source_snapshot" / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        run_training(snapshot, tmp_path)
