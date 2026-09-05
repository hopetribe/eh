import numpy as np
import pandas as pd


def test_r7_training_scores_confirmed_events_and_only_selects_new_candidates(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from gcn.backtest.signal_research_r7 import run_training, CHALLENGERS, RULES
    from gcn.backtest.signal_research_r2 import candidate_failures

    root = Path(__file__).resolve().parents[1]
    decision = run_training(root / "reports/signal-audit-v5-review-20260904", tmp_path)
    rows = pd.read_csv(tmp_path / "training.csv").to_dict("records")
    assert [r["rule"] for r in rows] == list(RULES)
    assert rows[0]["trades"] == 50 and rows[1]["trades"] == 84
    eligible = [r for r in rows if r["rule"] in CHALLENGERS and not candidate_failures(r, rows[0])]
    eligible.sort(key=lambda r: (-r["calmar"], CHALLENGERS.index(r["rule"])))
    assert decision["selected"] == (eligible[0]["rule"] if eligible else None)
    assert set(decision["failures"]) == set(CHALLENGERS)
    events = pd.read_csv(tmp_path / "events.csv")
    assert events.outcome_date.max() <= "2024-08-26"
    trades = pd.read_csv(tmp_path / "trades.csv")
    extra = trades[trades.rule.isin(CHALLENGERS) & trades.entry_origin.eq("additional")]
    assert len(extra) > 0 and extra.hold_days.le(20).all()
    assert extra.entry_stop_pct.eq(5).all() and extra.entry_limit.eq(20).all()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["research_version"] == "gcn-historical-r7"
    for name in ("gcn/backtest/signal_research_r6.py", "gcn/backtest/signal_research_r7.py"):
        assert hashlib.sha256((tmp_path / "source_snapshot" / name).read_bytes()).hexdigest() == manifest["algorithm_sources"][name]


def test_r7_confirmation_expires_causally_and_preserves_v5_priority(monkeypatch):
    from gcn.backtest import signal_research_r7 as research

    frame = pd.DataFrame({"CLOSE": 100., "HIGH": 101., "MID": 99.,
                          "B_SIGNAL": False, "ICON_JUEFAN": False, "S_SIGNAL": False}, index=range(26))
    frame.loc[4, ["CLOSE", "HIGH"]] = [102., 103.]
    frame.loc[21, ["CLOSE", "HIGH"]] = [103., 104.]
    frame.loc[5, "B_SIGNAL"] = True
    frame.loc[21, "ICON_JUEFAN"] = True
    frame.loc[25, "S_SIGNAL"] = True

    def baseline(f):
        original = f[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]].copy()
        original["ENTRY_STOP"] = original["ENTRY_LIMIT"] = np.nan
        original["USE_EXTRA"] = original["EXTRA_EXIT"] = False
        extra = original.copy()
        extra.loc[extra.index.intersection([0, 20]), ["B_SIGNAL", "ENTRY_STOP", "ENTRY_LIMIT"]] = [True, .05, 20.]
        return {"v5": original, "P-mid5-hold20-stop5": extra}

    monkeypatch.setattr(research, "baseline_signals", baseline)
    candidates = research.candidate_signals(frame)
    assert list(candidates) == ["v5", "P-mid5-hold20-stop5", "P-confirm3", "P-confirm5"]
    assert not candidates["P-confirm3"].B_SIGNAL.iloc[4]
    assert candidates["P-confirm5"].B_SIGNAL.iloc[4]
    assert candidates["P-confirm5"].ENTRY_STOP.iloc[4] == .05
    assert candidates["P-confirm5"].ENTRY_LIMIT.iloc[4] == 20.
    for rule in research.CHALLENGERS:
        signals = candidates[rule]
        assert not signals.B_SIGNAL.iloc[0]
        assert signals.B_SIGNAL.iloc[5] and signals.ICON_JUEFAN.iloc[21]
        assert signals.ENTRY_STOP.iloc[[5, 21]].isna().all()
        assert signals.ENTRY_LIMIT.iloc[[5, 21]].isna().all()
        assert signals.S_SIGNAL.equals(frame.S_SIGNAL)
        for cutoff in (3, 7, 22, 26):
            prefix = research.candidate_signals(frame.iloc[:cutoff])[rule]
            pd.testing.assert_frame_equal(prefix, signals.iloc[:cutoff])
