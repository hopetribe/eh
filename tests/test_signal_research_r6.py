import numpy as np
import pandas as pd


def test_r6_training_freezes_the_correct_control_and_two_candidates(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from gcn.backtest.signal_research_r6 import run_training, RULES, CHALLENGERS
    from gcn.backtest.signal_research_r2 import candidate_failures

    root = Path(__file__).resolve().parents[1]
    decision = run_training(root / "reports/signal-audit-v5-review-20260904", tmp_path)
    rows = pd.read_csv(tmp_path / "training.csv").to_dict("records")
    assert [r["rule"] for r in rows] == list(RULES)
    assert rows[0]["trades"] == 50 and rows[1]["trades"] == 70
    eligible = [r for r in rows if r["rule"] in CHALLENGERS and not candidate_failures(r, rows[0])]
    eligible.sort(key=lambda r: (-r["calmar"], CHALLENGERS.index(r["rule"])))
    assert decision["selected"] == (eligible[0]["rule"] if eligible else None)
    assert decision["research_version"] == "gcn-historical-r6"
    assert set(decision["failures"]) == set(CHALLENGERS)
    trades = pd.read_csv(tmp_path / "trades.csv")
    native = trades[trades.rule.eq(CHALLENGERS[1])]
    assert native.entry_stop_pct.isna().all()
    assert native.query("entry_origin=='additional'").entry_limit.eq(20).all()
    assert native.query("entry_origin=='v5'").entry_limit.isna().all()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for name in ("gcn/backtest/signal_research_r5.py", "gcn/backtest/signal_research_r6.py"):
        assert hashlib.sha256((tmp_path / "source_snapshot" / name).read_bytes()).hexdigest() == manifest["algorithm_sources"][name]
    for filename, digest in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest() == digest


def test_r6_source_horizon_survives_removing_initial_stop(monkeypatch):
    from gcn.backtest import signal_research_r6 as research
    from gcn.backtest.historical_research import evaluate_rule

    idx = pd.date_range("2024-01-01", periods=30)
    frame = pd.DataFrame({"OPEN": 100., "CLOSE": 100., "B_SIGNAL": False,
                          "ICON_JUEFAN": False, "S_SIGNAL": False}, index=idx)
    frame.loc[idx[2], "CLOSE"] = 94.
    frame.loc[idx[22], "B_SIGNAL"] = True

    def baseline(f):
        original = f[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]].copy()
        original["ENTRY_STOP"] = original["ENTRY_LIMIT"] = np.nan
        original["USE_EXTRA"] = original["EXTRA_EXIT"] = False
        extra = original.copy()
        extra.loc[idx[0], ["B_SIGNAL", "ENTRY_STOP"]] = [True, .05]
        return {"v5": original, "P-mid5": extra}

    monkeypatch.setattr(research, "baseline_signals", baseline)
    candidates = research.candidate_signals(frame)
    assert list(candidates) == ["v5", "P-mid5", "P-mid5-hold20-stop5", "P-mid5-hold20-trail20"]
    prepared = {"AAA": {"frame": frame, "rules": candidates}}
    for rule in research.CHALLENGERS:
        signals = candidates[rule]
        assert signals.ENTRY_LIMIT.iloc[0] == 20
        assert signals.ENTRY_STOP.iloc[[22]].isna().all() and signals.ENTRY_LIMIT.iloc[[22]].isna().all()
        assert signals.B_SIGNAL.equals(candidates["P-mid5"].B_SIGNAL)
        result = evaluate_rule(prepared, rule, idx[0], idx[-1], entry_hard_stop_col="ENTRY_STOP",
                               entry_max_hold_col="ENTRY_LIMIT", include_positions=True)
        first = result["trades"][0]
        if rule.endswith("stop5"):
            assert first["exit_date"] == "2024-01-04" and first["exit_reason"] == "hard_stop"
        else:
            assert signals.ENTRY_STOP.isna().all()
            assert first["exit_date"] == "2024-01-22" and first["hold_days"] == 20
            assert first["exit_reason"] == "max_hold"
        assert result["trades"][1]["exit_reason"] == "terminal"
        for cutoff in (10, 22, 30):
            prefix = research.candidate_signals(frame.iloc[:cutoff])[rule]
            pd.testing.assert_frame_equal(prefix, signals.iloc[:cutoff])
