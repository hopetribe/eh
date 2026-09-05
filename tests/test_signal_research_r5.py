import numpy as np
import pandas as pd


def test_r5_training_uses_its_frozen_candidates_and_captures_provenance(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from gcn.backtest.signal_research_r5 import run_training, RULES, CHALLENGERS
    from gcn.backtest.signal_research_r2 import candidate_failures

    root = Path(__file__).resolve().parents[1]
    decision = run_training(root / "reports/signal-audit-v5-review-20260904", tmp_path)
    assert decision["research_version"] == "gcn-historical-r5"
    rows = pd.read_csv(tmp_path / "training.csv").to_dict("records")
    assert [r["rule"] for r in rows] == list(RULES)
    assert rows[0]["trades"] == 50 and rows[1]["trades"] == 84
    eligible = [r for r in rows if r["rule"] in CHALLENGERS and not candidate_failures(r, rows[0])]
    eligible.sort(key=lambda r: (-r["calmar"], CHALLENGERS.index(r["rule"])))
    assert decision["selected"] == (eligible[0]["rule"] if eligible else None)
    assert set(decision["failures"]) == set(CHALLENGERS)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    name = "gcn/backtest/signal_research_r5.py"
    assert hashlib.sha256((tmp_path / "source_snapshot" / name).read_bytes()).hexdigest() == manifest["algorithm_sources"][name]
    assert (tmp_path / "protocol.md").read_bytes() == (root / "reports/gcn-historical-r5-20260905/protocol.md").read_bytes()
    for filename, digest in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest() == digest


def test_r5_slopes_only_filter_added_entries_and_use_past_values(monkeypatch):
    from gcn.backtest import signal_research_r5 as research

    frame = pd.DataFrame({"CLOSE": np.r_[np.full(230, 100.), np.arange(101., 121.)],
                          "MID": np.r_[np.full(240, 100.), np.arange(97., 107.)],
                          "B_SIGNAL": False, "ICON_JUEFAN": False, "S_SIGNAL": False})
    frame.loc[214, "B_SIGNAL"] = True
    frame.loc[242, "ICON_JUEFAN"] = True

    def baseline(f):
        original = f[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]].copy()
        original["ENTRY_STOP"] = np.nan
        extra = original.copy()
        extra["B_SIGNAL"] = True
        extra.loc[~(f.B_SIGNAL | f.ICON_JUEFAN), "ENTRY_STOP"] = .05
        return {"v5": original, "P-stop5": extra}

    monkeypatch.setattr(research, "baseline_signals", baseline)
    candidates = research.candidate_signals(frame)
    assert list(candidates) == ["v5", "P-stop5", "P-mid5", "P-long20", "P-dual"]
    assert not candidates["P-mid5"].B_SIGNAL.iloc[240]
    assert candidates["P-long20"].B_SIGNAL.iloc[240]
    for rule in research.CHALLENGERS:
        signal = candidates[rule]
        assert not signal.B_SIGNAL.iloc[100] and not signal.B_SIGNAL.iloc[220]
        assert signal.B_SIGNAL.iloc[249]
        assert signal.B_SIGNAL.iloc[214] and signal.ICON_JUEFAN.iloc[242]
        assert signal.ENTRY_STOP.iloc[[214, 242]].isna().all()
        assert signal.ENTRY_LIMIT.isna().all() and not signal.USE_EXTRA.any()
        assert signal.S_SIGNAL.equals(frame.S_SIGNAL)
        for cutoff in (215, 245, 250):
            prefix = research.candidate_signals(frame.iloc[:cutoff])[rule]
            pd.testing.assert_frame_equal(prefix, signal.iloc[:cutoff])
