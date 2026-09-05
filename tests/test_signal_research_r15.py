"""r15纯来源消融：保留原生状态、绝反与多来源重合。"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_r15_only_filters_confirmed_crash_only_setup_and_preserves_collisions_and_prefix(monkeypatch):
    from gcn.backtest import signal_research_r15 as research
    from gcn.backtest.signal_research_r14 import COMPONENTS
    from gcn.recipes.gcn_main import _stage_confirmation
    index = pd.bdate_range("2025-01-01", periods=20)
    frame = pd.DataFrame({"OPEN": 9., "HIGH": 10., "LOW": 8., "CLOSE": 9., "VOLUME": 1000.,
                          "MID": 8., "B_SETUP": False, "ICON_JUEFAN": False, "S_SIGNAL": False}, index=index)
    for col in COMPONENTS:
        frame[col] = False
    for pos in (0, 3, 6, 9, 12):
        frame.loc[index[pos], ["B_SETUP", "B_CRASH_RECOVER"]] = True
    for pos, col in ((3, "B_BEAR_RECOVER"), (6, "B_BASE_BULL"), (9, "B_STAGE_COMPONENT")):
        frame.loc[index[pos], col] = True
    frame.loc[index[[1, 4, 7, 10]], ["HIGH", "CLOSE"]] = [12., 11.]
    frame.loc[index[1], "ICON_JUEFAN"] = True  # Still independently eligible after B is filtered.
    frame.loc[index[15], "S_SIGNAL"] = True
    frame["B_SIGNAL"], frame["B_SETUP_EXPIRED"] = _stage_confirmation(
        frame.B_SETUP, frame.HIGH, frame.CLOSE, frame.MID)
    frame["B_ENTRY_SIGNAL"] = frame.B_SIGNAL
    def diagnostic(raw, *, version, diagnostics):
        assert version == "v5" and diagnostics
        assert list(raw.columns) == ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        return frame.loc[raw.index].copy()
    monkeypatch.setattr(research, "compute_ehopt10", diagnostic)
    original = frame.copy(deep=True)
    result = research.candidate_signals(frame)
    assert list(result) == ["v5", "B-exclude-crash-only"] == list(research.RULES)
    assert research.CHALLENGERS == ("B-exclude-crash-only",)
    old, new = result.values()
    assert np.flatnonzero(new.B_SIGNAL).tolist() == [4, 7, 10]
    assert new.ICON_JUEFAN.iloc[1] and not new.B_SIGNAL.iloc[1]
    for col in ("B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"):
        pd.testing.assert_series_equal(old[col], frame[col])
    for col in ("ICON_JUEFAN", "S_SIGNAL"):
        pd.testing.assert_series_equal(new[col], frame[col])
    for signals in result.values():
        assert signals[["ENTRY_STOP", "ENTRY_LIMIT"]].isna().all().all()
        assert not signals[["USE_EXTRA", "EXTRA_EXIT"]].any().any()
    for length in range(1, len(frame) + 1):
        prefix = research.candidate_signals(frame.iloc[:length])
        for rule in research.RULES:
            pd.testing.assert_frame_equal(prefix[rule], result[rule].iloc[:length])
    pd.testing.assert_frame_equal(frame, original)


def test_r15_rejects_noncanonical_input_signals_instead_of_mixing_versions():
    import pytest
    from gcn.data.sample import make_sample_data
    from gcn.recipes.gcn_main import compute_ehopt10
    from gcn.backtest.signal_research_r15 import candidate_signals
    native = compute_ehopt10(make_sample_data(900, seed=11), version="v5")
    for col in ("B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"):
        changed = native.copy()
        changed.loc[changed.index[500], col] = not changed[col].iloc[500]
        with pytest.raises(ValueError, match="规范v5"):
            candidate_signals(changed)


def test_r15_coverage_gate_is_explicit_without_changing_legacy_selection():
    from gcn.backtest.signal_research_r15 import candidate_failures, CHALLENGERS
    from gcn.backtest.signal_research_r4 import _choose_training
    archived = pd.read_csv(ROOT / "reports/gcn-historical-r13-20260905/results/training.csv")
    base = archived[archived.rule.eq("v5")].iloc[0].to_dict()
    candidate = {**base, "rule": CHALLENGERS[0], "entry_win": base["entry_win"] + 2,
                 "buy_covered": base["buy_covered"] - 1, "calmar": base["calmar"] + .1}
    # Legacy R/P-name gating stays unchanged; r15 must use the explicit protocol checker.
    assert _choose_training([base, candidate], CHALLENGERS) == CHALLENGERS[0]
    assert candidate_failures(candidate, base) == ["buy_covered"]
    assert _choose_training([base, candidate], CHALLENGERS, failure_checker=candidate_failures) is None
    candidate["buy_covered"] = base["buy_covered"]
    assert not candidate_failures(candidate, base)
    assert _choose_training([base, candidate], CHALLENGERS, failure_checker=candidate_failures) == CHALLENGERS[0]
    candidate["entry_events"] = 0
    assert "entry_coverage" in candidate_failures(candidate, base)


def test_r15_training_binds_explicit_gates_and_replays_native_trades_with_no_extra_risk(tmp_path, monkeypatch):
    import hashlib
    import json
    import pytest
    from gcn.backtest import signal_research_r15 as research
    from gcn.backtest.signal_research_r15 import run_training
    shared = research._run_training
    def checked(*args, **kwargs):
        assert kwargs["failure_checker"] is research.candidate_failures
        return shared(*args, **kwargs)
    monkeypatch.setattr(research, "_run_training", checked)
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    decision = run_training(snapshot, tmp_path)
    rows = pd.read_csv(tmp_path / "training.csv").to_dict("records")
    assert [r["rule"] for r in rows] == list(research.RULES)
    failures = research.candidate_failures(rows[1], rows[0])
    assert decision["failures"] == {research.CHALLENGERS[0]: failures}
    assert decision["selected"] == (None if failures else research.CHALLENGERS[0])
    assert decision["recommended"] == "v5" and not decision["production_changed"]
    events = pd.read_csv(tmp_path / "events.csv")
    assert events.date.min() >= "2021-08-27" and events.outcome_date.max() <= "2024-08-26"
    rejected = pd.read_csv(ROOT / "reports/gcn-historical-r14-20260905/results/events.csv")
    rejected = rejected[rejected.window.eq("training") & rejected.signal.eq("b") & rejected.B_CRASH_RECOVER
                        & ~rejected[["B_BASE_BULL", "B_STAGE_COMPONENT", "B_BEAR_RECOVER"]].any(axis=1)]
    excluded = set(zip(rejected.symbol, rejected.date))
    assert len(excluded) == 7
    base = events[events.rule.eq("v5")]
    candidate = events[events.rule.eq(research.CHALLENGERS[0])]
    for signal in ("b", "jf", "s"):
        old = base[base.signal.eq(signal)].drop(columns="rule")
        new = candidate[candidate.signal.eq(signal)].drop(columns="rule")
        if signal == "b":
            old = old[~old.apply(lambda r: (r.symbol, r.date) in excluded, axis=1)]
        pd.testing.assert_frame_equal(new.reset_index(drop=True), old.reset_index(drop=True))
    trades = pd.read_csv(tmp_path / "trades.csv")
    assert trades.entry_origin.eq("v5").all()
    assert trades[["entry_stop_pct", "entry_limit"]].isna().all().all()
    assert not trades.use_extra_exit.any()
    assert not trades.exit_reason.isin(["hard_stop", "max_hold", "profit_lock"]).any()
    prior = pd.read_csv(ROOT / "reports/gcn-historical-r13-20260905/results/trades.csv")
    fields = ["symbol", "entry_date", "exit_date", "return_pct", "hold_days", "exit_reason"]
    pd.testing.assert_frame_equal(trades.loc[trades.rule.eq("v5"), fields].reset_index(drop=True),
                                   prior.loc[prior.rule.eq("v5"), fields].reset_index(drop=True), atol=1e-10, rtol=0)
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    assert manifest["research_version"] == "gcn-historical-r15"
    for name in ("gcn/backtest/signal_research_r14.py", "gcn/backtest/signal_research_r15.py"):
        assert name in manifest["algorithm_sources"]
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((tmp_path / "source_snapshot" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        run_training(snapshot, tmp_path)


def test_r15_real_price_prefixes_recompute_sources_without_future_confirmation():
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.backtest.signal_research_r15 import candidate_signals, RULES
    from gcn.recipes.gcn_main import compute_ehopt10
    frames, _ = load_snapshot(ROOT / "reports/signal-audit-v5-review-20260904")
    for symbol in CORE:
        raw = frames[symbol].loc[:"2024-08-26"]
        whole = candidate_signals(compute_ehopt10(raw, version="v5"))
        for end in ("2022-11-07", "2023-01-17", "2024-07-10"):
            prefix_raw = raw.loc[:end]
            prefix = candidate_signals(compute_ehopt10(prefix_raw, version="v5"))
            for rule in RULES:
                pd.testing.assert_frame_equal(prefix[rule], whole[rule].loc[:end])
        if symbol == "YINN":
            assert whole[RULES[1]].loc["2022-11-07", "B_SIGNAL"]  # bear+crash overlap.
        if symbol == "TSLA":
            assert whole["v5"].loc["2023-01-17", "B_SIGNAL"]
            assert not whole[RULES[1]].loc["2023-01-17", "B_SIGNAL"]
