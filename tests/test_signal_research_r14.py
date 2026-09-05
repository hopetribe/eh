"""r14只诊断来源，不改变交易信号。"""
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.recipes.gcn_main import VERSIONS, compute_ehopt10


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("B_BASE_BULL", "B_STAGE_COMPONENT", "B_BEAR_RECOVER", "B_CRASH_RECOVER")


def test_r14_diagnostic_components_preserve_frozen_default_for_every_version():
    import importlib.util
    from gcn.backtest.historical_research import load_snapshot
    source = ROOT / "reports/gcn-historical-r13-20260905/results/source_snapshot/gcn/recipes/gcn_main.py"
    spec = importlib.util.spec_from_file_location("frozen_r13_recipe", source)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    frames, _ = load_snapshot(ROOT / "reports/signal-audit-v5-review-20260904")
    for version in VERSIONS:
        frame = frames["TQQQ"]
        original = frozen.compute_ehopt10(frame, version=version)
        default = compute_ehopt10(frame, version=version)
        diagnostic = compute_ehopt10(frame, version=version, diagnostics=True)
        pd.testing.assert_frame_equal(default, original, check_exact=True)
        pd.testing.assert_frame_equal(diagnostic[default.columns], default, check_exact=True)
        assert set(COMPONENTS).isdisjoint(default.columns)
        assert set(COMPONENTS).issubset(diagnostic.columns)
        pd.testing.assert_series_equal(diagnostic[list(COMPONENTS)].any(axis=1),
                                       diagnostic.B_ALL_RAW, check_names=False)
        expected_stage = default.B_STAGE_ENTRY_SIGNAL if version == "v4-exp" else default.B_STAGE_SIGNAL
        pd.testing.assert_series_equal(diagnostic.B_STAGE_COMPONENT, expected_stage, check_names=False)


def test_r14_setup_trace_handles_replacement_expiry_last_bar_and_pending_causally():
    from gcn.backtest.signal_research_r14 import trace_setups
    from gcn.recipes.gcn_main import _stage_confirmation
    index = pd.date_range("2025-01-01", periods=20)
    frame = pd.DataFrame({"HIGH": 10., "CLOSE": 9., "MID": 8., "B_SETUP": False}, index=index)
    for col in COMPONENTS:
        frame[col] = False
    frame.loc[index[[0, 2, 8, 14, 18]], "B_SETUP"] = True
    frame.loc[index[[0, 8, 14, 18]], "B_BASE_BULL"] = True
    frame.loc[index[2], ["B_BEAR_RECOVER", "B_CRASH_RECOVER"]] = True
    # New setup wins over a possible confirmation of the old setup.
    frame.loc[index[2], ["HIGH", "CLOSE"]] = [12., 11.]
    frame.loc[index[7], ["HIGH", "CLOSE"]] = [13., 12.5]
    frame.loc[index[8], "HIGH"] = np.nan  # A nonfinite setup high cannot confirm.
    frame.loc[index[15], "CLOSE"] = 10.  # Strictly greater, not equal.
    frame.loc[index[16], ["HIGH", "CLOSE", "MID"]] = [11., 10.5, 10.5]
    frame.loc[index[17], ["HIGH", "CLOSE"]] = [11., 10.5]
    rows = trace_setups(frame)
    assert rows.status.tolist() == ["replaced", "confirmed", "expired", "confirmed", "pending"]
    assert rows.wait_bars.tolist() == [2, 5, 5, 3, 1]
    assert rows.setup_i.tolist() == [0, 2, 8, 14, 18]
    assert rows.resolution_i.iloc[:4].tolist() == [2, 7, 13, 17]
    assert pd.isna(rows.resolution_i.iloc[-1])
    assert rows.loc[1, "B_BEAR_RECOVER"] and rows.loc[1, "B_CRASH_RECOVER"]
    assert not rows.loc[1, "B_BASE_BULL"]  # Taken from setup day, not confirmation day.
    for length in range(1, len(frame) + 1):
        prefix = frame.iloc[:length]
        trace = trace_setups(prefix)
        entry, expired = _stage_confirmation(prefix.B_SETUP, prefix.HIGH, prefix.CLOSE, prefix.MID)
        assert trace.loc[trace.status.eq("confirmed"), "resolution_i"].tolist() == np.flatnonzero(entry).tolist()
        assert trace.loc[trace.status.eq("expired"), "resolution_i"].tolist() == np.flatnonzero(expired).tolist()
        resolved = trace[trace.status.ne("pending")].reset_index(drop=True)
        pd.testing.assert_frame_equal(resolved, rows.iloc[:len(resolved)].reset_index(drop=True))
    assert trace_setups(frame.iloc[:0]).empty


def test_r14_frame_diagnostic_reconciles_sources_and_keeps_incomplete_outcomes_missing():
    import pytest
    from gcn.backtest.signal_research_r14 import diagnose_frame
    from gcn.recipes.gcn_main import _stage_confirmation
    index = pd.bdate_range("2024-01-01", periods=250)
    frame = pd.DataFrame({"OPEN": 100., "HIGH": 104., "LOW": 98., "CLOSE": 100.,
                          "MID": 99., "B_SETUP": False, "ICON_JUEFAN": False,
                          "S_SIGNAL": False}, index=index)
    for col in COMPONENTS:
        frame[col] = False
    frame.loc[index[[200, 212, 240]], "B_SETUP"] = True
    frame.loc[index[[200, 212, 240]], "B_BEAR_RECOVER"] = True
    frame.loc[index[200], "B_CRASH_RECOVER"] = True
    frame.loc[index[200], ["HIGH", "CLOSE", "LOW"]] = [100., 90., 89.]
    frame.loc[index[[212, 240]], "HIGH"] = 110.
    frame.loc[index[205], "CLOSE"] = 101.
    frame.loc[index[220], "ICON_JUEFAN"] = True
    frame.loc[index[[210, 225]], "S_SIGNAL"] = True
    frame.loc[index[[225, 230]], ["CLOSE", "HIGH"]] = [110., 111.]
    frame.loc[index[231], "HIGH"] = 120.
    frame.loc[index[232], "LOW"] = 80.
    frame.loc[index[240], ["CLOSE", "LOW"]] = [90., 89.]
    frame["B_ENTRY_SIGNAL"], frame["B_SETUP_EXPIRED"] = _stage_confirmation(
        frame.B_SETUP, frame.HIGH, frame.CLOSE, frame.MID)
    frame["B_SIGNAL"] = frame.B_ENTRY_SIGNAL
    events, states, check = diagnose_frame("TEST", frame, index[204], index[242])
    assert check["confirmed"] == 1 and check["expired"] == 1 and check["reconciled"]
    b = events[events.signal.eq("b")].iloc[0]
    assert b.setup_date == index[200].date().isoformat() and b.wait_bars == 5
    assert b.setup_regime == "bear" and b.regime == "bull"
    assert b.B_BEAR_RECOVER and b.B_CRASH_RECOVER and not b.B_BASE_BULL
    assert b.outcome_complete and b.win and np.isclose(b.ret20_pct, 10.)
    jf = events[events.signal.eq("jf")].iloc[0]
    assert jf.interference and not jf.win and np.isclose(jf.ret20_pct, -10.)
    sells = events[events.signal.eq("s")]
    assert sells.iloc[0].interference and not sells.iloc[0].win
    incomplete = sells.iloc[1]
    assert not incomplete.outcome_complete
    for col in ("ret20_pct", "mfe20_pct", "mae20_pct", "win", "interference", "outcome_date"):
        assert pd.isna(incomplete[col])
    assert states.status.tolist() == ["confirmed", "expired", "pending"]
    assert states.iloc[-1].wait_bars == 2
    # No future tail is allowed to fill an outcome beyond the fixed end date.
    short = diagnose_frame("TEST", frame.loc[:index[242]], index[204], index[242])
    pd.testing.assert_frame_equal(events, short[0])
    pd.testing.assert_frame_equal(states, short[1])
    bad = frame.copy()
    bad.loc[index[205], "B_ENTRY_SIGNAL"] = False
    with pytest.raises(ValueError, match="确认.*不一致"):
        diagnose_frame("TEST", bad, index[204], index[242])


def test_r14_summary_uses_complete_denominators_and_keeps_multilabel_overlap():
    from gcn.backtest.signal_research_r14 import summarize_events
    rows = []
    for symbol, complete, win, noise, ret in (("A", True, True, False, 10.),
                                               ("B", True, False, True, -12.),
                                               ("B", False, None, None, np.nan)):
        rows.append({"symbol": symbol, "signal": "b", "setup_regime": "bear", "regime": "bull",
                     "wait_bars": 5, "outcome_complete": complete, "win": win,
                     "interference": noise, "ret20_pct": ret, "mfe20_pct": 15. if complete else np.nan,
                     "mae20_pct": -14. if complete else np.nan,
                     **{col: col in ("B_BEAR_RECOVER", "B_CRASH_RECOVER") for col in COMPONENTS}})
    summary = summarize_events(pd.DataFrame(rows))
    overall = summary[(summary.signal == "b") & (summary.group_by == "all")].iloc[0]
    assert overall.events == 3 and overall.complete == 2 and overall.incomplete == 1
    assert overall.win_rate_pct == 50. and overall.interference_rate_pct == 50.
    assert overall.mean_ret20_pct == -1. and overall.median_ret20_pct == -1.
    components = summary[summary.group_by.eq("component")].set_index("group")
    assert components.loc["B_BEAR_RECOVER", "events"] == 3
    assert components.loc["B_CRASH_RECOVER", "events"] == 3
    assert components.events.sum() == 6  # Overlapping labels, not six distinct events.
    empty = summary[(summary.signal == "s") & (summary.group_by == "all")].iloc[0]
    assert empty.events == 0 and pd.isna(empty.win_rate_pct)
    stock = summary[(summary.signal == "b") & (summary.group_by == "symbol")].set_index("group")
    assert stock.loc["B", "complete"] == 1 and stock.loc["B", "win_rate_pct"] == 0.


def test_r14_frozen_snapshot_reconciles_all_native_events_and_archives_only_diagnostics(tmp_path):
    import hashlib
    import json
    import pytest
    from gcn.backtest.signal_research_r14 import run_diagnostic, WINDOWS, trace_setups
    from gcn.backtest.historical_research import CORE, load_snapshot, event_quality
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    decision = run_diagnostic(snapshot, tmp_path)
    assert decision["stage"] == "diagnostic_only" and decision["recommended"] == "v5"
    assert not decision["production_changed"] and "selected" not in decision
    events = pd.read_csv(tmp_path / "events.csv")
    states = pd.read_csv(tmp_path / "states.csv")
    checks = pd.read_csv(tmp_path / "reconciliation.csv")
    assert len(checks) == len(CORE) * 4 and checks.reconciled.all()
    assert set(events.signal) == {"b", "jf", "s"} and set(events.symbol) == set(CORE)
    frames, quality = load_snapshot(snapshot)
    for window, first, last in WINDOWS:
        start, end = pd.Timestamp(first), pd.Timestamp(last)
        window_events = events[events.window.eq(window)]
        assert window_events.date.min() >= first and window_events.date.max() <= last
        complete = window_events[window_events.outcome_complete]
        assert complete.outcome_date.max() <= last
        assert window_events.loc[~window_events.outcome_complete,
                                 ["ret20_pct", "mfe20_pct", "mae20_pct", "win", "interference"]].isna().all().all()
        prepared = {}
        for symbol in CORE:
            frame = compute_ehopt10(frames[symbol].loc[:end], version="v5", diagnostics=True)
            assert frame[list(COMPONENTS)].any(axis=1).equals(frame.B_ALL_RAW)
            prepared[symbol] = {"frame": frame, "rules": {"v5": frame}}
            traced = trace_setups(frame)
            confirm = traced[traced.status.eq("confirmed")].set_index("resolution_date")
            selected = window_events[window_events.symbol.eq(symbol) & window_events.signal.eq("b")]
            assert len(selected) == int(frame.loc[start:end].B_SIGNAL.sum())
            for event in selected.itertuples():
                row = confirm.loc[event.date]
                setup = frame.loc[event.setup_date]
                assert row.setup_date == event.setup_date and row.wait_bars == event.wait_bars
                assert tuple(bool(setup[col]) for col in COMPONENTS) == tuple(getattr(event, col) for col in COMPONENTS)
        # Independently recover legacy event metrics for every native signal and window.
        old = pd.DataFrame(event_quality(prepared, "v5", start, end)["events"])
        old = old[old.signal.ne("entry")]
        fields = ["symbol", "date", "signal", "ret20_pct", "mfe20_pct", "mae20_pct", "win", "interference"]
        order = ["symbol", "date", "signal"]
        pd.testing.assert_frame_equal(complete[fields].sort_values(order).reset_index(drop=True),
                                       old[fields].sort_values(order).reset_index(drop=True), check_dtype=False)
    assert not states[states.status.eq("pending")].resolution_date.notna().any()
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    assert manifest["research_version"] == "gcn-historical-r14" and manifest["source_quality"] == quality
    assert manifest["parent_manifest_sha256"] == hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((tmp_path / "source_snapshot" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    before = (tmp_path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_diagnostic(snapshot, tmp_path)
    assert (tmp_path / "manifest.json").read_bytes() == before
