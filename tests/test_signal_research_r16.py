"""r16仅诊断S来源和实际退出，不改变策略。"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("S_BLOWOFF", "S_BEAR_RALLY", "S_CROSS", "S_DELAY")


def test_r16_s_components_preserve_frozen_defaults_union_and_causal_prefixes():
    import importlib.util
    from gcn.backtest.historical_research import load_snapshot
    from gcn.recipes.gcn_main import VERSIONS, compute_ehopt10
    from gcn.core.tdx import COUNT
    source = ROOT / "reports/gcn-historical-r15-20260905/results/source_snapshot/gcn/recipes/gcn_main.py"
    spec = importlib.util.spec_from_file_location("frozen_r15_recipe", source)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    frames, _ = load_snapshot(ROOT / "reports/signal-audit-v5-review-20260904")
    raw = frames["AAOI"]
    for version in VERSIONS:
        default = compute_ehopt10(raw, version=version)
        diagnostic = compute_ehopt10(raw, version=version, diagnostics=True)
        pd.testing.assert_frame_equal(default, frozen.compute_ehopt10(raw, version=version), check_exact=True)
        pd.testing.assert_frame_equal(diagnostic[default.columns], default, check_exact=True)
        assert set(SOURCES).isdisjoint(default.columns) and set(SOURCES).issubset(diagnostic.columns)
        assert diagnostic[list(SOURCES)].any(axis=1).equals(diagnostic.S_RAW)
        assert (diagnostic.S_RAW & COUNT(diagnostic.S_RAW, 40).eq(1)).equals(default.S_SIGNAL)
        for end in ("2023-06-21", "2024-08-26"):
            prefix = compute_ehopt10(raw.loc[:end], version=version, diagnostics=True)
            columns = [*SOURCES, "S_RAW", "S_SIGNAL"]
            pd.testing.assert_frame_equal(prefix[columns], diagnostic.loc[:end, columns])


def _exit_fixture():
    from gcn.backtest.signal_research_r14 import COMPONENTS
    from gcn.recipes.gcn_main import _stage_confirmation
    from gcn.core.tdx import COUNT
    index = pd.bdate_range("2024-01-01", periods=131)
    frame = pd.DataFrame({"OPEN": 100., "HIGH": 101., "LOW": 99., "CLOSE": 100., "VOLUME": 1000.,
                          "MID": 90., "B_SETUP": False, "ICON_JUEFAN": False}, index=index)
    for col in (*COMPONENTS, *SOURCES):
        frame[col] = False
    frame.loc[index[60], ["HIGH", "CLOSE", "LOW"]] = [100., 90., 89.]
    frame.loc[index[60], ["B_SETUP", "B_BEAR_RECOVER", "B_CRASH_RECOVER"]] = True
    for pos, close in {61: 101., 62: 120., 63: 130., 64: 125., 65: 300.,
                       68: 120., 69: 130., 70: 100., 71: 300., 127: 99., 128: 105., 129: 108., 130: 110.}.items():
        frame.loc[index[pos], ["CLOSE", "HIGH", "LOW"]] = [close, max(101., close + 1), min(99., close - 1)]
    frame.loc[index[65], ["OPEN", "LOW"]] = [118., 117.]
    frame.loc[index[71], ["OPEN", "LOW"]] = [98., 97.]
    frame.loc[index[[67, 126]], "ICON_JUEFAN"] = True
    frame.loc[index[64], ["S_CROSS", "S_DELAY"]] = True
    frame.loc[index[70], "S_DELAY"] = True
    frame.loc[index[110], "S_BLOWOFF"] = True
    frame["S_RAW"] = frame[list(SOURCES)].any(axis=1)
    frame["S_SIGNAL"] = frame.S_RAW & COUNT(frame.S_RAW, 40).eq(1)
    frame["B_SIGNAL"], frame["B_SETUP_EXPIRED"] = _stage_confirmation(
        frame.B_SETUP, frame.HIGH, frame.CLOSE, frame.MID)
    frame["B_ENTRY_SIGNAL"] = frame.B_SIGNAL
    return frame


def test_r16_trade_audit_uses_entry_setup_exit_signal_previous_bar_and_correct_peak_fees():
    from gcn.backtest.signal_research_r16 import audit_trades
    frame = _exit_fixture(); index = frame.index
    trades, result = audit_trades("TEST", frame, index[61], index[130])
    assert trades.exit_reason.tolist() == ["signal", "trail", "terminal"]
    assert trades.entry_date.tolist() == [index[i].date().isoformat() for i in (62, 68, 127)]
    assert trades.exit_date.tolist() == [index[i].date().isoformat() for i in (65, 71, 130)]
    assert trades.hold_bars.tolist() == [3, 3, 4]
    assert trades.peak_close.tolist() == [130., 130., 110.]  # Not the exit-day CLOSE of 300.
    assert trades.peak_close_date.tolist() == [index[i].date().isoformat() for i in (63, 69, 130)]
    assert np.allclose(trades.peak_close_gain_pct, [30., 30., 10.])
    assert np.allclose(trades.giveback_pp, [12., 32., 0.])
    assert np.allclose(trades.peak_to_exit_drawdown_pct, [(1-118/130)*100, (1-98/130)*100, 0.])
    assert np.allclose(trades.return_pct, (np.array([1.18, .98, 1.10]) * .999**2 - 1) * 100)
    first = trades.iloc[0]
    assert first.entry_b and not first.entry_jf
    assert first.setup_date == index[60].date().isoformat()  # Before this independent window.
    assert first.B_BEAR_RECOVER and first.B_CRASH_RECOVER
    assert first.exit_signal_date == index[64].date().isoformat() and first.S_CROSS and first.S_DELAY
    assert trades.iloc[1].entry_jf and not trades.iloc[1].entry_b
    assert trades.iloc[1].exit_raw_s and trades.iloc[1].exit_s_suppressed
    assert not trades.iloc[1][list(SOURCES)].any()
    assert trades.iloc[1].profit_to_loss
    assert trades.suppressed_s_count.tolist() == [0, 1, 0]
    assert pd.isna(trades.iloc[2].exit_signal_date) and not trades.iloc[2][list(SOURCES)].any()
    assert len(result["trades"]) == 3


def test_r16_events_separate_executed_flat_suppressed_and_cutoff_pending_without_future_labels():
    import pytest
    from gcn.backtest.signal_research_r16 import audit_frame
    frame = _exit_fixture(); index = frame.index
    events, trades, check = audit_frame("TEST", frame, index[61], index[130])
    assert events.date.tolist() == [index[i].date().isoformat() for i in (64, 70, 110)]
    assert events.status.tolist() == ["executed", "suppressed_held", "ignored_flat"]
    assert events.emitted.tolist() == [True, False, True]
    assert events.iloc[0].S_CROSS and events.iloc[0].S_DELAY
    assert events.iloc[0].trade_id == trades.iloc[0].trade_id
    assert events.iloc[1].trade_id == trades.iloc[1].trade_id
    assert pd.isna(events.iloc[2].trade_id)
    assert events.iloc[2].interference and np.isclose(events.iloc[2].ret20_pct, 10.)
    assert check["raw"] == 3 and check["emitted"] == 2 and check["executed"] == 1
    assert check["signal_exits"] == 1 and check["reconciled"]
    short, short_trades, _ = audit_frame("TEST", frame, index[61], index[64])
    pending = short.iloc[0]
    assert pending.status == "pending_at_cutoff" and not pending.outcome_complete
    assert short_trades.exit_reason.tolist() == ["terminal"]
    for col in ("outcome_date", "ret20_pct", "mfe20_pct", "mae20_pct", "win", "interference"):
        assert pd.isna(pending[col])
    # The supplied future tail may not change this as-of report.
    truncated = audit_frame("TEST", frame.loc[:index[64]], index[61], index[64])
    pd.testing.assert_frame_equal(short, truncated[0])
    pd.testing.assert_frame_equal(short_trades, truncated[1])
    for last in (70, 110):
        prefix_events, prefix_trades, _ = audit_frame("TEST", frame, index[61], index[last])
        causal = ["date", "emitted", "held", *SOURCES]
        pd.testing.assert_frame_equal(prefix_events[causal], events.loc[events.date.le(index[last].date().isoformat()), causal])
        closed = prefix_trades[prefix_trades.exit_reason.ne("terminal")]
        pd.testing.assert_frame_equal(closed, trades.iloc[:len(closed)])
    bad = frame.copy()
    bad.loc[index[70], "S_SIGNAL"] = True
    with pytest.raises(ValueError, match="去重"):
        audit_frame("TEST", bad, index[61], index[130])
    bad = frame.copy()
    bad.loc[index[85], "S_CROSS"] = True
    with pytest.raises(ValueError, match="组件"):
        audit_frame("TEST", bad, index[61], index[130])


def test_r16_summaries_keep_emitted_raw_executed_and_trade_return_denominators_separate():
    from gcn.backtest.signal_research_r16 import audit_frame, summarize_events, summarize_trades
    frame = _exit_fixture()
    events, trades, _ = audit_frame("TEST", frame, frame.index[61], frame.index[130])
    summary = summarize_events(events)
    totals = summary[summary.group_by.eq("all")].set_index("scope")
    assert totals.loc["raw", "events"] == 3
    assert totals.loc["emitted", "events"] == 2 and totals.loc["emitted", "win_rate_pct"] == 50.
    assert totals.loc["executed", "events"] == 1 and totals.loc["executed", "win_rate_pct"] == 100.
    assert totals.loc["ignored_flat", "interference"] == 1
    assert totals.loc["suppressed_held", "interference"] == 1
    assert totals.loc["pending_at_cutoff", "events"] == 0 and pd.isna(totals.loc["pending_at_cutoff", "win_rate_pct"])
    components = summary[summary.scope.eq("emitted") & summary.group_by.eq("component")].set_index("group")
    assert components.events.sum() == 3  # One emitted event has CROSS+DELAY, not two independent exits.
    assert components.loc["S_BEAR_RALLY", "complete"] == 0
    trade_summary = summarize_trades(trades)
    overall = trade_summary[trade_summary.group_by.eq("all")].iloc[0]
    assert overall.trades == 3 and overall.wins == 2 and overall.profit_to_loss == 1
    assert np.isclose(overall.win_rate_pct, 200/3)
    assert np.isclose(overall.max_giveback_pp, 32.)
    exit_groups = trade_summary[trade_summary.group_by.eq("exit_reason")].set_index("group")
    assert exit_groups.loc["terminal", "trades"] == 1 and exit_groups.loc["signal", "trades"] == 1
    assert exit_groups.loc["trail", "suppressed_s_count"] == 1


def test_r16_frozen_archive_reconciles_native_events_trades_costs_sources_and_hashes(tmp_path):
    import hashlib
    import json
    import pytest
    from gcn.backtest.signal_research_r16 import run_diagnostic, WINDOWS
    from gcn.backtest.historical_research import CORE, load_snapshot, evaluate_rule
    from gcn.recipes.gcn_main import compute_ehopt10
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    decision = run_diagnostic(snapshot, tmp_path)
    assert decision["stage"] == "diagnostic_only" and decision["recommended"] == "v5"
    assert not decision["production_changed"] and "selected" not in decision
    events = pd.read_csv(tmp_path / "events.csv")
    trades = pd.read_csv(tmp_path / "trades.csv")
    checks = pd.read_csv(tmp_path / "reconciliation.csv")
    assert len(checks) == len(CORE) * 4 and checks.reconciled.all()
    assert checks.executed.equals(checks.signal_exits)
    archived = pd.read_csv(ROOT / "reports/gcn-historical-r14-20260905/results/events.csv")
    archived = archived[archived.signal.eq("s") & archived.outcome_complete]
    fields = ["window", "symbol", "date", "ret20_pct", "mfe20_pct", "mae20_pct", "win", "interference"]
    order = ["window", "symbol", "date"]
    actual = events[events.emitted & events.outcome_complete]
    pd.testing.assert_frame_equal(actual[fields].sort_values(order).reset_index(drop=True),
                                  archived[fields].sort_values(order).reset_index(drop=True), check_dtype=False)
    frames, quality = load_snapshot(snapshot)
    for window, first, last in WINDOWS:
        start, end = pd.Timestamp(first), pd.Timestamp(last)
        selected = events[events.window.eq(window)]
        window_trades = trades[trades.window.eq(window)]
        assert selected.date.min() >= first and selected.date.max() <= last
        assert selected.loc[selected.outcome_complete, "outcome_date"].max() <= last
        assert selected.loc[~selected.outcome_complete,
                            ["ret20_pct", "mfe20_pct", "mae20_pct", "win", "interference"]].isna().all().all()
        prepared = {}
        for symbol in CORE:
            frame = compute_ehopt10(frames[symbol].loc[:end], version="v5", diagnostics=True)
            prepared[symbol] = {"frame": frame, "rules": {"v5": frame}}
            base = frame.loc[start:end]
            for row in window_trades[window_trades.symbol.eq(symbol)].itertuples():
                i = base.index.get_loc(pd.Timestamp(row.entry_date))
                j = len(base) if row.exit_reason == "terminal" else base.index.get_loc(pd.Timestamp(row.exit_date))
                peak = base.CLOSE.iloc[i:j].max()
                exit_price = base.CLOSE.iloc[-1] if row.exit_reason == "terminal" else base.OPEN.iloc[j]
                assert row.entry_signal_date == base.index[i - 1].date().isoformat()
                assert row.entry_b == bool(base.B_SIGNAL.iloc[i - 1])
                assert row.entry_jf == bool(base.ICON_JUEFAN.iloc[i - 1])
                assert np.isclose(row.peak_close, peak) and np.isclose(row.exit_price, exit_price)
                assert row.peak_close_date == base.CLOSE.iloc[i:j].idxmax().date().isoformat()
                assert np.isclose(row.return_pct, (exit_price / base.OPEN.iloc[i] * .999**2 - 1) * 100)
                assert np.isclose(row.giveback_pp, (peak - exit_price) / base.OPEN.iloc[i] * 100)
                if row.exit_reason == "signal":
                    event = selected[selected.trade_id.eq(row.trade_id) & selected.status.eq("executed")]
                    assert len(event) == 1 and event.iloc[0].date == row.exit_signal_date
                    assert tuple(getattr(row, c) for c in SOURCES) == tuple(bool(base[c].iloc[j - 1]) for c in SOURCES)
                else:
                    assert not any(getattr(row, c) for c in SOURCES)
                if row.exit_reason == "terminal":
                    assert pd.isna(row.exit_signal_date) and pd.isna(row.exit_raw_s)
        original = pd.DataFrame(evaluate_rule(prepared, "v5", start, end)["trades"])
        original = original.rename(columns={"hold_days": "hold_bars", "peak_close_pct": "peak_close_gain_pct"})
        fields = list(original.columns)
        pd.testing.assert_frame_equal(window_trades[fields].reset_index(drop=True), original,
                                      check_dtype=False)
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    assert manifest["source_quality"] == quality
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


def test_r16_empty_prefix_and_unmatured_reports_keep_schema_and_missing_denominators():
    from gcn.backtest.signal_research_r16 import audit_frame, summarize_events, summarize_trades
    frame = _exit_fixture(); index = frame.index
    full_events, full_trades, _ = audit_frame("TEST", frame, index[60], index[130])
    events, trades, check = audit_frame("TEST", frame, index[60], index[61])
    assert events.empty and trades.empty and check["trades"] == check["raw"] == 0
    pd.testing.assert_frame_equal(events, full_events.iloc[:0])
    pd.testing.assert_frame_equal(trades, full_trades.iloc[:0])
    assert summarize_events(events).complete.eq(0).all()
    assert summarize_events(events).win_rate_pct.isna().all()
    assert summarize_trades(trades).trades.eq(0).all()
    assert summarize_trades(trades).win_rate_pct.isna().all()
    short, _, _ = audit_frame("TEST", frame, index[60], index[64])
    summary = summarize_events(short)
    pending = summary[summary.scope.eq("pending_at_cutoff") & summary.group_by.eq("all")].iloc[0]
    assert pending.events == pending.incomplete == 1 and pending.complete == 0
    assert pd.isna(pending.win_rate_pct) and pd.isna(pending.mean_ret20_pct)
