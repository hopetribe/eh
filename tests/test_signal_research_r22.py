"""r22原v5持仓执行归因；真实OPEN边界与价格因子还原。"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    from gcn.backtest.signal_research_r14 import COMPONENTS
    from gcn.backtest.signal_research_r16 import SOURCES
    ix = pd.bdate_range("2024-01-01", periods=9)
    o = np.array([50., 50., 50., 100., 120., 90., 80., 200., 100.])
    c = np.array([50., 50., 50., 110., 130., 100., 900., 100., 100.])
    f = pd.DataFrame({"OPEN": o, "CLOSE": c, "HIGH": np.maximum(o,c)+1,
                      "LOW": np.minimum(o,c)-1, "MID": 40.}, index=ix)
    for col in ("B_SETUP", "B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL", "S_RAW", *COMPONENTS, *SOURCES):
        f[col] = False
    f.loc[ix[2], "ICON_JUEFAN"] = True
    f.loc[ix[5], ["S_SIGNAL", "S_RAW", "S_DELAY"]] = True
    return f


def test_r22_reconstructs_original_trade_from_held_intraday_overnight_and_fees_not_entry_gap():
    from gcn.backtest.signal_research_r22 import audit_frame
    from gcn.backtest.signal_research_r16 import audit_trades
    f = _fixture(); ix = f.index
    trades, paths, check = audit_frame("TEST", f, ix[2], ix[-1])
    original, _ = audit_trades("TEST", f, ix[2], ix[-1])
    pd.testing.assert_frame_equal(trades[list(original.columns)], original, check_exact=True)
    r = trades.iloc[0]
    assert r.exit_reason == "signal" and r.exit_price == 80.  # S retains priority over trail.
    assert r.entry_gap_pct == 100. and r.entry_kind == "JF"
    assert np.isclose(r.overnight_factor, 120/110*90/130*80/100)
    assert np.isclose(r.intraday_factor, 110/100*130/120*100/90)
    assert r.cost_factor == .999**2
    assert np.isclose(r.reconstructed_return_pct, (.8*.999**2-1)*100)
    assert np.isclose(r.return_pct, r.reconstructed_return_pct)
    assert np.isclose(r.overnight_log_pct+r.intraday_log_pct+r.cost_log_pct, np.log(.8*.999**2)*100)
    assert r.last_held_close == 100. and r.last_held_close_date == str(ix[5].date())
    assert r.trail_reference == 104. and r.close_below_trail
    assert np.isclose(r.last_close_net_pct, -.1999) and np.isclose(r.exit_overnight_pct, -20.)
    assert np.isclose(r.exit_gap_impact_pp, r.return_pct-r.last_close_net_pct)
    assert np.isclose(r.worst_overnight_pct, (90/130-1)*100) and r.worst_overnight_date == str(ix[5].date())
    assert np.isclose(r.worst_intraday_pct, (130/120-1)*100) and r.worst_intraday_date == str(ix[4].date())
    assert paths.date.tolist() == [str(d.date()) for d in ix[3:7]]
    assert paths.observation.tolist() == ["close", "close", "close", "open"]
    assert pd.isna(paths.overnight_factor.iloc[0]) and pd.isna(paths.intraday_factor.iloc[-1])
    assert pd.isna(paths.close.iloc[-1]) and paths.open.iloc[-1] == 80.
    assert np.allclose(paths.running_gross_factor, [1.1, 1.3, 1., .8])
    assert check["reconciled"] and check["trades"] == 1 and check["path_rows"] == 4


def test_r22_open_exit_excludes_later_prices_and_terminal_prefixes_keep_causal_paths_and_empty_schema():
    from gcn.backtest.signal_research_r22 import audit_frame
    f = _fixture(); ix = f.index
    full, paths, _ = audit_frame("TEST", f, ix[2], ix[-1])
    changed = f.copy()
    changed.loc[ix[6], ["CLOSE", "HIGH", "LOW"]] = [1., 9999., .1]
    after, after_paths, _ = audit_frame("TEST", changed, ix[2], ix[-1])
    pd.testing.assert_frame_equal(after, full, check_exact=True)
    pd.testing.assert_frame_equal(after_paths, paths, check_exact=True)
    for last in range(3, len(f)):
        trades, prefix, _ = audit_frame("TEST", f, ix[2], ix[last])
        pd.testing.assert_frame_equal(prefix, paths[paths.date.le(str(ix[last].date()))], check_exact=True)
        pd.testing.assert_frame_equal(trades, audit_frame("TEST", f.loc[:ix[last]], ix[2], ix[last])[0], check_exact=True)
        r = trades.iloc[0]
        if last < 6:
            assert r.exit_reason == "terminal" and pd.isna(r.exit_overnight_pct) and r.exit_gap_impact_pp == 0.
            assert r.last_held_close == f.CLOSE.iloc[last]
            assert np.isclose(r.reconstructed_return_pct, (f.CLOSE.iloc[last]/100*.999**2-1)*100)
        else:
            pd.testing.assert_frame_equal(trades, full, check_exact=True)
        if last == 3:
            assert r.overnight_factor == 1. and pd.isna(r.worst_overnight_pct) and pd.isna(r.worst_overnight_date)
    empty = f.copy(); empty.ICON_JUEFAN = False
    trades, prefix, check = audit_frame("TEST", empty, ix[2], ix[-1])
    pd.testing.assert_frame_equal(trades, full.iloc[:0], check_exact=True)
    pd.testing.assert_frame_equal(prefix, paths.iloc[:0], check_exact=True)
    assert check["trades"] == check["path_rows"] == 0 and check["reconciled"]


def test_r22_actual_trade_chain_keeps_b_collision_multisource_and_counts_only_flat_bars():
    from gcn.backtest.signal_research_r22 import audit_frame
    f = _fixture(); ix = f.index
    f.loc[ix[6], ["CLOSE", "HIGH"]] = [90., 91.]
    f.loc[ix[6], ["B_SETUP", "B_BEAR_RECOVER", "B_CRASH_RECOVER"]] = True
    f.loc[ix[7], ["B_SIGNAL", "ICON_JUEFAN"]] = True
    f.loc[ix[8], ["CLOSE", "HIGH"]] = [110., 111.]
    trades, _, check = audit_frame("TEST", f, ix[2], ix[-1])
    a, b = trades.iloc[0], trades.iloc[1]
    assert len(trades) == 2 and a.entry_kind == "JF" and b.entry_kind == "B"
    assert b.entry_b and b.entry_jf and b.B_BEAR_RECOVER and b.B_CRASH_RECOVER
    assert b.setup_date == str(ix[6].date()) and b.entry_signal_date == str(ix[7].date())
    assert pd.isna(a.previous_trade_id) and pd.isna(a.flat_bars) and pd.isna(a.pair_return_pct)
    assert b.previous_trade_id == a.trade_id and b.previous_exit_date == a.exit_date
    assert b.previous_exit_reason == "signal" and b.previous_entry_kind == "JF"
    assert b.flat_bars == 2 and b.exit_reason == "terminal"
    expected = (.8 * 1.1 * .999**4 - 1) * 100
    assert np.isclose(b.pair_return_pct, expected) and np.isclose(b.chain_return_pct, expected)
    assert np.isclose(a.chain_return_pct, a.return_pct) and check["chain_reconciled"]


def test_r22_fixed_strata_preserve_denominators_source_overlap_and_signals_separate_from_orders():
    from gcn.backtest.signal_research_r22 import audit_frame, summarize_trades
    f = _fixture(); ix = f.index
    loss, _, _ = audit_frame("LOSS", f, ix[2], ix[-1])
    winner, _, _ = audit_frame("WIN", f, ix[2], ix[4])
    b = f.copy()
    b.loc[ix[1], ["B_SETUP", "B_BEAR_RECOVER", "B_CRASH_RECOVER"]] = True
    b.loc[ix[2], ["CLOSE", "HIGH"]] = [52., 53.]
    b.loc[ix[2], "B_SIGNAL"] = True
    b.loc[ix[4], "ICON_JUEFAN"] = True  # Native event while already holding is not a second entry.
    comparator, _, check = audit_frame("B", b, ix[2], ix[-1])
    assert check["b_signals"] == check["b_trades"] == 1 and check["jf_signals"] == 2 and check["jf_trades"] == 0
    assert check["entry_signals"] == 2 and check["s_signals"] == check["signal_exits"] == 1
    trades = pd.concat([loss, winner, comparator], ignore_index=True)
    summary = summarize_trades(trades)
    overall = summary[summary.group_by.eq("all")].set_index("scope")
    assert overall.loc["all", "trades"] == 3 and overall.loc["all", "wins"] == 1
    assert overall.loc["JF", "trades"] == 2 and overall.loc["JF", "negative_overnight_trades"] == 1
    assert overall.loc["all", "negative_intraday_trades"] == 0
    assert overall.loc["all", "exit_gap_worsened_trades"] == 2
    sources = summary[summary.group_by.eq("source")].set_index("group")
    assert sources.loc["B_BEAR_RECOVER", "trades"] == sources.loc["B_CRASH_RECOVER", "trades"] == 1
    assert sources.loc["multiple", "trades"] == 1  # Not mutually exclusive groups.
    empty = summarize_trades(trades.iloc[:0])
    assert empty.trades.eq(0).all() and empty.win_rate_pct.isna().all()
    assert set(empty.columns) == set(summary.columns)


def test_r22_training_archive_matches_frozen_v5_orders_and_binds_exact_protocol_window_and_sources(tmp_path):
    import hashlib
    import json
    import pytest
    from gcn.backtest.signal_research_r22 import run_diagnostic, WINDOWS
    from gcn.backtest.signal_research_r16 import TRADE_COLUMNS
    from gcn.backtest.historical_research import load_snapshot
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    decision = run_diagnostic(snapshot, tmp_path)
    assert decision["stage"] == "diagnostic_only" and not decision["production_changed"]
    assert decision["recommended"] == "v5" and decision["window"] == ["training", "2021-08-27", "2024-08-26"]
    assert WINDOWS[-2:] == (("recent", "2025-08-27", "2026-08-26"), ("full", "2021-08-27", "2026-08-26"))
    original = pd.read_csv(ROOT / "reports/gcn-historical-r20-20260905/training/trades.csv", float_precision="round_trip")
    trades = pd.read_csv(tmp_path / "trades.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(trades[["window", *TRADE_COLUMNS]], original[["window", *TRADE_COLUMNS]], check_exact=True)
    assert np.allclose(trades.return_pct, trades.reconstructed_return_pct, rtol=1e-11, atol=1e-10)
    assert np.allclose(trades.overnight_factor*trades.intraday_factor*trades.cost_factor, 1+trades.return_pct/100)
    paths = pd.read_csv(tmp_path / "paths.csv", float_precision="round_trip")
    assert paths.date.max() <= "2024-08-26" and paths.loc[paths.observation.eq("open"), "close"].isna().all()
    checks = pd.read_csv(tmp_path / "reconciliation.csv")
    assert checks.reconciled.all() and checks.chain_reconciled.all()
    assert checks.trades.sum() == len(trades) and checks.path_rows.sum() == len(paths)
    assert checks.b_trades.sum()+checks.jf_trades.sum() == len(trades)
    _, quality = load_snapshot(snapshot)
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    assert manifest["source_quality"] == quality
    assert manifest["parent_manifest_sha256"] == hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    assert manifest["protocol_sha256"] == hashlib.sha256((ROOT / "reports/gcn-historical-r22-20260905/protocol.md").read_bytes()).hexdigest()
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((tmp_path / "source_snapshot" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        run_diagnostic(snapshot, tmp_path)
    with pytest.raises(ValueError, match="固定窗口"):
        run_diagnostic(snapshot, tmp_path / "bad", window="custom")


def test_r22_real_training_price_prefixes_preserve_observed_paths_and_closed_trade_chains():
    from gcn.backtest.signal_research_r22 import audit_frame
    from gcn.backtest.historical_research import load_snapshot, CORE
    from gcn.recipes.gcn_main import compute_ehopt10
    frames, _ = load_snapshot(ROOT / "reports/signal-audit-v5-review-20260904")
    start, end = pd.Timestamp("2021-08-27"), pd.Timestamp("2024-08-26")
    checked = 0
    for symbol in CORE:
        raw = frames[symbol].loc[:end]
        frame = compute_ehopt10(raw, version="v5", diagnostics=True)
        trades, paths, _ = audit_frame(symbol, frame, start, end)
        if trades.empty:
            cut = frame.loc[start:].index[20]
            prefix = compute_ehopt10(raw.loc[:cut], version="v5", diagnostics=True)
            actual, observed, check = audit_frame(symbol, prefix, start, cut)
            pd.testing.assert_frame_equal(actual, trades, check_exact=True)
            pd.testing.assert_frame_equal(observed, paths, check_exact=True)
            assert check["trades"] == 0 and check["chain_reconciled"]
            checked += 1
            continue
        first = trades.iloc[0]
        i = frame.index.get_loc(pd.Timestamp(first.entry_date))
        j = frame.index.get_loc(pd.Timestamp(first.exit_date))
        for pos in sorted({min(i+1,j), max(i,j-1), j}):
            cut = frame.index[pos]
            prefix = compute_ehopt10(raw.loc[:cut], version="v5", diagnostics=True)
            actual, observed, _ = audit_frame(symbol, prefix, start, cut)
            pd.testing.assert_frame_equal(observed, paths[paths.date.le(str(cut.date()))], check_exact=True)
            closed = actual[actual.exit_reason.ne("terminal")]
            expected = trades[trades.trade_id.isin(closed.trade_id)]
            pd.testing.assert_frame_equal(closed, expected, check_exact=True)
            checked += 1
    assert checked >= 20


def test_r22_other_fixed_windows_truncate_before_indicators_and_reconcile_same_source_v5_orders(tmp_path, monkeypatch):
    import io
    import json
    from gcn.backtest import signal_research_r22 as research
    from gcn.backtest.signal_research_r16 import audit_trades, TRADE_COLUMNS
    from gcn.backtest.historical_research import load_snapshot, CORE
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    research.run_diagnostic(snapshot, tmp_path / "training")
    baseline = json.loads((tmp_path / "training/manifest.json").read_bytes())
    original_compute = research.compute_ehopt10
    observed_frames = []
    cutoff = None
    def checked(raw, *args, **kwargs):
        assert raw.index.max() <= cutoff
        result = original_compute(raw, *args, **kwargs)
        observed_frames.append(result)
        return result
    monkeypatch.setattr(research, "compute_ehopt10", checked)
    for name, first, last in research.WINDOWS[1:]:
        cutoff = pd.Timestamp(last); start = pd.Timestamp(first)
        observed_frames.clear()
        output = tmp_path / name
        decision = research.run_diagnostic(snapshot, output, window=name)
        assert decision["window"] == [name, first, last] and len(observed_frames) == 10
        trades = pd.read_csv(output / "trades.csv", float_precision="round_trip")
        expected = pd.concat([audit_trades(symbol, frame, start, cutoff)[0]
                              for symbol, frame in zip(CORE, observed_frames)], ignore_index=True)
        normalized = pd.read_csv(io.StringIO(expected.to_csv(index=False)), float_precision="round_trip")
        pd.testing.assert_frame_equal(trades[list(TRADE_COLUMNS)], normalized, check_exact=True)
        assert np.allclose(trades.return_pct, trades.reconstructed_return_pct, rtol=1e-11, atol=1e-10)
        paths = pd.read_csv(output / "paths.csv", float_precision="round_trip")
        assert paths.date.between(first,last).all() and paths.loc[paths.observation.eq("open"), "close"].isna().all()
        for row in trades.itertuples():
            held = paths[paths.trade_id.eq(row.trade_id)]
            assert np.isclose(held.overnight_factor.prod()*held.intraday_factor.prod()*.999**2, 1+row.return_pct/100)
            assert len(held) == row.hold_bars + int(row.exit_reason != "terminal")
        manifest = json.loads((output / "manifest.json").read_bytes())
        for key in ("parent_manifest_sha256", "source_quality", "protocol_sha256", "algorithm_sources", "environment"):
            assert manifest[key] == baseline[key]
