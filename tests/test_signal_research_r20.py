"""r20诊断原v5绝反风险路径，不产生新买卖订单。"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _path_fixture():
    from gcn.backtest.signal_research_r14 import COMPONENTS
    from gcn.backtest.signal_research_r16 import SOURCES
    index = pd.bdate_range("2024-01-01", periods=11)
    close = np.array([94., 95., 95., 101., 112., 100., 120., 96., 300., 99., 100.])
    frame = pd.DataFrame({"OPEN": 100., "CLOSE": close, "HIGH": np.maximum(close, 100.) + 1,
                          "LOW": np.minimum(close, 100.) - 1, "MID": 90.}, index=index)
    for col in ("B_SETUP", "B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL", "S_RAW", *COMPONENTS, *SOURCES):
        frame[col] = False
    frame.loc[index[0], "LOW"] = 90.
    frame.loc[index[2], "ICON_JUEFAN"] = True
    frame.loc[index[7], ["S_SIGNAL", "S_RAW", "S_DELAY"]] = True
    frame.loc[index[8], "OPEN"] = 98.
    return frame


def test_r20_uses_original_orders_actual_entry_and_held_close_only_for_risk_and_events():
    from gcn.backtest.signal_research_r20 import audit_frame
    from gcn.backtest.signal_research_r16 import audit_trades
    frame = _path_fixture(); ix = frame.index
    trades, paths, check = audit_frame("TEST", frame, ix[2], ix[-1])
    original, _ = audit_trades("TEST", frame, ix[2], ix[-1])
    pd.testing.assert_frame_equal(trades[list(original.columns)], original, check_exact=True)
    trade = trades.iloc[0]
    assert check["reconciled"] and check["trades"] == check["jf_trades"] == 1
    assert check["path_bars"] == trade.hold_bars == 5
    assert trade.entry_kind == "JF" and trade.risk_status == "valid"
    assert trade.signal_close == 95. and trade.signal_high == 101.
    assert trade.base_low3 == 90. and trade.risk_r == trade.risk_pct == 10.
    assert np.isclose(trade.entry_gap_pct, (100/95-1)*100)
    assert np.isclose(trade.break_even, 100/.999**2)
    assert trade.first_net_positive_date == ix[3].date().isoformat()
    assert trade.first_return_to_be_date == ix[5].date().isoformat()
    assert pd.isna(trade.first_trail_covers_be_date)
    assert trade.post_return_peak_close == 120. and trade.post_return_bars == 2
    assert trade.post_return_peak_date == ix[6].date().isoformat()
    assert trade.net_profit_to_loss and trade.observation_state == "closed"
    assert np.isclose(trade.return_pct, (.98*.999**2-1)*100)
    assert np.isclose(trade.giveback_pp, 22.)
    assert paths.date.tolist() == [i.date().isoformat() for i in ix[3:8]]
    assert paths.close.tolist() == [101., 112., 100., 120., 96.]
    assert paths.running_peak_close.tolist() == [101., 112., 112., 120., 120.]
    assert paths.running_trough_close.tolist() == [101., 101., 100., 100., 96.]
    assert np.allclose(paths.trail_reference, paths.running_peak_close * .8)
    assert np.allclose(paths.mfe_close_r, [.1, 1.2, 1.2, 2., 2.])
    assert np.allclose(paths.mae_close_r, [0., 0., 0., 0., .4])
    assert paths.net_positive.tolist() == [True, True, False, True, False]
    assert paths.returned_to_be.tolist() == [False, False, True, True, True]
    assert not paths.trail_covers_be.any()
    assert np.allclose(paths.close_net_pct, (paths.close/100*.999**2-1)*100)


def test_r20_invalid_risk_zero_events_and_b_collision_never_invent_r_multiples():
    from gcn.backtest.signal_research_r20 import audit_frame, EXTRA_SCHEMA
    frame = _path_fixture(); ix = frame.index
    full, full_paths, _ = audit_frame("TEST", frame, ix[2], ix[-1])
    for entry, status in ((90., "nonpositive"), (89., "nonpositive"), (100., "invalid")):
        bad = frame.copy()
        bad.loc[ix[3], "OPEN"] = entry
        if status == "invalid":
            bad.loc[ix[0], "LOW"] = np.nan
        trades, paths, _ = audit_frame("TEST", bad, ix[2], ix[-1])
        row = trades.iloc[0]
        assert row.risk_status == status
        assert row.risk_r == entry-90 if status == "nonpositive" else pd.isna(row.risk_r)
        assert paths[["mfe_close_r", "mae_close_r"]].isna().all().all()
        assert np.isfinite(paths.close_net_pct).all()
    empty = frame.copy()
    empty["ICON_JUEFAN"] = False
    trades, paths, check = audit_frame("TEST", empty, ix[2], ix[-1])
    pd.testing.assert_frame_equal(trades, full.iloc[:0])
    pd.testing.assert_frame_equal(paths, full_paths.iloc[:0])
    assert check["trades"] == check["jf_trades"] == check["path_bars"] == 0
    assert check["reconciled"]
    collision = frame.copy()
    collision.loc[ix[1], ["B_SETUP", "B_CRASH_RECOVER"]] = True
    collision.loc[ix[1], "HIGH"] = 94.
    collision.loc[ix[2], "B_SIGNAL"] = True
    trades, paths, check = audit_frame("TEST", collision, ix[2], ix[-1])
    assert len(trades) == check["b_trades"] == 1 and paths.empty
    assert trades.iloc[0].entry_b and trades.iloc[0].entry_jf and trades.iloc[0].entry_kind == "B"
    assert trades.iloc[0].risk_status == "not_applicable"
    columns = set(EXTRA_SCHEMA) - {"entry_kind", "risk_status", "observation_state"}
    assert trades[list(columns)].isna().all().all()


def test_r20_causal_path_prefixes_strict_breakeven_and_terminal_pending_state():
    from gcn.backtest.signal_research_r20 import audit_frame
    frame = _path_fixture(); ix = frame.index
    full, paths, _ = audit_frame("TEST", frame, ix[2], ix[-1])
    for last in range(3, len(frame)):
        trades, prefix, _ = audit_frame("TEST", frame, ix[2], ix[last])
        truncated = audit_frame("TEST", frame.loc[:ix[last]], ix[2], ix[last])
        pd.testing.assert_frame_equal(prefix, paths[paths.date.le(ix[last].date().isoformat())])
        pd.testing.assert_frame_equal(trades, truncated[0])
        pd.testing.assert_frame_equal(prefix, truncated[1])
        if last < 8:
            assert trades.iloc[0].exit_reason == "terminal"
            assert trades.iloc[0].exit_price == frame.CLOSE.iloc[last]
            assert trades.iloc[0].observation_state == ("pending_signal" if last == 7 else "open")
        else:
            pd.testing.assert_frame_equal(trades, full)
    # Equality is not net-positive; a later equality is a return after positive.
    exact = frame.copy()
    exact[["S_SIGNAL", "S_RAW", "S_DELAY"]] = False
    be = 100/.999**2
    exact.loc[ix[3:6], "CLOSE"] = [be, be/.8, be]
    trades, paths, _ = audit_frame("TEST", exact, ix[2], ix[5])
    row = trades.iloc[0]
    assert paths.net_positive.tolist() == [False, True, False]
    assert row.first_net_positive_date == row.first_trail_covers_be_date == ix[4].date().isoformat()
    assert row.first_return_to_be_date == ix[5].date().isoformat()
    assert row.observation_state == "pending_trail" and row.exit_reason == "terminal"
    assert row.post_return_bars == 0 and pd.isna(row.post_return_peak_close)
    assert pd.isna(row.post_return_peak_date)
    after, after_paths, _ = audit_frame("TEST", exact, ix[2], ix[6])
    assert after.iloc[0].exit_reason == "trail" and after.iloc[0].observation_state == "closed"
    assert after.iloc[0].exit_price == 100.  # Actual next OPEN, not the break-even reference.
    pd.testing.assert_frame_equal(paths, after_paths)


def test_r20_fixed_strata_keep_b_comparator_and_post_return_winner_counterexamples():
    from gcn.backtest.signal_research_r20 import audit_frame, summarize_trades
    frame = _path_fixture(); ix = frame.index
    loser = audit_frame("LOSS", frame, ix[2], ix[-1])[0]
    winner = audit_frame("WIN", frame, ix[2], ix[6])[0]
    b = frame.copy()
    b.loc[ix[1], ["B_SETUP", "B_CRASH_RECOVER"]] = True
    b.loc[ix[1], "HIGH"] = 94.
    b.loc[ix[2], "B_SIGNAL"] = True
    comparator = audit_frame("B", b, ix[2], ix[-1])[0]
    trades = pd.concat([loser, winner, comparator], ignore_index=True)
    summary = summarize_trades(trades)
    overall = summary[summary.group_by.eq("all")].set_index("scope")
    assert overall.loc["all", "trades"] == 3 and overall.loc["all", "wins"] == 1
    assert overall.loc["B", "trades"] == 1 and pd.isna(overall.loc["B", "median_risk_pct"])
    assert overall.loc["JF", "trades"] == 2 and overall.loc["JF", "wins"] == 1
    assert overall.loc["JF", "returned_to_be"] == 2 and overall.loc["JF", "net_profit_to_loss"] == 1
    assert overall.loc["JF", "median_risk_pct"] == 10.
    assert overall.loc["JF", "trail_covers_be"] == 0
    invalid = summary[summary.group_by.eq("risk_status") & summary.group.eq("invalid")].iloc[0]
    assert invalid.trades == 0 and pd.isna(invalid.win_rate_pct)
    recovered = summary[summary.group_by.eq("be_path") & summary.group.eq("returned")].iloc[0]
    assert recovered.trades == 2 and recovered.wins == 1
    assert set(summary.group_by) == {"all", "exit_reason", "risk_status", "outcome", "be_path",
                                     "trail_covers_be", "symbol"}
    empty = summarize_trades(trades.iloc[:0])
    assert empty.trades.eq(0).all() and empty.win_rate_pct.isna().all()
    assert empty.mean_return_pct.isna().all()


def test_r20_training_archive_reconciles_frozen_orders_fees_risk_sources_and_window_cutoff(tmp_path, monkeypatch):
    import hashlib
    import json
    import pytest
    from gcn.backtest import signal_research_r20 as research
    from gcn.backtest.signal_research_r20 import run_diagnostic
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.recipes.gcn_main import compute_ehopt10
    seen = []
    def bounded(raw, **kwargs):
        assert raw.index.max() <= pd.Timestamp("2024-08-26")
        seen.append(len(raw))
        return compute_ehopt10(raw, **kwargs)
    monkeypatch.setattr(research, "compute_ehopt10", bounded)
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    decision = run_diagnostic(snapshot, tmp_path)
    assert decision["window"] == ["training", "2021-08-27", "2024-08-26"]
    assert decision["stage"] == "diagnostic_only" and decision["recommended"] == "v5"
    assert not decision["production_changed"] and "selected" not in decision
    assert len(seen) == len(CORE)
    # Preserve emitted float bytes on re-read before exact arithmetic comparisons.
    trades = pd.read_csv(tmp_path / "trades.csv", float_precision="round_trip")
    paths = pd.read_csv(tmp_path / "paths.csv", float_precision="round_trip")
    checks = pd.read_csv(tmp_path / "reconciliation.csv")
    frozen = pd.read_csv(ROOT / "reports/gcn-historical-r16-20260905/results/trades.csv",
                         float_precision="round_trip")
    expected = frozen[frozen.window.eq("training")].reset_index(drop=True)
    pd.testing.assert_frame_equal(trades[list(expected.columns)], expected, check_exact=True)
    assert len(checks) == len(CORE) and checks.reconciled.all()
    assert checks.trades.sum() == len(trades)
    assert checks.path_bars.sum() == len(paths) == trades.loc[trades.entry_kind.eq("JF"), "hold_bars"].sum()
    assert paths.window.eq("training").all() and paths.date.max() <= "2024-08-26"
    frames, quality = load_snapshot(snapshot)
    for row in trades.itertuples():
        raw = frames[row.symbol]
        s = raw.index.get_loc(pd.Timestamp(row.entry_signal_date))
        i = raw.index.get_loc(pd.Timestamp(row.entry_date))
        terminal = row.exit_reason == "terminal"
        j = raw.index.get_loc(pd.Timestamp(row.exit_date)) + int(terminal)
        assert np.isclose(row.return_pct, (row.exit_price / row.entry_open * .999**2-1)*100)
        held = paths[paths.trade_id.eq(row.trade_id)]
        if row.entry_b:
            assert row.entry_kind == "B" and held.empty and pd.isna(row.risk_r)
            continue
        assert row.base_low3 == min(raw.low.iloc[s-2:s+1])
        assert row.risk_r == row.entry_open - row.base_low3
        assert row.signal_close == raw.close.iloc[s] and row.signal_high == raw.high.iloc[s]
        assert len(held) == j-i and np.allclose(held.close, raw.close.iloc[i:j])
        assert np.allclose(held.running_peak_close, raw.close.iloc[i:j].cummax())
        assert np.allclose(held.running_trough_close, raw.close.iloc[i:j].cummin())
        assert np.allclose(held.close_net_pct, (held.close/row.entry_open*.999**2-1)*100)
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    assert manifest["source_quality"] == quality and manifest["window"] == decision["window"]
    assert manifest["parent_manifest_sha256"] == hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    assert manifest["protocol_sha256"] == hashlib.sha256((ROOT / "reports/gcn-historical-r20-20260905/protocol.md").read_bytes()).hexdigest()
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((tmp_path / "source_snapshot" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    before = (tmp_path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_diagnostic(snapshot, tmp_path)
    assert (tmp_path / "manifest.json").read_bytes() == before
    with pytest.raises(ValueError, match="窗口"):
        run_diagnostic(snapshot, tmp_path / "invalid", window="optimized")


def test_r20_remaining_fixed_windows_match_r16_orders_and_same_training_source_contract(tmp_path, monkeypatch):
    import hashlib
    import json
    from gcn.backtest import signal_research_r20 as research
    from gcn.backtest.signal_research_r14 import WINDOWS
    from gcn.backtest.historical_research import CORE
    from gcn.recipes.gcn_main import compute_ehopt10
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    # Compare windows generated by this runtime, not new source hashes to an old frozen runtime.
    research.run_diagnostic(snapshot, tmp_path / "training")
    training = json.loads((tmp_path / "training/manifest.json").read_bytes())
    frozen = pd.read_csv(ROOT / "reports/gcn-historical-r16-20260905/results/trades.csv",
                         float_precision="round_trip")
    for window, first, last in WINDOWS[1:]:
        calls = []
        def bounded(raw, **kwargs):
            assert raw.index.max() <= pd.Timestamp(last)
            calls.append(len(raw))
            return compute_ehopt10(raw, **kwargs)
        monkeypatch.setattr(research, "compute_ehopt10", bounded)
        output = tmp_path / window
        result = research.run_diagnostic(snapshot, output, window=window)
        assert result["window"] == [window, first, last] and not result["production_changed"]
        assert len(calls) == len(CORE)
        trades = pd.read_csv(output / "trades.csv", float_precision="round_trip")
        paths = pd.read_csv(output / "paths.csv", float_precision="round_trip")
        expected = frozen[frozen.window.eq(window)].reset_index(drop=True)
        pd.testing.assert_frame_equal(trades[list(expected.columns)], expected, check_exact=True)
        assert paths.date.min() >= first and paths.date.max() <= last
        for row in trades.itertuples():
            held = paths[paths.trade_id.eq(row.trade_id)]
            if row.entry_kind == "B":
                assert held.empty and pd.isna(row.risk_r)
                continue
            assert len(held) == row.hold_bars
            assert held.date.min() == row.entry_date
            assert held.date.max() == (row.exit_date if row.exit_reason == "terminal" else row.exit_signal_date)
            assert np.isclose(row.peak_close, held.close.max())
            if pd.notna(row.first_return_to_be_date):
                subsequent = held[held.date.gt(row.first_return_to_be_date)]
                assert row.post_return_bars == len(subsequent)
                assert np.isclose(row.post_return_peak_close, subsequent.close.max(), equal_nan=True)
        manifest = json.loads((output / "manifest.json").read_bytes())
        for field in ("algorithm_sources", "environment", "parent_manifest_sha256", "protocol_sha256", "source_quality"):
            assert manifest[field] == training[field]
        for name, digest in manifest["outputs"].items():
            assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
        archived = ROOT / "reports/gcn-historical-r20-20260905" / window
        old_manifest = json.loads((archived / "manifest.json").read_bytes())
        for name, digest in old_manifest["outputs"].items():
            assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
        for name, digest in old_manifest["algorithm_sources"].items():
            assert hashlib.sha256((archived / "source_snapshot" / name).read_bytes()).hexdigest() == digest


def test_r20_real_price_prefixes_preserve_only_causal_path_columns_not_future_trade_labels():
    from gcn.backtest.signal_research_r20 import audit_frame
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.recipes.gcn_main import compute_ehopt10
    frames, _ = load_snapshot(ROOT / "reports/signal-audit-v5-review-20260904")
    start, end = pd.Timestamp("2021-08-27"), pd.Timestamp("2024-08-26")
    checked = 0
    for symbol in CORE:
        raw = frames[symbol].loc[:end]
        full = compute_ehopt10(raw, version="v5", diagnostics=True)
        trades, paths, _ = audit_frame(symbol, full, start, end)
        jf = trades[trades.entry_kind.eq("JF")]
        if jf.empty:
            continue
        first = jf.iloc[0]
        dates = {first.entry_date, first.exit_date}
        if pd.notna(first.first_return_to_be_date):
            dates.add(first.first_return_to_be_date)
        for date in sorted(dates):
            last = pd.Timestamp(date)
            frame = compute_ehopt10(raw.loc[:last], version="v5", diagnostics=True)
            prefix_trades, prefix_paths, _ = audit_frame(symbol, frame, start, last)
            pd.testing.assert_frame_equal(prefix_paths, paths[paths.date.le(date)].reset_index(drop=True),
                                          check_exact=True)
            closed = prefix_trades[prefix_trades.exit_reason.ne("terminal")]
            pd.testing.assert_frame_equal(closed, trades.iloc[:len(closed)], check_exact=True)
            checked += 1
    assert checked >= 12
