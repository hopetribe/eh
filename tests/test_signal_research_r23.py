"""r23持仓日内压力/恢复；状态与最终交易标签严格分离。"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _paths(factors=(.8, 1., 1.25, 1., .75, 1.5), gross=(.8, 1.1, 1.2, 1.1, .9, 1.6)):
    from gcn.backtest.signal_research_r22 import PATH_SCHEMA
    dates = pd.bdate_range("2024-01-01", periods=len(factors)+1).strftime("%Y-%m-%d").tolist()
    trades = pd.DataFrame([{"symbol": "TEST", "trade_id": "TEST-1", "entry_kind": "JF",
                            "entry_date": dates[0], "exit_date": dates[-1], "exit_reason": "signal",
                            "return_pct": (.7*.999**2-1)*100, "intraday_factor": factors[-1],
                            "source_trusted": False}])
    rows = [{"symbol": "TEST", "trade_id": "TEST-1", "date": date, "observation": "close",
             "running_intraday_factor": factor, "running_gross_factor": net,
             "running_overnight_factor": net/factor} for date, factor, net in zip(dates, factors, gross)]
    rows.append({"symbol": "TEST", "trade_id": "TEST-1", "date": dates[-1], "observation": "open",
                 "running_intraday_factor": factors[-1], "running_gross_factor": .7})
    return trades, pd.DataFrame(rows, columns=PATH_SCHEMA).astype(PATH_SCHEMA), dates


def test_r23_observes_strict_pressure_and_distinct_positive_return_without_open_exit_close():
    from gcn.backtest.signal_research_r23 import audit_paths
    trades, paths, dates = _paths()
    audited, observations = audit_paths(trades, paths)
    pd.testing.assert_frame_equal(audited[list(trades.columns)], trades, check_exact=True)
    assert observations.date.tolist() == dates[:-1]
    assert observations.pressure.tolist() == [True, False, False, False, True, False]
    assert observations.nonnegative.tolist() == [False, True, True, True, False, True]
    assert observations.ever_positive_intraday.tolist() == [False, False, True, True, True, True]
    assert observations.pressure_streak.tolist() == [1, 0, 0, 0, 1, 0]
    assert observations.held_bars.tolist() == list(range(1, 7))
    assert observations.first_pressure_date.tolist() == [dates[0]]*6
    assert pd.isna(observations.first_recovery_date.iloc[0])
    assert observations.first_recovery_date.iloc[1:].eq(dates[1]).all()
    assert observations.first_return_date.iloc[:3].isna().all()
    assert observations.first_return_date.iloc[3:].eq(dates[3]).all()
    assert observations.first_return_recovery_date.iloc[:5].isna().all()
    assert observations.first_return_recovery_date.iloc[5] == dates[5]
    assert np.allclose(observations.net_return_pct, (paths.running_gross_factor.iloc[:-1]*.999**2-1)*100)
    row = audited.iloc[0]
    assert row.pressure_observed and row.recovered and not row.recovery_censored
    assert row.first_pressure_bar == 1 and row.first_pressure_net_pct < 0 and not row.first_pressure_after_net_positive
    assert row.recovery_bars == 1 and row.observed_bars_after_pressure == 5
    assert row.pressure_bars == 2 and row.max_pressure_streak == 1
    assert row.first_return_bar == 4 and row.first_return_after_net_positive
    assert row.first_return_recovery_date == dates[5] and row.return_recovery_bars == 2 and not row.return_censored
    assert row.post_pressure_peak_intraday_factor == row.post_return_peak_intraday_factor == 1.5
    assert np.isclose(row.post_return_peak_net_pct, (1.6*.999**2-1)*100)
    assert row.return_pct < 0  # Original OPEN result remains distinct from held CLOSE diagnostics.


def test_r23_prefixes_strict_unit_boundary_reset_terminal_censor_and_empty_schema():
    from gcn.backtest.signal_research_r23 import audit_paths, OBS_EXTRA_SCHEMA
    trades, paths, dates = _paths((1., np.nextafter(1., 0.), 1., np.nextafter(1., 2.), 1.), (1.3,)*5)
    audited, observations = audit_paths(trades, paths)
    assert observations.pressure.tolist() == [False, True, False, False, False]
    assert observations.ever_positive_intraday.tolist() == [False, False, False, True, True]
    a = audited.iloc[0]
    assert a.first_pressure_date == dates[1] and a.first_pressure_after_net_positive
    assert a.first_recovery_date == dates[2] and a.first_return_date == dates[4]
    assert a.return_censored and pd.isna(a.return_recovery_bars) and a.observed_bars_after_return == 0
    assert pd.isna(a.post_return_peak_net_pct) and pd.isna(a.post_return_peak_intraday_factor)
    for cut in dates:
        partial = paths[paths.date.le(cut)]
        _, observed = audit_paths(trades, partial)
        pd.testing.assert_frame_equal(observed, observations[observations.date.le(cut)], check_exact=True)
        for name in OBS_EXTRA_SCHEMA:
            if name.endswith("_date"):
                assert observed[name].dropna().le(cut).all()
    altered = paths.copy()
    altered.loc[altered.observation.eq("open"), ["running_intraday_factor", "close", "running_gross_factor"]] = [50., 9999., 80.]
    changed = trades.copy(); changed.return_pct = 10000.
    pd.testing.assert_frame_equal(audit_paths(changed, altered)[1], observations, check_exact=True)
    second, second_paths, second_dates = _paths((.9, .8), (.9, .8))
    second.trade_id = "TEST-2"; second_paths.trade_id = "TEST-2"
    second.exit_reason = "terminal"; second.exit_date = second_dates[1]
    second.return_pct = (.8*.999**2-1)*100
    second_paths = second_paths[second_paths.observation.eq("close")]
    combined, combined_obs = audit_paths(pd.concat([trades, second], ignore_index=True),
                                          pd.concat([paths, second_paths], ignore_index=True))
    b = combined.iloc[1]
    assert b.first_pressure_bar == 1 and b.pressure_bars == b.max_pressure_streak == 2
    assert not b.first_pressure_after_net_positive and pd.isna(b.first_return_date)
    assert b.recovery_censored and not b.recovered and b.observed_bars_after_pressure == 1
    assert pd.isna(b.recovery_bars) and b.return_pct == second.return_pct.iloc[0]
    assert not combined_obs[combined_obs.trade_id.eq("TEST-2")].ever_positive_intraday.any()
    one, _ = audit_paths(second, second_paths.iloc[:1])
    assert one.recovery_censored.iloc[0] and one.observed_bars_after_pressure.iloc[0] == 0
    empty, empty_obs = audit_paths(trades.iloc[:0], paths.iloc[:0])
    pd.testing.assert_frame_equal(empty, audited.iloc[:0], check_exact=True)
    pd.testing.assert_frame_equal(empty_obs, observations.iloc[:0], check_exact=True)


def test_r23_fixed_strata_use_actual_trade_denominators_and_keep_censored_winners_and_source_quality():
    from gcn.backtest.signal_research_r23 import audit_paths, summarize_trades
    from gcn.backtest.signal_research_r22 import COMPONENTS
    winner, winner_paths, _ = _paths()
    winner.return_pct = 12.
    loss, loss_paths, _ = _paths((.9, .8), (.9, .8))
    loss.trade_id = "TEST-2"; loss_paths.trade_id = "TEST-2"
    loss.entry_kind = "B"; loss.source_trusted = True
    loss.exit_reason = "trail"
    trades = pd.concat([winner, loss], ignore_index=True)
    for col in COMPONENTS:
        trades[col] = False
    trades.loc[1, ["B_BEAR_RECOVER", "B_CRASH_RECOVER"]] = True
    audited, _ = audit_paths(trades, pd.concat([winner_paths, loss_paths], ignore_index=True))
    summary = summarize_trades(audited)
    overall = summary[summary.group_by.eq("all")].set_index("scope")
    assert overall.loc["all", "trades"] == 2 and overall.loc["all", "wins"] == 1
    assert overall.loc["all", "pressure_trades"] == 2 and overall.loc["all", "recovered_trades"] == 1
    assert overall.loc["all", "recovery_censored_trades"] == 1 and overall.loc["all", "recovery_rate_pct"] == 50.
    assert overall.loc["all", "final_negative_intraday_trades"] == 1
    assert overall.loc["JF", "wins"] == 1 and overall.loc["B", "wins"] == 0
    groups = summary[summary.scope.eq("all")].set_index(["group_by", "group"])
    assert groups.loc[("recovery", "recovered"), "wins"] == 1
    assert groups.loc[("recovery", "censored"), "trades"] == 1
    assert groups.loc[("source_trusted", "true"), "trades"] == 1
    assert groups.loc[("source_trusted", "false"), "wins"] == 1
    assert groups.loc[("symbol", "NVDA"), "trades"] == 0
    sources = summary[summary.group_by.eq("source")].set_index("group")
    assert sources.loc["B_BEAR_RECOVER", "trades"] == sources.loc["B_CRASH_RECOVER", "trades"] == sources.loc["multiple", "trades"] == 1
    empty = summarize_trades(audited.iloc[:0])
    assert empty.trades.eq(0).all() and empty.recovery_rate_pct.isna().all() and empty.win_rate_pct.isna().all()
    assert list(empty.columns) == list(summary.columns)


def test_r23_training_reuses_frozen_orders_and_paths_and_binds_inputs_protocol_sources_and_environment(tmp_path, monkeypatch):
    import hashlib
    import json
    import pytest
    from gcn.backtest.signal_research_r23 import run_diagnostic
    import gcn.backtest.signal_research_r22 as r22
    def forbidden(*args, **kwargs):
        raise AssertionError("r23 must reuse frozen paths, not recompute signals or orders")
    monkeypatch.setattr(r22, "compute_ehopt10", forbidden)
    monkeypatch.setattr(r22, "audit_frame", forbidden)
    parent = ROOT / "reports/gcn-historical-r22-20260905"
    decision = run_diagnostic(parent, tmp_path)
    assert decision["stage"] == "diagnostic_only" and decision["recommended"] == "v5" and not decision["production_changed"]
    assert decision["window"] == ["training", "2021-08-27", "2024-08-26"]
    read = lambda p: pd.read_csv(p, float_precision="round_trip")
    original = read(parent / "training/trades.csv")
    original_paths = read(parent / "training/paths.csv")
    trades, observed = read(tmp_path / "trades.csv"), read(tmp_path / "observations.csv")
    pd.testing.assert_frame_equal(trades[list(original.columns)], original, check_exact=True)
    pd.testing.assert_frame_equal(observed[list(original_paths.columns)],
                                  original_paths[original_paths.observation.eq("close")].reset_index(drop=True), check_exact=True)
    assert len(trades) == 50 and len(observed) == int(original.hold_bars.sum())
    assert observed.date.max() <= "2024-08-26"
    checks = read(tmp_path / "reconciliation.csv")
    assert checks.reconciled.all() and checks.chain_reconciled.all() and checks.pressure_path_reconciled.all()
    assert checks.trades.sum() == len(trades) and checks.held_close_rows.sum() == len(observed)
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert manifest["input_manifest_sha256"] == digest(parent / "training/manifest.json")
    old = json.loads((parent / "training/manifest.json").read_bytes())
    assert manifest["parent_manifest_sha256"] == old["parent_manifest_sha256"]
    assert manifest["source_quality"] == old["source_quality"] and manifest["input_environment"] == old["environment"]
    assert manifest["input_algorithm_sources"] == old["algorithm_sources"]
    assert manifest["protocol_sha256"] == digest(ROOT / "reports/gcn-historical-r23-20260905/protocol.md")
    assert all(trades.source_trusted == trades.symbol.map(manifest["source_quality"]))
    for name, expected in manifest["outputs"].items():
        assert digest(tmp_path / name) == expected
    for name, expected in manifest["algorithm_sources"].items():
        assert digest(tmp_path / "source_snapshot" / name) == expected == digest(ROOT / name)
    for name, expected in manifest["input_files"].items():
        assert digest(tmp_path / "input_snapshot" / name) == expected == digest(parent / "training" / name)
    for name, expected in manifest["input_algorithm_sources"].items():
        assert digest(tmp_path / "input_source_snapshot" / name) == expected
    with pytest.raises(FileExistsError):
        run_diagnostic(parent, tmp_path)
    with pytest.raises(ValueError, match="固定窗口"):
        run_diagnostic(parent, tmp_path / "custom", window="custom")


def test_r23_rejects_changed_frozen_inputs_manifest_source_and_midrun_mutation_before_output(tmp_path, monkeypatch):
    import json
    import shutil
    import pytest
    import gcn.backtest.signal_research_r23 as r23
    parent = ROOT / "reports/gcn-historical-r22-20260905"
    original_audit = r23.audit_paths
    def forbidden(*args, **kwargs):
        raise AssertionError("Changed frozen input must be rejected before audit")
    monkeypatch.setattr(r23, "audit_paths", forbidden)
    for case, name, error in (("price", "paths.csv", "输入内容"),
                               ("source", "source_snapshot/gcn/backtest/engine.py", "冻结源码"),
                               ("manifest", "manifest.json", "manifest")):
        target = tmp_path / case
        shutil.copytree(parent / "training", target / "training")
        path = target / "training" / name
        if case == "manifest":
            manifest = json.loads(path.read_bytes())
            manifest["window"][2] = "2024-08-27"
            path.write_text(json.dumps(manifest))
        else:
            path.write_bytes(path.read_bytes()+b"\n")
        output = tmp_path / (case + "-output")
        with pytest.raises(ValueError, match=error):
            r23.run_diagnostic(target, output)
        assert not output.exists()
    target = tmp_path / "midrun"
    shutil.copytree(parent / "training", target / "training")
    def mutate(trades, paths):
        result = original_audit(trades, paths)
        path = target / "training/source_snapshot/gcn/backtest/engine.py"
        path.write_bytes(path.read_bytes()+b"\n")
        return result
    monkeypatch.setattr(r23, "audit_paths", mutate)
    with pytest.raises(ValueError, match="计算期间r22冻结源码变化"):
        r23.run_diagnostic(target, tmp_path / "midrun-output")
    assert not (tmp_path / "midrun-output").exists()


def test_r23_real_training_price_prefixes_keep_only_then_observed_pressure_and_recovery_states():
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.backtest.signal_research_r22 import audit_frame
    from gcn.backtest.signal_research_r23 import audit_paths
    from gcn.recipes.gcn_main import compute_ehopt10
    frames, _ = load_snapshot(ROOT / "reports/signal-audit-v5-review-20260904")
    start, end = pd.Timestamp("2021-08-27"), pd.Timestamp("2024-08-26")
    tested = 0
    for symbol in CORE:
        raw = frames[symbol].loc[:end]
        frame = compute_ehopt10(raw, version="v5", diagnostics=True)
        trades, paths, _ = audit_frame(symbol, frame, start, end)
        audited, full = audit_paths(trades, paths)
        if audited.empty:
            assert full.empty
            continue
        first = audited.iloc[0]
        cuts = {first[name] for name in ("entry_date", "first_pressure_date", "first_recovery_date",
                                         "first_return_date", "first_return_recovery_date", "last_held_close_date", "exit_date")
                if pd.notna(first[name])}
        for date in sorted(cuts):
            cut = pd.Timestamp(date)
            truncated = compute_ehopt10(raw.loc[:cut], version="v5", diagnostics=True)
            early_trades, early_paths, _ = audit_frame(symbol, truncated, start, cut)
            _, early = audit_paths(early_trades, early_paths)
            pd.testing.assert_frame_equal(early, full[full.date.le(date)].reset_index(drop=True), check_exact=True)
            tested += 1
    assert tested >= 27  # Ten-stock coverage includes a legitimate zero-trade NVDA branch.


def test_r23_other_fixed_windows_keep_original_orders_close_paths_and_same_run_provenance(tmp_path):
    import hashlib
    import json
    from gcn.backtest.signal_research_r23 import run_diagnostic, WINDOWS, R22_MANIFESTS
    parent = ROOT / "reports/gcn-historical-r22-20260905"
    read = lambda path: pd.read_csv(path, float_precision="round_trip")
    run_diagnostic(parent, tmp_path / "training")
    training = json.loads((tmp_path / "training/manifest.json").read_bytes())
    for window, start, end in WINDOWS[1:]:
        output = tmp_path / window
        decision = run_diagnostic(parent, output, window=window)
        assert decision["window"] == [window, start, end] and not decision["production_changed"]
        original = read(parent / window / "trades.csv")
        original_paths = read(parent / window / "paths.csv")
        trades, observed = read(output / "trades.csv"), read(output / "observations.csv")
        pd.testing.assert_frame_equal(trades[list(original.columns)], original, check_exact=True)
        pd.testing.assert_frame_equal(observed[list(original_paths.columns)],
                                      original_paths[original_paths.observation.eq("close")].reset_index(drop=True), check_exact=True)
        assert observed.date.between(start, end).all()
        for name in ("first_pressure_date", "first_recovery_date", "first_return_date", "first_return_recovery_date"):
            marked = observed[observed[name].notna()]
            assert marked[name].le(marked.date).all()
        assert trades.recovery_censored.eq(trades.first_pressure_date.notna() & trades.first_recovery_date.isna()).all()
        assert trades.return_censored.eq(trades.first_return_date.notna() & trades.first_return_recovery_date.isna()).all()
        checks = read(output / "reconciliation.csv")
        assert checks.pressure_path_reconciled.all() and checks.held_close_rows.sum() == len(observed)
        m = json.loads((output / "manifest.json").read_bytes())
        assert m["input_manifest_sha256"] == R22_MANIFESTS[window]
        for name in ("algorithm_sources", "environment", "source_quality", "parent_manifest_sha256", "protocol_sha256"):
            assert m[name] == training[name]
        for name, expected in m["outputs"].items():
            assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
