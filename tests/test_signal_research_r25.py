"""r25只读真实退出/再入场；信号状态不重置、不回填未来结果。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _frame():
    from gcn.backtest.signal_research_r14 import COMPONENTS
    index = pd.bdate_range("2024-01-01", periods=10)
    frame = pd.DataFrame({"OPEN": 100., "HIGH": 110., "LOW": 80., "CLOSE": 90., "MID": 100.}, index=index)
    for col in ("B_ALL_RAW", "JF_RAW", "B_SETUP", "B_ENTRY_SIGNAL", "B_SETUP_EXPIRED",
                "B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL", *COMPONENTS):
        frame[col] = False
    frame.loc[index[[0, 2]], "B_ALL_RAW"] = True
    frame.loc[index[0], ["B_SETUP", "B_CRASH_RECOVER"]] = True
    frame.loc[index[4], "CLOSE"] = 111.
    frame.loc[index[4], ["B_ENTRY_SIGNAL", "B_SIGNAL", "ICON_JUEFAN", "JF_RAW"]] = True
    return frame


def test_r25_signal_states_keep_full_history_suppression_pending_and_resolution_without_future_backfill():
    from gcn.backtest.signal_research_r25 import signal_states
    frame = _frame(); dates = frame.index.strftime("%Y-%m-%d")
    states = signal_states(frame)
    assert states.raw_b_count20.tolist() == [1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
    assert states.raw_b_suppressed.tolist() == [False, False, True, False, False, False, False, False, False, False]
    assert states.pending_setup_date.iloc[:4].eq(dates[0]).all()
    assert states.pending_setup_age.iloc[:4].tolist() == [0, 1, 2, 3]
    assert states.pending_B_CRASH_RECOVER.iloc[:4].all()
    assert states.pending_setup_date.iloc[4:].isna().all()
    assert states.resolved_setup_status.iloc[:4].isna().all()
    assert states.resolved_setup_status.iloc[4] == "confirmed"
    assert states.resolved_setup_date.iloc[4] == dates[0]
    assert states.resolved_B_CRASH_RECOVER.iloc[4]
    assert states.B_SIGNAL.iloc[4] and states.ICON_JUEFAN.iloc[4]
    assert states.no_raw_buy.iloc[1] and not states.no_raw_buy.iloc[4]
    for cut in frame.index:
        pd.testing.assert_frame_equal(signal_states(frame.loc[:cut]), states.loc[:cut], check_exact=True)
    pd.testing.assert_frame_equal(signal_states(frame.iloc[:0]), states.iloc[:0], check_exact=True)


def test_r25_setup_expiry_replacement_and_signal_reconciliation_preserve_known_only_states():
    import pytest
    from gcn.backtest.signal_research_r25 import signal_states
    for replacement in (False, True):
        frame = _frame(); index = frame.index
        frame.loc[index[4], ["B_ENTRY_SIGNAL", "B_SIGNAL"]] = False
        frame.loc[index[4], "CLOSE"] = 90.
        frame.loc[index[7 if replacement else 5], "B_SETUP_EXPIRED"] = True
        if replacement:
            frame.loc[index[2], ["B_SETUP", "B_BEAR_RECOVER"]] = True
        states = signal_states(frame)
        expiry = 7 if replacement else 5
        assert states.resolved_setup_status.iloc[expiry] == "expired"
        assert pd.isna(states.pending_setup_date.iloc[expiry])
        if replacement:
            assert states.resolved_setup_status.iloc[2] == "replaced"
            assert states.resolved_setup_date.iloc[2] == str(index[0].date())
            assert states.pending_setup_date.iloc[2] == str(index[2].date())
            assert states.pending_B_BEAR_RECOVER.iloc[2] and states.resolved_B_CRASH_RECOVER.iloc[2]
        for cut in index:
            pd.testing.assert_frame_equal(signal_states(frame.loc[:cut]), states.loc[:cut], check_exact=True)
        broken = frame.copy(); broken.loc[index[expiry], "B_SETUP_EXPIRED"] = False
        with pytest.raises(ValueError, match="还原"):
            signal_states(broken)
        broken = frame.copy(); broken.loc[index[8], "B_SIGNAL"] = True
        with pytest.raises(ValueError, match="确认"):
            signal_states(broken)


def _trades():
    frame = _frame(); idx = frame.index
    frame.loc[idx[0], ["ICON_JUEFAN", "JF_RAW"]] = True
    frame.loc[idx[3], "OPEN"] = 80.
    frame.loc[idx[5], "OPEN"] = 120.
    frame.loc[idx[8], "OPEN"] = 90.
    rows = [dict(rule="v5", symbol="TEST", entry_date=str(idx[1].date()), exit_date=str(idx[8].date()),
                 exit_reason="trail", return_pct=(.9*.999**2-1)*100, entry_b=False, entry_jf=True),
            dict(rule="JF-joint-pressure", symbol="TEST", entry_date=str(idx[1].date()), exit_date=str(idx[3].date()),
                 exit_reason="joint_pressure", return_pct=(.8*.999**2-1)*100, entry_b=False, entry_jf=True),
            dict(rule="JF-joint-pressure", symbol="TEST", entry_date=str(idx[5].date()), exit_date=str(idx[8].date()),
                 exit_reason="trail", return_pct=(.75*.999**2-1)*100, entry_b=True, entry_jf=True)]
    return frame, pd.DataFrame(rows)


def test_r25_actual_open_exit_observes_same_close_stops_before_next_entry_and_keeps_real_chain():
    from gcn.backtest.signal_research_r25 import audit_symbol
    frame, trades = _trades(); dates = frame.index.strftime("%Y-%m-%d")
    episodes, observations = audit_symbol("TEST", frame, trades, source_trusted=True)
    assert len(episodes) == 1
    row = episodes.iloc[0]
    assert observations.date.tolist() == dates[3:5].tolist()
    assert observations.flat_bar.tolist() == [1, 2]
    assert observations.pending_setup_date.iloc[0] == dates[0]
    assert observations.resolved_setup_status.iloc[1] == "confirmed"
    assert row.flat_bars == 2 and row.end_kind == "reentry" and row.source_trusted
    assert row.entry_date == dates[1] and row.exit_date == dates[3]
    assert row.reference_price == 100/.999**2
    assert observations.reference_reclaimed.tolist() == [False, True]
    assert pd.isna(observations.first_reference_reclaim_date.iloc[0])
    assert row.first_reference_reclaim_date == observations.first_reference_reclaim_date.iloc[1] == dates[4]
    assert row.next_entry_date == dates[5] and row.next_signal_date == dates[4] and row.next_entry_kind == "B"
    assert row.next_entry_b and row.next_entry_jf and row.next_B_CRASH_RECOVER
    assert row.next_setup_date == dates[0]  # Setup began before the exit; it was never reset.
    assert row.next_return_pct == trades.return_pct.iloc[2] and row.original_return_pct == trades.return_pct.iloc[0]
    assert row.chain_same_original_end
    assert np.isclose(row.chain_return_pct, ((1+trades.return_pct.iloc[1]/100)*(1+trades.return_pct.iloc[2]/100)-1)*100)
    assert row.chain_return_pct < row.original_return_pct
    assert row.raw_buy_rows == row.tradable_buy_rows == 1  # B/JF collision is one date, not two entries.
    assert row.original_horizon_bars == 5  # Retrospective original horizon can extend beyond actual reentry.
    assert not any("return" in col or col.startswith("next_") or col.startswith("original_") for col in observations)


def test_r25_real_episode_prefixes_censor_future_reentry_outcomes_and_keep_strict_reference_boundary():
    from gcn.backtest.signal_research_r25 import audit_symbol
    frame, trades = _trades(); dates = frame.index.strftime("%Y-%m-%d")
    full, observations = audit_symbol("TEST", frame, trades)
    for cut in frame.index:
        early, observed = audit_symbol("TEST", frame.loc[:cut], trades)
        date = cut.date().isoformat()
        pd.testing.assert_frame_equal(observed, observations[observations.date.le(date)], check_exact=True)
        if len(early):
            row = early.iloc[0]
            assert row.end_kind == ("right_censored" if date < dates[5] else "reentry")
            for col in ("first_reference_reclaim_date", "next_entry_date", "next_signal_date", "next_exit_date", "original_exit_date"):
                assert pd.isna(row[col]) or row[col] <= date
            if date < dates[8]:
                assert pd.isna(row.next_return_pct) and pd.isna(row.chain_return_pct) and pd.isna(row.original_return_pct)
            if date < dates[5]:
                assert pd.isna(row.next_entry_kind) and pd.isna(row.next_setup_date) and not row.next_entry_b
    altered = trades.copy(); altered.loc[altered.exit_reason.eq("trail"), "return_pct"] = 999.
    pd.testing.assert_frame_equal(audit_symbol("TEST", frame, altered)[1], observations, check_exact=True)
    without_next = trades.iloc[:2]
    censored, observed = audit_symbol("TEST", frame, without_next)
    assert observed.date.tolist() == dates[3:].tolist()
    assert censored.end_kind.iloc[0] == "right_censored" and pd.isna(censored.next_entry_date.iloc[0])
    assert observed.first_reference_reclaim_date.iloc[1:].eq(dates[4]).all()
    reference = 100/.999**2
    for value, hit in ((reference, True), (np.nextafter(reference, 0.), False)):
        small = frame.iloc[:4].copy(); small.iloc[3, small.columns.get_loc("CLOSE")] = value
        row, observed = audit_symbol("TEST", small, trades)
        assert observed.reference_reclaimed.iloc[0] == hit
        assert pd.notna(row.first_reference_reclaim_date.iloc[0]) == hit
    empty, empty_obs = audit_symbol("TEST", frame.iloc[:0], trades)
    pd.testing.assert_frame_equal(empty, full.iloc[:0], check_exact=True)
    pd.testing.assert_frame_equal(empty_obs, observations.iloc[:0], check_exact=True)
    no_joint = trades.copy(); no_joint.exit_reason = "trail"
    pd.testing.assert_frame_equal(audit_symbol("TEST", frame, no_joint)[0], full.iloc[:0], check_exact=True)


def test_r25_rejects_non_jf_exit_sources_ambiguous_pairing_overlaps_and_wrong_next_source():
    import pytest
    from gcn.backtest.signal_research_r25 import audit_symbol
    frame, trades = _trades()
    wrong = trades.copy(); wrong.loc[1, "entry_b"] = True
    with pytest.raises(ValueError, match="纯JF"):
        audit_symbol("TEST", frame, wrong)
    wrong_frame = frame.copy(); wrong_frame.loc[frame.index[0], "ICON_JUEFAN"] = False
    with pytest.raises(ValueError, match="纯JF"):
        audit_symbol("TEST", wrong_frame, trades)
    with pytest.raises(ValueError, match="重复"):
        audit_symbol("TEST", frame, pd.concat([trades, trades.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError, match="配对"):
        audit_symbol("TEST", frame, trades.iloc[1:])
    wrong = trades.copy(); wrong.loc[0, "exit_date"] = wrong.loc[1, "exit_date"]
    with pytest.raises(ValueError, match="提前"):
        audit_symbol("TEST", frame, wrong)
    wrong = trades.copy(); wrong.loc[2, "entry_date"] = wrong.loc[1, "exit_date"]
    with pytest.raises(ValueError, match="重叠"):
        audit_symbol("TEST", frame, wrong)
    wrong = trades.copy(); wrong.loc[2, "entry_b"] = False
    with pytest.raises(ValueError, match="下一实际入场"):
        audit_symbol("TEST", frame, wrong)


def test_r25_fixed_episode_strata_keep_overlap_zero_symbols_censoring_and_aligned_chain_denominators():
    from gcn.backtest.historical_research import CORE
    from gcn.backtest.signal_research_r25 import audit_symbol, summarize_episodes
    frame, trades = _trades()
    episodes, observations = audit_symbol("TEST", frame, trades, source_trusted=True)
    summary = summarize_episodes(episodes)
    all_rows = summary[summary.group_by.eq("all")].iloc[0]
    assert all_rows.episodes == all_rows.reference_reclaims == all_rows.has_reentry == 1
    assert all_rows.flat_bars == 2 and all_rows.original_horizon_bars == 5
    assert all_rows.raw_buy_rows == all_rows.tradable_buy_rows == all_rows.raw_jf_rows == 1
    assert all_rows.raw_b_rows == 0 and all_rows.confirmed_b_rows == all_rows.tradable_jf_rows == 1
    assert all_rows.next_completed == all_rows.aligned_chains == all_rows.worse_aligned_chains == 1
    assert all_rows.next_wins == all_rows.original_wins == 0
    assert summary[summary.group_by.eq("symbol") & summary.group.isin(CORE)].episodes.eq(0).all()
    assert set(CORE) <= set(summary[summary.group_by.eq("symbol")].group)
    assert summary[summary.group_by.eq("next_source") & summary.group.eq("B_CRASH_RECOVER")].episodes.iloc[0] == 1
    assert summary[summary.group_by.eq("source_trusted") & summary.group.eq("false")].episodes.iloc[0] == 0
    early, _ = audit_symbol("TEST", frame.iloc[:4], trades)
    row = summarize_episodes(early).iloc[0]
    assert row.episodes == row.right_censored == row.reference_censored == 1
    assert row.next_completed == row.aligned_chains == row.original_wins == 0
    empty = summarize_episodes(episodes.iloc[:0])
    assert empty.episodes.eq(0).all() and empty.next_completed.eq(0).all()


def test_r25_training_archive_binds_both_original_windows_parent_prices_sources_and_actual_flat_rows(tmp_path, monkeypatch):
    import hashlib
    import json
    from gcn.backtest import engine, historical_research
    from gcn.backtest.signal_research_r25 import run_diagnostic, R24_MANIFESTS
    def forbidden(*args, **kwargs):
        raise AssertionError("r25 must not simulate any orders")
    monkeypatch.setattr(engine, "_one_strategy", forbidden)
    monkeypatch.setattr(historical_research, "_one_strategy", forbidden)
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    prior = ROOT / "reports/gcn-historical-r24-20260905"
    decision = run_diagnostic(snapshot, prior, tmp_path, window="training")
    assert decision["stage"] == "diagnostic_only" and decision["recommended"] == "v5"
    assert not decision["production_changed"] and decision["window"] == ["training", "2021-08-27", "2024-08-26"]
    read = lambda name: pd.read_csv(tmp_path / name, float_precision="round_trip")
    episodes, observations, checks = read("episodes.csv"), read("observations.csv"), read("reconciliation.csv")
    original = pd.read_csv(prior / "results/trades.csv", float_precision="round_trip")
    expected = original[original.rule.eq("JF-joint-pressure") & original.exit_reason.eq("joint_pressure")]
    assert len(episodes) == 9 and len(checks) == 10 and checks.reconciled.all()
    pd.testing.assert_frame_equal(episodes[["symbol", "entry_date", "exit_date", "return_pct"]].reset_index(drop=True),
                                  expected[["symbol", "entry_date", "exit_date", "return_pct"]].reset_index(drop=True), check_exact=True)
    assert observations.date.max() <= "2024-08-26"
    for row in episodes.itertuples():
        obs = observations[observations.episode_id.eq(row.episode_id)]
        assert len(obs) == row.flat_bars and obs.date.iloc[0] == row.exit_date
        if pd.notna(row.next_entry_date):
            assert obs.date.lt(row.next_entry_date).all()
        assert obs.first_reference_reclaim_date.dropna().le(obs.date[obs.first_reference_reclaim_date.notna()]).all()
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    frozen = ROOT / "reports/gcn-historical-r25-20260905/training"
    for name in manifest["outputs"]:
        assert (tmp_path / name).read_bytes() == (frozen / name).read_bytes(), name
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert manifest["input_manifest_sha256"] == R24_MANIFESTS
    assert manifest["parent_manifest_sha256"] == digest(snapshot / "manifest.json")
    for group, prefix in (("input_files", "input_snapshot"), ("input_algorithm_sources", "input_source_snapshot"),
                          ("parent_files", "parent_snapshot"), ("algorithm_sources", "source_snapshot")):
        for name, expected_hash in manifest[group].items():
            assert digest(tmp_path / prefix / name) == expected_hash
            if group == "algorithm_sources":
                assert digest(ROOT / name) == expected_hash
    assert len(manifest["input_files"]) == 14 and len(manifest["input_algorithm_sources"]) == 25
    assert len(manifest["parent_files"]) == 23
    for name, expected_hash in manifest["outputs"].items():
        assert digest(tmp_path / name) == expected_hash
    import pytest
    with pytest.raises(FileExistsError):
        run_diagnostic(snapshot, prior, tmp_path, window="training")


def test_r25_archive_rejects_changed_either_window_parent_sources_and_midrun_changes_before_output(tmp_path, monkeypatch):
    import json
    import shutil
    import pytest
    from gcn.backtest import signal_research_r25 as research
    snapshot = tmp_path / "parent"; prior = tmp_path / "r24"; output = tmp_path / "output"
    original_snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    snapshot.mkdir()
    shutil.copyfile(original_snapshot / "manifest.json", snapshot / "manifest.json")
    shutil.copytree(original_snapshot / "input_snapshot", snapshot / "input_snapshot")
    shutil.copytree(ROOT / "reports/gcn-historical-r24-20260905", prior)
    for relative, message in (("results/trades.csv", "输入内容"), ("validation/trades.csv", "输入内容"),
                              ("results/manifest.json", "冻结manifest"),
                              ("validation/source_snapshot/gcn/backtest/engine.py", "冻结源码")):
        path = prior / relative; raw = path.read_bytes(); path.write_bytes(raw+b"\n")
        with pytest.raises(ValueError, match=message):
            research.run_diagnostic(snapshot, prior, output)
        assert not output.exists(); path.write_bytes(raw)
    path = snapshot / "input_snapshot/TQQQ_1d.csv"; raw = path.read_bytes(); path.write_bytes(raw+b"\n")
    with pytest.raises(ValueError, match="父输入快照"):
        research.run_diagnostic(snapshot, prior, output)
    assert not output.exists(); path.write_bytes(raw)
    with pytest.raises(ValueError, match="固定"):
        research.run_diagnostic(snapshot, prior, output, window="full")
    saved = research.audit_symbol
    changed = prior / "validation/comparisons.csv"; before = changed.read_bytes()
    mutated = False
    def mutate(*args, **kwargs):
        nonlocal mutated
        if not mutated:
            changed.write_bytes(before+b"\n"); mutated = True
        return saved(*args, **kwargs)
    with monkeypatch.context() as context:
        context.setattr(research, "audit_symbol", mutate)
        with pytest.raises(ValueError, match="计算期间"):
            research.run_diagnostic(snapshot, prior, output)
    assert not output.exists(); changed.write_bytes(before)
    original_read = Path.read_bytes
    for target, expected_error in ((ROOT / "gcn/backtest/signal_research_r25.py", "计算期间"),
                                   (ROOT / "reports/gcn-historical-r25-20260905/protocol.md", "计算期间")):
        reads = 0
        def altered(path):
            nonlocal reads
            value = original_read(path)
            if path == target:
                reads += 1
                if reads > 1:
                    return value+b"\n"
            return value
        with monkeypatch.context() as context:
            context.setattr(Path, "read_bytes", altered)
            with pytest.raises(ValueError, match=expected_error):
                research.run_diagnostic(snapshot, prior, output)
        assert not output.exists()
    output.mkdir(); marker = output / "keep.txt"; marker.write_text("keep")
    with pytest.raises(FileExistsError):
        research.run_diagnostic(snapshot, prior, output)
    assert marker.read_text() == "keep" and len(list(output.iterdir())) == 1


@pytest.mark.parametrize("stage,end", [("results", "2024-08-26"), ("validation", "2025-08-26")])
def test_r25_real_price_prefixes_preserve_observations_and_only_then_known_reentry_sources(stage, end):
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.backtest.signal_research_r25 import audit_symbol
    from gcn.recipes.gcn_main import compute_ehopt10
    frames, quality = load_snapshot(ROOT / "reports/signal-audit-v5-review-20260904")
    trades = pd.read_csv(ROOT / "reports/gcn-historical-r24-20260905" / stage / "trades.csv", float_precision="round_trip")
    checked = 0
    for symbol in CORE:
        raw = frames[symbol].loc[:end]
        def audit(cut):
            frame = compute_ehopt10(raw.loc[:cut], version="v5", diagnostics=True)
            return audit_symbol(symbol, frame, trades, source_trusted=quality[symbol])
        episodes, observations = audit(raw.index[-1])
        cuts = {raw.index[0], raw.index[-1]}
        for row in episodes.itertuples():
            for date in (row.exit_date, row.first_reference_reclaim_date, row.next_entry_date):
                if pd.notna(date):
                    pos = raw.index.get_loc(pd.Timestamp(date))
                    cuts.update(raw.index[max(0, pos-1):pos+1])
            obs = observations[observations.episode_id.eq(row.episode_id)]
            for mask in (obs.raw_b_suppressed, obs.pending_setup_date.notna(), obs.resolved_setup_status.notna()):
                if mask.any():
                    cuts.add(pd.Timestamp(obs[mask].date.iloc[0]))
        for cut in sorted(cuts):
            early, observed = audit(cut); date = cut.date().isoformat()
            pd.testing.assert_frame_equal(observed, observations[observations.date.le(date)], check_exact=True)
            for col in ("first_reference_reclaim_date", "next_entry_date", "next_signal_date", "next_exit_date", "original_exit_date"):
                assert early[col].dropna().le(date).all()
            for row in early.itertuples():
                expected = episodes[episodes.episode_id.eq(row.episode_id)].iloc[0]
                if pd.notna(row.next_entry_date):
                    for col in ("next_entry_date", "next_signal_date", "next_entry_kind", "next_entry_b", "next_entry_jf"):
                        assert getattr(row, col) == expected[col]
                else:
                    assert pd.isna(row.next_return_pct) and not row.next_entry_b and not row.next_entry_jf
                if pd.isna(row.next_exit_date):
                    assert pd.isna(row.next_return_pct) and pd.isna(row.chain_return_pct)
            checked += 1
    assert checked >= 40


def test_r25_known_validation_keeps_all_four_exits_censored_nvda_and_actual_tsla_chain(tmp_path):
    import hashlib
    import json
    from gcn.backtest.signal_research_r25 import run_diagnostic
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    prior = ROOT / "reports/gcn-historical-r24-20260905"
    before = (ROOT / "reports/gcn-historical-r25-20260905/training/manifest.json").read_bytes()
    decision = run_diagnostic(snapshot, prior, tmp_path, window="validation")
    assert decision["window"] == ["validation", "2024-08-27", "2025-08-26"]
    assert decision["stage"] == "diagnostic_only" and not decision["production_changed"]
    e = pd.read_csv(tmp_path / "episodes.csv", float_precision="round_trip")
    o = pd.read_csv(tmp_path / "observations.csv", float_precision="round_trip")
    original = pd.read_csv(prior / "validation/trades.csv", float_precision="round_trip")
    expected = original[original.rule.eq("JF-joint-pressure") & original.exit_reason.eq("joint_pressure")]
    columns = ["symbol", "entry_date", "exit_date", "return_pct"]
    pd.testing.assert_frame_equal(e[columns].reset_index(drop=True), expected[columns].reset_index(drop=True), check_exact=True)
    assert len(e) == 4 and pd.read_csv(tmp_path / "reconciliation.csv").reconciled.all()
    nvda = e[e.symbol.eq("NVDA")].iloc[0]
    assert nvda.end_kind == "right_censored" and pd.isna(nvda.next_entry_date) and pd.isna(nvda.next_return_pct)
    assert nvda.first_reference_reclaim_date == nvda.exit_date == "2025-04-24"
    assert nvda.raw_b_rows == nvda.suppressed_b_rows == 2 and nvda.setup_rows == nvda.tradable_buy_rows == 0
    assert o[o.symbol.eq("NVDA")].date.max() == "2025-08-26"
    tsla = e[e.symbol.eq("TSLA")].iloc[0]
    assert tsla.next_entry_date == "2025-03-26" and tsla.next_B_BEAR_RECOVER and not tsla.next_B_CRASH_RECOVER
    assert tsla.chain_same_original_end and tsla.chain_return_pct < tsla.original_return_pct
    assert o.first_reference_reclaim_date.dropna().le(o.date[o.first_reference_reclaim_date.notna()]).all()
    assert (ROOT / "reports/gcn-historical-r25-20260905/training/manifest.json").read_bytes() == before
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    for name, expected_hash in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected_hash
        assert (tmp_path / name).read_bytes() == (ROOT / "reports/gcn-historical-r25-20260905/validation" / name).read_bytes()
