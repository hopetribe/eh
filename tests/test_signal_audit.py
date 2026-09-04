# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from gcn.backtest.signal_audit import (
    Candidate, DEFAULT_AUDIT_SYMBOLS, _audit_window_line, _confirm_sell,
    _data_coverage, _manifest_run_id, _non_overlapping_positions,
    _partition_selection_universe,
    _snapshot_run_materials,
    _portfolio_stats, _turn_coverage,
    audit_data, candidate_grid, missed_turn_table, signal_event_table,
    choose_incremental_recommendation, choose_recommendation, selection_score,
)
from gcn.backtest.engine import DEFAULT_SYMBOLS


def test_audit_default_symbols_match_the_production_watchlist():
    assert DEFAULT_AUDIT_SYMBOLS == DEFAULT_SYMBOLS


def test_data_audit_marks_a_symbol_that_ends_before_the_requested_window(tmp_path):
    pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "open": [10.0, 11.0], "high": [12.0, 12.0],
        "low": [9.0, 10.0], "close": [11.0, 11.5],
        "volume": [100.0, 120.0],
    }).to_csv(tmp_path / "AAA_1d.csv", index=False)

    _, rows = audit_data(
        tmp_path, ("AAA",), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03")
    )

    assert rows[0]["status"] == "stale-end"


def test_data_coverage_finds_internal_calendar_gap_without_partial_history_drag():
    idx = pd.date_range("2025-01-01", periods=4, freq="D")
    frames = {
        "AAA": pd.DataFrame(index=idx),
        "BBB": pd.DataFrame(index=idx.delete(2)),
        "IPO": pd.DataFrame(index=idx[2:]),
    }
    rows = [
        {"symbol": "AAA", "status": "ok", "metadata_hash": "match"},
        {"symbol": "BBB", "status": "ok", "metadata_hash": "match"},
        {"symbol": "IPO", "status": "partial-history", "metadata_hash": "match"},
    ]

    coverage = _data_coverage(frames, rows, idx[0], idx[-1])

    assert coverage["reference_symbols"] == ["AAA", "BBB"]
    assert coverage["common_complete_end"] == "2025-01-02"
    assert coverage["calendar_gaps"] == {"BBB": ["2025-01-03"]}
    assert coverage["partial_history_symbols"] == ["IPO"]
    assert coverage["requested_end_complete"] is False


def test_data_coverage_does_not_call_a_common_stale_end_complete():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    requested_end = idx[-1] + pd.Timedelta(days=1)
    frames = {
        "AAA": pd.DataFrame(index=idx),
        "BBB": pd.DataFrame(index=idx),
    }
    rows = [
        {"symbol": "AAA", "status": "stale-end", "metadata_hash": "match"},
        {"symbol": "BBB", "status": "stale-end", "metadata_hash": "no-meta"},
    ]

    coverage = _data_coverage(frames, rows, idx[0], requested_end)

    assert coverage["calendar_gaps"] == {}
    assert coverage["requested_end_complete"] is False
    assert coverage["metadata_untrusted_symbols"] == ["BBB"]
    assert rows[0]["complete_window"] is False


def test_partial_history_symbols_are_external_validation_not_selection_inputs():
    prepared = {"AAA": object(), "BBB": object(), "IPO": object()}
    coverage = {"reference_symbols": ["AAA", "BBB"]}

    selection, external = _partition_selection_universe(prepared, coverage)

    assert list(selection) == ["AAA", "BBB"]
    assert list(external) == ["IPO"]


def test_selection_pool_requires_a_complete_window_and_excludes_stale_inputs():
    prepared = {"AAA": object(), "STALE": object(), "IPO": object()}
    coverage = {
        "reference_symbols": ["AAA", "STALE"],
        "complete_window_symbols": ["AAA"],
        "partial_history_symbols": ["IPO"],
    }

    selection, external = _partition_selection_universe(prepared, coverage)

    assert list(selection) == ["AAA"]
    assert list(external) == ["IPO"]


def test_run_materials_are_snapshotted_before_audit(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "report"
    data_dir.mkdir()
    csv = data_dir / "AAA_1d.csv"
    metadata = data_dir / "AAA_1d.csv.meta.json"
    csv.write_bytes(b"date,open,high,low,close,volume\n")
    metadata.write_bytes(b"{}")

    snapshot_dir, source_hashes = _snapshot_run_materials(
        data_dir, output_dir, ("AAA", "MISSING")
    )

    assert (snapshot_dir / csv.name).read_bytes() == csv.read_bytes()
    assert (snapshot_dir / metadata.name).read_bytes() == metadata.read_bytes()
    assert (output_dir / "source_snapshot/gcn/backtest/engine.py").is_file()
    assert "gcn/backtest/engine.py" in source_hashes


def test_data_audit_counts_infinite_ohlcv_as_invalid(tmp_path):
    pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "open": [10.0, 11.0], "high": [12.0, np.inf],
        "low": [9.0, 10.0], "close": [11.0, 11.5],
        "volume": [100.0, 120.0],
    }).to_csv(tmp_path / "AAA_1d.csv", index=False)

    frames, rows = audit_data(
        tmp_path, ("AAA",), pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
    )
    coverage = _data_coverage(
        frames, rows, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")
    )

    assert rows[0]["bad_ohlcv"] == 1
    assert rows[0]["complete_window"] is False
    assert coverage["invalid_quality_symbols"] == ["AAA"]


def test_manifest_run_id_is_stable_and_changes_with_inputs_config_or_environment():
    code = {"b.py": "bbb", "a.py": "aaa"}
    config = {"cost": 0.001, "symbols": ["AAA"]}
    coverage = {"common_complete_end": "2025-01-02"}
    inputs = {"AAA": {"sha256": "data-a"}}

    environment = {"python": "3.11.14", "numpy": "2.4.4", "pandas": "3.0.2"}
    first = _manifest_run_id(code, config, coverage, inputs, environment)
    reordered = _manifest_run_id(
        {"a.py": "aaa", "b.py": "bbb"}, config, coverage, inputs, environment
    )

    assert first == reordered
    assert first != _manifest_run_id(
        code, {**config, "cost": 0.0025}, coverage, inputs, environment
    )
    assert first != _manifest_run_id(
        code, config, coverage, {"AAA": {"sha256": "data-b"}}, environment
    )
    assert first != _manifest_run_id(
        code, config, coverage, inputs, {**environment, "numpy": "2.5.0"}
    )


def test_audit_report_window_line_uses_the_requested_cutoff():
    splits = {"full": (pd.Timestamp("2021-08-27"), pd.Timestamp("2026-08-27"))}

    line = _audit_window_line(splits)

    assert "2026-08-27" in line
    assert "2026-08-28" not in line


def test_signal_events_separate_v4_setup_from_v5_confirmation_and_censor_at_end():
    idx = pd.date_range("2025-01-01", periods=35, freq="D")
    frame = pd.DataFrame({
        "OPEN": 100.0, "HIGH": 101.0, "LOW": 99.0, "CLOSE": 100.0,
        "B_SIGNAL": False, "ICON_JUEFAN": False, "S_SIGNAL": False,
        "B_CONDITION": False, "B_STAGE_SIGNAL": False,
        "B_SCORE": 0.0, "S_SCORE": 0.0,
    }, index=idx)
    frame.loc[[idx[0], idx[10]], "B_SIGNAL"] = True
    frame.loc[idx[2], "ICON_JUEFAN"] = True
    frame.loc[idx[3], "S_SIGNAL"] = True
    confirmed = pd.Series(False, index=idx)
    confirmed.iloc[1] = True
    prepared = {"AAA": {
        "v4": frame,
        "entries": {"b-confirm5-ma20": confirmed},
    }}

    events = signal_event_table(prepared, idx[0], idx[25])

    assert set(events["signal"]) == {"B买Setup(v4)", "B确认(v5)", "绝反", "S卖"}
    assert set(events["signal_role"]) == {"setup", "entry", "exit"}
    late_setup = events[
        (events["signal"] == "B买Setup(v4)") & (events["date"] == "2025-01-11")
    ].iloc[0]
    assert np.isnan(late_setup["ret_20d_pct"])
    assert not late_setup["outcome_complete_20d"]
    assert pd.isna(late_setup["interference"])


def test_missed_turns_mark_only_tradable_position_states_as_actionable():
    idx = pd.date_range("2025-01-01", periods=70, freq="D")
    frame = pd.DataFrame({
        "OPEN": 100.0, "HIGH": 101.0, "LOW": 99.0, "CLOSE": 100.0,
        "B_SIGNAL": False, "ICON_JUEFAN": False, "S_SIGNAL": False,
    }, index=idx)
    frame.loc[idx[20], "LOW"] = 80.0
    frame.loc[idx[30], "HIGH"] = 125.0
    frame.loc[idx[40], "LOW"] = 80.0
    entry = pd.Series(False, index=idx)
    exit_ = pd.Series(False, index=idx)
    entry.iloc[5] = True
    exit_.iloc[35] = True
    prepared = {"AAA": {
        "v4": frame,
        "entries": {"incumbent": entry},
        "exits": {"S": exit_},
    }}

    turns = missed_turn_table(
        prepared, idx[0], idx[-1], candidate=Candidate("incumbent", "S", None, None)
    )

    buy_while_held = turns[(turns["kind"] == "buy") & (turns["date"] == idx[10].date().isoformat())]
    sell_while_held = turns[(turns["kind"] == "sell") & (turns["date"] == idx[30].date().isoformat())]
    assert buy_while_held["actionable"].tolist() == [False]
    assert sell_while_held["actionable"].tolist() == [True]


def test_missed_turn_positions_apply_the_candidate_initial_hard_stop():
    idx = pd.date_range("2025-01-01", periods=70, freq="D")
    frame = pd.DataFrame({
        "OPEN": 100.0, "HIGH": 101.0, "LOW": 99.0, "CLOSE": 100.0,
        "B_SIGNAL": False, "ICON_JUEFAN": False, "S_SIGNAL": False,
    }, index=idx)
    frame.loc[idx[6], "CLOSE"] = 80.0
    frame.loc[idx[20], "LOW"] = 80.0
    frame.loc[idx[30], "HIGH"] = 125.0
    entry = pd.Series(False, index=idx)
    exit_ = pd.Series(False, index=idx)
    entry.iloc[5] = True
    prepared = {"AAA": {
        "v4": frame,
        "entries": {"incumbent": entry},
        "exits": {"S": exit_},
    }}

    turns = missed_turn_table(
        prepared, idx[0], idx[-1],
        candidate=Candidate("incumbent", "S", None, None, hard_stop=0.15),
    )

    buy_after_stop = turns[(turns["kind"] == "buy") & (turns["date"] == idx[10].date().isoformat())]
    assert buy_after_stop["actionable"].tolist() == [True]


def test_missed_turn_sell_coverage_includes_actual_trailing_stop_decision():
    idx = pd.date_range("2025-01-01", periods=70, freq="D")
    frame = pd.DataFrame({
        "OPEN": 100.0, "HIGH": 101.0, "LOW": 99.0, "CLOSE": 100.0,
        "B_SIGNAL": False, "ICON_JUEFAN": False, "S_SIGNAL": False,
    }, index=idx)
    frame.loc[idx[7], "CLOSE"] = 130.0
    frame.loc[[idx[8], idx[9]], "CLOSE"] = 110.0
    frame.loc[idx[10], ["HIGH", "CLOSE"]] = [150.0, 100.0]
    frame.loc[idx[20], "LOW"] = 60.0
    entry = pd.Series(False, index=idx)
    entry.iloc[5] = True
    no_exit = pd.Series(False, index=idx)
    prepared = {"AAA": {
        "v4": frame,
        "entries": {"incumbent": entry},
        "exits": {"S": no_exit},
    }}

    turns = missed_turn_table(
        prepared, idx[0], idx[-1],
        candidate=Candidate("incumbent", "S", 0.20, None),
    )

    sell_turn = turns[(turns["kind"] == "sell")
                      & (turns["date"] == idx[10].date().isoformat())].iloc[0]
    assert sell_turn["actionable"]
    assert sell_turn["covered"]
    assert sell_turn["nearest_signal_date"] == idx[10].date().isoformat()


def test_missed_turn_labels_do_not_read_prices_after_the_audit_cutoff():
    idx = pd.date_range("2025-01-01", periods=80, freq="D")
    frame = pd.DataFrame({
        "OPEN": 100.0, "HIGH": 101.0, "LOW": 99.0, "CLOSE": 100.0,
        "B_SIGNAL": False, "ICON_JUEFAN": False, "S_SIGNAL": False,
    }, index=idx)
    frame.loc[idx[10], "LOW"] = 80.0
    frame.loc[idx[20], "HIGH"] = 130.0
    frame.loc[idx[25], "LOW"] = 70.0
    frame.loc[idx[35], "LOW"] = 80.0
    frame.loc[idx[55], "HIGH"] = 140.0
    cutoff = idx[54]

    full = missed_turn_table({"AAA": {"v4": frame}}, idx[0], cutoff)
    truncated = missed_turn_table(
        {"AAA": {"v4": frame.loc[:cutoff].copy()}}, idx[0], cutoff
    )

    pd.testing.assert_frame_equal(
        full.reset_index(drop=True), truncated.reset_index(drop=True)
    )


def test_turn_coverage_can_restrict_counts_to_actionable_events():
    turns = pd.DataFrame({
        "covered": [True, False, False],
        "actionable": [True, True, False],
    })

    summary = _turn_coverage(turns, actionable_only=True)

    assert summary == {"total": 2, "covered": 1, "missed": 1, "rate": 50.0}


def test_non_overlapping_positions_keep_strongest_event_in_cluster():
    strength = pd.Series([0.1, 0.5, 0.3, 0.0, 0.8, 0.1])
    kept = _non_overlapping_positions(np.array([0, 1, 2, 4]), strength, gap=2)
    assert kept == [1, 4]


def test_portfolio_stats_include_initial_period_return_and_drawdown():
    stats = _portfolio_stats(pd.Series([-0.10, 0.05, 0.02]))
    assert stats["total"] < 0
    assert np.isclose(stats["mdd"], 10.0)
    assert set(stats) == {"total", "cagr", "mdd", "sharpe", "calmar"}


def test_selection_score_ignores_test_metrics_and_rejects_tiny_samples():
    baseline = {"trades": 20}
    train = {"trades": 12, "sharpe": 1.0, "calmar": 1.0,
             "cagr": 8.0, "median_symbol_total": 10.0, "mdd": 20.0,
             "positive_symbols": 6, "symbols": 10}
    validation = {"trades": 6, "sharpe": 0.8, "calmar": 0.7,
                  "cagr": 10.0, "median_symbol_total": 8.0, "mdd": 18.0,
                  "positive_symbols": 6, "symbols": 10}
    assert selection_score(train, validation, baseline) > -1_000
    assert selection_score({**train, "trades": 2}, validation, baseline) == -1_000
    assert selection_score(train, {**validation, "positive_symbols": 4}, baseline) == -1_000


def test_candidate_name_is_stable_and_explicit():
    candidate = Candidate("v4-b+jf", "S", 0.15, 60)
    assert candidate.name == "v4-b+jf|exit=S|trail=15%|hold=60"


def test_candidate_name_appends_initial_hard_stop_when_present():
    candidate = Candidate("v5", "S", 0.20, None, hard_stop=0.125)

    assert candidate.name == "v5|exit=S|trail=20%|hold=none|hard=12.5%"


def test_candidate_grid_adds_only_the_targeted_initial_hard_stop_neighborhood():
    candidates = candidate_grid()
    hard_stop_candidates = [candidate for candidate in candidates if candidate.hard_stop]

    assert {candidate.hard_stop for candidate in hard_stop_candidates} == {0.125, 0.15}
    assert all(candidate.entry == "b-confirm5-ma20+jf" for candidate in hard_stop_candidates)
    assert all(candidate.exit == "S" and candidate.trail == 0.20 for candidate in hard_stop_candidates)
    assert len({candidate.name for candidate in candidates}) == len(candidates)


def test_sell_confirmation_is_causal_and_expires():
    idx = pd.date_range("2025-01-01", periods=9, freq="D")
    frame = pd.DataFrame({
        "LOW": [10.0] * 9,
        "CLOSE": [11.0, 10.5, 9.8, 9.5, 11.0, 10.8, 10.7, 10.6, 9.0],
    }, index=idx)
    setup = pd.Series([True, False, False, False, True, False, False, False, False],
                      index=idx)
    confirmed = _confirm_sell(setup, frame, ma_days=2, window=3)
    assert confirmed.tolist() == [False, False, True, False, False, False, False, False, False]


def test_recommendation_requires_return_retention_and_holdout_risk_improvement():
    baseline = {
        "name": "v4-b+jf|exit=S|trail=none|hold=none", "selection_score": 0.1,
        "full_cagr": 20.0, "full_mdd": 40.0, "test_cagr": 30.0,
        "test_sharpe": 1.0, "test_mdd": 12.0, "max_hold": np.nan,
        "trail": np.nan, "exit": "S",
    }
    too_defensive = {
        **baseline, "name": "defensive", "selection_score": 2.0,
        "full_cagr": 10.0, "full_mdd": 5.0, "test_cagr": 10.0,
        "test_sharpe": 1.5, "test_mdd": 4.0,
    }
    accepted = {
        **baseline, "name": "accepted", "selection_score": 1.0,
        "full_cagr": 16.0, "full_mdd": 18.0, "test_cagr": 16.0,
        "test_sharpe": 1.1, "test_mdd": 10.0, "trail": 0.2,
    }
    name, gates = choose_recommendation(pd.DataFrame([baseline, too_defensive, accepted]))
    assert name == "accepted"
    assert gates["eligible_candidates"] == 1


def test_recommendation_prefers_equal_evidence_without_an_extra_hard_stop():
    baseline = {
        "name": Candidate("v4-b+jf", "S", None, None).name, "selection_score": 0.1,
        "full_cagr": 20.0, "full_mdd": 40.0, "test_cagr": 30.0,
        "test_sharpe": 1.0, "test_mdd": 12.0, "max_hold": np.nan,
        "trail": np.nan, "hard_stop": np.nan, "exit": "S",
    }
    common = {
        **baseline, "selection_score": 1.0, "full_cagr": 16.0,
        "full_mdd": 18.0, "test_cagr": 16.0, "test_sharpe": 1.1,
        "test_mdd": 10.0, "trail": 0.20,
    }
    hard = {**common, "name": "hard", "hard_stop": 0.125}
    simple = {**common, "name": "simple", "hard_stop": np.nan}

    name, _ = choose_recommendation(pd.DataFrame([baseline, hard, simple]))

    assert name == "simple"


def test_incremental_review_keeps_v5_when_challenger_only_wins_known_full_sample():
    incumbent_name = Candidate("b-confirm5-ma20+jf", "S", 0.20, None).name
    incumbent = {
        "name": incumbent_name, "selection_score": 1.0,
        "train_cagr": 9.0, "train_mdd": 15.0, "train_sharpe": 0.7,
        "train_positive_symbols": 7, "train_symbols": 10, "train_trades": 50,
        "train_median_symbol_total": 8.0, "train_worst_trade": -20.0,
        "validation_cagr": 27.0, "validation_mdd": 11.0,
        "validation_sharpe": 1.4, "validation_positive_symbols": 8,
        "validation_symbols": 10, "validation_trades": 20,
        "validation_median_symbol_total": 12.0,
        "validation_worst_trade": -18.0,
        "full_cagr": 21.0, "full_mdd": 15.0, "full_sharpe": 1.3,
        "test_cagr": 35.0, "test_mdd": 9.0, "test_sharpe": 2.4,
        "max_hold": np.nan, "trail": 0.20, "hard_stop": np.nan, "exit": "S",
    }
    challenger = {
        **incumbent, "name": f"{incumbent_name}|hard=15%", "selection_score": 1.2,
        "train_cagr": 11.0, "train_mdd": 14.0, "train_sharpe": 0.9,
        "validation_cagr": 26.0, "validation_mdd": 12.0,
        "validation_sharpe": 1.3,
        "full_cagr": 24.0, "full_mdd": 13.0, "full_sharpe": 1.5,
        "test_cagr": 50.0, "test_mdd": 6.0, "test_sharpe": 3.0,
        "hard_stop": 0.15,
    }

    name, gates = choose_incremental_recommendation(
        pd.DataFrame([challenger, incumbent]), incumbent_name, [challenger["name"]]
    )

    assert name == incumbent_name
    assert gates["eligible_challengers"] == 0
    assert gates["known_test_used_for_promotion"] is False


def test_incremental_review_promotes_only_when_train_and_validation_both_pass():
    incumbent_name = Candidate("b-confirm5-ma20+jf", "S", 0.20, None).name

    def split(prefix, cagr, mdd, sharpe, worst):
        return {
            f"{prefix}_cagr": cagr, f"{prefix}_mdd": mdd,
            f"{prefix}_sharpe": sharpe, f"{prefix}_worst_trade": worst,
            f"{prefix}_trades": 20, f"{prefix}_symbols": 10,
            f"{prefix}_positive_symbols": 7,
            f"{prefix}_median_symbol_total": 8.0,
        }

    incumbent = {
        "name": incumbent_name, "selection_score": 1.0,
        "max_hold": np.nan, "trail": 0.20, "hard_stop": np.nan, "exit": "S",
        **split("train", 10.0, 15.0, 1.0, -20.0),
        **split("validation", 12.0, 12.0, 1.1, -18.0),
    }
    challenger_name = f"{incumbent_name}|hard=15%"
    challenger = {
        **incumbent, "name": challenger_name, "hard_stop": 0.15,
        **split("train", 10.0, 13.5, 1.1, -18.0),
        **split("validation", 12.0, 10.5, 1.2, -16.0),
    }

    name, gates = choose_incremental_recommendation(
        pd.DataFrame([incumbent, challenger]), incumbent_name, [challenger_name]
    )

    assert name == challenger_name
    assert gates["train"]["passed"] is True
    assert gates["validation"]["passed"] is True
    assert gates["promoted"] is True
