# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from gcn.backtest.signal_audit import (
    Candidate, _confirm_sell, _non_overlapping_positions, _portfolio_stats,
    choose_recommendation, selection_score,
)


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
