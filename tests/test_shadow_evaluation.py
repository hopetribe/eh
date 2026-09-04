# -*- coding: utf-8 -*-
"""v6 正式评估器的纯函数合成测试。"""
import copy
import json
import math
from pathlib import Path

import numpy as np

from gcn.backtest.shadow_evaluation import (
    calculate_evaluation,
    cross_symbol_robustness,
    derive_challenger_cohorts,
    evaluate_promotion_gates,
    fixed_order_artifact_sha256,
    formal_evaluate,
    leave_one_out_robustness,
    metrics_from_symbol_daily_returns,
    nonoverlapping_downside,
    paired_two_axis_bootstrap,
    path_metrics,
    replay_fixed_orders,
    safe_ratio,
    symbol_daily_returns,
)
from gcn.backtest.shadow_validation import canonical_spec_hash


_SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "gcn"
    / "backtest"
    / "shadow_specs"
    / "v6-profit-arm20-keep50-20260905.json"
)


def _spec(*, replications=200, block_sessions=20):
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    spec = copy.deepcopy(spec)
    spec["evaluation"]["bootstrap"]["replications"] = replications
    spec["evaluation"]["bootstrap"]["time_block_sessions"] = block_sessions
    spec["evaluation"]["downside"]["block_sessions"] = block_sessions
    return spec


def test_symbol_daily_returns_use_one_as_each_symbols_initial_equity():
    equity = np.array([
        [1.10, 1.21],
        [0.90, 0.81],
    ])

    actual = symbol_daily_returns(equity)

    np.testing.assert_allclose(actual, [[0.10, 0.10], [-0.10, -0.10]])


def test_path_metrics_follow_frozen_equal_weight_formulas_exactly():
    # 两个标的每日收益都为 -10%, +20%；组合等权收益也相同。
    equity = np.array([
        [0.90, 1.08],
        [0.90, 1.08],
    ])
    held = np.array([
        [True, True],
        [True, False],
    ])

    actual = path_metrics(equity, held, annual_sessions=252)
    daily = np.array([-0.10, 0.20])
    expected_log = 252 / 2 * np.log1p(daily).sum()

    np.testing.assert_allclose(actual["portfolio_daily_returns"], daily)
    assert math.isclose(actual["annualized_log_return"], expected_log)
    assert math.isclose(actual["cagr"], math.exp(expected_log) - 1)
    assert math.isclose(
        actual["sharpe"],
        math.sqrt(252) * daily.mean() / daily.std(ddof=1),
    )
    # 回撤序列必须包含初始净值 1，所以首日即产生 10% 回撤。
    assert math.isclose(actual["mdd"], 0.10)
    assert math.isclose(actual["exposure"], 3 / 4)
    np.testing.assert_allclose(actual["symbol_total_returns"], [0.08, 0.08])


def test_path_metrics_return_null_sharpe_for_zero_sample_volatility():
    equity = np.ones((2, 4))
    held = np.zeros_like(equity, dtype=bool)

    actual = path_metrics(equity, held)

    assert actual["sharpe"] is None


def test_replay_fixed_orders_revalues_identical_fills_at_requested_cost():
    opens = np.array([[100.0, 110.0, 120.0]])
    closes = np.array([[105.0, 115.0, 130.0]])
    entries = np.array([[True, False, False]])
    exits = np.array([[False, False, True]])

    base = replay_fixed_orders(opens, closes, entries, exits, cost_bps=10)
    stress = replay_fixed_orders(opens, closes, entries, exits, cost_bps=25)

    np.testing.assert_allclose(
        base["equity"],
        [[0.999 / 100 * 105, 0.999 / 100 * 115, 0.999 ** 2 * 1.2]],
    )
    np.testing.assert_array_equal(base["held_at_close"], [[True, True, False]])
    assert math.isclose(stress["equity"][0, -1], 0.9975 ** 2 * 1.2)
    np.testing.assert_array_equal(stress["entry_fills"], entries)
    np.testing.assert_array_equal(stress["exit_fills"], exits)


def test_replay_fixed_orders_marks_terminal_position_without_exit_cost():
    opens = np.array([[100.0, 100.0]])
    closes = np.array([[100.0, 130.0]])
    entries = np.array([[True, False]])
    exits = np.array([[False, False]])

    stress = replay_fixed_orders(opens, closes, entries, exits, cost_bps=25)

    assert math.isclose(stress["equity"][0, -1], 0.9975 * 1.3)
    assert stress["held_at_close"][0, -1]


def test_replay_fixed_orders_rejects_impossible_fill_sequence():
    prices = np.ones((1, 3)) * 100
    entries = np.array([[True, True, False]])
    exits = np.zeros((1, 3), dtype=bool)

    try:
        replay_fixed_orders(prices, prices, entries, exits, cost_bps=10)
    except ValueError as error:
        assert "已持仓" in str(error)
    else:
        raise AssertionError("已持仓时不能再次执行入场成交")


def test_nonoverlapping_downside_is_anchored_and_excludes_partial_tail():
    daily = np.concatenate([
        np.full(20, 0.01),
        np.full(20, -0.01),
        np.full(5, -0.50),
    ])

    actual = nonoverlapping_downside(daily, block_sessions=20)

    expected_returns = np.array([1.01 ** 20 - 1, 0.99 ** 20 - 1])
    np.testing.assert_allclose(actual["block_returns"], expected_returns)
    np.testing.assert_allclose(actual["losses"], np.maximum(-expected_returns, 0))
    assert actual["full_blocks"] == 2
    assert actual["excluded_sessions"] == 5


def test_safe_ratio_returns_null_for_zero_or_nonfinite_denominator():
    assert safe_ratio(3.0, 2.0) == 1.5
    assert safe_ratio(1.0, 0.0) is None
    assert safe_ratio(1.0, float("nan")) is None


def _naive_paired_bootstrap(
    incumbent,
    challenger,
    *,
    annual_sessions,
    replications,
    seed,
    time_block_sessions,
    downside_block_sessions,
    symbol_resample_count,
):
    symbol_count, session_count = incumbent.shape
    blocks_per_replication = math.ceil(session_count / time_block_sessions)
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = rng.integers(
        0,
        session_count - time_block_sessions + 1,
        size=(replications, blocks_per_replication),
    )
    symbols = rng.integers(
        0,
        symbol_count,
        size=(replications, symbol_resample_count),
    )
    annualized = []
    downside = []
    for replication in range(replications):
        time_positions = np.concatenate([
            np.arange(start, start + time_block_sessions)
            for start in starts[replication]
        ])[:session_count]
        symbol_positions = symbols[replication]
        incumbent_portfolio = incumbent[
            symbol_positions[:, None], time_positions[None, :]
        ].mean(axis=0)
        challenger_portfolio = challenger[
            symbol_positions[:, None], time_positions[None, :]
        ].mean(axis=0)
        annualized.append(
            annual_sessions / session_count
            * (
                np.log1p(challenger_portfolio).sum()
                - np.log1p(incumbent_portfolio).sum()
            )
        )
        incumbent_loss = nonoverlapping_downside(
            incumbent_portfolio, block_sessions=downside_block_sessions,
        )["mean_loss"]
        challenger_loss = nonoverlapping_downside(
            challenger_portfolio, block_sessions=downside_block_sessions,
        )["mean_loss"]
        downside.append(incumbent_loss - challenger_loss)
    return np.array(annualized), np.array(downside)


def test_paired_bootstrap_matches_literal_pcg64_two_axis_algorithm():
    incumbent = np.array([
        np.linspace(-0.010, 0.012, 25),
        np.linspace(0.008, -0.009, 25),
        np.sin(np.arange(25)) * 0.004 - 0.001,
    ])
    challenger = incumbent + np.array([[0.0010], [0.0005], [0.0015]])
    arguments = {
        "annual_sessions": 252,
        "replications": 7,
        "seed": 123,
        "time_block_sessions": 4,
        "downside_block_sessions": 5,
        "symbol_resample_count": 3,
    }

    actual = paired_two_axis_bootstrap(incumbent, challenger, **arguments)
    expected_log, expected_downside = _naive_paired_bootstrap(
        incumbent, challenger, **arguments,
    )

    np.testing.assert_allclose(
        actual["annualized_log_return_delta_samples"], expected_log,
    )
    np.testing.assert_allclose(
        actual["downside_improvement_samples"], expected_downside,
    )
    assert math.isclose(
        actual["annualized_log_return_delta_q05"],
        np.quantile(expected_log, 0.05, method="linear"),
    )
    assert math.isclose(
        actual["downside_improvement_q05"],
        np.quantile(expected_downside, 0.05, method="linear"),
    )


def test_paired_bootstrap_is_reproducible_and_batch_size_independent():
    incumbent = np.tile(np.linspace(-0.02, 0.02, 41), (4, 1))
    challenger = incumbent + np.arange(1, 5)[:, None] * 0.0001
    arguments = {
        "annual_sessions": 252,
        "replications": 31,
        "seed": 20260905,
        "time_block_sessions": 7,
        "downside_block_sessions": 5,
        "symbol_resample_count": 4,
    }

    one = paired_two_axis_bootstrap(incumbent, challenger, batch_size=1, **arguments)
    seven = paired_two_axis_bootstrap(incumbent, challenger, batch_size=7, **arguments)

    np.testing.assert_array_equal(
        one["annualized_log_return_delta_samples"],
        seven["annualized_log_return_delta_samples"],
    )
    np.testing.assert_array_equal(
        one["downside_improvement_samples"],
        seven["downside_improvement_samples"],
    )


def test_paired_bootstrap_rejects_circular_or_too_short_implicit_paths():
    returns = np.zeros((2, 3))

    try:
        paired_two_axis_bootstrap(
            returns,
            returns,
            annual_sessions=252,
            replications=5,
            seed=1,
            time_block_sessions=4,
            downside_block_sessions=2,
            symbol_resample_count=2,
        )
    except ValueError as error:
        assert "time_block_sessions" in str(error)
    else:
        raise AssertionError("普通非循环MBB不得绕回样本起点")


def test_cross_symbol_robustness_uses_final_mark_equity_and_positive_sum():
    incumbent = np.ones((10, 3))
    deltas = np.array([0.10, 0.09, 0.08, 0.07, 0.06, 0.05,
                       -0.01, -0.02, -0.03, -0.04])
    challenger = incumbent.copy()
    challenger[:, -1] += deltas
    symbols = [f"S{position}" for position in range(10)]

    actual = cross_symbol_robustness(incumbent, challenger, symbols=symbols)

    np.testing.assert_allclose(actual["total_return_deltas"], deltas)
    assert math.isclose(actual["median_total_return_delta"], 0.055)
    assert actual["positive_symbols"] == 6
    assert math.isclose(actual["positive_contribution"], 0.10 / 0.45)
    assert math.isclose(actual["by_symbol"]["S0"], deltas[0])


def test_cross_symbol_robustness_has_null_contribution_without_positive_delta():
    incumbent = np.ones((2, 2))
    challenger = np.array([[1.0, 0.9], [1.0, 1.0]])

    actual = cross_symbol_robustness(incumbent, challenger)

    assert actual["positive_symbols"] == 0
    assert actual["positive_contribution"] is None


def test_leave_one_out_drops_exactly_one_and_equal_weights_the_other_symbols():
    session_count = 40
    incumbent = np.tile(
        np.where(np.arange(session_count) % 2 == 0, -0.004, 0.003),
        (10, 1),
    )
    challenger = incumbent + np.arange(1, 11)[:, None] * 0.0001
    symbols = [f"S{position}" for position in range(10)]
    gates = {
        "annualized_log_return_delta_min_bps": -500,
        "mdd_delta_max_bps": 0,
        "sharpe_delta_min_milli": -100,
    }

    actual = leave_one_out_robustness(
        incumbent,
        challenger,
        symbols=symbols,
        annual_sessions=252,
        gates=gates,
    )

    assert len(actual["rows"]) == 10
    assert actual["passes"] == 10
    first = actual["rows"][0]
    assert first["dropped_symbol"] == "S0"
    expected_incumbent = metrics_from_symbol_daily_returns(
        incumbent[1:], annual_sessions=252,
    )
    expected_challenger = metrics_from_symbol_daily_returns(
        challenger[1:], annual_sessions=252,
    )
    assert math.isclose(
        first["annualized_log_return_delta"],
        expected_challenger["annualized_log_return"]
        - expected_incumbent["annualized_log_return"],
    )
    assert math.isclose(
        first["mdd_delta"],
        expected_challenger["mdd"] - expected_incumbent["mdd"],
    )
    assert math.isclose(
        first["sharpe_delta"],
        expected_challenger["sharpe"] - expected_incumbent["sharpe"],
    )
    assert first["passed"]


def test_leave_one_out_treats_null_sharpe_as_gate_failure():
    unchanged = np.zeros((3, 20))
    gates = {
        "annualized_log_return_delta_min_bps": -500,
        "mdd_delta_max_bps": 0,
        "sharpe_delta_min_milli": -100,
    }

    actual = leave_one_out_robustness(
        unchanged,
        unchanged,
        symbols=["A", "B", "C"],
        annual_sessions=252,
        gates=gates,
    )

    assert actual["passes"] == 0
    assert all(row["sharpe_delta"] is None for row in actual["rows"])
    assert not any(row["passed"] for row in actual["rows"])


def _passing_evidence():
    return {
        "base": {
            "annualized_log_return_delta_q05": -0.049999,
            "point_cagr_delta": -0.03,
            "downside_improvement_q05": 0.000001,
            "downside_loss_ratio": 0.90,
            "mdd_delta": 0.0,
            "sharpe_delta": -0.10,
            "entry_count_ratio": 0.75,
            "exposure_ratio": 0.75,
        },
        "stress": {
            "annualized_log_return_delta": -0.05,
            "mdd_delta": 0.0,
            "sharpe_delta": -0.10,
        },
        "cross_symbol": {
            "median_total_return_delta": 0.000001,
            "positive_symbols": 6,
            "positive_contribution": 0.35,
        },
        "leave_one_out": {"passes": 9},
    }


def test_all_promotion_gates_obey_strict_gt_and_inclusive_min_max():
    actual = evaluate_promotion_gates(_spec()["evaluation"]["gates"], _passing_evidence())

    assert actual["all"]
    assert all(actual["base"].values())
    assert all(actual["stress"].values())
    assert all(actual["cross_symbol"].values())
    assert all(actual["leave_one_out"].values())


def test_gt_gates_fail_at_exact_boundary_but_min_max_still_pass():
    gate_spec = _spec()["evaluation"]["gates"]
    cases = [
        ("base", "annualized_log_return_delta_q05", -0.05),
        ("base", "downside_improvement_q05", 0.0),
        ("cross_symbol", "median_total_return_delta", 0.0),
    ]

    for section, field, value in cases:
        evidence = _passing_evidence()
        evidence[section][field] = value
        actual = evaluate_promotion_gates(gate_spec, evidence)
        assert not actual[section][field]
        assert not actual["all"]


def test_null_nonfinite_and_zero_denominator_evidence_are_gate_failures():
    gate_spec = _spec()["evaluation"]["gates"]
    cases = [
        ("base", "entry_count_ratio", None),
        ("base", "exposure_ratio", None),
        ("base", "downside_loss_ratio", None),
        ("base", "sharpe_delta", float("nan")),
        ("stress", "mdd_delta", float("inf")),
        ("cross_symbol", "positive_contribution", None),
        ("leave_one_out", "passes", None),
    ]

    for section, field, value in cases:
        evidence = _passing_evidence()
        evidence[section][field] = value
        actual = evaluate_promotion_gates(gate_spec, evidence)
        assert not actual[section][field]
        assert not actual["all"]


def test_promotion_is_all_sections_and_not_any_section():
    evidence = _passing_evidence()
    evidence["base"]["mdd_delta"] = 0.00001

    actual = evaluate_promotion_gates(_spec()["evaluation"]["gates"], evidence)

    assert all(actual["stress"].values())
    assert all(actual["cross_symbol"].values())
    assert all(actual["leave_one_out"].values())
    assert not actual["base"]["mdd_delta"]
    assert not actual["all"]


def _cyclical_fixed_order_case():
    symbol_count = 10
    session_count = 80
    opens = np.empty(session_count)
    closes = np.empty(session_count)
    previous_close = 100.0
    for session in range(session_count):
        opens[session] = previous_close
        daily_return = -0.05 if session % 4 == 3 else 0.005
        closes[session] = opens[session] * (1 + daily_return)
        previous_close = closes[session]
    opens = np.tile(opens, (symbol_count, 1))
    closes = np.tile(closes, (symbol_count, 1))

    incumbent_entries = np.zeros((symbol_count, session_count), dtype=bool)
    incumbent_exits = np.zeros_like(incumbent_entries)
    incumbent_entries[:, 0] = True

    challenger_entries = np.zeros_like(incumbent_entries)
    challenger_exits = np.zeros_like(incumbent_entries)
    challenger_entries[:, np.arange(0, session_count, 4)] = True
    challenger_exits[:, np.arange(3, session_count, 4)] = True
    return (
        opens,
        closes,
        incumbent_entries,
        incumbent_exits,
        challenger_entries,
        challenger_exits,
    )


def test_low_level_calculation_integrates_all_gates_without_formal_state():
    spec = _spec(replications=200)
    inputs = _cyclical_fixed_order_case()

    actual = calculate_evaluation(spec, *inputs)

    assert actual["calculation_status"] == "METRIC_GATES_EVALUATED"
    assert actual["eligible_by_metrics"]
    assert "state" not in actual
    assert "decision" not in actual
    assert "formal_evaluation_consumed" not in actual
    assert actual["gates"]["all"]
    assert actual["order_replay"] == {
        "base_cost_bps_per_side": 10,
        "stress_cost_bps_per_side": 25,
        "stress_reuses_base_orders": True,
        "incumbent_entry_fills": 10,
        "incumbent_exit_fills": 0,
        "challenger_entry_fills": 200,
        "challenger_exit_fills": 200,
    }
    assert actual["metrics"]["bootstrap"]["replications"] == 200
    assert actual["evidence"]["base"]["downside_improvement_q05"] > 0


def test_formal_evaluator_uses_unsampled_point_downside_ratio():
    spec = _spec(replications=31)
    inputs = _cyclical_fixed_order_case()
    opens, closes, inc_entries, inc_exits, chal_entries, chal_exits = inputs
    incumbent = replay_fixed_orders(
        opens, closes, inc_entries, inc_exits, cost_bps=10,
    )
    challenger = replay_fixed_orders(
        opens, closes, chal_entries, chal_exits, cost_bps=10,
    )
    incumbent_metrics = path_metrics(
        incumbent["equity"], incumbent["held_at_close"],
    )
    challenger_metrics = path_metrics(
        challenger["equity"], challenger["held_at_close"],
    )
    incumbent_loss = nonoverlapping_downside(
        incumbent_metrics["portfolio_daily_returns"], block_sessions=20,
    )["mean_loss"]
    challenger_loss = nonoverlapping_downside(
        challenger_metrics["portfolio_daily_returns"], block_sessions=20,
    )["mean_loss"]

    actual = calculate_evaluation(spec, *inputs)

    assert math.isclose(
        actual["evidence"]["base"]["downside_loss_ratio"],
        challenger_loss / incumbent_loss,
    )
    assert actual["metrics"]["downside"]["point_estimate"] == (
        "unsampled_original_series"
    )


def test_formal_evaluator_replays_stress_cost_without_terminal_exit_fee():
    spec = _spec(replications=31)
    inputs = _cyclical_fixed_order_case()
    expected_stress = replay_fixed_orders(
        inputs[0], inputs[1], inputs[2], inputs[3], cost_bps=25,
    )

    actual = calculate_evaluation(spec, *inputs)

    expected_totals = expected_stress["equity"][:, -1] - 1
    np.testing.assert_allclose(
        actual["metrics"]["stress"]["incumbent"]["symbol_total_returns"],
        expected_totals,
    )
    # 在职老策略末日仅盯市，因此只有入场时的 25bps。
    assert math.isclose(
        expected_stress["equity"][0, -1],
        0.9975 * inputs[1][0, -1] / inputs[0][0, 0],
    )


def test_zero_incumbent_downside_is_not_mature_and_does_not_consume_once():
    spec = _spec(replications=31)
    prices = np.full((10, 40), 100.0)
    empty = np.zeros_like(prices, dtype=bool)

    actual = calculate_evaluation(spec, prices, prices, empty, empty, empty, empty)

    assert actual["calculation_status"] == "NOT_MATURE_ZERO_INCUMBENT_DOWNSIDE"
    assert not actual["eligible_by_metrics"]
    assert "state" not in actual
    assert "formal_evaluation_consumed" not in actual
    assert actual["metrics"]["bootstrap"] is None
    assert actual["evidence"]["base"]["downside_loss_ratio"] is None
    assert not actual["gates"]["all"]


def test_formal_evaluator_rejects_protocol_that_recalculates_stress_path():
    spec = _spec(replications=31)
    spec["evaluation"]["stress_policy"]["path_recalculation"] = True

    try:
        calculate_evaluation(spec, *_cyclical_fixed_order_case())
    except ValueError as error:
        assert "path_recalculation" in str(error)
    else:
        raise AssertionError("压力情景不得重算触发路径")


def _registered_formal_case(*, locked_end=None, next_common_session=None):
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    start = spec["boundaries"]["expected_accrual_start"]
    locked_end = locked_end or spec["boundaries"]["expected_minimum_accrual_end"]
    next_common_session = next_common_session or "2029-09-17"
    raw_dates = np.arange(
        np.datetime64(start),
        np.datetime64(locked_end) + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    dates = tuple(str(value) for value in raw_dates[np.is_busday(raw_dates)])
    assert dates[0] == start
    assert dates[-1] == locked_end
    symbol_count = len(spec["universe"]["core_symbols"])
    session_count = len(dates)

    opens = np.empty(session_count)
    closes = np.empty(session_count)
    previous_close = 100.0
    for session in range(session_count):
        opens[session] = previous_close
        daily_return = -0.10 if session % 4 in (0, 3) else 0.10
        closes[session] = opens[session] * (1 + daily_return)
        previous_close = closes[session]
    opens = np.tile(opens, (symbol_count, 1))
    closes = np.tile(closes, (symbol_count, 1))

    incumbent_entries = np.zeros((symbol_count, session_count), dtype=bool)
    incumbent_exits = np.zeros_like(incumbent_entries)
    incumbent_entry_positions = np.arange(1, session_count, 160)
    incumbent_exit_positions = incumbent_entry_positions[1:] - 1
    incumbent_entries[:, incumbent_entry_positions] = True
    incumbent_exits[:, incumbent_exit_positions] = True

    challenger_entries = np.zeros_like(incumbent_entries)
    challenger_exits = np.zeros_like(incumbent_entries)
    challenger_entries[:, np.arange(1, session_count, 4)] = True
    challenger_exits[:, np.arange(4, session_count, 4)] = True
    challenger_exit_reasons = np.full(
        challenger_exits.shape, "", dtype=object,
    )
    challenger_exit_reasons[challenger_exits] = "profit_lock"
    inputs = (
        opens,
        closes,
        incumbent_entries,
        incumbent_exits,
        challenger_entries,
        challenger_exits,
        challenger_exit_reasons,
    )
    symbols = tuple(spec["universe"]["core_symbols"])
    source_hashes = {"frozen-algorithm-manifest": "a" * 64}
    accepted_bar_hashes = {
        symbol: f"{position + 1:064x}"
        for position, symbol in enumerate(symbols)
    }
    incumbent_path = replay_fixed_orders(
        opens, closes, incumbent_entries, incumbent_exits, cost_bps=10,
    )
    incumbent_path_metrics = path_metrics(
        incumbent_path["equity"], incumbent_path["held_at_close"],
    )
    incumbent_blocks = nonoverlapping_downside(
        incumbent_path_metrics["portfolio_daily_returns"], block_sessions=20,
    )["block_returns"]
    cohort_evidence = derive_challenger_cohorts(
        spec,
        opens,
        closes,
        incumbent_exits,
        challenger_entries,
        challenger_exits,
        challenger_exit_reasons,
        symbols=symbols,
        session_dates=dates,
    )
    context = {
        "state": "READY_ONCE",
        "spec_hash": canonical_spec_hash(spec),
        "actual_accrual_start": start,
        "locked_end": locked_end,
        "performance_end": locked_end,
        "locked_months": 36,
        "maturity_36_passed": True,
        "maturity_summary": {
            "incumbent_reference_entries": int(incumbent_entries.sum()),
            "challenger_armed_cohorts": cohort_evidence[
                "challenger_armed_cohorts"
            ],
            "challenger_armed_symbols": cohort_evidence[
                "challenger_armed_symbols"
            ],
            "incumbent_active_symbols": 10,
            "incumbent_negative_20_session_blocks": int(
                np.count_nonzero(incumbent_blocks < 0)
            ),
            "affected_exits": cohort_evidence["affected_exits"],
            "affected_symbols": cohort_evidence["affected_symbols"],
        },
        "post_lock_common_sessions": 60,
        "pending_20_session_labels": 0,
        "pending_60_session_labels": 0,
        "formal_evaluation_count": 0,
        "source_hashes": source_hashes,
        "accepted_bar_hashes": accepted_bar_hashes,
        "order_artifact_sha256": "0" * 64,
    }
    context["next_common_session_after_locked_end"] = next_common_session
    context["order_artifact_sha256"] = fixed_order_artifact_sha256(
        *inputs,
        spec_hash=context["spec_hash"],
        symbols=symbols,
        session_dates=dates,
        locked_end=locked_end,
        source_hashes=source_hashes,
        accepted_bar_hashes=accepted_bar_hashes,
    )
    return spec, inputs, symbols, dates, context


def test_controlled_formal_entry_accepts_last_common_session_before_expected_endpoint():
    spec, inputs, symbols, dates, context = _registered_formal_case(
        locked_end="2029-09-13",
        next_common_session="2029-09-17",
    )

    actual = formal_evaluate(
        spec,
        *inputs,
        symbols=symbols,
        session_dates=dates,
        readiness_context=context,
    )

    assert actual["provenance"]["locked_end"] == "2029-09-13"
    assert actual["provenance"]["next_common_session_after_locked_end"] == (
        "2029-09-17"
    )


def test_controlled_formal_entry_validates_ready_snapshot_and_returns_cas_intent():
    spec, inputs, symbols, dates, context = _registered_formal_case()

    actual = formal_evaluate(
        spec,
        *inputs,
        symbols=symbols,
        session_dates=dates,
        readiness_context=context,
    )

    assert actual["state"] == "ELIGIBLE_FOR_V6_IMPLEMENTATION"
    assert actual["decision"] == spec["decision"]["promotion_result"]
    assert actual["eligible"]
    assert actual["formal_evaluation_consumed"]
    assert actual["cas_transition"] == {
        "expected_state": "READY_ONCE",
        "expected_formal_evaluation_count": 0,
        "next_formal_evaluation_count": 1,
        "next_state": "ELIGIBLE_FOR_V6_IMPLEMENTATION",
    }
    assert actual["provenance"]["spec_hash"] == canonical_spec_hash(spec)
    assert actual["provenance"]["locked_end"] == dates[-1]
    assert actual["provenance"]["order_artifact_sha256"] == (
        context["order_artifact_sha256"]
    )
    assert actual["metrics"]["bootstrap"]["replications"] == 20_000


def test_controlled_formal_entry_rejects_shortened_bootstrap_spec():
    spec, inputs, symbols, dates, context = _registered_formal_case()
    spec["evaluation"]["bootstrap"]["replications"] = 31

    try:
        formal_evaluate(
            spec,
            *inputs,
            symbols=symbols,
            session_dates=dates,
            readiness_context=context,
        )
    except ValueError as error:
        assert "注册哈希" in str(error)
    else:
        raise AssertionError("缩水bootstrap协议不得产生正式决策")


def test_controlled_formal_entry_requires_ready_once_and_unused_count():
    spec, inputs, symbols, dates, context = _registered_formal_case()
    cases = [
        ("state", "OUTCOME_EMBARGO_60", "READY_ONCE"),
        ("formal_evaluation_count", 1, "次数已消耗"),
        ("performance_end", "2030-09-13", "performance_end"),
    ]

    for field, value, message in cases:
        changed = copy.deepcopy(context)
        changed[field] = value
        try:
            formal_evaluate(
                spec,
                *inputs,
                symbols=symbols,
                session_dates=dates,
                readiness_context=changed,
            )
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"readiness_context.{field}必须阻断正式评估")


def test_controlled_formal_entry_binds_frozen_symbol_order():
    spec, inputs, symbols, dates, context = _registered_formal_case()
    swapped = list(symbols)
    swapped[0], swapped[-1] = swapped[-1], swapped[0]

    try:
        formal_evaluate(
            spec,
            *inputs,
            symbols=swapped,
            session_dates=dates,
            readiness_context=context,
        )
    except ValueError as error:
        assert "symbols" in str(error)
    else:
        raise AssertionError("核心标的行序不得静默互换")


def test_controlled_formal_entry_rejects_order_artifact_tampering():
    spec, inputs, symbols, dates, context = _registered_formal_case()
    changed = copy.deepcopy(context)
    changed["order_artifact_sha256"] = "f" * 64

    try:
        formal_evaluate(
            spec,
            *inputs,
            symbols=symbols,
            session_dates=dates,
            readiness_context=changed,
        )
    except ValueError as error:
        assert "order_artifact_sha256" in str(error)
    else:
        raise AssertionError("固定订单工件篡改必须阻断正式评估")


def test_controlled_formal_entry_detects_price_or_fill_path_tampering():
    spec, inputs, symbols, dates, context = _registered_formal_case()
    changed_inputs = list(inputs)
    changed_inputs[1] = changed_inputs[1].copy()
    changed_inputs[1][0, 10] *= 1.001

    try:
        formal_evaluate(
            spec,
            *changed_inputs,
            symbols=symbols,
            session_dates=dates,
            readiness_context=context,
        )
    except ValueError as error:
        assert "order_artifact_sha256" in str(error)
    else:
        raise AssertionError("锁窗价格篡改必须改变订单工件哈希")


def test_controlled_formal_entry_binds_maturity_counts_to_order_path():
    spec, inputs, symbols, dates, context = _registered_formal_case()
    cases = [
        ("incumbent_reference_entries", 1, "参考入场计数"),
        ("incumbent_active_symbols", -1, "活跃标的数"),
        ("incumbent_negative_20_session_blocks", 1, "负20日块计数"),
    ]

    for field, delta, message in cases:
        changed = copy.deepcopy(context)
        changed["maturity_summary"][field] += delta
        try:
            formal_evaluate(
                spec,
                *inputs,
                symbols=symbols,
                session_dates=dates,
                readiness_context=changed,
            )
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"maturity_summary.{field}必须与锁窗路径一致")


def test_controlled_formal_entry_rejects_endpoint_one_day_early_or_late():
    spec, inputs, symbols, dates, context = _registered_formal_case()
    cases = []
    early_context = copy.deepcopy(context)
    early_context["locked_end"] = dates[-2]
    early_context["performance_end"] = dates[-2]
    early_context["next_common_session_after_locked_end"] = dates[-1]
    cases.append((tuple(value[:, :-1] for value in inputs), dates[:-1], early_context))
    late_dates = (*dates[:-1], "2029-09-15")
    late_context = copy.deepcopy(context)
    late_context["locked_end"] = late_dates[-1]
    late_context["performance_end"] = late_dates[-1]
    cases.append((inputs, late_dates, late_context))

    for changed_inputs, changed_dates, changed_context in cases:
        try:
            formal_evaluate(
                spec,
                *changed_inputs,
                symbols=symbols,
                session_dates=changed_dates,
                readiness_context=changed_context,
            )
        except ValueError as error:
            assert "冻结端点" in str(error)
        else:
            raise AssertionError("正式锁窗不得提前或推迟一日")


def test_controlled_formal_entry_derives_armed_and_affected_counts():
    spec, inputs, symbols, dates, context = _registered_formal_case()
    cases = [
        ("challenger_armed_symbols", 9),
        ("affected_symbols", 7),
    ]

    for field, fake_value in cases:
        changed = copy.deepcopy(context)
        changed["maturity_summary"][field] = fake_value
        try:
            formal_evaluate(
                spec,
                *inputs,
                symbols=symbols,
                session_dates=dates,
                readiness_context=changed,
            )
        except ValueError as error:
            assert field in str(error)
            assert "cohort" in str(error)
        else:
            raise AssertionError(f"{field}不得由调用方自报")


def test_profit_lock_reason_must_match_arm_and_floor_path():
    spec = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    symbols = tuple(spec["universe"]["core_symbols"])
    prices = np.full((10, 4), 100.0)
    incumbent_exits = np.zeros((10, 4), dtype=bool)
    challenger_entries = np.zeros((10, 4), dtype=bool)
    challenger_exits = np.zeros((10, 4), dtype=bool)
    challenger_entries[:, 1] = True
    challenger_exits[:, 3] = True
    reasons = np.full((10, 4), "", dtype=object)
    reasons[:, 3] = "profit_lock"

    try:
        derive_challenger_cohorts(
            spec,
            prices,
            prices,
            incumbent_exits,
            challenger_entries,
            challenger_exits,
            reasons,
            symbols=symbols,
            session_dates=("2029-09-11", "2029-09-12", "2029-09-13", "2029-09-14"),
        )
    except ValueError as error:
        assert "武装阈值" in str(error)
    else:
        raise AssertionError("普通退出不得伪造为profit_lock")


def test_controlled_formal_entry_rejects_first_session_open_fill():
    spec, inputs, symbols, dates, context = _registered_formal_case()
    changed_inputs = list(inputs)
    changed_inputs[2] = changed_inputs[2].copy()
    changed_inputs[2][:, 0] = True

    try:
        formal_evaluate(
            spec,
            *changed_inputs,
            symbols=symbols,
            session_dates=dates,
            readiness_context=context,
        )
    except ValueError as error:
        assert "首列" in str(error)
    else:
        raise AssertionError("累积起点收盘前不得存在成交")
