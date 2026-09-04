# -*- coding: utf-8 -*-
"""回测引擎不变量与成交口径测试。"""
import copy
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from gcn.backtest.engine import (
    PRESETS, _buy_hold, _exposure, _one_strategy, _perf, presets_for_version,
    event_study, slice_years, run_backtest,
)
from gcn.data.sample import make_sample_data
from gcn.recipes.gcn_main import compute_ehopt10


def _res():
    return compute_ehopt10(make_sample_data(600))


def test_slice_years():
    res = _res()
    s1 = slice_years(res, 1)
    assert s1.index[0] >= res.index[-1] - pd.DateOffset(years=1)
    assert len(s1) < len(res)
    assert slice_years(res, None) is res
    half = slice_years(res, 0.5)
    assert half.index[0] >= res.index[-1] - pd.Timedelta(days=365.2425 / 2)

    undated = res.reset_index(drop=True)
    assert len(slice_years(undated, 1, interval="1d")) == min(252, len(undated))


def test_run_backtest_shape_and_consistency():
    res = _res()
    rep = run_backtest(res, cost=0.001)
    assert set(rep) >= {"events", "strategies", "equity"}
    assert len(rep["equity"][list(rep["equity"])[0]]) == len(res)
    for s in rep["strategies"]:
        assert set(s) >= {"name", "total", "cagr", "mdd", "sharpe", "trades", "exposure"}
    bh = next(s for s in rep["strategies"] if s["name"].startswith("基准"))
    assert bh["trades"] == 0 and bh["exposure"] == 1.0
    # 事件研究结构
    ev = rep["events"]
    assert any(e["signal"] == "B_SIGNAL" for e in ev)
    assert any(e["signal"] == "_BASE" for e in ev)


def test_experimental_stage_strategy_is_only_available_for_experimental_version():
    stable = run_backtest(compute_ehopt10(make_sample_data(600), version="v4"))
    experiment = run_backtest(compute_ehopt10(make_sample_data(600), version="v4-exp"))

    stable_names = {row["name"] for row in stable["strategies"]}
    experiment_names = {row["name"] for row in experiment["strategies"]}
    assert "阶段Setup → S卖" not in stable_names
    assert "阶段确认 → S卖" not in stable_names
    assert {"阶段Setup → S卖", "阶段确认 → S卖"} <= experiment_names
    assert any(row["signal"] == "B_STAGE_ENTRY_SIGNAL" for row in experiment["events"])


def test_v5_presets_include_the_validated_trailing_stop_strategy():
    assert presets_for_version("v4") is PRESETS
    recommended = presets_for_version("v5")[0]
    assert recommended == {
        "name": "v5推荐: B确认+绝反 → S卖 + 20%止损",
        "entry": ["B_SIGNAL", "ICON_JUEFAN"],
        "exit": ["S_SIGNAL"],
        "trail": 0.20,
    }


def test_preset_columns_missing_skips():
    res = _res()
    presets = [{"name": "不存在列", "entry": ["NOPE_COL"], "exit": ["B_SIGNAL"]},
               {"name": "B买 → S卖", "entry": ["B_SIGNAL"], "exit": ["S_SIGNAL"]}]
    rep = run_backtest(res, cost=0.001, presets=presets)
    assert [s["name"] for s in rep["strategies"]] == ["B买 → S卖", "基准: 买入持有"]


def _trade_frame(n=5):
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "OPEN": np.arange(10.0, 10.0 + n),
        "CLOSE": np.arange(10.5, 10.5 + n),
        "ENTRY": False,
        "EXIT": False,
    }, index=idx)


def test_strategy_terminal_close_cost_and_exposure():
    res = _trade_frame(4)
    res.loc[res.index[0], "ENTRY"] = True
    bt = _one_strategy(res, ["ENTRY"], ["EXIT"], cost=0.01, max_hold=None)
    assert len(bt["trades"]) == 1
    trade = bt["trades"][0]
    assert trade["exit_reason"] == "terminal"
    assert trade["hold"] == 3
    expected = (0.99 / 11.0) * 13.5 * 0.99
    assert np.isclose(bt["equity"][-1], expected)
    assert np.isclose(_exposure(bt, res), 3 / 4)


def test_mark_terminal_policy_keeps_open_position_unsettled():
    res = _trade_frame(4)
    res.loc[res.index[0], "ENTRY"] = True

    bt = _one_strategy(
        res, ["ENTRY"], ["EXIT"], cost=0.01, max_hold=None,
        terminal_policy="mark",
    )

    assert bt["trades"] == []
    assert np.isclose(bt["equity"][-1], (0.99 / 11.0) * 13.5)


def test_terminal_policy_rejects_unknown_value():
    _assert_value_error(
        "terminal_policy",
        lambda: _one_strategy(
            _trade_frame(), ["ENTRY"], ["EXIT"], cost=0, max_hold=None,
            terminal_policy="ignore",
        ),
    )


def test_mark_terminal_policy_reports_open_and_pending_exit_state():
    res = _trade_frame(4)
    res.loc[res.index[0], "ENTRY"] = True
    res.loc[res.index[-1], "EXIT"] = True

    state = _one_strategy(
        res, ["ENTRY"], ["EXIT"], cost=0, max_hold=None,
        terminal_policy="mark",
    )["state"]

    assert state["position"] == "open"
    assert state["entry_i"] == 1
    assert state["entry_open"] == 11.0
    assert state["highest_close"] == 13.5
    assert state["pending_buy"] is False
    assert state["pending_sell_reason"] == "signal"


def test_mark_state_reports_pending_profit_lock_without_settling_it():
    res = _trade_frame(3)
    res["OPEN"] = [10.0, 100.0, 110.0]
    res["CLOSE"] = [10.0, 120.0, 104.0]
    res.loc[res.index[0], "ENTRY"] = True

    bt = _one_strategy(
        res, ["ENTRY"], ["EXIT"], cost=0, max_hold=None,
        trail=0.20, profit_keep=0.20, terminal_policy="mark",
    )
    state = bt["state"]

    assert bt["trades"] == []
    assert state["pending_sell_reason"] == "profit_lock"
    assert state["profit_armed"] is True
    assert np.isclose(state["profit_floor"], 104.0)
    assert np.isclose(state["mark_equity"], 1.04)


def test_keep50_profit_floor_uses_the_frozen_rule_cost():
    res = _trade_frame(3)
    res["OPEN"] = [10.0, 100.0, 110.0]
    res["CLOSE"] = [10.0, 120.0, 110.05]
    res.loc[res.index[0], "ENTRY"] = True

    state = _one_strategy(
        res, ["ENTRY"], ["EXIT"], cost=0.001, max_hold=None,
        trail=0.20, profit_keep=0.50, terminal_policy="mark",
    )["state"]
    break_even = 100.0 / (1 - 0.001) ** 2

    assert state["pending_sell_reason"] == "profit_lock"
    assert state["profit_armed"] is True
    assert np.isclose(state["profit_floor"], break_even + 0.50 * (120 - break_even))


def test_mark_state_distinguishes_a_pending_entry_from_flat_cash():
    res = _trade_frame(3)
    res.loc[res.index[-1], "ENTRY"] = True

    state = _one_strategy(
        res, ["ENTRY"], ["EXIT"], cost=0, max_hold=None,
        terminal_policy="mark",
    )["state"]

    assert state["status"] == "pending_entry"
    assert state["pending_buy"] is True


def test_exit_bar_not_exposed_and_max_hold_is_exact():
    res = _trade_frame(5)
    res.loc[res.index[0], "ENTRY"] = True
    res.loc[res.index[1], "EXIT"] = True
    bt = _one_strategy(res, ["ENTRY"], ["EXIT"], cost=0, max_hold=None)
    assert bt["trades"][0]["j"] == 2
    assert bt["trades"][0]["hold"] == 1
    assert np.isclose(_exposure(bt, res), 1 / 5)

    res.loc[:, "EXIT"] = False
    bt = _one_strategy(res, ["ENTRY"], ["EXIT"], cost=0, max_hold=1)
    assert bt["trades"][0]["hold"] == 1


def test_initial_hard_stop_confirms_at_close_and_exits_at_next_open():
    res = _trade_frame(5)
    res["OPEN"] = [10.0, 100.0, 80.0, 80.0, 80.0]
    res["CLOSE"] = [10.0, 84.0, 80.0, 80.0, 80.0]
    res.loc[res.index[0], "ENTRY"] = True

    bt = _one_strategy(
        res, ["ENTRY"], ["EXIT"], cost=0, max_hold=None, hard_stop=0.15
    )

    trade = bt["trades"][0]
    assert trade["exit_reason"] == "hard_stop"
    assert trade["j"] == 2
    assert np.isclose(trade["ret"], -0.20)


def test_profit_lock_arms_at_trail_gain_and_exits_at_next_open():
    res = _trade_frame(5)
    res["OPEN"] = [10.0, 100.0, 110.0, 90.0, 90.0]
    res["CLOSE"] = [10.0, 120.0, 104.0, 90.0, 90.0]
    res.loc[res.index[0], "ENTRY"] = True

    bt = _one_strategy(
        res, ["ENTRY"], ["EXIT"], cost=0, max_hold=None,
        trail=0.20, profit_keep=0.20,
    )

    trade = bt["trades"][0]
    assert trade["exit_reason"] == "profit_lock"
    assert trade["j"] == 3
    assert np.isclose(trade["ret"], -0.10)


def test_existing_trail_owns_a_same_day_profit_floor_breach():
    res = _trade_frame(5)
    res["OPEN"] = [10.0, 100.0, 100.0, 90.0, 90.0]
    res["CLOSE"] = [10.0, 120.0, 90.0, 90.0, 90.0]
    res.loc[res.index[0], "ENTRY"] = True

    trade = _one_strategy(
        res, ["ENTRY"], ["EXIT"], cost=0, max_hold=None,
        trail=0.20, profit_keep=0.20,
    )["trades"][0]

    assert trade["exit_reason"] == "trail"


def test_profit_keep_requires_a_trailing_stop():
    _assert_value_error(
        "profit_keep",
        lambda: _one_strategy(
            _trade_frame(), ["ENTRY"], ["EXIT"], cost=0, max_hold=None,
            profit_keep=0.20,
        ),
    )


def test_profit_keep_must_be_a_positive_fraction():
    for invalid in (False, "0.20", -0.1, 0, 1, np.inf, np.nan):
        _assert_value_error(
            "profit_keep",
            lambda invalid=invalid: _one_strategy(
                _trade_frame(), ["ENTRY"], ["EXIT"], cost=0,
                max_hold=None, trail=0.20, profit_keep=invalid,
            ),
        )


def test_profit_keep_requires_a_valid_trailing_fraction():
    for invalid in (False, "0.20", -0.1, 0, 1, np.inf, np.nan):
        _assert_value_error(
            "trail",
            lambda invalid=invalid: _one_strategy(
                _trade_frame(), ["ENTRY"], ["EXIT"], cost=0,
                max_hold=None, trail=invalid, profit_keep=0.20,
            ),
        )


def test_run_backtest_applies_a_preset_initial_hard_stop():
    res = _trade_frame(5)
    res["OPEN"] = [10.0, 100.0, 80.0, 120.0, 120.0]
    res["CLOSE"] = [10.0, 84.0, 100.0, 120.0, 120.0]
    res.loc[res.index[0], "ENTRY"] = True
    presets = [{
        "name": "带初始止损", "entry": ["ENTRY"], "exit": ["EXIT"],
        "hard_stop": 0.15,
    }]

    report = run_backtest(res, cost=0, presets=presets)

    strategy = report["strategies"][0]
    assert strategy["trades"] == 1
    assert strategy["total"] == -20.0


def test_run_backtest_applies_a_preset_profit_lock():
    res = _trade_frame(5)
    res["OPEN"] = [10.0, 100.0, 110.0, 90.0, 70.0]
    res["CLOSE"] = [10.0, 120.0, 104.0, 90.0, 70.0]
    res.loc[res.index[0], "ENTRY"] = True
    presets = [{
        "name": "盈利保护", "entry": ["ENTRY"], "exit": ["EXIT"],
        "trail": 0.20, "profit_keep": 0.20,
    }]

    strategy = run_backtest(res, cost=0, presets=presets)["strategies"][0]

    assert strategy["total"] == -10.0


def _assert_value_error(message, callback):
    try:
        callback()
    except ValueError as error:
        assert message in str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def test_initial_hard_stop_must_be_a_positive_fraction():
    for invalid in (False, "0.15", -0.1, 0, 1, np.inf, np.nan):
        _assert_value_error(
            "hard_stop",
            lambda invalid=invalid: _one_strategy(
                _trade_frame(), ["ENTRY"], ["EXIT"], cost=0,
                max_hold=None, hard_stop=invalid,
            ),
        )


def test_strategy_rejects_non_finite_execution_prices():
    for column in ("OPEN", "CLOSE"):
        res = _trade_frame()
        res.loc[res.index[2], column] = np.nan
        _assert_value_error(
            "OPEN/CLOSE",
            lambda res=res: _one_strategy(
                res, ["ENTRY"], ["EXIT"], cost=0, max_hold=None,
            ),
        )


def test_strategy_rejects_non_positive_execution_prices():
    for value in (0.0, -1.0):
        res = _trade_frame()
        res.loc[res.index[2], "OPEN"] = value
        _assert_value_error(
            "OPEN/CLOSE",
            lambda res=res: _one_strategy(
                res, ["ENTRY"], ["EXIT"], cost=0, max_hold=None,
            ),
        )


def test_hard_stop_boundary_is_inclusive_and_signal_has_priority():
    res = _trade_frame(4)
    res["OPEN"] = [10.0, 100.0, 85.0, 85.0]
    res["CLOSE"] = [10.0, 85.0, 85.0, 85.0]
    res.loc[res.index[0], "ENTRY"] = True
    res.loc[res.index[1], "EXIT"] = True

    bt = _one_strategy(
        res, ["ENTRY"], ["EXIT"], cost=0, max_hold=None,
        trail=0.10, hard_stop=0.15,
    )

    assert bt["trades"][0]["exit_reason"] == "signal"
    assert bt["trades"][0]["j"] == 2


def test_perf_uses_initial_capital_and_dollar_profit_factor():
    st = _perf(np.array([0.8]), [], periods_per_year=252)
    assert st["total"] == -20.0
    assert st["mdd"] == 20.0

    trades = [
        {"ret": 0.5, "pnl": 1.0, "hold": 1},
        {"ret": -0.5, "pnl": -0.5, "hold": 1},
    ]
    st = _perf(np.array([1.0, 1.5, 1.0]), trades, periods_per_year=252)
    assert st["pf"] == 2.0


def test_buy_hold_charges_both_sides_and_tracks_initial_drawdown():
    res = pd.DataFrame({"OPEN": [10.0], "CLOSE": [8.0]})
    bt = _buy_hold(res, cost=0.01)
    assert np.isclose(bt["equity"][-1], 8.0 / 10.0 * 0.99 * 0.99)
    assert _perf(bt["equity"], [], periods_per_year=252)["mdd"] > 20.0


def test_presets_are_pure_and_empty_list_means_no_strategies():
    res = _res()
    presets = [{"name": "one", "entry": ["B_SIGNAL"], "exit": ["S_SIGNAL"]}]
    before = copy.deepcopy(presets)
    run_backtest(res, presets=presets)
    assert presets == before
    rep = run_backtest(res, presets=[])
    assert [s["name"] for s in rep["strategies"]] == ["基准: 买入持有"]
    assert set(PRESETS[0]) == {"name", "entry", "exit"}


def test_parallel_backtests_do_not_cross_contaminate_curves():
    res = _res()

    def run(n):
        report = run_backtest(res.iloc[:n], presets=PRESETS)
        return {name: len(curve) for name, curve in report["equity"].items()}

    with ThreadPoolExecutor(max_workers=2) as pool:
        small, large = pool.map(run, (300, 500))
    assert set(small.values()) == {300}
    assert set(large.values()) == {500}
    assert all("_equity" not in preset for preset in PRESETS)


def test_interval_controls_annualization_and_metadata():
    res = _res().iloc[:100]
    daily = run_backtest(res, presets=[], interval="1d")
    weekly = run_backtest(res, presets=[], interval="1wk")
    assert daily["timeframe"] == {
        "interval": "1d", "period_label": "日", "periods_per_year": 252,
    }
    assert weekly["timeframe"] == {
        "interval": "1wk", "period_label": "周", "periods_per_year": 52,
    }
    assert daily["strategies"][0]["cagr"] != weekly["strategies"][0]["cagr"]


def test_event_study_custom_horizon_and_split_no_leakage():
    n = 10
    res = pd.DataFrame({
        "OPEN": np.arange(10.0, 20.0),
        "CLOSE": np.arange(10.5, 20.5),
        "B_SIGNAL": False,
    })
    # split=6, h=3: T=1 is safe in-sample; T=4 exits after split and must be excluded.
    res.loc[[1, 4, 6], "B_SIGNAL"] = True
    rows = event_study(res, horizons=(3,))
    row = next(x for x in rows if x["signal"] == "B_SIGNAL")
    assert row["split"]["horizon"] == 3
    assert row["split"]["in_sample"]["n"] == 1
    assert row["split"]["out_sample"]["n"] == 1
    assert set(row["split"]["in_sample"]) >= {"base_win", "base_mean", "excess"}
    assert set(row["split"]["out_sample"]) >= {"base_win", "base_mean", "excess"}
    assert row["split5"] is None
    assert set(row["horizons"]["3"]) >= {"p", "q", "base_mean", "excess"}
