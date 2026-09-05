"""r24纯绝反双分量压力退出；真实次OPEN、来源锁定与旧默认兼容。"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    return pd.DataFrame({
        "OPEN": [100., 50., 100., 121., 90., 80., 100.],
        "CLOSE": [200., 50., 110., 100., 90., 900., 100.],
        "B_SIGNAL": False, "ICON_JUEFAN": [False, True, False, True, False, False, False],
        "S_SIGNAL": False, "JOINT": [False, True, False, False, False, False, False],
    })


def test_r24_locks_actual_entry_source_excludes_entry_gap_and_waits_for_both_factors_then_real_open():
    from gcn.backtest.engine import _one_strategy
    frame = _fixture()
    result = _one_strategy(frame, ["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"], .001, None,
                           trail=.20, entry_joint_pressure_col="JOINT")
    assert [(t["i"], t["j"], t["hold"], t["exit_reason"]) for t in result["trades"]] == [(2, 5, 3, "joint_pressure")]
    trade = result["trades"][0]
    assert trade["joint_pressure_enabled"] and trade["joint_pressure_ever_net_positive"]
    assert trade["joint_pressure_first_profit_i"] == 2 and trade["joint_pressure_trigger_i"] == 4
    assert np.isclose(trade["joint_pressure_intraday_factor"], 110/100*100/121*90/90)
    assert np.isclose(trade["joint_pressure_overnight_factor"], 121/110*90/100)
    assert np.isclose(trade["joint_pressure_intraday_factor"]*trade["joint_pressure_overnight_factor"], .9)
    assert np.isclose(trade["ret"], .8*.999**2-1)  # Entry gap and exit-day 900 CLOSE are not held PnL.
    assert result["held"].tolist() == [False, False, True, True, True, False, False]
    assert result["state"]["joint_pressure"] is None


def test_r24_boolean_marker_and_cost_contract_rejects_invalid_values_and_null_disables():
    import pytest
    from gcn.backtest.engine import _one_strategy
    frame = _fixture(); frame.JOINT = frame.JOINT.astype(object)
    args = (["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"], .001, None)
    for column in ("missing", 1, ["JOINT"]):
        with pytest.raises(ValueError, match="entry_joint_pressure_col"):
            _one_strategy(frame, *args, entry_joint_pressure_col=column)
    for value in (0, 1, .1, "False", "", [True], {}, 1+0j):
        bad = frame.copy(); bad.at[1, "JOINT"] = value
        with pytest.raises(ValueError, match="entry_joint_pressure_col"):
            _one_strategy(bad, *args, entry_joint_pressure_col="JOINT")
    for value in (False, np.bool_(False), None, np.nan, pd.NA):
        disabled = frame.copy(); disabled.at[1, "JOINT"] = value
        result = _one_strategy(disabled, *args, entry_joint_pressure_col="JOINT")
        assert result["trades"][0]["exit_reason"] == "terminal"
        assert not result["trades"][0]["joint_pressure_enabled"]
        assert result["trades"][0]["joint_pressure_first_profit_i"] is None
    for cost in (1., -1., True, "0.001", None, np.nan, np.inf):
        with pytest.raises(ValueError, match="cost"):
            _one_strategy(frame, ["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"], cost, None,
                           entry_joint_pressure_col="JOINT")


def test_r24_strict_unit_boundaries_prior_net_profit_and_fixed_reference_fee_not_actual_cost():
    from gcn.backtest.engine import _one_strategy
    def run(o, c, cost=.001):
        frame = pd.DataFrame({"OPEN": o, "CLOSE": c, "B_SIGNAL": [True]+[False]*(len(o)-1),
                              "S_SIGNAL": False, "JOINT": True})
        return _one_strategy(frame, ["B_SIGNAL"], ["S_SIGNAL"], cost, None,
                              entry_joint_pressure_col="JOINT", terminal_policy="mark")
    unit_night = run([100., 100., 110., 90.], [200., 110., 90., 90.])
    a = unit_night["state"]["joint_pressure"]
    assert a["joint_pressure_intraday_factor"] < 1 and a["joint_pressure_overnight_factor"] == 1
    assert a["joint_pressure_ever_net_positive"] and unit_night["state"]["status"] == "open"
    unit_day = run([100., 100., 100.], [200., 125., 80.])
    b = unit_day["state"]["joint_pressure"]
    assert b["joint_pressure_intraday_factor"] == 1 and b["joint_pressure_overnight_factor"] == .8
    assert unit_day["state"]["status"] == "open"
    below_day = run([100., 100., 100.], [200., 125., np.nextafter(80., 0.)])
    assert below_day["state"]["status"] == "pending_exit"
    assert below_day["state"]["pending_sell_reason"] == "joint_pressure"
    never_profited = run([100., 100., 81., 70.], [200., 90., 80., 75.])
    c = never_profited["state"]["joint_pressure"]
    assert c["joint_pressure_intraday_factor"] < 1 and c["joint_pressure_overnight_factor"] < 1
    assert not c["joint_pressure_ever_net_positive"] and c["joint_pressure_first_profit_i"] is None
    assert never_profited["state"]["status"] == "open"  # Pre-entry 200 CLOSE is irrelevant.
    assert 1/.999**2*.999**2 == 1.
    equal_net = run([1., .999**2, .99, .98], [10., 1., .98, .97])
    assert not equal_net["state"]["joint_pressure"]["joint_pressure_ever_net_positive"]
    f = _fixture(); f.loc[2, "CLOSE"] = 100.3; f.loc[3, ["OPEN", "CLOSE"]] = [120., 99.]
    f.loc[4, ["OPEN", "CLOSE"]] = [80., 80.]
    outputs = [_one_strategy(f, ["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"], fee, None,
                              entry_joint_pressure_col="JOINT") for fee in (.001, .0025)]
    assert 1.003*.999**2 > 1 and 1.003*.9975**2 < 1
    for fee, result in zip((.001, .0025), outputs):
        row = result["trades"][0]
        assert (row["i"], row["j"], row["exit_reason"]) == (2, 5, "joint_pressure")
        assert row["joint_pressure_first_profit_i"] == 2 and row["joint_pressure_trigger_i"] == 4
        assert np.isclose(row["ret"], .8*(1-fee)**2-1)
    for key in outputs[0]["trades"][0]:
        if key.startswith("joint_pressure_"):
            assert outputs[0]["trades"][0][key] == outputs[1]["trades"][0][key]


def test_r24_reentry_resets_factors_profit_memory_and_preserves_mark_prefixes_terminal_and_old_exit_priority():
    from gcn.backtest.engine import _one_strategy
    f = pd.DataFrame({
        "OPEN": [100., 50., 100., 121., 90., 80., 100., 81., 90., 90., 100., 85., 100.],
        "CLOSE": [200., 50., 110., 100., 90., 100., 90., 80., 110., 90., 90., 500., 100.],
        "B_SIGNAL": False, "ICON_JUEFAN": False, "S_SIGNAL": False, "JOINT": False,
    })
    f.loc[[1, 3, 5, 7, 11], "ICON_JUEFAN"] = True
    f.loc[[1, 5], "JOINT"] = True
    args = (["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"], .001, None)
    options = {"entry_joint_pressure_col": "JOINT", "trail": .20}
    result = _one_strategy(f, *args, **options)
    assert [(t["i"], t["j"], t["exit_reason"], t["joint_pressure_first_profit_i"], t["joint_pressure_trigger_i"])
            for t in result["trades"]] == [(2, 5, "joint_pressure", 2, 4), (6, 11, "joint_pressure", 8, 10),
                                           (12, 13, "terminal", None, None)]
    marked = _one_strategy(f, *args, **options, terminal_policy="mark")
    for length in range(1, len(f)+1):
        early = _one_strategy(f.iloc[:length], *args, **options, terminal_policy="mark")
        np.testing.assert_array_equal(early["equity"], marked["equity"][:length])
        np.testing.assert_array_equal(early["held"], marked["held"][:length])
        assert early["trades"] == [t for t in marked["trades"] if t["j"] < length]
    reset = _one_strategy(f.iloc[:7], *args, **options, terminal_policy="mark")["state"]["joint_pressure"]
    assert reset["joint_pressure_intraday_factor"] == .9 and reset["joint_pressure_overnight_factor"] == 1.
    assert not reset["joint_pressure_ever_net_positive"] and reset["joint_pressure_first_profit_i"] is None
    pending = _one_strategy(f.iloc[:5], *args, **options, terminal_policy="mark")
    assert pending["state"]["pending_sell_reason"] == "joint_pressure" and not pending["trades"]
    terminal = _one_strategy(f.iloc[:5], *args, **options)
    assert terminal["trades"][0]["exit_reason"] == "terminal"
    assert np.isclose(terminal["trades"][0]["ret"], .9*.999**2-1) and terminal["state"]["joint_pressure"] is None
    old = pd.DataFrame({"OPEN": [100., 100., 119., 93.], "CLOSE": [100., 120., 99., 900.],
                        "B_SIGNAL": [True, False, False, False], "S_SIGNAL": False, "JOINT": True,
                        "FLOOR": 100., "BE_BASE": 90., "USE_EXTRA": True, "EXTRA_EXIT": [False, False, True, False]})
    cases = [(True, None, {}, "signal"), (False, None, {"trail": .15}, "trail"),
             (False, None, {"hard_stop": .005}, "hard_stop"),
             (False, None, {"entry_floor_col": "FLOOR"}, "entry_floor"),
             (False, None, {"entry_exit_cols": ("USE_EXTRA", "EXTRA_EXIT")}, "entry_signal"),
             (False, None, {"profit_keep": .5}, "profit_lock"), (False, 2, {}, "max_hold"),
             (False, None, {"entry_breakeven_base_col": "BE_BASE"}, "breakeven"),
             (False, None, {}, "joint_pressure")]
    for signal, hold, config, reason in cases:
        copy = old.copy(); copy.loc[2, "S_SIGNAL"] = signal
        out = _one_strategy(copy, ["B_SIGNAL"], ["S_SIGNAL"], .001, hold,
                              **{**options, **config})
        row = out["trades"][0]
        assert (row["i"], row["j"], row["exit_reason"]) == (1, 3, reason)
        assert row["joint_pressure_trigger_i"] == (2 if reason == "joint_pressure" else None)


def test_r24_disabled_guard_preserves_frozen_engine_including_old_breakeven_and_all_false_null_markers():
    import importlib.util
    from gcn.backtest.engine import _one_strategy
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.backtest.signal_research_r18 import candidate_signals as floor_signals, CHALLENGERS as floors
    from gcn.backtest.signal_research_r21 import candidate_signals as be_signals, CHALLENGERS as be_rules
    from gcn.recipes.gcn_main import compute_ehopt10
    source = ROOT / "reports/gcn-historical-r23-20260905/training/input_source_snapshot/gcn/backtest/engine.py"
    spec = importlib.util.spec_from_file_location("r24_before_engine", source)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    frames, _ = load_snapshot(ROOT / "reports/signal-audit-v5-review-20260904")
    configurations = ({}, {"profit_keep": .5}, {"hard_stop": .125}, {"entry_floor_col": "FLOOR"},
                      {"entry_floor_col": "FLOOR", "entry_floor_confirm_bars": 2},
                      {"entry_breakeven_base_col": "BE_BASE"})
    for symbol in CORE:
        f = compute_ehopt10(frames[symbol].loc[:"2024-08-26"], version="v5")
        f["FLOOR"] = floor_signals(f)[floors[0]].ENTRY_FLOOR
        f["BE_BASE"] = be_signals(f)[be_rules[0]].ENTRY_BE_BASE
        f["OFF"] = False; f["EMPTY"] = pd.Series(pd.NA, index=f.index, dtype="boolean")
        args = (["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"], .001, None)
        for config in configurations:
            expected = old._one_strategy(f, *args, trail=.20, **config)
            for optional in ({}, {"entry_joint_pressure_col": None}):
                actual = _one_strategy(f, *args, trail=.20, **config, **optional)
                np.testing.assert_array_equal(actual["equity"], expected["equity"])
                np.testing.assert_array_equal(actual["held"], expected["held"])
                assert actual["trades"] == expected["trades"] and actual["state"] == expected["state"]
            for column in ("OFF", "EMPTY"):
                actual = _one_strategy(f, *args, trail=.20, **config, entry_joint_pressure_col=column)
                np.testing.assert_array_equal(actual["equity"], expected["equity"])
                np.testing.assert_array_equal(actual["held"], expected["held"])
                assert {key: actual["state"][key] for key in expected["state"]} == expected["state"]
                assert len(actual["trades"]) == len(expected["trades"])
                for row, prior in zip(actual["trades"], expected["trades"]):
                    assert {key: row[key] for key in prior} == prior
                    assert not row["joint_pressure_enabled"] and row["joint_pressure_trigger_i"] is None


def test_r24_factory_retains_native_events_and_only_marks_pure_jf_not_b_collision():
    from gcn.backtest.signal_research_r24 import candidate_signals, CHALLENGERS, CONTROLS, RULES
    f = _fixture(); f["LOW"] = f[["OPEN", "CLOSE"]].min(axis=1)-1
    f.loc[3, "B_SIGNAL"] = True
    assert CHALLENGERS == ("JF-joint-pressure",) and CONTROLS == ("v5",) and RULES == CONTROLS+CHALLENGERS
    rules = candidate_signals(f)
    native = ["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]
    for rule in RULES:
        pd.testing.assert_frame_equal(rules[rule][native], f[native], check_exact=True)
        assert rules[rule].ENTRY_STOP.isna().all() and rules[rule].ENTRY_LIMIT.isna().all()
        assert rules[rule].ENTRY_FLOOR.isna().all() and not rules[rule].USE_EXTRA.any()
        assert "ENTRY_BE_BASE" not in rules[rule]
    assert not rules["v5"].ENTRY_JOINT_PRESSURE.any()
    assert rules[CHALLENGERS[0]].ENTRY_JOINT_PRESSURE.tolist() == [False, True, False, False, False, False, False]
    for length in range(1, len(f)+1):
        pd.testing.assert_frame_equal(candidate_signals(f.iloc[:length])[CHALLENGERS[0]],
                                      rules[CHALLENGERS[0]].iloc[:length], check_exact=True)


def test_r24_evaluator_carries_actual_factors_dates_positions_and_reprices_same_guard_orders():
    from gcn.backtest.historical_research import evaluate_rule
    from gcn.backtest.signal_research_r24 import candidate_signals, CHALLENGERS
    f = _fixture(); f.index = pd.bdate_range("2024-01-02", periods=len(f))
    f["LOW"] = f[["OPEN", "CLOSE"]].min(axis=1)-1
    prepared = {"TEST": {"frame": f, "rules": candidate_signals(f)}}
    options = {"entry_joint_pressure_col": "ENTRY_JOINT_PRESSURE", "include_positions": True}
    start, end = f.index[0], f.index[-1]
    outputs = [evaluate_rule(prepared, CHALLENGERS[0], start, end, fee, **options) for fee in (.001, .0025)]
    for fee, result in zip((.001, .0025), outputs):
        row = result["trades"][0]
        assert row["entry_date"] == f.index[2].date().isoformat() and row["exit_date"] == f.index[5].date().isoformat()
        assert row["exit_reason"] == "joint_pressure" and row["joint_pressure_enabled"]
        assert row["joint_pressure_first_profit_date"] == f.index[2].date().isoformat()
        assert row["joint_pressure_trigger_date"] == f.index[4].date().isoformat()
        assert np.isclose(row["joint_pressure_intraday_factor"]*row["joint_pressure_overnight_factor"], .9)
        assert np.isclose(row["return_pct"], (.8*(1-fee)**2-1)*100)
        assert result["positions"]["TEST"].tolist() == [False, False, True, True, True, False, False]
    for key in outputs[0]["trades"][0]:
        if key != "return_pct":
            assert outputs[0]["trades"][0][key] == outputs[1]["trades"][0][key]
    old = evaluate_rule(prepared, "v5", start, end)
    disabled = evaluate_rule(prepared, "v5", start, end, **options)
    pd.testing.assert_frame_equal(old["returns"], disabled["returns"], check_exact=True)
    assert old["stats"] == disabled["stats"]
    assert not any(key.startswith("joint_pressure_") for key in old["trades"][0])
    assert {key: disabled["trades"][0][key] for key in old["trades"][0]} == old["trades"][0]


def test_r24_training_binds_fixed_rule_and_reconciles_actual_orders_factors_sources_and_old_v5(tmp_path):
    import hashlib
    import json
    import pytest
    from gcn.backtest.signal_research_r24 import run_training, CHALLENGERS, candidate_failures
    from gcn.backtest.historical_research import load_snapshot
    from gcn.recipes.gcn_main import compute_ehopt10
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    decision = run_training(snapshot, tmp_path)
    read = lambda p: pd.read_csv(p, float_precision="round_trip")
    rows = read(tmp_path / "training.csv").set_index("rule")
    failures = candidate_failures(rows.loc[CHALLENGERS[0]].to_dict(), rows.loc["v5"].to_dict())
    assert decision["failures"][CHALLENGERS[0]] == failures
    assert decision["selected"] == (None if failures else CHALLENGERS[0])
    assert decision["recommended"] == "v5" and not decision["production_changed"]
    frozen = ROOT / "reports/gcn-historical-r21-20260905/results"
    old = read(frozen / "training.csv").set_index("rule")
    pd.testing.assert_series_equal(rows.loc["v5"].drop("joint_pressure_exits"),
                                    old.loc["v5"].drop("breakeven_exits"), check_exact=True)
    trades = read(tmp_path / "trades.csv")
    old_trades = read(frozen / "trades.csv")
    columns = ["rule", "symbol", "entry_date", "exit_date", "return_pct", "hold_days", "exit_reason", "peak_close_pct"]
    pd.testing.assert_frame_equal(trades[trades.rule.eq("v5")][columns].reset_index(drop=True),
                                  old_trades[old_trades.rule.eq("v5")][columns].reset_index(drop=True), check_exact=True)
    frames, quality = load_snapshot(snapshot)
    prepared = {symbol: compute_ehopt10(raw.loc[:"2024-08-26"], version="v5") for symbol, raw in frames.items()}
    for row in trades.itertuples():
        frame = prepared[row.symbol]
        i = frame.index.get_loc(pd.Timestamp(row.entry_date))
        j = frame.index.get_loc(pd.Timestamp(row.exit_date)) + int(row.exit_reason == "terminal")
        signal = frame.iloc[i-1]; c = frame.CLOSE.iloc[i:j]; o = frame.OPEN.iloc[i:j]
        enabled = row.rule == CHALLENGERS[0] and bool(signal.ICON_JUEFAN) and not bool(signal.B_SIGNAL)
        assert row.entry_b == bool(signal.B_SIGNAL) and row.entry_jf == bool(signal.ICON_JUEFAN)
        assert row.entry_signal_date == frame.index[i-1].date().isoformat()
        assert row.joint_pressure_enabled == enabled
        price = c.iloc[-1] if row.exit_reason == "terminal" else frame.OPEN.iloc[j]
        assert np.isclose(row.return_pct, (price/o.iloc[0]*.999**2-1)*100)
        if not enabled:
            assert pd.isna(row.joint_pressure_intraday_factor) and pd.isna(row.joint_pressure_overnight_factor)
            assert not row.joint_pressure_ever_net_positive and pd.isna(row.joint_pressure_trigger_date)
            continue
        day = (c/o).cumprod(); night = (o/c.shift(1)).fillna(1.).cumprod()
        profit = c/o.iloc[0]*.999**2 > 1
        hits = profit.cummax() & day.lt(1) & night.lt(1)
        assert np.isclose(row.joint_pressure_intraday_factor, day.iloc[-1])
        assert np.isclose(row.joint_pressure_overnight_factor, night.iloc[-1])
        assert row.joint_pressure_ever_net_positive == profit.any()
        if profit.any():
            assert row.joint_pressure_first_profit_date == c.index[np.flatnonzero(profit)[0]].date().isoformat()
        else:
            assert pd.isna(row.joint_pressure_first_profit_date)
        if hits.any():
            assert np.flatnonzero(hits)[0] == len(c)-1
        if row.exit_reason == "joint_pressure":
            assert hits.iloc[-1] and row.joint_pressure_trigger_date == c.index[-1].date().isoformat()
            assert not frame.S_SIGNAL.iloc[j-1] and c.iloc[-1] > c.max()*.8
    assert rows.loc["v5", "joint_pressure_exits"] == 0
    assert rows.loc[CHALLENGERS[0], "joint_pressure_exits"] == trades[trades.rule.eq(CHALLENGERS[0])].exit_reason.eq("joint_pressure").sum()
    events = read(tmp_path / "events.csv")
    pd.testing.assert_frame_equal(events[events.rule.eq("v5")].drop(columns="rule").reset_index(drop=True),
                                  events[events.rule.eq(CHALLENGERS[0])].drop(columns="rule").reset_index(drop=True), check_exact=True)
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    assert manifest["entry_joint_pressure_col"] == "ENTRY_JOINT_PRESSURE"
    assert manifest["joint_pressure_reference_cost"] == .001 and manifest["joint_pressure_factor_threshold"] == 1.
    assert manifest["source_quality"] == quality
    assert manifest["parent_manifest_sha256"] == hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    for name, expected in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected
    for name, expected in manifest["algorithm_sources"].items():
        assert hashlib.sha256((tmp_path / "source_snapshot" / name).read_bytes()).hexdigest() == expected
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError):
        run_training(snapshot, tmp_path)


def test_r24_real_training_prefixes_preserve_candidate_orders_equity_and_current_factor_state():
    from gcn.backtest.engine import _one_strategy
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.backtest.signal_research_r24 import candidate_signals, CHALLENGERS
    from gcn.recipes.gcn_main import compute_ehopt10
    frames, _ = load_snapshot(ROOT / "reports/signal-audit-v5-review-20260904")
    start, end = pd.Timestamp("2021-08-27"), pd.Timestamp("2024-08-26")
    args = (["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"], .001, None)
    options = {"trail": .20, "entry_joint_pressure_col": "ENTRY_JOINT_PRESSURE", "terminal_policy": "mark"}
    checked = 0
    for symbol in CORE:
        raw = frames[symbol].loc[:end]
        def prepare(cut):
            frame = compute_ehopt10(raw.loc[:cut], version="v5")
            frame["ENTRY_JOINT_PRESSURE"] = candidate_signals(frame)[CHALLENGERS[0]].ENTRY_JOINT_PRESSURE
            return frame.loc[start:cut]
        full = prepare(end)
        expected = _one_strategy(full, *args, **options)
        cuts = {0, len(full)-1}
        if expected["trades"]:
            selected = [expected["trades"][0]]
            joint = [t for t in expected["trades"] if t["exit_reason"] == "joint_pressure"]
            selected += joint[:1]
            for trade in selected:
                cuts.update((trade["i"], trade["j"]-1, trade["j"]))
        for pos in sorted(cuts):
            frame = prepare(full.index[pos])
            early = _one_strategy(frame, *args, **options)
            np.testing.assert_array_equal(early["equity"], expected["equity"][:pos+1])
            np.testing.assert_array_equal(early["held"], expected["held"][:pos+1])
            assert early["trades"] == [t for t in expected["trades"] if t["j"] <= pos]
            state = early["state"]
            if state["position"] == "open" and state["joint_pressure"]["joint_pressure_enabled"]:
                i = state["entry_i"]; o = frame.OPEN.iloc[i:]; c = frame.CLOSE.iloc[i:]
                assert np.isclose(state["joint_pressure"]["joint_pressure_intraday_factor"], (c/o).prod())
                assert np.isclose(state["joint_pressure"]["joint_pressure_overnight_factor"], (o/c.shift(1)).fillna(1).prod())
                positive = c/o.iloc[0]*.999**2 > 1
                assert state["joint_pressure"]["joint_pressure_ever_net_positive"] == positive.any()
                assert state["joint_pressure"]["joint_pressure_first_profit_i"] == (
                    i+np.flatnonzero(positive)[0] if positive.any() else None)
            checked += 1
    assert checked >= 40
