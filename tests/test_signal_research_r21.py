"""r21：纯绝反1R成熟后净保本，严格保留原订单费用与旧默认。"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_r21_locks_actual_entry_r_waits_for_one_r_and_fills_next_open_not_reference():
    from gcn.backtest.engine import _one_strategy
    be = 100/.999**2
    frame = pd.DataFrame({
        "OPEN": [100., 100., 100., 100., 100., 100., 100., 93., 100.],
        "CLOSE": [180., 95., 101., 100., 110., 105., be, 150., 150.],
        "LOW": 80., "B_SIGNAL": False, "ICON_JUEFAN": False, "S_SIGNAL": False,
        "BE_BASE": [np.nan, 90., np.nan, 99., np.nan, np.nan, np.nan, np.nan, np.nan],
    })
    frame.loc[[1, 3], "ICON_JUEFAN"] = True
    result = _one_strategy(frame, ["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"], .001, None,
                           trail=.20, entry_breakeven_base_col="BE_BASE")
    assert [(t["i"], t["j"], t["hold"], t["exit_reason"]) for t in result["trades"]] == [
        (2, 7, 5, "breakeven")]
    trade = result["trades"][0]
    assert np.isclose(trade["ret"], .93*.999**2-1)
    assert trade["breakeven_base_price"] == 90. and trade["breakeven_risk"] == 10.
    assert trade["breakeven_risk_status"] == "valid" and trade["breakeven_armed"]
    assert trade["breakeven_arm_i"] == 4
    assert result["held"].tolist() == [False, False, True, True, True, True, True, False, False]


def test_r21_base_contract_rejects_bad_types_but_disables_nonfinite_or_nonpositive_r():
    import pytest
    from gcn.backtest.engine import _one_strategy
    frame = pd.DataFrame({"OPEN": 100., "CLOSE": [100., 110., 100.], "B_SIGNAL": [True, False, False],
                          "S_SIGNAL": False, "BE_BASE": pd.Series([None]*3, dtype=object)})
    args = (["B_SIGNAL"], ["S_SIGNAL"], .001, None)
    for column in ("missing", 1, ["BE_BASE"]):
        with pytest.raises(ValueError, match="entry_breakeven_base_col"):
            _one_strategy(frame, *args, entry_breakeven_base_col=column)
    for value in (0, -1, True, np.bool_(False), "90", [90], {}, 90+1j):
        bad = frame.copy(); bad.at[0, "BE_BASE"] = value
        with pytest.raises(ValueError, match="entry_breakeven_base_col"):
            _one_strategy(bad, *args, entry_breakeven_base_col="BE_BASE")
    for value in (None, np.nan, pd.NA, np.inf, -np.inf, 100., 101.):
        bad_r = frame.copy(); bad_r.at[0, "BE_BASE"] = value
        result = _one_strategy(bad_r, *args, entry_breakeven_base_col="BE_BASE")
        row = result["trades"][0]
        assert row["exit_reason"] == "terminal" and not row["breakeven_armed"]
        assert row["breakeven_arm_i"] is None
        if value is not None and value is not pd.NA and np.isfinite(value):
            assert row["breakeven_risk_status"] == "nonpositive" and row["breakeven_risk"] == 100-value
        else:
            assert row["breakeven_risk_status"] == "invalid" and row["breakeven_risk"] is None
    for cost in (1., -1., True, "0.001", np.nan, np.inf, None):
        with pytest.raises(ValueError, match="cost"):
            _one_strategy(frame, ["B_SIGNAL"], ["S_SIGNAL"], cost, None,
                           entry_breakeven_base_col="BE_BASE")


def test_r21_reentry_clears_arm_state_preserves_original_exit_priority_and_causal_mark_prefixes():
    from gcn.backtest.engine import _one_strategy
    be = 100/.999**2
    frame = pd.DataFrame({
        "OPEN": [100., 100., 100., 95., 100., 100., 100., 100., 99., 100., 100., 100.],
        "CLOSE": [100., 110., be, 100., 101., 100., 110., 100., 100., 120., 100., 102.],
        "B_SIGNAL": [True, False, False, True, False, False, False, False, True, False, False, False],
        "S_SIGNAL": False,
        "BE_BASE": [90., np.nan, np.nan, 90., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    })
    args = (["B_SIGNAL"], ["S_SIGNAL"], .001, None)
    options = {"trail": .20, "entry_breakeven_base_col": "BE_BASE"}
    result = _one_strategy(frame, *args, **options)
    assert [(t["i"], t["j"], t["exit_reason"], t["breakeven_arm_i"]) for t in result["trades"]] == [
        (1, 3, "breakeven", 1), (4, 8, "breakeven", 6), (9, 12, "terminal", None)]
    marked = _one_strategy(frame, *args, **options, terminal_policy="mark")
    assert marked["state"]["breakeven"]["breakeven_risk_status"] == "invalid"
    for length in range(1, len(frame)+1):
        prefix = _one_strategy(frame.iloc[:length], *args, **options, terminal_policy="mark")
        np.testing.assert_array_equal(prefix["equity"], marked["equity"][:length])
        np.testing.assert_array_equal(prefix["held"], marked["held"][:length])
        assert prefix["trades"] == [t for t in marked["trades"] if t["j"] < length]
    pending = _one_strategy(frame.iloc[:3], *args, **options, terminal_policy="mark")
    assert pending["state"]["status"] == "pending_exit" and pending["state"]["pending_sell_reason"] == "breakeven"
    assert pending["state"]["breakeven"]["breakeven_armed"] and not pending["trades"]
    terminal = _one_strategy(frame.iloc[:3], *args, **options)
    assert terminal["trades"][0]["exit_reason"] == "terminal"
    assert np.isclose(terminal["trades"][0]["ret"], 0.)
    assert terminal["state"]["breakeven"] is None
    for s, peak, hold, reason in ((True, 130., None, "signal"), (False, 130., None, "trail"),
                                  (False, 110., 2, "max_hold"), (False, 110., None, "breakeven")):
        original = frame.iloc[:4].copy()
        original.loc[1, "CLOSE"] = peak
        original.loc[2, "S_SIGNAL"] = s
        actual = _one_strategy(original, ["B_SIGNAL"], ["S_SIGNAL"], .001, hold, **options)
        assert actual["trades"][0]["j"] == 3 and actual["trades"][0]["exit_reason"] == reason


def test_r21_disabled_feature_preserves_frozen_engine_and_all_empty_prices_keep_orders():
    import importlib.util
    from gcn.backtest.engine import _one_strategy
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.backtest.signal_research_r18 import candidate_signals, CHALLENGERS
    from gcn.recipes.gcn_main import compute_ehopt10
    source = ROOT / "reports/gcn-historical-r20-20260905/training/source_snapshot/gcn/backtest/engine.py"
    spec = importlib.util.spec_from_file_location("r20_frozen_engine", source)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    frames, _ = load_snapshot(ROOT / "reports/signal-audit-v5-review-20260904")
    configurations = ({}, {"profit_keep": .5}, {"hard_stop": .125},
                      {"entry_floor_col": "ENTRY_FLOOR"},
                      {"entry_floor_col": "ENTRY_FLOOR", "entry_floor_confirm_bars": 2})
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:"2024-08-26"], version="v5")
        frame["ENTRY_FLOOR"] = candidate_signals(frame)[CHALLENGERS[0]].ENTRY_FLOOR
        frame["EMPTY_BASE"] = np.nan
        args = (["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"], .001, None)
        for config in configurations:
            expected = old._one_strategy(frame, *args, trail=.20, **config)
            for optional in ({}, {"entry_breakeven_base_col": None}):
                actual = _one_strategy(frame, *args, trail=.20, **config, **optional)
                np.testing.assert_array_equal(actual["equity"], expected["equity"])
                np.testing.assert_array_equal(actual["held"], expected["held"])
                assert actual["trades"] == expected["trades"] and actual["state"] == expected["state"]
            empty = _one_strategy(frame, *args, trail=.20, **config, entry_breakeven_base_col="EMPTY_BASE")
            np.testing.assert_array_equal(empty["equity"], expected["equity"])
            np.testing.assert_array_equal(empty["held"], expected["held"])
            assert len(empty["trades"]) == len(expected["trades"])
            for actual, prior in zip(empty["trades"], expected["trades"]):
                assert {key: actual[key] for key in prior} == prior
                assert actual["breakeven_risk_status"] == "invalid" and not actual["breakeven_armed"]


def test_r21_factory_keeps_native_signals_and_only_selects_pure_jf_three_bar_bases():
    from gcn.backtest.signal_research_r21 import candidate_signals, RULES, CHALLENGERS
    from gcn.backtest.signal_research_r17 import candidate_signals as previous
    frame = pd.DataFrame({"LOW": [90., 88., 95., 97., 89., 90.],
                          "B_SIGNAL": [False, False, False, True, False, False],
                          "ICON_JUEFAN": [True, False, True, True, True, False], "S_SIGNAL": False})
    saved = frame.copy(deep=True)
    rules = candidate_signals(frame)
    assert list(rules) == list(RULES) == ["v5", "JF-1R-breakeven"]
    original = previous(frame)["v5"]
    for rule in RULES:
        pd.testing.assert_frame_equal(rules[rule][original.columns], original, check_exact=True)
    assert rules["v5"].ENTRY_BE_BASE.isna().all()
    np.testing.assert_allclose(rules[CHALLENGERS[0]].ENTRY_BE_BASE, [np.nan, np.nan, 88., np.nan, 89., np.nan])
    for length in range(1, len(frame)+1):
        prefix = candidate_signals(frame.iloc[:length])
        for rule in RULES:
            pd.testing.assert_frame_equal(prefix[rule], rules[rule].iloc[:length], check_exact=True)
    pd.testing.assert_frame_equal(frame, saved)


def test_r21_training_binds_fixed_guard_and_reconciles_actual_sources_arm_dates_fees_and_archives(tmp_path):
    import hashlib
    import json
    from gcn.backtest.signal_research_r21 import run_training, RULES, CHALLENGERS
    from gcn.backtest.signal_research_r17 import candidate_failures
    from gcn.backtest.historical_research import load_snapshot
    from gcn.recipes.gcn_main import compute_ehopt10
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    decision = run_training(snapshot, tmp_path)
    rows = pd.read_csv(tmp_path / "training.csv").set_index("rule")
    assert list(rows.index) == list(RULES)
    failures = candidate_failures(rows.iloc[1].to_dict(), rows.iloc[0].to_dict())
    assert decision["failures"] == {CHALLENGERS[0]: failures}
    assert decision["selected"] == (None if failures else CHALLENGERS[0])
    assert not decision["production_changed"] and decision["recommended"] == "v5"
    frozen = ROOT / "reports/gcn-historical-r19-20260905/results"
    original = pd.read_csv(frozen / "training.csv").set_index("rule")
    columns = [col for col in original.columns if col != "entry_floor_exits"]
    pd.testing.assert_series_equal(rows.loc["v5", columns], original.loc["v5", columns], check_exact=True)
    trades = pd.read_csv(tmp_path / "trades.csv", float_precision="round_trip")
    old = pd.read_csv(frozen / "trades.csv", float_precision="round_trip")
    columns = ["rule", "symbol", "entry_date", "exit_date", "return_pct", "hold_days", "exit_reason", "peak_close_pct"]
    pd.testing.assert_frame_equal(trades.loc[trades.rule.eq("v5"), columns].reset_index(drop=True),
                                  old.loc[old.rule.eq("v5"), columns].reset_index(drop=True), check_exact=True)
    frames, quality = load_snapshot(snapshot)
    for row in trades.itertuples():
        frame = compute_ehopt10(frames[row.symbol].loc[:"2024-08-26"], version="v5")
        signal = frame.loc[row.entry_signal_date]
        s = frame.index.get_loc(pd.Timestamp(row.entry_signal_date))
        i = frame.index.get_loc(pd.Timestamp(row.entry_date))
        terminal = row.exit_reason == "terminal"
        j = frame.index.get_loc(pd.Timestamp(row.exit_date)) + int(terminal)
        held = frame.CLOSE.iloc[i:j]
        entry = frame.OPEN.iloc[i]
        exit_price = held.iloc[-1] if terminal else frame.OPEN.iloc[j]
        assert row.entry_b == bool(signal.B_SIGNAL) and row.entry_jf == bool(signal.ICON_JUEFAN)
        assert np.isclose(row.return_pct, (exit_price/entry*.999**2-1)*100)
        enabled = row.rule == CHALLENGERS[0] and signal.ICON_JUEFAN and not signal.B_SIGNAL
        if not enabled:
            assert pd.isna(row.breakeven_risk) and not row.breakeven_armed
            continue
        base = frame.LOW.iloc[s-2:s+1].min()
        assert row.breakeven_base_price == base and row.breakeven_risk == entry-base
        risk = entry-base
        armed = held.cummax().ge(entry+risk) if risk > 0 else pd.Series(False, index=held.index)
        assert row.breakeven_armed == armed.any()
        assert pd.notna(row.breakeven_arm_date) == armed.any()
        if armed.any():
            assert row.breakeven_arm_date == held.index[np.flatnonzero(armed)[0]].date().isoformat()
        if row.exit_reason == "breakeven":
            hits = np.flatnonzero(armed & held.le(entry/.999**2))
            assert len(hits) and hits[0] == len(held)-1
            assert not frame.S_SIGNAL.iloc[j-1] and held.iloc[-1] > held.max()*.8
    events = pd.read_csv(tmp_path / "events.csv")
    pd.testing.assert_frame_equal(events[events.rule.eq("v5")].drop(columns="rule").reset_index(drop=True),
                                  events[events.rule.eq(CHALLENGERS[0])].drop(columns="rule").reset_index(drop=True))
    m = json.loads((tmp_path / "manifest.json").read_bytes())
    assert m["entry_breakeven_base_col"] == "ENTRY_BE_BASE"
    assert m["breakeven_arm_r"] == 1. and m["breakeven_reference_cost"] == .001
    assert m["source_quality"] == quality
    assert m["parent_manifest_sha256"] == hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    for name, digest in m["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    for name, digest in m["algorithm_sources"].items():
        assert hashlib.sha256((tmp_path / "source_snapshot" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_r21_cost_stress_reprices_identical_guard_orders_using_fixed_reference_fee():
    from gcn.backtest.historical_research import evaluate_rule
    frame = pd.DataFrame({"OPEN": [100., 100., 100., 100., 100., 100., 95.],
                          "CLOSE": [100., 105., 100., 110., 101., 100., 120.],
                          "B_SIGNAL": False, "ICON_JUEFAN": [True, False, False, False, False, False, False],
                          "S_SIGNAL": False, "ENTRY_BE_BASE": [90., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]},
                         index=pd.bdate_range("2025-01-01", periods=7))
    prepared = {"TEST": {"frame": frame, "rules": {"JF-1R-breakeven": frame}}}
    results = [evaluate_rule(prepared, "JF-1R-breakeven", frame.index[0], frame.index[-1], cost,
                             entry_breakeven_base_col="ENTRY_BE_BASE", include_positions=True)
               for cost in (.001, .0025)]
    pd.testing.assert_frame_equal(results[0]["positions"], results[1]["positions"])
    for result, cost in zip(results, (.001, .0025)):
        assert len(result["trades"]) == 1
        trade = result["trades"][0]
        assert trade["entry_date"] == frame.index[1].date().isoformat()
        assert trade["exit_date"] == frame.index[6].date().isoformat() and trade["exit_reason"] == "breakeven"
        assert trade["breakeven_arm_date"] == frame.index[3].date().isoformat()
        assert np.isclose(trade["return_pct"], (.95*(1-cost)**2-1)*100)
    assert {k:v for k,v in results[0]["trades"][0].items() if k != "return_pct"} == {
        k:v for k,v in results[1]["trades"][0].items() if k != "return_pct"}


def test_r21_shared_training_extension_preserves_old_r19_output_bytes_and_schema(tmp_path):
    import hashlib
    import json
    from gcn.backtest.signal_research_r19 import run_training
    frozen = ROOT / "reports/gcn-historical-r19-20260905/results"
    expected = json.loads((frozen / "manifest.json").read_bytes())
    run_training(ROOT / "reports/signal-audit-v5-review-20260904", tmp_path)
    for name, digest in expected["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    current = json.loads((tmp_path / "manifest.json").read_bytes())
    assert "entry_breakeven_base_col" not in current and "breakeven_arm_r" not in current
    assert not any(col.startswith("breakeven_") for col in pd.read_csv(tmp_path / "trades.csv").columns)
