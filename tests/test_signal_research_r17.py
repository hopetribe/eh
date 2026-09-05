"""r17绝反结构失效：真实入场锁定价格，默认策略不变。"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_r17_fixed_floor_is_locked_at_entry_uses_close_and_executes_next_open_with_fees():
    from gcn.backtest.engine import _one_strategy
    index = pd.bdate_range("2025-01-01", periods=10)
    frame = pd.DataFrame({"OPEN": 100., "CLOSE": [100., 100., 95., 90., 89., 110., 95., 80., 80., 80.],
                          "B_SIGNAL": False, "ICON_JUEFAN": False, "S_SIGNAL": False,
                          "ENTRY_FLOOR": np.nan}, index=index)
    frame.loc[index[[0, 2, 5]], "ICON_JUEFAN"] = True
    frame.loc[index[[0, 2]], "ENTRY_FLOOR"] = [90., 97.]
    frame.loc[index[5], "OPEN"] = 85.  # Gap through the floor: fill at actual OPEN, not 90.
    frame.loc[index[9], "OPEN"] = 82.
    frame.loc[index[8], "S_SIGNAL"] = True
    frame["LOW"] = frame[["OPEN", "CLOSE"]].min(axis=1) - 1
    frame.loc[index[3], "LOW"] = 80.  # Intraday breach, but CLOSE equals the floor.
    result = _one_strategy(frame, ["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"],
                           .001, None, trail=.20, entry_floor_col="ENTRY_FLOOR")
    trades = result["trades"]
    assert [(t["i"], t["j"], t["hold"], t["exit_reason"]) for t in trades] == [
        (1, 5, 4, "entry_floor"), (6, 9, 3, "signal")]
    assert np.allclose([t["ret"] for t in trades], np.array([.85, .82]) * .999**2 - 1)
    assert result["held"].tolist() == [False, True, True, True, True, False, True, True, True, False]
    assert result["state"]["status"] == "flat"


def test_r17_entry_floor_requires_existing_column_and_finite_positive_price_or_null():
    import pytest
    from gcn.backtest.engine import _one_strategy
    frame = pd.DataFrame({"OPEN": 100., "CLOSE": 100., "B_SIGNAL": [True, False, False],
                          "S_SIGNAL": False, "ENTRY_FLOOR": pd.Series([None]*3, dtype=object)})
    for column in ("missing", 1, ["ENTRY_FLOOR"]):
        with pytest.raises(ValueError, match="entry_floor_col"):
            _one_strategy(frame, ["B_SIGNAL"], ["S_SIGNAL"], .001, None, entry_floor_col=column)
    for value in (0, -1, np.inf, -np.inf, True, np.bool_(False), "90", [90], {}, 90+1j):
        changed = frame.copy()
        changed.at[0, "ENTRY_FLOOR"] = value
        with pytest.raises(ValueError, match="entry_floor_col"):
            _one_strategy(changed, ["B_SIGNAL"], ["S_SIGNAL"], .001, None, entry_floor_col="ENTRY_FLOOR")
    for value in (None, np.nan, pd.NA, 90, np.float64(90), np.int64(90)):
        changed = frame.copy()
        changed.at[0, "ENTRY_FLOOR"] = value
        result = _one_strategy(changed, ["B_SIGNAL"], ["S_SIGNAL"], .001, None,
                               entry_floor_col="ENTRY_FLOOR")
        assert result["trades"][0]["exit_reason"] == "terminal"


def test_r17_gap_entry_priority_terminal_and_causal_mark_prefixes():
    from gcn.backtest.engine import _one_strategy
    frame = pd.DataFrame({"OPEN": [100., 85., 75., 80.], "CLOSE": [100., 88., 80., 80.],
                          "B_SIGNAL": [True, False, False, False], "S_SIGNAL": False,
                          "ENTRY_FLOOR": [90., 99., np.nan, np.nan], "USE_EXTRA": True, "EXTRA": True})
    args = (["B_SIGNAL"], ["S_SIGNAL"], .001, None)
    result = _one_strategy(frame, *args, entry_floor_col="ENTRY_FLOOR", terminal_policy="mark")
    assert [(t["i"], t["j"], t["exit_reason"]) for t in result["trades"]] == [(1, 2, "entry_floor")]
    assert np.isclose(result["trades"][0]["ret"], 75/85 * .999**2 - 1)
    for length in range(1, len(frame) + 1):
        prefix = _one_strategy(frame.iloc[:length], *args, entry_floor_col="ENTRY_FLOOR", terminal_policy="mark")
        np.testing.assert_array_equal(prefix["equity"], result["equity"][:length])
        assert prefix["trades"] == [t for t in result["trades"] if t["j"] < length]
    short = _one_strategy(frame.iloc[:2], *args, entry_floor_col="ENTRY_FLOOR", terminal_policy="mark")
    assert short["state"]["pending_sell_reason"] == "entry_floor" and not short["trades"]
    terminal = _one_strategy(frame.iloc[:2], *args, entry_floor_col="ENTRY_FLOOR")
    assert terminal["trades"][0]["exit_reason"] == "terminal"
    assert np.isclose(terminal["trades"][0]["ret"], 88/85 * .999**2 - 1)
    crossed = frame.copy(); crossed.loc[1, "CLOSE"] = 80.
    for s, stop, expected in ((True, .01, "signal"), (False, .01, "hard_stop"), (False, None, "entry_floor")):
        crossed.loc[1, "S_SIGNAL"] = s
        actual = _one_strategy(crossed, *args, trail=.20, hard_stop=stop,
                               entry_floor_col="ENTRY_FLOOR", entry_exit_cols=("USE_EXTRA", "EXTRA"))
        assert actual["trades"][0]["exit_reason"] == expected


def test_r17_omitted_or_empty_floor_exactly_preserves_frozen_engine_for_core_prices():
    import importlib.util
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.backtest.engine import _one_strategy
    from gcn.recipes.gcn_main import compute_ehopt10
    source = ROOT / "reports/gcn-historical-r16-20260905/results/source_snapshot/gcn/backtest/engine.py"
    spec = importlib.util.spec_from_file_location("r16_frozen_engine", source)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    frames, _ = load_snapshot(ROOT / "reports/signal-audit-v5-review-20260904")
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol], version="v5")
        frame["EMPTY_FLOOR"] = np.nan
        for settings in ({"trail": .20}, {"trail": .20, "profit_keep": .50},
                         {"hard_stop": .05, "terminal_policy": "mark"}):
            args = (["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"], .001, None)
            expected = old._one_strategy(frame, *args, **settings)
            for optional in ({}, {"entry_floor_col": None}, {"entry_floor_col": "EMPTY_FLOOR"}):
                actual = _one_strategy(frame, *args, **settings, **optional)
                np.testing.assert_array_equal(actual["equity"], expected["equity"])
                np.testing.assert_array_equal(actual["held"], expected["held"])
                assert actual["trades"] == expected["trades"] and actual["state"] == expected["state"]


def test_r17_candidate_keeps_all_native_signals_and_only_marks_pure_jf_signal_low():
    from gcn.backtest.signal_research_r17 import candidate_signals, RULES, CHALLENGERS
    frame = pd.DataFrame({"LOW": [90., 89., 88., 95., 98.],
                          "B_SIGNAL": [True, True, False, False, False],
                          "ICON_JUEFAN": [True, False, True, True, False], "S_SIGNAL": False},
                         index=pd.bdate_range("2025-01-01", periods=5))
    original = frame.copy(deep=True)
    result = candidate_signals(frame)
    assert list(result) == list(RULES) == ["v5", "JF-low-invalidation"]
    assert CHALLENGERS == ("JF-low-invalidation",)
    assert result["v5"].ENTRY_FLOOR.isna().all()
    assert result[CHALLENGERS[0]].ENTRY_FLOOR.iloc[:2].isna().all()  # B wins a collision.
    assert result[CHALLENGERS[0]].ENTRY_FLOOR.iloc[2:4].tolist() == [88., 95.]
    assert pd.isna(result[CHALLENGERS[0]].ENTRY_FLOOR.iloc[4])
    columns = ["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]
    for signals in result.values():
        pd.testing.assert_frame_equal(signals[columns], frame[columns])
        assert signals[["ENTRY_STOP", "ENTRY_LIMIT"]].isna().all().all()
        assert not signals[["USE_EXTRA", "EXTRA_EXIT"]].any().any()
    for length in range(1, len(frame) + 1):
        prefix = candidate_signals(frame.iloc[:length])
        for rule in RULES:
            pd.testing.assert_frame_equal(prefix[rule], result[rule].iloc[:length])
    pd.testing.assert_frame_equal(frame, original)


def test_r17_training_checker_enforces_coverage_explicitly_without_candidate_name_heuristics():
    from gcn.backtest.signal_research_r17 import candidate_failures, CHALLENGERS
    from gcn.backtest.signal_research_r4 import _choose_training
    table = pd.read_csv(ROOT / "reports/gcn-historical-r15-20260905/results/training.csv")
    base = table[table.rule.eq("v5")].iloc[0].to_dict()
    candidate = {**base, "rule": CHALLENGERS[0], "buy_covered": base["buy_covered"] - 1}
    assert candidate_failures(candidate, base) == ["buy_covered"]
    assert _choose_training([base, candidate], CHALLENGERS, failure_checker=candidate_failures) is None
    candidate["buy_covered"] = base["buy_covered"]
    assert not candidate_failures(candidate, base)
    assert _choose_training([base, candidate], CHALLENGERS, failure_checker=candidate_failures) == CHALLENGERS[0]
    candidate["buy_covered"] = np.nan
    assert "buy_covered" in candidate_failures(candidate, base)
    candidate["entry_events"] = 0
    assert "entry_coverage" in candidate_failures(candidate, base)


def test_r17_training_archives_explicit_floor_and_reconciles_native_control_and_actual_triggers(tmp_path, monkeypatch):
    import hashlib
    import json
    import pytest
    from gcn.backtest import signal_research_r17 as research
    from gcn.backtest.signal_research_r17 import run_training
    from gcn.backtest.historical_research import load_snapshot
    from gcn.recipes.gcn_main import compute_ehopt10
    shared = research._run_training
    def checked(*args, **kwargs):
        assert kwargs["entry_floor_col"] == "ENTRY_FLOOR"
        assert kwargs["failure_checker"] is research.candidate_failures
        return shared(*args, **kwargs)
    monkeypatch.setattr(research, "_run_training", checked)
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    decision = run_training(snapshot, tmp_path)
    rows = pd.read_csv(tmp_path / "training.csv").set_index("rule")
    assert list(rows.index) == list(research.RULES)
    failures = research.candidate_failures(rows.iloc[1].to_dict(), rows.iloc[0].to_dict())
    assert decision["failures"] == {research.CHALLENGERS[0]: failures}
    assert decision["selected"] == (None if failures else research.CHALLENGERS[0])
    assert decision["recommended"] == "v5" and not decision["production_changed"]
    trades = pd.read_csv(tmp_path / "trades.csv")
    original = pd.read_csv(ROOT / "reports/gcn-historical-r15-20260905/results/trades.csv")
    fields = ["symbol", "entry_date", "exit_date", "return_pct", "hold_days", "exit_reason", "peak_close_pct"]
    pd.testing.assert_frame_equal(trades.loc[trades.rule.eq("v5"), fields].reset_index(drop=True),
                                  original.loc[original.rule.eq("v5"), fields].reset_index(drop=True), check_exact=True)
    assert trades.entry_origin.eq("v5").all()
    assert trades.entry_stop_pct.isna().all() and trades.entry_limit.isna().all()
    assert not trades.use_extra_exit.any()
    assert trades.loc[trades.rule.eq("v5"), "entry_floor_price"].isna().all()
    frames, quality = load_snapshot(snapshot)
    selected = trades[trades.rule.eq(research.CHALLENGERS[0])]
    assert selected.exit_reason.eq("entry_floor").any()
    for symbol in selected.symbol.unique():
        frame = compute_ehopt10(frames[symbol].loc[:"2024-08-26"], version="v5")
        base = frame.loc["2021-08-27":"2024-08-26"]
        for row in selected[selected.symbol.eq(symbol)].itertuples():
            signal = frame.loc[row.entry_signal_date]
            enabled = signal.ICON_JUEFAN and not signal.B_SIGNAL
            assert row.entry_b == bool(signal.B_SIGNAL) and row.entry_jf == bool(signal.ICON_JUEFAN)
            assert pd.notna(row.entry_floor_price) == bool(enabled)
            if enabled:
                assert np.isclose(row.entry_floor_price, signal.LOW, rtol=1e-12)
            i = base.index.get_loc(pd.Timestamp(row.entry_date))
            terminal = row.exit_reason == "terminal"
            j = len(base) if terminal else base.index.get_loc(pd.Timestamp(row.exit_date))
            assert frame.index[frame.index.get_loc(pd.Timestamp(row.entry_date))-1].date().isoformat() == row.entry_signal_date
            exit_price = base.CLOSE.iloc[-1] if terminal else base.OPEN.iloc[j]
            assert np.isclose(row.return_pct, (exit_price/base.OPEN.iloc[i] * .999**2-1)*100)
            if row.exit_reason == "entry_floor":
                assert enabled and base.CLOSE.iloc[j-1] < signal.LOW and not base.S_SIGNAL.iloc[j-1]
                assert base.CLOSE.iloc[i:j-1].ge(signal.LOW).all()
    events = pd.read_csv(tmp_path / "events.csv")
    pd.testing.assert_frame_equal(events[events.rule.eq("v5")].drop(columns="rule").reset_index(drop=True),
                                  events[events.rule.eq(research.CHALLENGERS[0])].drop(columns="rule").reset_index(drop=True))
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    assert manifest["entry_floor_col"] == "ENTRY_FLOOR" and manifest["source_quality"] == quality
    assert manifest["parent_manifest_sha256"] == hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((tmp_path / "source_snapshot" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        run_training(snapshot, tmp_path)
