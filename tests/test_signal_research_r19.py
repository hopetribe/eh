"""r19：持仓内连续破底确认，不改变原生信号与旧默认执行。"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_r19_floor_counts_only_consecutive_held_closes_and_resets_on_reclaim_or_equality():
    from gcn.backtest.engine import _one_strategy
    frame = pd.DataFrame({
        "OPEN": [100., 100., 100., 85., 90., 90., 90., 90., 90., 80., 90., 90.],
        "CLOSE": [88., 87., 89., 89., 91., 89., 90., 89., 88., 94., 94., 94.],
        "LOW": 75., "B_SIGNAL": False, "ICON_JUEFAN": False, "S_SIGNAL": False,
        "ENTRY_FLOOR": np.nan,
    }, index=pd.bdate_range("2025-01-01", periods=12))
    frame.loc[frame.index[[2, 5]], "ICON_JUEFAN"] = True
    frame.loc[frame.index[[2, 5]], "ENTRY_FLOOR"] = [90., 70.]
    result = _one_strategy(frame, ["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"], .001, None,
                           trail=.20, entry_floor_col="ENTRY_FLOOR", entry_floor_confirm_bars=2)
    assert [(t["i"], t["j"], t["hold"], t["exit_reason"]) for t in result["trades"]] == [
        (3, 9, 6, "entry_floor")]
    assert np.isclose(result["trades"][0]["ret"], 80/85 * .999**2 - 1)
    assert result["held"].tolist() == [False]*3 + [True]*6 + [False]*3
    assert result["state"]["status"] == "flat"


def test_r19_floor_confirmation_requires_positive_non_boolean_integer():
    import pytest
    from gcn.backtest.engine import _one_strategy
    frame = pd.DataFrame({"OPEN": 100., "CLOSE": 95., "B_SIGNAL": [True, False, False],
                          "S_SIGNAL": False, "ENTRY_FLOOR": [90., np.nan, np.nan]})
    for value in (None, 0, -1, 1.5, 2., True, np.bool_(False), "2", np.nan, np.inf, [], {}):
        with pytest.raises(ValueError, match="entry_floor_confirm_bars"):
            _one_strategy(frame, ["B_SIGNAL"], ["S_SIGNAL"], .001, None,
                           entry_floor_col="ENTRY_FLOOR", entry_floor_confirm_bars=value)
    for value in (1, 2, np.int64(2)):
        result = _one_strategy(frame, ["B_SIGNAL"], ["S_SIGNAL"], .001, None,
                               entry_floor_col="ENTRY_FLOOR", entry_floor_confirm_bars=value)
        assert result["trades"][0]["exit_reason"] == "terminal"


def test_r19_counter_clears_between_trades_preserves_exit_priority_and_mark_prefixes():
    from gcn.backtest.engine import _one_strategy
    frame = pd.DataFrame({
        "OPEN": [100., 100., 90., 80., 100., 100., 90., 80., 70., 75.],
        "CLOSE": [80., 89., 88., 89., 89., 91., 89., 88., 75., 80.],
        "B_SIGNAL": [True, False, False, True, False, False, False, False, False, False],
        "S_SIGNAL": False, "ENTRY_FLOOR": [90., np.nan, np.nan, 90., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    })
    args = (["B_SIGNAL"], ["S_SIGNAL"], .001, None)
    settings = {"trail": .20, "entry_floor_col": "ENTRY_FLOOR", "entry_floor_confirm_bars": 2}
    result = _one_strategy(frame, *args, **settings, terminal_policy="mark")
    assert [(t["i"], t["j"], t["exit_reason"]) for t in result["trades"]] == [
        (1, 3, "entry_floor"), (4, 8, "entry_floor")]
    for length in range(1, len(frame)+1):
        prefix = _one_strategy(frame.iloc[:length], *args, **settings, terminal_policy="mark")
        np.testing.assert_array_equal(prefix["equity"], result["equity"][:length])
        np.testing.assert_array_equal(prefix["held"], result["held"][:length])
        assert prefix["trades"] == [t for t in result["trades"] if t["j"] < length]
    one = _one_strategy(frame.iloc[:2], *args, **settings, terminal_policy="mark")
    two = _one_strategy(frame.iloc[:3], *args, **settings, terminal_policy="mark")
    assert one["state"]["status"] == "open" and one["state"]["pending_sell_reason"] is None
    assert two["state"]["status"] == "pending_exit" and two["state"]["pending_sell_reason"] == "entry_floor"
    terminal = _one_strategy(frame.iloc[:3], *args, **settings)
    assert terminal["trades"][0]["exit_reason"] == "terminal"
    assert np.isclose(terminal["trades"][0]["ret"], .88 * .999**2 - 1)
    for signal, stop, expected in ((True, .12, "signal"), (False, .12, "hard_stop"), (False, None, "entry_floor")):
        crossed = frame.copy(); crossed.loc[2, "S_SIGNAL"] = signal
        actual = _one_strategy(crossed, *args, **settings, hard_stop=stop)
        assert actual["trades"][0]["j"] == 3 and actual["trades"][0]["exit_reason"] == expected
    waiting = frame.copy(); waiting.loc[[1, 2], "CLOSE"] = [110., 87.]
    actual = _one_strategy(waiting, *args, **settings)
    assert actual["trades"][0]["j"] == 3 and actual["trades"][0]["exit_reason"] == "trail"
    waiting.loc[2, "S_SIGNAL"] = True
    actual = _one_strategy(waiting, *args, **settings)
    assert actual["trades"][0]["exit_reason"] == "signal"


def test_r19_default_one_bar_and_disabled_floor_exactly_preserve_frozen_r18_engine():
    import importlib.util
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.backtest.signal_research_r18 import candidate_signals, CHALLENGERS
    from gcn.backtest.engine import _one_strategy
    from gcn.recipes.gcn_main import compute_ehopt10
    spec = importlib.util.spec_from_file_location("r18_frozen_engine", ROOT / "reports/gcn-historical-r18-20260905/results/source_snapshot/gcn/backtest/engine.py")
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    frames, _ = load_snapshot(ROOT / "reports/signal-audit-v5-review-20260904")
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:"2024-08-26"], version="v5")
        frame["ENTRY_FLOOR"] = candidate_signals(frame)[CHALLENGERS[0]].ENTRY_FLOOR
        args = (["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"], .001, None)
        for column in (None, "ENTRY_FLOOR"):
            expected = old._one_strategy(frame, *args, trail=.20, entry_floor_col=column)
            options = [{}, {"entry_floor_confirm_bars": 1}]
            if column is None:
                options.append({"entry_floor_confirm_bars": 2})
            for optional in options:
                actual = _one_strategy(frame, *args, trail=.20, entry_floor_col=column, **optional)
                np.testing.assert_array_equal(actual["equity"], expected["equity"])
                np.testing.assert_array_equal(actual["held"], expected["held"])
                assert actual["trades"] == expected["trades"] and actual["state"] == expected["state"]


def test_r19_candidate_reuses_exact_r18_native_prices_and_only_changes_execution_confirmation():
    from gcn.backtest.signal_research_r19 import candidate_signals, RULES, CHALLENGERS, CONFIRM_BARS
    from gcn.backtest.signal_research_r18 import candidate_signals as previous, CHALLENGERS as OLD
    frame = pd.DataFrame({"LOW": [90., 88., 95., 97., 89., 90.],
                          "B_SIGNAL": [False, False, False, True, False, False],
                          "ICON_JUEFAN": [True, False, True, True, True, False], "S_SIGNAL": False})
    saved = frame.copy(deep=True)
    old, new = previous(frame), candidate_signals(frame)
    assert list(new) == list(RULES) == ["v5", "JF-base-low-confirm2"]
    assert CHALLENGERS == ("JF-base-low-confirm2",) and CONFIRM_BARS == 2
    pd.testing.assert_frame_equal(new["v5"], old["v5"], check_exact=True)
    pd.testing.assert_frame_equal(new[CHALLENGERS[0]], old[OLD[0]], check_exact=True)
    assert pd.isna(new[CHALLENGERS[0]].ENTRY_FLOOR.iloc[3])  # B wins the collision.
    for length in range(1, len(frame)+1):
        prefix = candidate_signals(frame.iloc[:length])
        for rule in RULES:
            pd.testing.assert_frame_equal(prefix[rule], new[rule].iloc[:length], check_exact=True)
    pd.testing.assert_frame_equal(frame, saved)


def test_r19_training_binds_two_bar_configuration_and_reconciles_first_confirmed_breaks(tmp_path, monkeypatch):
    import hashlib
    import json
    import pytest
    from gcn.backtest import signal_research_r19 as research
    from gcn.backtest.signal_research_r19 import run_training
    from gcn.backtest.signal_research_r17 import candidate_failures
    from gcn.backtest.historical_research import load_snapshot
    from gcn.recipes.gcn_main import compute_ehopt10
    shared = research._run_training
    def checked(*args, **kwargs):
        assert kwargs["entry_floor_confirm_bars"] == 2 and kwargs["entry_floor_col"] == "ENTRY_FLOOR"
        assert kwargs["failure_checker"] is candidate_failures and kwargs["controls"] == ("v5",)
        return shared(*args, **kwargs)
    monkeypatch.setattr(research, "_run_training", checked)
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    decision = run_training(snapshot, tmp_path)
    rows = pd.read_csv(tmp_path / "training.csv").set_index("rule")
    assert list(rows.index) == list(research.RULES)
    failures = candidate_failures(rows.iloc[1].to_dict(), rows.iloc[0].to_dict())
    assert decision["failures"] == {research.CHALLENGERS[0]: failures}
    assert decision["selected"] == (None if failures else research.CHALLENGERS[0])
    assert not decision["production_changed"] and decision["recommended"] == "v5"
    frozen = ROOT / "reports/gcn-historical-r18-20260905/results"
    base_rows = pd.read_csv(frozen / "training.csv").set_index("rule")
    pd.testing.assert_series_equal(rows.loc["v5"], base_rows.loc["v5"], check_exact=True)
    trades = pd.read_csv(tmp_path / "trades.csv")
    original = pd.read_csv(frozen / "trades.csv")
    pd.testing.assert_frame_equal(trades.loc[trades.rule.eq("v5"), original.columns].reset_index(drop=True),
                                  original[original.rule.eq("v5")].reset_index(drop=True), check_exact=True)
    assert trades.entry_floor_confirm_bars.eq(2).all() and trades.entry_origin.eq("v5").all()
    assert trades.entry_stop_pct.isna().all() and trades.entry_limit.isna().all() and not trades.use_extra_exit.any()
    frames, quality = load_snapshot(snapshot)
    for row in trades[trades.rule.eq(research.CHALLENGERS[0])].itertuples():
        frame = compute_ehopt10(frames[row.symbol].loc[:"2024-08-26"], version="v5")
        signal_pos = frame.index.get_loc(pd.Timestamp(row.entry_date)) - 1
        signal = frame.iloc[signal_pos]
        enabled = bool(signal.ICON_JUEFAN and not signal.B_SIGNAL)
        assert row.entry_b == bool(signal.B_SIGNAL) and row.entry_jf == bool(signal.ICON_JUEFAN)
        assert row.entry_signal_date == frame.index[signal_pos].date().isoformat()
        assert pd.notna(row.entry_floor_price) == enabled
        floor = float(frame.LOW.iloc[signal_pos-2:signal_pos+1].min())
        if enabled:
            assert signal_pos >= 2 and np.isclose(row.entry_floor_price, floor, rtol=1e-12)
        base = frame.loc["2021-08-27":"2024-08-26"]
        i = base.index.get_loc(pd.Timestamp(row.entry_date))
        terminal = row.exit_reason == "terminal"
        j = len(base) if terminal else base.index.get_loc(pd.Timestamp(row.exit_date))
        exit_price = base.CLOSE.iloc[-1] if terminal else base.OPEN.iloc[j]
        assert np.isclose(row.return_pct, (exit_price/base.OPEN.iloc[i] * .999**2 - 1)*100)
        if row.exit_reason == "entry_floor":
            below = base.CLOSE.iloc[i:j].lt(floor).to_numpy()
            hits = np.flatnonzero(below[1:] & below[:-1]) + 1
            assert enabled and len(hits) and hits[0] == j-i-1
            assert not base.S_SIGNAL.iloc[j-1]
    events = pd.read_csv(tmp_path / "events.csv")
    pd.testing.assert_frame_equal(events[events.rule.eq("v5")].drop(columns="rule").reset_index(drop=True),
                                  events[events.rule.eq(research.CHALLENGERS[0])].drop(columns="rule").reset_index(drop=True))
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    assert manifest["entry_floor_confirm_bars"] == 2 and manifest["entry_floor_col"] == "ENTRY_FLOOR"
    assert manifest["source_quality"] == quality
    assert manifest["parent_manifest_sha256"] == hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    assert manifest["protocol_sha256"] == hashlib.sha256((ROOT / "reports/gcn-historical-r19-20260905/protocol.md").read_bytes()).hexdigest()
    assert {f"gcn/backtest/signal_research_r{n}.py" for n in (17, 18, 19)} <= set(manifest["algorithm_sources"])
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((tmp_path / "source_snapshot" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        run_training(snapshot, tmp_path)


def test_r19_shared_training_keeps_legacy_r18_outputs_and_default_configuration_schema(tmp_path):
    import json
    from gcn.backtest.signal_research_r18 import run_training
    frozen = ROOT / "reports/gcn-historical-r18-20260905/results"
    run_training(ROOT / "reports/signal-audit-v5-review-20260904", tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    assert "entry_floor_confirm_bars" not in manifest
    assert "entry_floor_confirm_bars" not in pd.read_csv(tmp_path / "trades.csv").columns
    for name in ("training.csv", "trades.csv", "events.csv", "missed_turns.csv", "decision.json", "protocol.md"):
        assert (tmp_path / name).read_bytes() == (frozen / name).read_bytes(), name


def test_r19_confirmation_cost_pressure_reprices_same_synthetic_order_path():
    from gcn.backtest.historical_research import evaluate_rule
    frame = pd.DataFrame({"OPEN": [100., 100., 95., 90., 90., 90., 80., 80.],
                          "CLOSE": [100., 95., 89., 91., 89., 88., 80., 82.],
                          "B_SIGNAL": False, "ICON_JUEFAN": [True]+[False]*7, "S_SIGNAL": False,
                          "ENTRY_FLOOR": [90.]+[np.nan]*7}, index=pd.bdate_range("2025-01-01", periods=8))
    prepared = {"SYNTHETIC": {"frame": frame, "rules": {"test": frame.copy()}}}
    baseline = evaluate_rule(prepared, "test", frame.index[0], frame.index[-1],
                             entry_floor_col="ENTRY_FLOOR", entry_floor_confirm_bars=2, include_positions=True)
    stress = evaluate_rule(prepared, "test", frame.index[0], frame.index[-1], cost=.0025,
                           entry_floor_col="ENTRY_FLOOR", entry_floor_confirm_bars=2, include_positions=True)
    pd.testing.assert_frame_equal(baseline["positions"], stress["positions"])
    assert len(baseline["trades"]) == len(stress["trades"]) == 1
    for result, cost in ((baseline, .001), (stress, .0025)):
        trade = result["trades"][0]
        assert trade["entry_date"] == frame.index[1].date().isoformat()
        assert trade["exit_date"] == frame.index[6].date().isoformat() and trade["exit_reason"] == "entry_floor"
        assert np.isclose(trade["return_pct"], (.8*(1-cost)**2-1)*100)
