"""r18仅改变绝反固定底部价，不新增信号或模拟引擎。"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_r18_uses_exact_native_three_bar_low_including_signal_bar_without_future_or_collision():
    from gcn.backtest.signal_research_r18 import candidate_signals, RULES, CHALLENGERS
    from gcn.backtest.signal_research_r17 import candidate_signals as old_signals
    frame = pd.DataFrame({"LOW": [5., 12., 11., 10., 7., 12., 14., 13., 9., 8.],
                          "B_SIGNAL": [False, False, False, True, False, True, False, False, False, False],
                          "ICON_JUEFAN": [True, True, True, True, True, False, True, True, False, True],
                          "S_SIGNAL": False}, index=pd.bdate_range("2025-01-01", periods=10))
    original = frame.copy(deep=True)
    result = candidate_signals(frame)
    assert list(result) == list(RULES) == ["v5", "JF-base-low-invalidation"]
    assert CHALLENGERS == ("JF-base-low-invalidation",)
    pd.testing.assert_frame_equal(result["v5"], old_signals(frame)["v5"])
    candidate = result[CHALLENGERS[0]]
    expected = pd.Series([np.nan, np.nan, 5., np.nan, 7., np.nan, 7., 12., np.nan, 8.],
                         index=frame.index, name="ENTRY_FLOOR")
    pd.testing.assert_series_equal(candidate.ENTRY_FLOOR, expected)
    pd.testing.assert_frame_equal(candidate.drop(columns="ENTRY_FLOOR"), result["v5"].drop(columns="ENTRY_FLOOR"))
    assert candidate.ENTRY_FLOOR.dropna().le(frame.LOW.loc[candidate.ENTRY_FLOOR.notna()]).all()
    for length in range(1, len(frame) + 1):
        prefix = candidate_signals(frame.iloc[:length])
        for rule in RULES:
            pd.testing.assert_frame_equal(prefix[rule], result[rule].iloc[:length])
    pd.testing.assert_frame_equal(frame, original)


def test_r18_native_floor_stays_locked_despite_later_jf_and_clears_before_b_collision_entry():
    from gcn.backtest.signal_research_r18 import candidate_signals, CHALLENGERS
    from gcn.backtest.engine import _one_strategy
    frame = pd.DataFrame({
        "OPEN": [100., 100., 100., 100., 95., 88., 77., 90., 83., 80., 82.],
        "CLOSE": [100., 100., 100., 90., 85., 79., 88., 85., 79., 81., 84.],
        "LOW": [80., 90., 95., 89., 70., 78., 76., 82., 78., 79., 81.],
        "B_SIGNAL": False, "ICON_JUEFAN": False, "S_SIGNAL": False,
    }, index=pd.bdate_range("2025-01-01", periods=11))
    frame.loc[frame.index[[2, 4, 6]], "ICON_JUEFAN"] = True
    frame.loc[frame.index[6], "B_SIGNAL"] = True
    frame.loc[frame.index[9], "S_SIGNAL"] = True
    signals = candidate_signals(frame)[CHALLENGERS[0]]
    assert signals.ENTRY_FLOOR.iloc[2] == 80. and signals.ENTRY_FLOOR.iloc[4] == 70.
    frame["ENTRY_FLOOR"] = signals.ENTRY_FLOOR
    args = (["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"], .001, None)
    result = _one_strategy(frame, *args, trail=.20, entry_floor_col="ENTRY_FLOOR", terminal_policy="mark")
    assert [(t["i"], t["j"], t["exit_reason"]) for t in result["trades"]] == [
        (3, 6, "entry_floor"), (7, 10, "signal")]
    np.testing.assert_allclose([t["ret"] for t in result["trades"]], np.array([77/100, 82/90]) * .999**2 - 1)
    assert frame.CLOSE.iloc[3] < frame.LOW.iloc[2] and frame.CLOSE.iloc[3:5].ge(80).all()
    assert frame.CLOSE.iloc[8] < 80  # The previous trade's floor must not leak into the B entry.
    for length in range(1, len(frame) + 1):
        prefix = frame.iloc[:length].copy()
        prefix["ENTRY_FLOOR"] = candidate_signals(prefix)[CHALLENGERS[0]].ENTRY_FLOOR
        actual = _one_strategy(prefix, *args, trail=.20, entry_floor_col="ENTRY_FLOOR", terminal_policy="mark")
        np.testing.assert_array_equal(actual["equity"], result["equity"][:length])
        assert actual["trades"] == [t for t in result["trades"] if t["j"] < length]


def test_r18_training_binds_native_base_price_and_reconciles_original_control_events_and_fills(tmp_path, monkeypatch):
    import hashlib
    import json
    import pytest
    from gcn.backtest import signal_research_r18 as research
    from gcn.backtest.signal_research_r18 import run_training
    from gcn.backtest.signal_research_r17 import candidate_failures
    from gcn.backtest.historical_research import load_snapshot
    from gcn.recipes.gcn_main import compute_ehopt10
    shared = research._run_training
    def checked(*args, **kwargs):
        assert kwargs["entry_floor_col"] == "ENTRY_FLOOR"
        assert kwargs["failure_checker"] is candidate_failures
        assert kwargs["controls"] == ("v5",) and kwargs["challengers"] == research.CHALLENGERS
        return shared(*args, **kwargs)
    monkeypatch.setattr(research, "_run_training", checked)
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    decision = run_training(snapshot, tmp_path)
    rows = pd.read_csv(tmp_path / "training.csv").set_index("rule")
    assert list(rows.index) == list(research.RULES)
    failures = candidate_failures(rows.iloc[1].to_dict(), rows.iloc[0].to_dict())
    assert decision["failures"] == {research.CHALLENGERS[0]: failures}
    assert decision["selected"] == (None if failures else research.CHALLENGERS[0])
    assert decision["recommended"] == "v5" and not decision["production_changed"]
    frozen = ROOT / "reports/gcn-historical-r17-20260905/results"
    baseline = pd.read_csv(frozen / "training.csv").set_index("rule")
    pd.testing.assert_series_equal(rows.loc["v5"], baseline.loc["v5"], check_exact=True)
    trades = pd.read_csv(tmp_path / "trades.csv")
    original = pd.read_csv(frozen / "trades.csv")
    pd.testing.assert_frame_equal(trades[trades.rule.eq("v5")].reset_index(drop=True),
                                  original[original.rule.eq("v5")].reset_index(drop=True), check_exact=True)
    assert trades.entry_origin.eq("v5").all()
    assert trades.entry_stop_pct.isna().all() and trades.entry_limit.isna().all()
    assert not trades.use_extra_exit.any()
    frames, quality = load_snapshot(snapshot)
    for row in trades[trades.rule.eq(research.CHALLENGERS[0])].itertuples():
        frame = compute_ehopt10(frames[row.symbol].loc[:"2024-08-26"], version="v5")
        signal_pos = frame.index.get_loc(pd.Timestamp(row.entry_date)) - 1
        signal = frame.iloc[signal_pos]
        enabled = signal.ICON_JUEFAN and not signal.B_SIGNAL
        assert row.entry_signal_date == frame.index[signal_pos].date().isoformat()
        assert row.entry_b == bool(signal.B_SIGNAL) and row.entry_jf == bool(signal.ICON_JUEFAN)
        assert pd.notna(row.entry_floor_price) == bool(enabled)
        floor = float(frame.LOW.iloc[signal_pos-2:signal_pos+1].min())
        if enabled:
            assert signal_pos >= 2 and np.isclose(row.entry_floor_price, floor, rtol=1e-12)
            assert row.entry_floor_price <= signal.LOW
        base = frame.loc["2021-08-27":"2024-08-26"]
        i = base.index.get_loc(pd.Timestamp(row.entry_date))
        terminal = row.exit_reason == "terminal"
        j = len(base) if terminal else base.index.get_loc(pd.Timestamp(row.exit_date))
        exit_price = base.CLOSE.iloc[-1] if terminal else base.OPEN.iloc[j]
        assert np.isclose(row.return_pct, (exit_price/base.OPEN.iloc[i] * .999**2 - 1)*100)
        if row.exit_reason == "entry_floor":
            assert enabled and base.CLOSE.iloc[j-1] < floor and not base.S_SIGNAL.iloc[j-1]
            assert base.CLOSE.iloc[i:j-1].ge(floor).all()
    events = pd.read_csv(tmp_path / "events.csv")
    pd.testing.assert_frame_equal(events[events.rule.eq("v5")].drop(columns="rule").reset_index(drop=True),
                                  events[events.rule.eq(research.CHALLENGERS[0])].drop(columns="rule").reset_index(drop=True))
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    assert manifest["entry_floor_col"] == "ENTRY_FLOOR" and manifest["source_quality"] == quality
    assert manifest["parent_manifest_sha256"] == hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    assert manifest["protocol_sha256"] == hashlib.sha256((ROOT / "reports/gcn-historical-r18-20260905/protocol.md").read_bytes()).hexdigest()
    assert {"gcn/backtest/signal_research_r17.py", "gcn/backtest/signal_research_r18.py"} <= set(manifest["algorithm_sources"])
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((tmp_path / "source_snapshot" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        run_training(snapshot, tmp_path)


def test_r18_core_training_prefixes_recompute_from_raw_prices_without_future_floor():
    from gcn.backtest.signal_research_r18 import candidate_signals, RULES, CHALLENGERS
    from gcn.backtest.historical_research import CORE, load_snapshot
    from gcn.recipes.gcn_main import compute_ehopt10
    frames, _ = load_snapshot(ROOT / "reports/signal-audit-v5-review-20260904")
    for symbol in CORE:
        raw = frames[symbol].loc[:"2024-08-26"]
        frame = compute_ehopt10(raw, version="v5")
        expected = candidate_signals(frame)
        manual = pd.concat([frame.LOW, frame.LOW.shift(1), frame.LOW.shift(2)], axis=1).min(axis=1, skipna=False)
        manual = manual.where(frame.ICON_JUEFAN & ~frame.B_SIGNAL).rename("ENTRY_FLOOR")
        pd.testing.assert_series_equal(expected[CHALLENGERS[0]].ENTRY_FLOOR, manual)
        for length in (70, 256, len(raw)-1):
            actual = candidate_signals(compute_ehopt10(raw.iloc[:length], version="v5"))
            for rule in RULES:
                pd.testing.assert_frame_equal(actual[rule], expected[rule].iloc[:length], check_exact=True)
