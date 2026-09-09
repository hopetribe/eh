"""r33原S卖缺口自然机制的训练筛查。"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_r33_first_trigger_uses_real_next_open_and_respects_old_exit_boundary():
    from gcn.backtest.signal_research_r33 import first_held_trigger

    index = pd.bdate_range("2024-01-02", periods=4)
    frame = pd.DataFrame(
        {
            "OPEN": [100.0, 101.0, 90.0, 80.0],
            "CLOSE": [100.0, 95.0, 85.0, 75.0],
        },
        index=index,
    )
    trade = {
        "symbol": "TEST",
        "trade_id": "TEST:2024-01-02",
        "entry_date": index[0].date().isoformat(),
        "exit_date": index[3].date().isoformat(),
        "exit_reason": "trail",
        "return_pct": -20.15992,
        "entry_b": True,
        "entry_jf": False,
    }
    signal = pd.Series([False, True, True, False], index=index)

    row = first_held_trigger(frame, trade, signal, "example")
    assert row["trigger_date"] == index[1].date().isoformat()
    assert row["counterfactual_exit_date"] == index[2].date().isoformat()
    assert np.isclose(row["counterfactual_return_pct"], (90 / 100 * 0.999**2 - 1) * 100)
    assert np.isclose(
        row["local_delta_pp"], row["counterfactual_return_pct"] - trade["return_pct"]
    )

    old_exit_day_only = pd.Series([False, False, True, False], index=index)
    assert first_held_trigger(frame, trade, old_exit_day_only, "example") is None


def test_r33_natural_mechanisms_use_fixed_existing_boundaries():
    from gcn.backtest.signal_research_r33 import natural_mechanisms

    index = pd.bdate_range("2024-01-02", periods=6)
    frame = pd.DataFrame(
        {
            "CLOSE": [10.0, 10.0, 10.0, 10.0, 12.0, 8.0],
            "MID": [9.0, 9.0, 9.0, 9.0, 9.0, 10.0],
            "LOWER": [8.0, 8.0, 8.0, 8.0, 8.0, 9.0],
            "UPPER": [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            "获利筹": [70.0, 70.0, 70.0, 70.0, 90.0, 80.0],
            "V3": [65.0, 65.0, 65.0, 65.0, 85.0, 85.0],
            "MACD": [1.0, 1.0, 1.0, 1.0, 2.0, 1.0],
            "RSI1": [70.0, 70.0, 70.0, 70.0, 80.0, 90.0],
            "S_CONDITION": [False, False, False, False, True, False],
            "S_CONDITION_DOWN_LONG": False,
            "S_CONDITION_UP_LAST": False,
            "NINE2_SELL_SIGNAL": [False, False, False, False, True, False],
            "NINE2_UP_9": [False, False, False, False, False, True],
        },
        index=index,
    )
    signals = natural_mechanisms(frame)

    assert list(signals) == [
        "close-cross-mid",
        "close-cross-lower",
        "ma5-cross-mid",
        "s-condition",
        "s-condition-union",
        "nine2-sell",
        "nine2-up9",
        "chip-rollover",
        "chip-rollover-after85",
        "macd-turn-down",
        "upper-rejection-score",
        "rsi-cross85",
    ]
    for name in ("close-cross-mid", "close-cross-lower", "chip-rollover",
                 "chip-rollover-after85", "macd-turn-down",
                 "rsi-cross85"):
        assert signals[name].iloc[-1]
    assert not signals["ma5-cross-mid"].iloc[-1]  # 严格边界，相等不下穿。
    assert not signals["upper-rejection-score"].iloc[-1]
    assert signals["s-condition"].iloc[-2]
    assert signals["nine2-sell"].iloc[-2]


def test_r33_real_training_screen_rejects_independent_rules_and_marks_r32_synonym():
    from gcn.backtest.signal_research_r33 import training_screen

    events, summary = training_screen(
        ROOT / "reports/signal-audit-v5-review-20260904",
        ROOT / "reports/gcn-historical-r32-20260908/training",
    )
    rows = summary.set_index("mechanism")

    assert len(events[events.mechanism.eq("close-cross-mid")]) == 37
    assert rows.loc["close-cross-mid", "original_losses"] == 19
    assert rows.loc["close-cross-mid", "original_winners"] == 18
    assert np.isclose(rows.loc["close-cross-mid", "local_delta_pp"], -67.820, atol=0.001)
    assert rows.loc["s-condition", "original_losses"] == 0
    assert rows.loc["s-condition", "original_winners"] == 7
    assert rows.loc["nine2-sell", "held_triggers"] == 4
    assert rows.loc["ma5-cross-mid-loss", "held_triggers"] == 16
    assert rows.loc["ma5-cross-mid-loss", "r32_overlap_trades"] == 15
    assert np.isclose(rows.loc["ma5-cross-mid-loss", "r32_overlap_pct"], 93.75)
    assert set(rows["status"]) == {"rejected"}


def test_r33_runner_writes_scoped_no_candidate_archive(tmp_path):
    from gcn.backtest.signal_research_r33 import run_training_screen

    output = tmp_path / "r33"
    result = run_training_screen(
        ROOT / "reports/signal-audit-v5-review-20260904",
        ROOT / "reports/gcn-historical-r32-20260908/training",
        output,
    )

    assert result["decision"] == {
        "research_version": "gcn-historical-r33",
        "stage": "training_mechanism_screen",
        "status": "no_candidate",
        "candidate": None,
        "reason": "no_independent_s_exit_mechanism",
        "candidate_backtest_run": False,
        "validation_run": False,
        "production_changed": False,
    }
    assert set(result["manifest"]["outputs"]) == {
        "events.csv", "summary.csv", "decision.json", "protocol.md"
    }
    assert result["manifest"]["r32_training_manifest_sha256"]
    for name in (*result["manifest"]["outputs"], "manifest.json"):
        assert (output / name).is_file()


def test_r33_formal_training_archive_reproduces_byte_for_byte(tmp_path):
    from gcn.backtest.signal_research_r33 import run_training_screen

    output = tmp_path / "reproduced"
    result = run_training_screen(
        ROOT / "reports/signal-audit-v5-review-20260904",
        ROOT / "reports/gcn-historical-r32-20260908/training",
        output,
    )
    formal = ROOT / "reports/gcn-historical-r33-20260908/training"
    for name in (*result["manifest"]["outputs"], "manifest.json"):
        assert (output / name).read_bytes() == (formal / name).read_bytes()
