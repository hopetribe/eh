"""r32持仓MACD零轴下穿且净亏的补充退出。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_r32_exits_at_real_next_open_after_strict_macd_cross_and_net_loss():
    from gcn.backtest.engine import _one_strategy

    frame = pd.DataFrame(
        {
            "OPEN": [100.0, 100.0, 90.0, 80.0],
            "CLOSE": [100.0, 90.0, 80.0, 80.0],
            "MACD": [1.0, -1.0, -2.0, -2.0],
            "B_SIGNAL": [True, False, False, False],
            "S_SIGNAL": False,
            "MACD_LOSS": [True, False, False, False],
        }
    )
    result = _one_strategy(
        frame,
        ["B_SIGNAL"],
        ["S_SIGNAL"],
        0.001,
        None,
        entry_macd_loss_col="MACD_LOSS",
    )

    assert [(trade["i"], trade["j"], trade["exit_reason"]) for trade in result["trades"]] == [
        (1, 2, "macd_loss")
    ]
    trade = result["trades"][0]
    assert trade["macd_loss_enabled"]
    assert trade["macd_loss_trigger_i"] == 1
    assert trade["macd_loss_prev"] == 1.0
    assert trade["macd_loss_current"] == -1.0
    assert np.isclose(trade["macd_loss_net_factor"], 90 / 100 * 0.999**2)
    assert np.isclose(trade["ret"], 90 / 100 * 0.999**2 - 1)


def test_r32_factory_preserves_native_signals_and_enables_all_real_entries(monkeypatch):
    import gcn.backtest.signal_research_r32 as research

    frame = pd.DataFrame(
        {
            "MACD": [1.0, -1.0, 1.0, -1.0],
            "B_SIGNAL": [True, False, True, False],
            "ICON_JUEFAN": [False, True, True, False],
            "S_SIGNAL": [False, False, True, False],
        }
    )
    original = frame[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]].copy()
    monkeypatch.setattr(research, "prior_signals", lambda source: {"v5": original.copy()})

    rules = research.candidate_signals(frame)
    assert list(rules) == ["v5", "held-macd-loss-exit"]
    assert not rules["v5"].ENTRY_MACD_LOSS.any()
    assert rules["held-macd-loss-exit"].ENTRY_MACD_LOSS.tolist() == [True, True, True, False]
    for signals in rules.values():
        pd.testing.assert_frame_equal(
            signals[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]], original
        )


def test_r32_evaluator_carries_macd_audit_and_reprices_same_orders(monkeypatch):
    from gcn.backtest.historical_research import evaluate_rule
    from gcn.backtest import signal_research_r32 as research

    index = pd.bdate_range("2024-01-02", periods=4)
    frame = pd.DataFrame(
        {
            "OPEN": [100.0, 100.0, 90.0, 80.0],
            "CLOSE": [100.0, 90.0, 80.0, 80.0],
            "LOW": [99.0, 89.0, 79.0, 79.0],
            "MACD": [1.0, -1.0, -2.0, -2.0],
            "B_SIGNAL": [True, False, False, False],
            "ICON_JUEFAN": False,
            "S_SIGNAL": False,
        },
        index=index,
    )
    native = frame[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]].copy()
    native["ENTRY_STOP"] = np.nan
    native["ENTRY_LIMIT"] = np.nan
    native["ENTRY_FLOOR"] = np.nan
    native["USE_EXTRA"] = False
    native["EXTRA_EXIT"] = False
    monkeypatch.setattr(research, "prior_signals", lambda source: {"v5": native.copy()})
    prepared = {"TEST": {"frame": frame, "rules": research.candidate_signals(frame)}}
    outputs = [
        evaluate_rule(
            prepared,
            research.CHALLENGERS[0],
            index[0],
            index[-1],
            fee,
            entry_macd_loss_col="ENTRY_MACD_LOSS",
            include_positions=True,
        )
        for fee in (0.001, 0.0025)
    ]

    for fee, output in zip((0.001, 0.0025), outputs):
        trade = output["trades"][0]
        assert trade["exit_reason"] == "macd_loss"
        assert trade["macd_loss_trigger_date"] == index[1].date().isoformat()
        assert np.isclose(trade["return_pct"], (90 / 100 * (1 - fee) ** 2 - 1) * 100)
    for key in outputs[0]["trades"][0]:
        if key != "return_pct":
            assert outputs[0]["trades"][0][key] == outputs[1]["trades"][0][key]


def test_r32_training_wrapper_binds_frozen_candidate_and_marker(monkeypatch, tmp_path):
    from gcn.backtest import signal_research_r32 as research

    captured = {}

    def inspect(*args, **kwargs):
        captured.update(kwargs)
        return {"checked": True}

    monkeypatch.setattr(research, "_run_training", inspect)
    assert research.run_training(tmp_path / "snapshot", tmp_path / "output") == {
        "checked": True
    }
    assert captured["research_version"] == "gcn-historical-r32"
    assert captured["protocol_relative"] == "reports/gcn-historical-r32-20260908/protocol.md"
    assert captured["candidate_builder"] is research.candidate_signals
    assert captured["challengers"] == research.CHALLENGERS
    assert captured["controls"] == research.CONTROLS
    assert captured["entry_macd_loss_col"] == "ENTRY_MACD_LOSS"


def test_r32_real_training_runs_full_candidate_chain(tmp_path):
    from gcn.backtest.signal_research_r32 import CHALLENGERS, run_training

    decision = run_training(
        ROOT / "reports/signal-audit-v5-review-20260904",
        tmp_path,
    )
    rows = pd.read_csv(tmp_path / "training.csv").set_index("rule")
    trades = pd.read_csv(tmp_path / "trades.csv")

    assert decision["research_version"] == "gcn-historical-r32"
    assert set(rows.index) == {"v5", CHALLENGERS[0]}
    exits = trades[
        trades.rule.eq(CHALLENGERS[0]) & trades.exit_reason.eq("macd_loss")
    ]
    assert len(exits) > 0
    assert exits.macd_loss_enabled.all()
    assert exits.macd_loss_trigger_date.notna().all()


def test_r32_mark_keeps_terminal_pending_exit_and_trigger_audit():
    from gcn.backtest.engine import _one_strategy

    frame = pd.DataFrame(
        {
            "OPEN": [100.0, 100.0],
            "CLOSE": [100.0, 90.0],
            "MACD": [0.0, -1.0],
            "B_SIGNAL": [True, False],
            "S_SIGNAL": False,
            "MACD_LOSS": [True, False],
        }
    )
    result = _one_strategy(
        frame,
        ["B_SIGNAL"],
        ["S_SIGNAL"],
        0.001,
        None,
        terminal_policy="mark",
        entry_macd_loss_col="MACD_LOSS",
    )

    assert result["state"]["status"] == "pending_exit"
    assert result["state"]["pending_sell_reason"] == "macd_loss"
    assert result["state"]["macd_loss"]["macd_loss_trigger_i"] == 1
    assert result["trades"] == []


def test_r32_all_false_marker_without_macd_is_exactly_disabled():
    from gcn.backtest.engine import _one_strategy

    frame = pd.DataFrame(
        {
            "OPEN": [100.0, 100.0, 110.0],
            "CLOSE": [100.0, 105.0, 110.0],
            "B_SIGNAL": [True, False, False],
            "S_SIGNAL": False,
            "MACD_LOSS": False,
        }
    )
    args = (["B_SIGNAL"], ["S_SIGNAL"], 0.001, None)
    expected = _one_strategy(frame, *args, terminal_policy="mark")
    actual = _one_strategy(
        frame,
        *args,
        terminal_policy="mark",
        entry_macd_loss_col="MACD_LOSS",
    )

    np.testing.assert_array_equal(actual["equity"], expected["equity"])
    np.testing.assert_array_equal(actual["held"], expected["held"])
    assert actual["trades"] == expected["trades"]
    assert actual["state"] == expected["state"]


def test_r32_rejects_nonnumeric_macd_with_feature_specific_error():
    import pytest
    from gcn.backtest.engine import _one_strategy

    frame = pd.DataFrame(
        {
            "OPEN": [100.0, 100.0],
            "CLOSE": [100.0, 90.0],
            "MACD": [1.0, "bad"],
            "B_SIGNAL": [True, False],
            "S_SIGNAL": False,
            "MACD_LOSS": [True, False],
        }
    )
    with pytest.raises(ValueError, match="MACD"):
        _one_strategy(
            frame,
            ["B_SIGNAL"],
            ["S_SIGNAL"],
            0.001,
            None,
            entry_macd_loss_col="MACD_LOSS",
        )


def test_r32_strict_boundaries_nonfinite_values_old_exit_priority_and_marker_contract():
    import pytest
    from gcn.backtest.engine import _one_strategy

    args = (["B_SIGNAL"], ["S_SIGNAL"], 0.001, None)

    def run(macd, closes, *, sell_at_cross=False):
        frame = pd.DataFrame(
            {
                "OPEN": [100.0, 100.0, 90.0, 80.0],
                "CLOSE": closes,
                "MACD": macd,
                "B_SIGNAL": [True, False, False, False],
                "S_SIGNAL": [False, sell_at_cross, False, False],
                "MACD_LOSS": [True, False, False, False],
            }
        )
        return _one_strategy(
            frame,
            *args,
            terminal_policy="mark",
            entry_macd_loss_col="MACD_LOSS",
        )

    assert run([0.0, 0.0, 0.0, 0.0], [100.0, 90.0, 80.0, 80.0])["state"]["status"] == "open"
    zero_previous = run([0.0, -1.0, -2.0, -3.0], [100.0, 90.0, 80.0, 80.0])
    assert zero_previous["trades"][0]["exit_reason"] == "macd_loss"
    assert run([-1.0, -2.0, -3.0, -4.0], [100.0, 90.0, 80.0, 80.0])["state"]["status"] == "open"
    assert run([0.0, -1.0, -2.0, -3.0], [100.0, 100.0 / 0.999**2, 80.0, 80.0])["state"]["status"] == "open"
    assert run([np.nan, -1.0, -2.0, -3.0], [100.0, 90.0, 80.0, 80.0])["state"]["status"] == "open"

    priority = run([0.0, -1.0, -2.0, -3.0], [100.0, 90.0, 80.0, 80.0], sell_at_cross=True)
    assert priority["trades"][0]["exit_reason"] == "signal"
    assert priority["trades"][0]["macd_loss_trigger_i"] is None

    base = pd.DataFrame(
        {
            "OPEN": [100.0, 100.0],
            "CLOSE": [100.0, 90.0],
            "MACD": [0.0, -1.0],
            "B_SIGNAL": [True, False],
            "S_SIGNAL": False,
            "MACD_LOSS": [True, False],
        }
    )
    for column in ("missing", 1, ["MACD_LOSS"]):
        with pytest.raises(ValueError, match="entry_macd_loss_col"):
            _one_strategy(base, *args, entry_macd_loss_col=column)
    for value in (0, 1, 0.1, "False", "", [True], {}, 1 + 0j):
        bad = base.copy()
        bad["MACD_LOSS"] = bad["MACD_LOSS"].astype(object)
        bad.at[0, "MACD_LOSS"] = value
        with pytest.raises(ValueError, match="entry_macd_loss_col"):
            _one_strategy(bad, *args, entry_macd_loss_col="MACD_LOSS")


def test_r32_formal_training_reproduces_byte_for_byte_and_preserves_entries(tmp_path):
    from gcn.backtest.signal_research_r32 import CHALLENGERS, run_training

    reproduced = tmp_path / "training"
    run_training(ROOT / "reports/signal-audit-v5-review-20260904", reproduced)
    formal = ROOT / "reports/gcn-historical-r32-20260908/training"
    manifest = json.loads((formal / "manifest.json").read_text())
    for name in list(manifest["outputs"]) + ["manifest.json"]:
        assert (reproduced / name).read_bytes() == (formal / name).read_bytes()

    rows = pd.read_csv(formal / "training.csv").set_index("rule")
    candidate = CHALLENGERS[0]
    assert rows.loc[candidate, "macd_loss_exits"] == 18
    assert rows.loc[candidate, "cagr"] > rows.loc["v5", "cagr"]
    assert rows.loc[candidate, "mdd"] < rows.loc["v5", "mdd"]
    assert rows.loc[candidate, "sharpe"] > rows.loc["v5", "sharpe"]
    assert rows.loc[candidate, "buy_covered"] == rows.loc["v5", "buy_covered"]
    trades = pd.read_csv(formal / "trades.csv")
    assert trades.loc[trades.rule.eq(candidate), "entry_origin"].eq("v5").all()
    assert rows.loc[candidate, "entry_events"] == rows.loc["v5", "entry_events"]
