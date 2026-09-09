"""r30原生B实际入场首日双分量拒绝；真实次OPEN与严格首日边界。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    return pd.DataFrame({
        "OPEN": [100., 100., 95., 90., 80.],
        "CLOSE": [100., 100., 90., 900., 80.],
        "B_SIGNAL": [False, True, False, False, False],
        "ICON_JUEFAN": False,
        "S_SIGNAL": False,
        "TWO_LEG": [False, True, False, False, False],
    })


def test_r30_locks_signal_close_checks_only_entry_day_and_exits_at_real_next_open():
    from gcn.backtest.engine import _one_strategy

    frame = _fixture()
    result = _one_strategy(
        frame,
        ["B_SIGNAL", "ICON_JUEFAN"],
        ["S_SIGNAL"],
        .001,
        None,
        trail=.20,
        entry_two_leg_rejection_col="TWO_LEG",
    )

    assert [(t["i"], t["j"], t["hold"], t["exit_reason"]) for t in result["trades"]] == [
        (2, 3, 1, "two_leg_rejection")
    ]
    trade = result["trades"][0]
    assert trade["two_leg_enabled"]
    assert trade["two_leg_signal_close"] == 100.
    assert np.isclose(trade["two_leg_entry_gap_factor"], .95)
    assert np.isclose(trade["two_leg_intraday_factor"], 90 / 95)
    assert trade["two_leg_trigger_i"] == 2
    assert np.isclose(trade["ret"], 90 / 95 * .999**2 - 1)
    assert result["held"].tolist() == [False, False, True, False, False]
    assert result["state"]["two_leg_rejection"] is None


def test_r30_factory_preserves_native_signals_and_marks_b_including_jf_collision_only():
    from gcn.backtest.signal_research_r30 import candidate_signals, CHALLENGERS, RULES

    frame = pd.DataFrame({
        "LOW": [90., 89., 88., 87.],
        "B_SIGNAL": [True, False, True, False],
        "ICON_JUEFAN": [False, True, True, False],
        "S_SIGNAL": False,
    }, index=pd.bdate_range("2025-01-01", periods=4))
    original = frame.copy(deep=True)
    result = candidate_signals(frame)

    assert list(result) == list(RULES) == ["v5", "B-entry-day-two-leg-rejection"]
    assert CHALLENGERS == ("B-entry-day-two-leg-rejection",)
    assert not result["v5"].ENTRY_TWO_LEG_REJECTION.any()
    assert result[CHALLENGERS[0]].ENTRY_TWO_LEG_REJECTION.tolist() == [True, False, True, False]
    for signals in result.values():
        pd.testing.assert_frame_equal(signals[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]],
                                      frame[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]])
    pd.testing.assert_frame_equal(frame, original)


def test_r30_evaluator_carries_two_leg_audit_and_reprices_the_same_orders():
    from gcn.backtest.historical_research import evaluate_rule
    from gcn.backtest.signal_research_r30 import candidate_signals, CHALLENGERS

    frame = _fixture()
    frame["LOW"] = frame[["OPEN", "CLOSE"]].min(axis=1) - 1
    frame.index = pd.bdate_range("2024-01-02", periods=len(frame))
    prepared = {"TEST": {"frame": frame, "rules": candidate_signals(frame)}}
    options = {"entry_two_leg_rejection_col": "ENTRY_TWO_LEG_REJECTION",
               "include_positions": True}
    outputs = [evaluate_rule(prepared, CHALLENGERS[0], frame.index[0], frame.index[-1], fee,
                             **options) for fee in (.001, .0025)]

    for fee, result in zip((.001, .0025), outputs):
        row = result["trades"][0]
        assert row["entry_date"] == frame.index[2].date().isoformat()
        assert row["exit_date"] == frame.index[3].date().isoformat()
        assert row["exit_reason"] == "two_leg_rejection"
        assert row["two_leg_enabled"] and row["two_leg_trigger_date"] == frame.index[2].date().isoformat()
        assert row["two_leg_signal_close"] == 100.
        assert np.isclose(row["return_pct"], (90 / 95 * (1-fee)**2 - 1) * 100)
        assert result["positions"]["TEST"].tolist() == [False, False, True, False, False]
    for key in outputs[0]["trades"][0]:
        if key != "return_pct":
            assert outputs[0]["trades"][0][key] == outputs[1]["trades"][0][key]


def test_r30_training_wrapper_binds_frozen_candidate_protocol_and_marker(monkeypatch, tmp_path):
    from gcn.backtest import signal_research_r30 as research

    captured = {}

    def inspect(*args, **kwargs):
        captured.update(kwargs)
        return {"checked": True}

    monkeypatch.setattr(research, "_run_training", inspect)
    assert research.run_training(tmp_path / "snapshot", tmp_path / "output") == {"checked": True}
    assert captured["research_version"] == "gcn-historical-r30"
    assert captured["protocol_relative"] == "reports/gcn-historical-r30-20260908/protocol.md"
    assert captured["candidate_builder"] is research.candidate_signals
    assert captured["challengers"] == research.CHALLENGERS
    assert captured["controls"] == research.CONTROLS
    assert captured["entry_two_leg_rejection_col"] == "ENTRY_TWO_LEG_REJECTION"


def test_r30_real_training_runs_full_candidate_chain_and_records_two_leg_exits(tmp_path):
    from gcn.backtest.signal_research_r30 import run_training, CHALLENGERS

    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    decision = run_training(snapshot, tmp_path)
    rows = pd.read_csv(tmp_path / "training.csv").set_index("rule")
    trades = pd.read_csv(tmp_path / "trades.csv")

    assert decision["research_version"] == "gcn-historical-r30"
    assert set(rows.index) == {"v5", CHALLENGERS[0]}
    candidate = trades[trades.rule.eq(CHALLENGERS[0])]
    exits = candidate[candidate.exit_reason.eq("two_leg_rejection")]
    assert len(exits) > 0
    assert exits.two_leg_enabled.all()
    assert exits.two_leg_trigger_date.notna().all()


def test_r30_all_false_marker_is_exactly_the_same_as_omitting_the_feature():
    from gcn.backtest.engine import _one_strategy

    frame = _fixture()
    frame["TWO_LEG"] = False
    args = (["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"], .001, None)
    expected = _one_strategy(frame, *args, trail=.20, terminal_policy="mark")
    actual = _one_strategy(frame, *args, trail=.20, terminal_policy="mark",
                           entry_two_leg_rejection_col="TWO_LEG")

    np.testing.assert_array_equal(actual["equity"], expected["equity"])
    np.testing.assert_array_equal(actual["held"], expected["held"])
    assert actual["trades"] == expected["trades"]
    assert actual["state"] == expected["state"]


def test_r30_strict_boundaries_first_day_only_old_exit_priority_terminal_and_marker_contract():
    import pytest
    from gcn.backtest.engine import _one_strategy

    args = (["B_SIGNAL"], ["S_SIGNAL"], .001, None)

    def run(signal_close, entry_open, entry_close, later_close=50., *, signal_on_entry=False,
            terminal_policy="mark"):
        frame = pd.DataFrame({
            "OPEN": [100., entry_open, 90., 80.],
            "CLOSE": [signal_close, entry_close, later_close, 80.],
            "B_SIGNAL": [True, False, False, False],
            "S_SIGNAL": [False, signal_on_entry, False, False],
            "TWO_LEG": [True, False, False, False],
        })
        return _one_strategy(frame, *args, terminal_policy=terminal_policy,
                             entry_two_leg_rejection_col="TWO_LEG")

    assert run(100., 100., 90.)["state"]["status"] == "open"  # Equal entry gap.
    assert run(101., 100., 100.)["state"]["status"] == "open"  # Equal intraday leg.
    late = run(101., 100., 101., later_close=50.)
    assert late["state"]["status"] == "open"  # A later red bar is outside the first-day boundary.
    priority = run(101., 100., 90., signal_on_entry=True, terminal_policy="liquidate")
    assert priority["trades"][0]["exit_reason"] == "signal"
    assert priority["trades"][0]["two_leg_trigger_i"] is None

    terminal_frame = pd.DataFrame({"OPEN": [100., 90.], "CLOSE": [100., 80.],
                                   "B_SIGNAL": [True, False], "S_SIGNAL": False,
                                   "TWO_LEG": [True, False]})
    marked = _one_strategy(terminal_frame, *args, terminal_policy="mark",
                           entry_two_leg_rejection_col="TWO_LEG")
    assert marked["state"]["status"] == "pending_exit"
    assert marked["state"]["pending_sell_reason"] == "two_leg_rejection"
    terminal = _one_strategy(terminal_frame, *args,
                             entry_two_leg_rejection_col="TWO_LEG")
    assert terminal["trades"][0]["exit_reason"] == "terminal"

    base = _fixture()
    for column in ("missing", 1, ["T"]):
        with pytest.raises(ValueError, match="entry_two_leg_rejection_col"):
            _one_strategy(base, *args, entry_two_leg_rejection_col=column)
    for value in (0, 1, .1, "False", "", [True], {}, 1+0j):
        bad = base.copy()
        bad["TWO_LEG"] = bad["TWO_LEG"].astype(object)
        bad.at[1, "TWO_LEG"] = value
        with pytest.raises(ValueError, match="entry_two_leg_rejection_col"):
            _one_strategy(bad, *args, entry_two_leg_rejection_col="TWO_LEG")


def test_r30_formal_training_reproduces_byte_for_byte_and_keeps_all_original_entries(tmp_path):
    from gcn.backtest.signal_research_r30 import run_training, CHALLENGERS

    reproduced = tmp_path / "training"
    run_training(ROOT / "reports/signal-audit-v5-review-20260904", reproduced)
    formal = ROOT / "reports/gcn-historical-r30-20260908/training"
    manifest = json.loads((formal / "manifest.json").read_text())
    for name in list(manifest["outputs"]) + ["manifest.json"]:
        assert (reproduced / name).read_bytes() == (formal / name).read_bytes()

    rows = pd.read_csv(formal / "training.csv").set_index("rule")
    assert rows.loc[CHALLENGERS[0], "two_leg_rejection_exits"] == 6
    assert rows.loc[CHALLENGERS[0], "cagr"] > rows.loc["v5", "cagr"]
    assert rows.loc[CHALLENGERS[0], "mdd"] < rows.loc["v5", "mdd"]
    trades = pd.read_csv(formal / "trades.csv")

    def keys(rule):
        selected = trades[trades.rule.eq(rule)]
        return set(zip(selected.symbol, selected.entry_date))

    assert keys("v5") == keys(CHALLENGERS[0])
