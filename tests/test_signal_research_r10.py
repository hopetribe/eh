import numpy as np
import pandas as pd


def test_r10_reconciles_all_windows_without_revising_r9_or_resetting_positions(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import pytest
    from gcn.backtest.signal_research_r9 import run_training
    from gcn.backtest.signal_research_r9_validation import run_validation
    from gcn.backtest.signal_research_r9_stress import run_stress
    from gcn.backtest.signal_research_r10 import run_diagnostic, RULES, MODES

    root = Path(__file__).resolve().parents[1]
    snapshot = root / "reports/signal-audit-v5-review-20260904"
    r9, output = tmp_path / "r9", tmp_path / "diagnostic"
    run_training(snapshot, r9 / "results")
    run_validation(snapshot, r9 / "results", r9 / "validation")
    run_stress(snapshot, r9 / "results", r9 / "validation", r9 / "stress")
    before = {s: (r9 / s / "manifest.json").read_bytes() for s in ("results", "validation", "stress")}
    decision = run_diagnostic(snapshot, r9, output)
    rows = pd.read_csv(output / "comparisons.csv")
    assert len(rows) == 36 and set(rows.rule) == set(RULES) and set(rows["mode"]) == set(MODES)
    archived = pd.read_csv(r9 / "stress/comparisons.csv")
    reset = rows[rows["mode"].eq("reset_liquidate")].merge(archived, on=["case", "rule"], suffixes=("", "_r9"))
    assert len(reset) == 12
    for col in ("cagr", "mdd", "total", "sharpe"):
        assert np.allclose(reset[col], reset[col + "_r9"], atol=1e-10, rtol=0, equal_nan=True)
    for rule in RULES:
        carry = rows[rows.rule.eq(rule) & rows["mode"].eq("carry_mark")].set_index("case")
        years = carry.loc[[f"year{y}" for y in range(2021, 2026)]]
        assert np.isclose((1 + years.total / 100).prod(), 1 + carry.loc["full5y", "total"] / 100, atol=1e-12, rtol=0)
    boundaries = pd.read_csv(output / "boundaries.csv")
    assert len(boundaries) == 120
    assert boundaries.query("case=='year2024' and symbol=='YINN'").prior_entry_date.eq("2024-08-06").all()
    assert decision["status"] == "diagnostic_only" and decision["recommended"] == "v5"
    assert not decision["production_changed"] and decision["r9_status"] == "rejected_keep_v5"
    manifest = json.loads((output / "manifest.json").read_text())
    for stage, raw in before.items():
        assert (r9 / stage / "manifest.json").read_bytes() == raw
        assert manifest["r9_manifest_sha256"][stage] == hashlib.sha256(raw).hexdigest()
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        run_diagnostic(snapshot, r9, output)


def test_window_modes_keep_pending_entry_and_isolate_terminal_exit_cost():
    from gcn.backtest.signal_research_r10 import window_replay

    idx = pd.date_range("2024-01-01", periods=5)
    frame = pd.DataFrame({"OPEN": [100., 100., 110., 125., 125.],
                          "CLOSE": [100., 100., 120., 130., 125.],
                          "B_SIGNAL": [True, True, False, False, False],
                          "ICON_JUEFAN": False, "S_SIGNAL": False}, index=idx)
    signals = frame[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]].copy()
    signals["ENTRY_STOP"] = signals["ENTRY_LIMIT"] = np.nan
    signals["USE_EXTRA"] = signals["EXTRA_EXIT"] = signals["ENTRY_PROFIT_ENABLED"] = False
    bundle = {"frame": frame, "rules": {"v5": signals}}
    results = {mode: window_replay(bundle, "v5", idx[0], idx[1], idx[3], mode)
               for mode in ("carry_mark", "reset_mark", "reset_liquidate")}
    assert results["carry_mark"]["prior_state"]["pending_buy"]
    assert np.isclose(results["carry_mark"]["returns"].iloc[0], -.001)
    assert results["reset_mark"]["returns"].iloc[0] == 0
    assert results["carry_mark"]["end_entry_date"] == "2024-01-02"
    assert results["reset_mark"]["end_entry_date"] == "2024-01-03"
    mark = 1 + results["reset_mark"]["returns"]
    liquidate = 1 + results["reset_liquidate"]["returns"]
    assert np.allclose(mark.iloc[:-1], liquidate.iloc[:-1])
    assert np.isclose(liquidate.prod(), mark.prod() * .999)
    assert results["reset_liquidate"]["trades"][-1]["exit_reason"] == "terminal"
    assert results["reset_mark"]["end_state"]["position"] == "open"
    assert results["reset_liquidate"]["end_state"]["position"] == "flat"


def test_carry_window_preserves_peak_and_source_and_uses_prior_close_equity():
    from gcn.backtest.signal_research_r10 import window_replay
    from gcn.backtest.signal_research_r9 import CHALLENGERS

    idx = pd.date_range("2024-01-01", periods=7)
    frame = pd.DataFrame({"OPEN": [100., 100., 130., 120., 110., 115., 120.],
                          "CLOSE": [100., 130., 120., 110., 115., 120., 125.],
                          "B_SIGNAL": [True, False, False, False, False, False, False],
                          "ICON_JUEFAN": False, "S_SIGNAL": False}, index=idx)
    signals = frame[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]].copy()
    signals["ENTRY_STOP"] = signals["ENTRY_LIMIT"] = np.nan
    signals["USE_EXTRA"] = signals["EXTRA_EXIT"] = False
    signals["ENTRY_PROFIT_ENABLED"] = [True, False, False, False, False, False, False]
    bundle = {"frame": frame, "rules": {CHALLENGERS[0]: signals}}
    carry = window_replay(bundle, CHALLENGERS[0], idx[0], idx[2], idx[4], "carry_mark")
    reset = window_replay(bundle, CHALLENGERS[0], idx[0], idx[2], idx[4], "reset_mark")
    assert carry["prior_state"]["highest_close"] == 130.
    assert carry["prior_state"]["profit_armed"]
    assert carry["prior_entry_date"] == "2024-01-02"
    assert carry["end_state"]["position"] == "flat"
    assert carry["trades"][-1]["exit_reason"] == "profit_lock"
    assert np.isclose(carry["returns"].iloc[0], 120 / 130 - 1)
    assert np.isclose((1 + carry["returns"]).prod() - 1, 1.1 * .999 / 1.3 - 1)
    assert reset["returns"].eq(0).all() and reset["prior_entry_date"] is None
    pending_exit = window_replay(bundle, CHALLENGERS[0], idx[0], idx[4], idx[5], "carry_mark")
    assert pending_exit["prior_state"]["pending_sell_reason"] == "profit_lock"
    assert np.isclose(pending_exit["returns"].iloc[0], -.001)
    assert not pending_exit["held"].any()
