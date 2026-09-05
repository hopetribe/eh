"""历史研究契约；不消费真实前向实验或修改行情缓存。"""
import numpy as np
import pandas as pd

from gcn.data.sample import make_sample_data
from gcn.recipes.gcn_main import compute_ehopt10
from gcn.core.tdx import COUNT, BARSLAST, REF


def test_raw_diagnostics_reproduce_v5_setups_and_exits_without_changing_defaults():
    data = make_sample_data(900, seed=11)
    plain = compute_ehopt10(data, version="v5")
    diagnostic = compute_ehopt10(data, version="v5", diagnostics=True)
    assert set(diagnostic) - set(plain) == {
        "B_ALL_RAW", "S_RAW", "JF_RAW", "B_BASE_BULL", "B_STAGE_COMPONENT",
        "B_BEAR_RECOVER", "B_CRASH_RECOVER",
    }
    for col in plain:
        assert np.allclose(plain[col], diagnostic[col], equal_nan=True), col
    b_raw, s_raw, jf_raw = (diagnostic[k] for k in ("B_ALL_RAW", "S_RAW", "JF_RAW"))
    assert diagnostic["B_SETUP"].equals(b_raw & (COUNT(b_raw, 20) == 1))
    assert diagnostic["S_SIGNAL"].equals(s_raw & (COUNT(s_raw, 40) == 1))
    gap = REF(BARSLAST(jf_raw), 1)
    assert diagnostic["ICON_JUEFAN"].equals(jf_raw & (gap.isna() | (gap >= 9)))


def test_cooldown_uses_last_emitted_signal_and_preserves_prefix():
    from gcn.backtest.historical_research import accepted_cooldown

    raw = pd.Series([True, True, False, True, True, True, False, True])
    actual = accepted_cooldown(raw, 3)
    assert actual.tolist() == [True, False, False, True, False, False, False, True]
    for end in range(1, len(raw) + 1):
        assert accepted_cooldown(raw.iloc[:end], 3).equals(actual.iloc[:end])


def test_research_candidates_are_separate_and_use_causal_signal_columns():
    from gcn.backtest.historical_research import candidate_signals

    frame = compute_ehopt10(make_sample_data(900, seed=11), version="v5", diagnostics=True)
    rules = candidate_signals(frame)
    assert list(rules) == ["v5", "b-cooldown20", "s-cooldown40", "jf-cooldown10",
                           "bs-cooldown", "b-momentum", "profit50", "bs-cooldown-profit50"]
    for name, signals in rules.items():
        assert list(signals) == ["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]
        assert signals.index.equals(frame.index)
        if name in {"v5", "profit50"}:
            pd.testing.assert_frame_equal(signals, frame[list(signals)])
        pd.testing.assert_frame_equal(candidate_signals(frame.iloc[:600])[name],
                                      signals.iloc[:600])
    assert rules["b-momentum"]["B_SIGNAL"].equals(
        frame["B_SIGNAL"] & (frame["MACD"] > frame["MACD"].shift(1)))
    assert rules["b-cooldown20"]["S_SIGNAL"].equals(frame["S_SIGNAL"])
    assert rules["s-cooldown40"]["B_SIGNAL"].equals(frame["B_SIGNAL"])


def test_rule_evaluation_fills_next_open_and_reprices_the_same_trades():
    from gcn.backtest.historical_research import evaluate_rule

    idx = pd.date_range("2024-01-01", periods=5)
    frame = pd.DataFrame({"OPEN": [80, 100, 115, 120, 125],
                          "CLOSE": [90, 110, 116, 121, 126]}, index=idx)
    signals = pd.DataFrame({"B_SIGNAL": [True, False, False, False, False],
                            "ICON_JUEFAN": False,
                            "S_SIGNAL": [False, False, True, False, False]}, index=idx)
    prepared = {"AAA": {"frame": frame, "rules": {"v5": signals}}}
    base = evaluate_rule(prepared, "v5", idx[0], idx[-1])
    stress = evaluate_rule(prepared, "v5", idx[0], idx[-1], cost=0.0025)
    assert base["trades"][0]["entry_date"] == "2024-01-02"
    assert base["trades"][0]["exit_date"] == "2024-01-04"
    assert stress["trades"][0]["entry_date"] == base["trades"][0]["entry_date"]
    assert stress["trades"][0]["exit_date"] == base["trades"][0]["exit_date"]
    assert np.isclose(base["stats"]["total"], (1.2 * 0.999**2 - 1) * 100)
    assert np.isclose(stress["stats"]["total"], (1.2 * 0.9975**2 - 1) * 100)


def test_event_quality_excludes_outcomes_crossing_the_evaluation_boundary():
    from gcn.backtest.historical_research import event_quality

    idx = pd.date_range("2024-01-01", periods=42)
    frame = pd.DataFrame({"OPEN": 100.0, "HIGH": 120.0, "LOW": 90.0,
                          "CLOSE": 110.0}, index=idx)
    signals = pd.DataFrame(False, index=idx, columns=["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"])
    signals.loc[idx[[0, 19, 20]], "B_SIGNAL"] = True
    prepared = {"AAA": {"frame": frame, "rules": {"v5": signals}}}
    result = event_quality(prepared, "v5", idx[0], idx[39])
    assert result["stats"]["entry_events"] == 2
    assert result["stats"]["entry_win"] == 100.0
    assert result["stats"]["entry_interference"] == 0.0
    assert len(result["events"]) == 4  # 同一事件分别计入entry并集与B子类
    assert all(row["outcome_date"] <= "2024-02-09" for row in result["events"])


def test_frozen_input_loader_rejects_changed_bytes(tmp_path):
    import hashlib
    import json
    import pytest
    from gcn.backtest.historical_research import load_snapshot

    csv = tmp_path / "AAA.csv"
    csv.write_text("date,open,high,low,close,volume\n2024-01-01,10,12,9,11,100\n")
    meta = tmp_path / "AAA.meta.json"
    meta.write_text(json.dumps({"sha256": "old", "source": "yahoo", "adjustment": "adjusted"}))
    manifest = {"inputs": {"AAA": {
        "snapshot_path": "AAA.csv", "metadata_snapshot_path": "AAA.meta.json",
        "sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(meta.read_bytes()).hexdigest()}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frames, quality = load_snapshot(tmp_path, digest)
    assert len(frames["AAA"]) == 1 and quality["AAA"] is False
    csv.write_text(csv.read_text().replace(",11,", ",11.5,"))
    with pytest.raises(ValueError, match="AAA.*摘要"):
        load_snapshot(tmp_path, digest)


def test_training_selection_keeps_accuracy_and_coverage_gates_before_calmar():
    from gcn.backtest.historical_research import choose_training

    base = {"rule": "v5", "trades": 50, "entry_events": 54, "entry_win": 50.,
            "entry_interference": 48., "cagr": 8.72, "mdd": 15., "calmar": 0.58}
    s = {**base, "rule": "s-cooldown40"}
    profit = {**base, "rule": "profit50", "cagr": 7.80, "calmar": 5.}
    low_coverage = {**base, "rule": "b-momentum", "entry_events": 20, "calmar": 9.}
    assert choose_training([base, profit, s, low_coverage]) == "s-cooldown40"
    assert choose_training([base, profit, low_coverage]) is None


def test_unchanged_validation_results_cannot_promote_a_candidate():
    from gcn.backtest.historical_research import validation_failures

    base = {"trades": 17, "entry_events": 22, "entry_win": 63.6,
            "entry_interference": 36.4, "cagr": 27.35, "mdd": 11.29,
            "sharpe": 1.414, "win": 64.7, "s_win": 50.0,
            "b_win": 50.0, "jf_win": 80.0}
    assert validation_failures(base.copy(), base) == ["no_material_improvement"]
    assert validation_failures({**base, "s_win": 60.0}, base) == []


def test_historical_run_reproduces_baseline_and_records_rejection(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from gcn.backtest.historical_research import run_research, SNAPSHOT_SHA

    root = Path(__file__).resolve().parents[1]
    result = run_research(root / "reports/signal-audit-v5-review-20260904", tmp_path)
    assert result["selected"] == "s-cooldown40"
    assert result["validation_failures"] == ["no_material_improvement"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["parent_manifest_sha256"] == SNAPSHOT_SHA
    for filename, digest in manifest["outputs"].items():
        assert hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest() == digest
    baseline = pd.read_csv(tmp_path / "comparisons.csv").query("case == 'full5y' and rule == 'v5'").iloc[0]
    assert np.isclose(baseline["cagr"], 21.0642, atol=0.01)
    assert baseline["trades"] == 82
