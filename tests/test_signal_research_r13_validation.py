def test_r13_validation_replays_locked_combination_without_changing_training_or_profit_scope(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import pandas as pd
    from gcn.backtest.signal_research_r13 import run_training, CHALLENGERS
    from gcn.backtest.signal_research_r13_validation import run_validation
    from gcn.backtest.signal_research_r7_validation import validation_failures
    from gcn.backtest.historical_research import load_snapshot
    from gcn.core.indicators import volume_ratio

    snapshot = Path(__file__).resolve().parents[1] / "reports/signal-audit-v5-review-20260904"
    training, output = tmp_path / "training", tmp_path / "validation"
    assert run_training(snapshot, training)["selected"] == CHALLENGERS[0]
    original = (training / "manifest.json").read_bytes()
    decision = run_validation(snapshot, training, output)
    rows = pd.read_csv(output / "comparisons.csv").to_dict("records")
    assert [row["rule"] for row in rows] == ["v5", CHALLENGERS[0]]
    assert decision["selected"] == CHALLENGERS[0] and decision["recommended"] == "v5"
    assert decision["failures"] == validation_failures(rows[1], rows[0])
    assert decision["status"] == ("rejected_keep_v5" if decision["failures"] else "passed_validation_pending_stress")
    trades = pd.read_csv(output / "trades.csv")
    candidate = trades[trades.rule.eq(CHALLENGERS[0])]
    assert not candidate.exit_reason.eq("profit_lock").any()
    extra = candidate[candidate.entry_origin.eq("additional")]
    assert len(extra) > 0 and extra.hold_days.le(20).all()
    frames, _ = load_snapshot(snapshot)
    for trade in extra.itertuples():
        frame = frames[trade.symbol].loc[:"2025-08-26"]
        ratio = volume_ratio(frame.volume, 20)
        pos = frame.index.get_loc(pd.Timestamp(trade.entry_date)) - 1
        assert ratio.iloc[pos] >= 1
    events = pd.read_csv(output / "events.csv")
    assert events.date.min() >= "2024-08-27" and events.outcome_date.max() <= "2025-08-26"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["training_manifest_sha256"] == hashlib.sha256(original).hexdigest()
    assert "profit_keeps" not in manifest and "entry_profit_enabled_col" not in manifest
    assert "gcn/backtest/signal_research_r13_validation.py" in manifest["algorithm_sources"]
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() == digest
    assert (training / "manifest.json").read_bytes() == original
