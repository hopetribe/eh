def test_r9_validation_replays_only_locked_training_winner_and_profit_scope(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import pandas as pd
    from gcn.backtest.signal_research_r9 import run_training, CHALLENGERS, PROFIT_KEEPS
    from gcn.backtest.signal_research_r9_validation import run_validation
    from gcn.backtest.signal_research_r7_validation import validation_failures

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
    assert candidate.entry_profit_enabled.equals(candidate.entry_origin.eq("v5"))
    assert not candidate.query("entry_origin=='additional'").exit_reason.eq("profit_lock").any()
    events = pd.read_csv(output / "events.csv")
    assert events.date.min() >= "2024-08-27" and events.outcome_date.max() <= "2025-08-26"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["training_manifest_sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["profit_keeps"] == PROFIT_KEEPS
    assert manifest["entry_profit_enabled_col"] == "ENTRY_PROFIT_ENABLED"
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() == digest
    assert (training / "manifest.json").read_bytes() == original
