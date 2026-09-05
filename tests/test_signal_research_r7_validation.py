def test_validation_requires_net_performance_and_entry_improvement():
    from gcn.backtest.signal_research_r7_validation import validation_failures

    base = {"rule": "v5", "trades": 20, "entry_events": 25, "entry_win": 60.,
            "entry_interference": 35., "cagr": 20., "mdd": 12., "buy_covered": 4,
            "win": 55., "sharpe": 1.2}
    passing = {**base, "rule": "P-confirm5", "entry_win": 61.}
    assert validation_failures(passing, base) == []
    assert "win" in validation_failures({**passing, "win": 54.}, base)
    assert "sharpe" in validation_failures({**passing, "sharpe": 1.1}, base)
    assert "buy_covered" in validation_failures({**passing, "buy_covered": 3}, base)
    assert "no_entry_improvement" in validation_failures({**passing, "entry_win": 60., "mdd": 10.}, base)
    assert validation_failures({**passing, "entry_win": 60., "buy_covered": 5}, base) == []


def test_validation_replays_only_frozen_winner_and_seals_separate_outputs(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import pandas as pd
    from gcn.backtest.signal_research_r7 import run_training
    from gcn.backtest.signal_research_r7_validation import run_validation, validation_failures

    root = Path(__file__).resolve().parents[1]
    # This integration exercises a same-source train/validate pair; the archived
    # r7 report intentionally retains its older, immutable source identity.
    training = tmp_path / "training"
    output = tmp_path / "validation"
    run_training(root / "reports/signal-audit-v5-review-20260904", training)
    original = (training / "manifest.json").read_bytes()
    decision = run_validation(root / "reports/signal-audit-v5-review-20260904", training, output)
    comparisons = pd.read_csv(output / "comparisons.csv").to_dict("records")
    assert [row["rule"] for row in comparisons] == ["v5", "P-confirm5"]
    assert decision["selected"] == "P-confirm5" and decision["recommended"] == "v5"
    assert decision["failures"] == validation_failures(comparisons[1], comparisons[0])
    assert decision["status"] == ("rejected_keep_v5" if decision["failures"] else "passed_validation_pending_stress")
    events = pd.read_csv(output / "events.csv")
    assert events.date.min() >= "2024-08-27" and events.outcome_date.max() <= "2025-08-26"
    assert (training / "manifest.json").read_bytes() == original
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["training_manifest_sha256"] == hashlib.sha256(original).hexdigest()
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
