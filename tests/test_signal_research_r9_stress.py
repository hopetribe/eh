def test_r9_stress_checks_every_frozen_subset_and_preserves_parent_artifacts(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import pandas as pd
    import pytest
    from gcn.backtest.historical_research import CORE
    from gcn.backtest.signal_research_r9 import run_training, CHALLENGERS
    from gcn.backtest.signal_research_r9_validation import run_validation
    from gcn.backtest.signal_research_r9_stress import run_stress, stress_failures

    root = Path(__file__).resolve().parents[1]
    snapshot = root / "reports/signal-audit-v5-review-20260904"
    training, validation, output = (tmp_path / s for s in ("training", "validation", "stress"))
    run_training(snapshot, training)
    assert run_validation(snapshot, training, validation)["failures"] == []
    originals = [(p / "manifest.json").read_bytes() for p in (training, validation)]
    decision = run_stress(snapshot, training, validation, output)
    comparisons = pd.read_csv(output / "comparisons.csv")
    expected = {"full5y", "early8", "trusted5", "unleveraged8", "cost025", "TEM_external",
                *(f"year{y}" for y in range(2021, 2026)),
                *("without_" + s for s in CORE), *("only_" + s for s in CORE)}
    assert set(comparisons[~comparisons.case.str.startswith("neighbor")].case) == expected
    assert comparisons.groupby("case").size().eq(2).all()
    assert set(comparisons.rule) == {"v5", CHALLENGERS[0]}
    assert decision["failures"] == stress_failures(comparisons.to_dict("records"))
    assert decision["recommended"] == "v5" and not decision["production_changed"]
    assert decision["status"] == ("rejected_keep_v5" if decision["failures"] else "passed_stress_pending_review")
    assert comparisons.query("case=='early8'").symbols.eq("TQQQ,MSFT,NFLX,YINN,TSLA,NVDA,GOOGL,AAOI").all()
    assert comparisons.query("case=='trusted5'").symbols.eq("TQQQ,SNOW,TSLA,NVDA,AAOI").all()
    assert comparisons.query("case=='cost025'").cost.eq(.0025).all()
    trades = pd.read_csv(output / "trades.csv")
    for rule in ("v5", CHALLENGERS[0]):
        base = trades[(trades.case == "full5y") & (trades.rule == rule)]
        cost = trades[(trades.case == "cost025") & (trades.rule == rule)]
        keys = ["symbol", "entry_date", "exit_date", "exit_reason", "entry_profit_enabled"]
        pd.testing.assert_frame_equal(base[keys].reset_index(drop=True), cost[keys].reset_index(drop=True))
    manifest = json.loads((output / "manifest.json").read_text())
    for stage, original in zip(("training", "validation"), originals):
        assert manifest[stage + "_manifest_sha256"] == hashlib.sha256(original).hexdigest()
        assert (tmp_path / stage / "manifest.json").read_bytes() == original
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        run_stress(snapshot, training, validation, output)


def test_r9_stress_gates_require_broad_returns_and_early_risk_retention():
    from copy import deepcopy
    import pytest
    from gcn.backtest.historical_research import CORE
    from gcn.backtest.signal_research_r9 import CHALLENGERS
    from gcn.backtest.signal_research_r9_stress import stress_failures

    cases = ["full5y", "cost025", "early8", *("without_" + s for s in CORE)]
    rows = [{"case": case, "rule": rule, "cagr": 10., "mdd": 10.}
            for case in cases for rule in ("v5", CHALLENGERS[0])]
    assert stress_failures(rows) == []
    changed = deepcopy(rows)
    for row in changed:
        if row["rule"] != "v5" and row["case"] in {"without_TQQQ", "without_MSFT", "without_NFLX"}:
            row["cagr"] = 9.9
    assert stress_failures(changed) == ["leave_one_out_cagr"]
    changed[1]["cagr"] = 9.99
    changed[3]["mdd"] = 11.01
    changed[5]["cagr"] = 8.99
    assert set(stress_failures(changed)) == {"full5y_cagr", "cost025_mdd", "early8_cagr", "leave_one_out_cagr"}
    with pytest.raises(ValueError, match="压力对照"):
        stress_failures(rows[:-1])
