"""r37冻结衰减算法、时间净化及不可变父工件。"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def sample_events():
    rows = []
    for symbol in ("A", "B", "C"):
        for year in range(2018, 2027):
            for signal in ("b", "jf", "s"):
                date = pd.Timestamp(year, 1, 1)
                rows.append({"symbol": symbol, "signal": signal, "date": date,
                             "outcome_date": date + pd.Timedelta(days=28), "warmup_complete": True,
                             "mature": True, "win": year > 2020, "gross_win": year > 2020,
                             "interference": year <= 2020, "source_trusted": True})
    return pd.DataFrame(rows)


def test_exact_half_life_weights_prior_and_effective_sample_size():
    from gcn.backtest.signal_research_r37 import decayed_base_rate
    cutoff = pd.Timestamp("2026-01-01")
    train = pd.DataFrame({"win": [True, False],
                          "outcome_date": [cutoff-pd.Timedelta(days=730), cutoff-pd.Timedelta(days=1460)]})
    result = decayed_base_rate(train, cutoff)
    assert result["weighted_events"] == .75
    assert np.isclose(result["effective_events"], .75**2 / (.5**2 + .25**2))
    assert result["probability"] == 2.5 / 4.75
    assert result["base_probability"] == .5
    assert decayed_base_rate(train.iloc[:0], cutoff)["probability"] == .5
    with pytest.raises(ValueError, match="成熟"):
        decayed_base_rate(train.assign(outcome_date=cutoff), cutoff)


def test_predictions_purge_future_labels_and_leave_out_the_symbol():
    from gcn.backtest.signal_research_r37 import cross_predictions
    events = sample_events()
    scored = cross_predictions(events)
    assert (pd.to_datetime(scored.fit_max_outcome_date) < pd.to_datetime(scored.fold_start)).all()
    held = scored[scored.scheme.eq("time_symbol")]
    assert all(row.symbol not in json.loads(row.fit_symbols) for row in held.itertuples())
    altered = events.copy()
    future = altered.outcome_date.ge(pd.Timestamp("2021-08-27"))
    altered.loc[future, "win"] = ~altered.loc[future, "win"]
    second = cross_predictions(altered)
    idx = scored.fold_start.eq("2021-08-27")
    np.testing.assert_array_equal(scored.loc[idx, "probability"], second.loc[idx, "probability"])
    assert not scored.duplicated(["scheme", "symbol", "signal", "date"]).any()


def test_archive_binds_parent_and_refuses_reuse(tmp_path):
    from gcn.backtest.signal_research_r37 import run_research
    parent = ROOT / "reports/gcn-historical-r36-20260912/results"
    output = tmp_path / "results"
    decision = run_research(parent, output)
    assert decision["production_changed"] is False
    assert decision["half_life_days"] == 730
    manifest = json.loads((output/"manifest.json").read_text())
    assert manifest["parent_manifest_sha256"] == hashlib.sha256((parent/"manifest.json").read_bytes()).hexdigest()
    for name, expected in manifest["outputs"].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError):
        run_research(parent, output)
    replay = tmp_path / "replay"
    run_research(parent, replay)
    for file in output.rglob("*"):
        if file.is_file():
            assert file.read_bytes() == (replay/file.relative_to(output)).read_bytes()


def test_parent_manifest_change_is_rejected_before_outputs(tmp_path):
    from gcn.backtest.signal_research_r37 import run_research
    parent = tmp_path/"parent"
    parent.mkdir()
    (parent/"manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="父manifest"):
        run_research(parent, tmp_path/"output")
    assert not (tmp_path/"output").exists()


def test_corrupted_parent_events_are_rejected_even_with_unchanged_manifest(tmp_path):
    from gcn.backtest.signal_research_r37 import run_research
    source = ROOT / "reports/gcn-historical-r36-20260912/results"
    parent = tmp_path/"parent"
    parent.mkdir()
    shutil.copy2(source/"manifest.json", parent/"manifest.json")
    manifest = json.loads((parent/"manifest.json").read_text())
    for name in manifest["outputs"]:
        shutil.copy2(source/name, parent/name)
    with (parent/"events.csv").open("ab") as handle:
        handle.write(b"corrupted\n")
    with pytest.raises(ValueError, match="r36输出摘要不匹配: events.csv"):
        run_research(parent, tmp_path/"output")
    assert not (tmp_path/"output").exists()
