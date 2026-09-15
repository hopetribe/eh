"""r36的指标因果性、概率基准及双重隔离回归。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def price_frame(n=260):
    close = 100 + np.arange(n) * .1 + np.sin(np.arange(n))
    return pd.DataFrame({"OPEN": close - .3, "HIGH": close + 1,
                         "LOW": close - 3, "CLOSE": close,
                         "VOLUME": 1000 + np.arange(n) * 2},
                        index=pd.bdate_range("2020-01-01", periods=n))


def synthetic_events():
    from gcn.backtest.signal_research_r36 import FAMILIES
    rows = []
    for symbol_i, symbol in enumerate(("A", "B", "C", "D")):
        for i, date in enumerate(pd.date_range("2018-01-01", "2026-01-01", freq="MS")):
            value = np.sin(i + symbol_i)
            rows.append({"symbol": symbol, "signal": ("b", "jf", "s")[i % 3],
                         "date": date, "outcome_date": date + pd.Timedelta(days=28),
                         "mature": True, "win": value > 0, "gross_win": value > 0,
                         "interference": value < -.3, "source_trusted": True,
                         **{feature: value for pair in FAMILIES.values() for feature in pair}})
    return pd.DataFrame(rows)


def test_indicators_are_causal_scale_invariant_and_preserve_missingness():
    from gcn.backtest.signal_research_r36 import indicator_features
    raw = price_frame()
    actual = indicator_features(raw)
    pd.testing.assert_frame_equal(indicator_features(raw.iloc[:220]), actual.iloc[:220])
    assert actual.iloc[:199].isna().all().all()
    assert np.isclose(actual.cmf21.iloc[-1], .5)
    assert np.isclose(actual.cmf_delta5.iloc[-1], 0)
    scaled = raw.copy()
    scaled[["OPEN", "HIGH", "LOW", "CLOSE"]] *= 7
    scaled.VOLUME *= 13
    np.testing.assert_allclose(indicator_features(scaled), actual, atol=1e-10, equal_nan=True)
    raw.VOLUME = 0
    missing = indicator_features(raw)
    assert missing.cmf21.iloc[-1:].isna().all()
    assert missing.obv_flow20.iloc[-1:].isna().all()


def test_labels_require_real_next_open_and_complete_twenty_bar_outcome():
    from gcn.backtest.signal_research_r36 import frame_events
    raw = price_frame(222)
    raw[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]] = False
    raw.loc[raw.index[200], ["B_SIGNAL", "S_SIGNAL"]] = True
    raw.loc[raw.index[205], "ICON_JUEFAN"] = True
    raw.loc[raw.index[201], "OPEN"] = 100
    raw.loc[raw.index[220], "CLOSE"] = 100.1
    events = frame_events("TEST", raw, True)
    assert len(events) == 3
    assert events.loc[events.signal.eq("b"), "gross_win"].item()
    assert not events.loc[events.signal.eq("b"), "win"].item()
    pending = events[events.signal.eq("jf")].iloc[0]
    assert not pending.mature and pd.isna(pending.win) and pd.isna(pending.outcome_date)
    assert events.loc[events.signal.eq("b"), "reference_open"].item() == 100
    assert events.loc[events.signal.eq("b"), "outcome_date"].item() == raw.index[220]


def test_probability_model_is_deterministic_and_falls_back_for_unavailable_features():
    from gcn.backtest.signal_research_r36 import fit_model, predict_model
    events = synthetic_events().query("signal == 'b'")
    model = fit_model(events, "cmf")
    assert model == fit_model(events, "cmf")
    scored = predict_model(events.drop(columns="win"), model)
    assert np.isfinite(scored.probability).all()
    assert scored.probability.between(0, 1).all()
    assert scored.loc[events.cmf21.gt(.5), "probability"].mean() > model["base_probability"]
    bad = events.iloc[:1].copy()
    bad.cmf21 = np.nan
    fallback = predict_model(bad, model).iloc[0]
    assert fallback.probability == model["base_probability"] and not fallback.high
    assert not fallback.feature_available
    one_class = events.assign(win=False)
    fitted = fit_model(one_class, "cmf")
    assert fitted["fallback"] and fitted["base_probability"] == 2 / (len(events) + 4)


def test_cross_predictions_purge_unmatured_labels_and_exclude_held_symbol():
    from gcn.backtest.signal_research_r36 import cross_predictions
    events = synthetic_events()
    first = cross_predictions(events)
    assert set(first.scheme) == {"time", "time_symbol"}
    assert (pd.to_datetime(first.fit_max_outcome_date) < pd.to_datetime(first.fold_start)).all()
    held = first[first.scheme.eq("time_symbol")]
    assert all(row.symbol not in json.loads(row.fit_symbols) for row in held.itertuples())
    assert not first.duplicated(["scheme", "candidate", "symbol", "signal", "date"]).any()
    changed = events.copy()
    future = changed.outcome_date.ge(pd.Timestamp("2021-08-27"))
    changed.loc[future, "win"] = ~changed.loc[future, "win"]
    second = cross_predictions(changed)
    select = first.fold_start.eq("2021-08-27")
    np.testing.assert_array_equal(first.loc[select, "probability"], second.loc[select, "probability"])


def test_r36_gates_reject_high_win_with_worse_probability_predictions():
    from gcn.backtest.signal_research_r36 import summarize
    scores = pd.DataFrame([{
        "scheme": "time", "candidate": "cmf", "signal": "b", "symbol": f"S{i%6}",
        "date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=i),
        "fold_start": "2021-08-27", "probability": .99 if i < 20 else .01,
        "base_probability": .6, "win": i % 3 != 0, "gross_win": i % 3 != 0,
        "interference": i % 3 == 0, "high": i < 20, "source_trusted": True,
    } for i in range(40)])
    metrics, _, _ = summarize(scores)
    assert metrics.brier_gain.iloc[0] < 0
    assert not metrics.passed.iloc[0]
    assert "brier" in metrics.failures.iloc[0]


def test_real_events_reconcile_prior_training_and_preserve_pending_labels():
    from gcn.backtest.signal_research_r36 import build_events
    events, quality = build_events(ROOT / "reports/signal-audit-v5-review-20260904")
    prior = events[events.date.between("2021-08-27", "2024-08-26")
                   & events.outcome_date.le(pd.Timestamp("2024-08-26"))]
    assert prior.groupby("signal").size().to_dict() == {"b": 31, "jf": 23, "s": 35}
    assert prior.groupby("signal").gross_win.sum().to_dict() == {"b": 14, "jf": 13, "s": 20}
    assert events.loc[~events.mature, "win"].isna().all()
    assert events.loc[events.mature, "outcome_date"].le(pd.Timestamp("2026-08-27")).all()
    assert len(quality) == 10 and quality.events.sum() == len(events)


def test_archive_binds_protocol_outputs_sources_and_replays_deterministically(tmp_path):
    from gcn.backtest.signal_research_r36 import run_research
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    output = tmp_path / "first"
    decision = run_research(snapshot, output)
    assert decision["production_changed"] is False
    assert set(decision["selected_by_signal"]) == {"b", "jf", "s"}
    assert "exploratory" in decision["evidence"]
    manifest = json.loads((output / "manifest.json").read_text())
    for group, prefix in (("outputs", output), ("algorithm_sources", output / "source_snapshot")):
        for name, expected in manifest[group].items():
            assert hashlib.sha256((prefix/name).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError, match="必须为空"):
        run_research(snapshot, output)
    replay = tmp_path / "second"
    run_research(snapshot, replay)
    for file in output.rglob("*"):
        if file.is_file():
            assert file.read_bytes() == (replay / file.relative_to(output)).read_bytes()
