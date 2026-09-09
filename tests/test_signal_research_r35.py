"""r35 v5信号质量置信分层。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_r35_features_use_only_current_and_trailing_values():
    from gcn.backtest.signal_research_r35 import causal_features

    index = pd.bdate_range("2024-01-02", periods=65)
    macd = np.arange(65, dtype=float)
    frame = pd.DataFrame(
        {
            "CLOSE": np.full(65, 11.0),
            "MID": np.full(65, 10.0),
            "UPPER": np.full(65, 12.0),
            "LOWER": np.full(65, 8.0),
            "MACD": macd,
            "RSI1": np.full(65, 75.0),
            "获利筹": np.full(65, 25.0),
        },
        index=index,
    )
    features = causal_features(frame)
    scale = np.std(macd[:60], ddof=0)

    assert list(features) == [
        "band_position", "macd_level", "macd_delta", "rsi_center", "chip_center"
    ]
    assert features.loc[index[58], "macd_level"] == 0.0
    assert np.isclose(features.loc[index[59], "band_position"], 0.25)
    assert np.isclose(features.loc[index[59], "macd_level"], 59 / scale)
    assert np.isclose(features.loc[index[59], "macd_delta"], 1 / scale)
    assert features.loc[index[59], "rsi_center"] == 1.0
    assert features.loc[index[59], "chip_center"] == -0.5

    changed = frame.copy()
    changed.loc[index[60]:, ["CLOSE", "MACD", "RSI1", "获利筹"]] = 999.0
    pd.testing.assert_series_equal(
        causal_features(changed).loc[index[59]], features.loc[index[59]]
    )


def test_r35_regularized_models_are_deterministic_and_tier_with_fit_thresholds():
    from gcn.backtest.signal_research_r35 import fit_quality_models, predict_quality

    rows = []
    for signal in ("b", "jf", "s"):
        for value in np.linspace(-2.0, 2.0, 12):
            rows.append(
                {
                    "signal": signal,
                    "win": bool(value > 0),
                    "band_position": value,
                    "macd_level": 0.0,
                    "macd_delta": 0.0,
                    "rsi_center": 0.0,
                    "chip_center": 0.0,
                }
            )
    events = pd.DataFrame(rows)
    first = fit_quality_models(events)
    second = fit_quality_models(events)
    assert first == second

    scored = predict_quality(events.drop(columns="win"), first)
    for signal in ("b", "jf", "s"):
        selected = scored[scored.signal.eq(signal)]
        assert selected.iloc[-1].quality_score > selected.iloc[0].quality_score
        assert {"low", "normal", "high"}.issubset(set(selected.quality_tier))


def test_r35_shared_ridge_model_handles_one_sided_signal_fold():
    from gcn.backtest.signal_research_r35 import fit_quality_models, predict_quality

    rows = []
    for signal in ("b", "jf", "s"):
        for value in np.linspace(-2.0, 2.0, 8):
            rows.append(
                {
                    "signal": signal,
                    "win": False if signal == "b" else bool(value > 0),
                    "band_position": value,
                    "macd_level": value / 2,
                    "macd_delta": 0.0,
                    "rsi_center": 0.0,
                    "chip_center": 0.0,
                }
            )
    events = pd.DataFrame(rows)
    model = fit_quality_models(events)
    scored = predict_quality(events.drop(columns="win"), model)

    assert set(model) == {"mean", "scale", "beta", "thresholds"}
    assert np.isfinite(scored.quality_score).all()
    assert set(scored.quality_tier) == {"low", "normal", "high"}


def test_r35_training_events_reproduce_frozen_v5_signal_quality():
    from gcn.backtest.signal_research_r35 import build_event_table

    events = build_event_table(
        ROOT / "reports/signal-audit-v5-review-20260904",
        pd.Timestamp("2021-08-27"),
        pd.Timestamp("2024-08-26"),
    )
    summary = events.groupby("signal").agg(
        events=("win", "size"), wins=("win", "sum"), interference=("interference", "sum")
    )

    assert len(events) == 89
    assert summary.loc["b"].tolist() == [31, 14, 16]
    assert summary.loc["jf"].tolist() == [23, 13, 10]
    assert summary.loc["s"].tolist() == [35, 20, 13]
    assert np.isfinite(events[[
        "band_position", "macd_level", "macd_delta", "rsi_center", "chip_center"
    ]].to_numpy()).all()


def test_r35_cross_validation_scores_only_held_time_or_symbol_rows():
    from gcn.backtest.signal_research_r35 import build_event_table, cross_validated_scores

    events = build_event_table(
        ROOT / "reports/signal-audit-v5-review-20260904",
        pd.Timestamp("2021-08-27"),
        pd.Timestamp("2024-08-26"),
    )
    temporal = cross_validated_scores(events, "expanding_time")
    symbol = cross_validated_scores(events, "leave_one_symbol")

    assert len(temporal) == 55
    assert pd.to_datetime(temporal.date).min() >= pd.Timestamp("2022-08-27")
    assert not temporal.duplicated(["symbol", "signal", "date"]).any()
    assert set(temporal.cv_scheme) == {"expanding_time"}
    assert len(symbol) == len(events)
    assert not symbol.duplicated(["symbol", "signal", "date"]).any()
    assert (symbol["fold"] == symbol["symbol"]).all()
    assert symbol.quality_tier.notna().all()


def test_r35_training_screen_applies_all_preregistered_gates_without_validation():
    from gcn.backtest.signal_research_r35 import training_screen

    result = training_screen(ROOT / "reports/signal-audit-v5-review-20260904")
    metrics = result["metrics"].set_index("scheme")

    assert len(result["events"]) == 89
    assert len(result["expanding_time"]) == 55
    assert len(result["leave_one_symbol"]) == 89
    assert set(metrics.index) == {"in_sample", "expanding_time", "leave_one_symbol"}
    assert metrics.loc["in_sample", "high_events"] == 31
    assert np.isclose(metrics.loc["in_sample", "high_win_rate_pct"], 67.742, atol=0.001)
    assert metrics.loc["in_sample", "failures"] == "symbols"
    assert metrics.loc["expanding_time", "high_events"] == 16
    assert metrics.loc["expanding_time", "high_win_rate_pct"] == 50.0
    assert np.isclose(
        metrics.loc["expanding_time", "high_interference_rate_pct"], 43.75
    )
    assert np.isclose(
        metrics.loc["leave_one_symbol", "high_win_rate_pct"], 51.724, atol=0.001
    )
    assert np.isclose(
        metrics.loc["leave_one_symbol", "high_interference_rate_pct"], 48.276, atol=0.001
    )
    assert result["decision"]["status"] == "no_candidate"
    assert result["decision"]["validation_run"] is False


def test_r35_runner_writes_training_only_no_candidate_archive(tmp_path):
    from gcn.backtest.signal_research_r35 import run_training_screen

    output = tmp_path / "r35"
    result = run_training_screen(
        ROOT / "reports/signal-audit-v5-review-20260904", output
    )

    assert result["decision"]["status"] == "no_candidate"
    assert result["decision"]["candidate"] is None
    assert result["decision"]["validation_run"] is False
    assert result["decision"]["quality_fields_implemented"] is False
    assert result["decision"]["production_changed"] is False
    assert set(result["manifest"]["outputs"]) == {
        "events.csv", "in_sample.csv", "expanding_time.csv",
        "leave_one_symbol.csv", "metrics.csv", "tier_summary.csv",
        "model.json", "decision.json", "protocol.md",
    }
    metrics = result["metrics"].set_index("scheme")
    for scheme in metrics.index:
        row = metrics.loc[scheme]
        assert np.isclose(row.high_coverage_pct, 100 * row.high_events / row.events)
        assert np.isclose(
            row.win_improvement_pp, row.high_win_rate_pct - row.base_win_rate_pct
        )
        assert np.isclose(
            row.interference_reduction_pp,
            row.base_interference_rate_pct - row.high_interference_rate_pct,
        )
    assert result["decision"]["training_passed"] == bool(metrics.passed.all())
    assert "validation" not in result


def test_r35_formal_training_archive_is_complete_and_self_verifying():
    archive = ROOT / "reports/gcn-historical-r35-20260908/training"
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((archive / "decision.json").read_text(encoding="utf-8"))

    assert set(path.name for path in archive.iterdir()) == {
        "events.csv", "in_sample.csv", "expanding_time.csv",
        "leave_one_symbol.csv", "metrics.csv", "tier_summary.csv",
        "model.json", "decision.json", "protocol.md", "manifest.json",
    }
    assert manifest["protocol_sha256"] == (
        "04cca422cedba4c74039cc7eb9d7bf80ca64091eb0ca5a10b91b469da9d579b9"
    )
    for name, expected in manifest["outputs"].items():
        actual = hashlib.sha256((archive / name).read_bytes()).hexdigest()
        assert actual == expected
    assert decision["status"] == "no_candidate"
    assert decision["validation_run"] is False
    assert decision["quality_fields_implemented"] is False
