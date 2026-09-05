"""r19固定验证：退出改善门槛与训练同源配置，不放宽旧入场规则。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_r19_validation_uses_frozen_exit_improvement_gates_without_relaxing_legacy_entry_gates():
    from gcn.backtest.signal_research_r19_validation import validation_failures
    from gcn.backtest.signal_research_r7_validation import validation_failures as legacy
    base = {"trades": 20, "entry_events": 30, "entry_win": 50., "entry_interference": 40.,
            "cagr": 10., "mdd": 10., "buy_covered": 5, "win": 50., "sharpe": 1., "rule": "v5"}
    assert validation_failures(base, base) == ["no_material_improvement"]
    improved = {**base, "rule": "JF-base-low-confirm2", "win": 55.}
    assert validation_failures(improved, base) == []
    assert "no_entry_improvement" in legacy(improved, base)
    assert validation_failures({**base, "mdd": 9.5}, base) == []
    assert validation_failures({**base, "mdd": 9.51}, base) == ["no_material_improvement"]
    for field, value, expected in (("win", 49., "win"), ("sharpe", .99, "sharpe"),
                                    ("buy_covered", 4, "buy_covered"), ("cagr", 8., "cagr"),
                                    ("entry_win", 49., "entry_win"), ("entry_interference", 41., "entry_interference")):
        assert expected in validation_failures({**improved, "mdd": 9., field: value}, base)
    for value in (None, float("nan")):
        assert "win" in validation_failures({**improved, "win": value}, base)


def test_r19_validation_replays_same_price_and_confirmation_config_with_explicit_exit_checker(tmp_path, monkeypatch):
    import hashlib
    import json
    import numpy as np
    import pandas as pd
    import pytest
    from gcn.backtest.signal_research_r19 import run_training, CHALLENGERS
    from gcn.backtest import signal_research_r19_validation as validation
    from gcn.backtest.signal_research_r19_validation import run_validation
    from gcn.backtest.signal_research_r17 import candidate_failures
    from gcn.backtest.historical_research import load_snapshot
    from gcn.recipes.gcn_main import compute_ehopt10
    shared = validation._run_validation
    def checked(*args, **kwargs):
        assert kwargs["entry_floor_col"] == "ENTRY_FLOOR" and kwargs["entry_floor_confirm_bars"] == 2
        assert kwargs["training_failure_checker"] is candidate_failures
        assert kwargs["validation_failure_checker"] is validation.validation_failures
        return shared(*args, **kwargs)
    monkeypatch.setattr(validation, "_run_validation", checked)
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    training, output = tmp_path / "training", tmp_path / "validation"
    assert run_training(snapshot, training)["selected"] == CHALLENGERS[0]
    before = (training / "manifest.json").read_bytes()
    decision = run_validation(snapshot, training, output)
    rows = pd.read_csv(output / "comparisons.csv").set_index("rule")
    assert list(rows.index) == ["v5", CHALLENGERS[0]]
    assert decision["failures"] == validation.validation_failures(rows.iloc[1].to_dict(), rows.iloc[0].to_dict())
    assert decision["status"] == ("rejected_keep_v5" if decision["failures"] else "passed_validation_pending_stress")
    assert not decision["production_changed"] and decision["recommended"] == "v5"
    frozen = ROOT / "reports/gcn-historical-r13-20260905/validation"
    original_rows = pd.read_csv(frozen / "comparisons.csv").set_index("rule")
    pd.testing.assert_series_equal(rows.loc["v5"], original_rows.loc["v5"], check_exact=True)
    trades = pd.read_csv(output / "trades.csv")
    old_trades = pd.read_csv(frozen / "trades.csv")
    pd.testing.assert_frame_equal(trades.loc[trades.rule.eq("v5"), old_trades.columns].reset_index(drop=True),
                                  old_trades[old_trades.rule.eq("v5")].reset_index(drop=True), check_exact=True)
    assert trades.entry_floor_confirm_bars.eq(2).all()
    assert trades.loc[trades.rule.eq("v5"), "entry_floor_price"].isna().all()
    frames, quality = load_snapshot(snapshot)
    for row in trades[trades.rule.eq(CHALLENGERS[0])].itertuples():
        frame = compute_ehopt10(frames[row.symbol].loc[:"2025-08-26"], version="v5")
        s = frame.index.get_loc(pd.Timestamp(row.entry_date)) - 1
        signal = frame.iloc[s]
        enabled = bool(signal.ICON_JUEFAN and not signal.B_SIGNAL)
        assert row.entry_b == bool(signal.B_SIGNAL) and row.entry_jf == bool(signal.ICON_JUEFAN)
        assert row.entry_signal_date == frame.index[s].date().isoformat()
        assert pd.notna(row.entry_floor_price) == enabled
        floor = frame.LOW.iloc[s-2:s+1].min()
        if enabled:
            assert np.isclose(row.entry_floor_price, floor, rtol=1e-12)
        base = frame.loc["2024-08-27":"2025-08-26"]
        i = base.index.get_loc(pd.Timestamp(row.entry_date))
        terminal = row.exit_reason == "terminal"
        j = len(base) if terminal else base.index.get_loc(pd.Timestamp(row.exit_date))
        price = base.CLOSE.iloc[-1] if terminal else base.OPEN.iloc[j]
        assert np.isclose(row.return_pct, (price/base.OPEN.iloc[i] * .999**2 - 1)*100)
        if row.exit_reason == "entry_floor":
            below = base.CLOSE.iloc[i:j].lt(floor).to_numpy()
            hits = np.flatnonzero(below[1:] & below[:-1]) + 1
            assert enabled and len(hits) and hits[0] == j-i-1 and not base.S_SIGNAL.iloc[j-1]
    events = pd.read_csv(output / "events.csv")
    pd.testing.assert_frame_equal(events[events.rule.eq("v5")].drop(columns="rule").reset_index(drop=True),
                                  events[events.rule.eq(CHALLENGERS[0])].drop(columns="rule").reset_index(drop=True))
    assert events.outcome_date.max() <= "2025-08-26"
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["entry_floor_confirm_bars"] == 2 and manifest["entry_floor_col"] == "ENTRY_FLOOR"
    assert manifest["training_manifest_sha256"] == hashlib.sha256(before).hexdigest()
    assert manifest["source_quality"] == quality and (training / "manifest.json").read_bytes() == before
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        run_validation(snapshot, training, output)


def test_r19_validation_rejects_missing_or_tampered_floor_config_and_rechecks_training_coverage(tmp_path, monkeypatch):
    import hashlib
    import json
    import pandas as pd
    import pytest
    from gcn.backtest.signal_research_r19 import run_training, CHALLENGERS
    from gcn.backtest.signal_research_r19_validation import run_validation
    from gcn.backtest import signal_research_r7_validation as shared
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    training, output = tmp_path / "training", tmp_path / "validation"
    run_training(snapshot, training)
    manifest_path = training / "manifest.json"
    original = manifest_path.read_bytes()
    def stop_before_prices(*args, **kwargs):
        raise AssertionError("stop before validation prices")
    monkeypatch.setattr(shared, "load_snapshot", stop_before_prices)
    changes = [("entry_floor_confirm_bars", value) for value in (None, True, 0, 1, 2., 3, "2")]
    changes += [("entry_floor_col", value) for value in (None, "OTHER_FLOOR")]
    for key, value in changes:
        changed = json.loads(original)
        if value is None:
            changed.pop(key)
        else:
            changed[key] = value
        manifest_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="底部失效配置"):
            run_validation(snapshot, training, output)
        assert not output.exists()
    manifest_path.write_bytes(original)
    table_path = training / "training.csv"
    raw_table = table_path.read_bytes()
    table = pd.read_csv(table_path)
    table.loc[table.rule.eq(CHALLENGERS[0]), "buy_covered"] = table.loc[table.rule.eq("v5"), "buy_covered"].iloc[0] - 1
    table.to_csv(table_path, index=False)
    changed = json.loads(original)
    changed["outputs"]["training.csv"] = hashlib.sha256(table_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="唯一候选"):
        run_validation(snapshot, training, output)
    table_path.write_bytes(raw_table)
    manifest_path.write_bytes(original)
    with pytest.raises(AssertionError, match="stop before validation prices"):
        run_validation(snapshot, training, output)
    assert not output.exists()


def test_r19_shared_validator_preserves_legacy_r7_results_and_absent_floor_schema(tmp_path):
    import json
    import pandas as pd
    from gcn.backtest.signal_research_r7 import run_training
    from gcn.backtest.signal_research_r7_validation import run_validation
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    training, output = tmp_path / "training", tmp_path / "validation"
    run_training(snapshot, training)
    run_validation(snapshot, training, output)
    frozen = ROOT / "reports/gcn-historical-r7-20260905/validation"
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert "entry_floor_col" not in manifest and "entry_floor_confirm_bars" not in manifest
    assert not {"entry_floor_price", "entry_floor_confirm_bars"} & set(pd.read_csv(output / "trades.csv").columns)
    for name in ("comparisons.csv", "trades.csv", "events.csv", "missed_turns.csv", "decision.json", "protocol.md"):
        assert (output / name).read_bytes() == (frozen / name).read_bytes(), name
