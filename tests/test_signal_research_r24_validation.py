"""r24固定验证；同源规则绑定及真实订单审计。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_r24_validator_reuses_frozen_exit_gates_and_binds_candidate_source_and_marker(monkeypatch):
    from gcn.backtest import signal_research_r24_validation as validation
    from gcn.backtest.signal_research_r19_validation import validation_failures
    from gcn.backtest.signal_research_r17 import candidate_failures
    from gcn.backtest.signal_research_r24 import candidate_signals, RULES, CHALLENGERS
    assert validation.validation_failures is validation_failures
    captured = {}
    def inspect(*args, **kwargs):
        captured.update(kwargs)
        return {"checked": True}
    monkeypatch.setattr(validation, "_run_validation", inspect)
    assert validation.run_validation(Path("snapshot"), Path("training"), Path("output")) == {"checked": True}
    assert captured["entry_joint_pressure_col"] == "ENTRY_JOINT_PRESSURE"
    assert captured["training_failure_checker"] is candidate_failures
    assert captured["validation_failure_checker"] is validation_failures
    assert captured["candidate_builder"] is candidate_signals
    assert captured["rules"] == RULES and captured["challengers"] == CHALLENGERS
    assert captured["extra_validator_sources"] == ("gcn/backtest/signal_research_r19_validation.py",)
    assert captured["validator_source"] == "gcn/backtest/signal_research_r24_validation.py"
    assert captured.get("entry_breakeven_base_col") is None and captured.get("entry_floor_col") is None


def test_r24_validation_binds_same_training_and_reconciles_actual_v5_candidate_orders_and_factors(tmp_path):
    import hashlib
    import json
    import numpy as np
    import pandas as pd
    from gcn.backtest.signal_research_r24 import run_training, CHALLENGERS
    from gcn.backtest.signal_research_r24_validation import run_validation, validation_failures
    from gcn.backtest.historical_research import load_snapshot
    from gcn.recipes.gcn_main import compute_ehopt10
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    frozen_training = ROOT / "reports/gcn-historical-r24-20260905/results"
    training, output = tmp_path / "training", tmp_path / "validation"
    assert run_training(snapshot, training)["selected"] == CHALLENGERS[0]
    for name in json.loads((frozen_training / "manifest.json").read_bytes())["outputs"]:
        assert (training / name).read_bytes() == (frozen_training / name).read_bytes()
    before = (training / "manifest.json").read_bytes()
    decision = run_validation(snapshot, training, output)
    frozen_validation = ROOT / "reports/gcn-historical-r24-20260905/validation"
    for name in json.loads((frozen_validation / "manifest.json").read_bytes())["outputs"]:
        assert (output / name).read_bytes() == (frozen_validation / name).read_bytes(), name
    read = lambda p: pd.read_csv(p, float_precision="round_trip")
    rows = read(output / "comparisons.csv").set_index("rule")
    failures = validation_failures(rows.iloc[1].to_dict(), rows.iloc[0].to_dict())
    assert decision["failures"] == failures and decision["recommended"] == "v5" and not decision["production_changed"]
    assert decision["status"] == ("rejected_keep_v5" if failures else "passed_validation_pending_stress")
    old = ROOT / "reports/gcn-historical-r21-20260905/validation"
    pd.testing.assert_series_equal(rows.loc["v5"], read(old / "comparisons.csv").set_index("rule").loc["v5"], check_exact=True)
    trades = read(output / "trades.csv"); old_trades = read(old / "trades.csv")
    columns = ["rule", "symbol", "entry_date", "exit_date", "return_pct", "hold_days", "exit_reason", "peak_close_pct"]
    pd.testing.assert_frame_equal(trades[trades.rule.eq("v5")][columns].reset_index(drop=True),
                                  old_trades[old_trades.rule.eq("v5")][columns].reset_index(drop=True), check_exact=True)
    frames, quality = load_snapshot(snapshot)
    prepared = {symbol: compute_ehopt10(raw.loc[:"2025-08-26"], version="v5") for symbol, raw in frames.items()}
    for row in trades.itertuples():
        frame = prepared[row.symbol]; i = frame.index.get_loc(pd.Timestamp(row.entry_date))
        j = frame.index.get_loc(pd.Timestamp(row.exit_date)) + int(row.exit_reason == "terminal")
        signal = frame.iloc[i-1]; c = frame.CLOSE.iloc[i:j]; o = frame.OPEN.iloc[i:j]
        assert row.entry_b == bool(signal.B_SIGNAL) and row.entry_jf == bool(signal.ICON_JUEFAN)
        assert row.entry_signal_date == frame.index[i-1].date().isoformat()
        enabled = row.rule == CHALLENGERS[0] and bool(signal.ICON_JUEFAN) and not bool(signal.B_SIGNAL)
        assert row.joint_pressure_enabled == enabled
        price = c.iloc[-1] if row.exit_reason == "terminal" else frame.OPEN.iloc[j]
        assert np.isclose(row.return_pct, (price/o.iloc[0]*.999**2-1)*100)
        if not enabled:
            assert pd.isna(row.joint_pressure_intraday_factor) and pd.isna(row.joint_pressure_trigger_date)
            continue
        day = (c/o).cumprod(); night = (o/c.shift(1)).fillna(1.).cumprod()
        profit = c/o.iloc[0]*.999**2 > 1; hits = profit.cummax() & day.lt(1) & night.lt(1)
        assert np.isclose(row.joint_pressure_intraday_factor, day.iloc[-1])
        assert np.isclose(row.joint_pressure_overnight_factor, night.iloc[-1])
        assert row.joint_pressure_ever_net_positive == profit.any()
        if profit.any():
            assert row.joint_pressure_first_profit_date == c.index[np.flatnonzero(profit)[0]].date().isoformat()
        else:
            assert pd.isna(row.joint_pressure_first_profit_date)
        if hits.any():
            assert np.flatnonzero(hits)[0] == len(c)-1
        if row.exit_reason == "joint_pressure":
            assert hits.iloc[-1] and row.joint_pressure_trigger_date == c.index[-1].date().isoformat()
            assert not frame.S_SIGNAL.iloc[j-1] and c.iloc[-1] > c.max()*.8
    events = read(output / "events.csv")
    pd.testing.assert_frame_equal(events[events.rule.eq("v5")].drop(columns="rule").reset_index(drop=True),
                                  events[events.rule.eq(CHALLENGERS[0])].drop(columns="rule").reset_index(drop=True), check_exact=True)
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["entry_joint_pressure_col"] == "ENTRY_JOINT_PRESSURE"
    assert manifest["joint_pressure_reference_cost"] == .001 and manifest["joint_pressure_factor_threshold"] == 1.
    assert manifest["training_manifest_sha256"] == hashlib.sha256(before).hexdigest()
    assert manifest["source_quality"] == quality and (training / "manifest.json").read_bytes() == before
    assert "gcn/backtest/signal_research_r19_validation.py" in manifest["algorithm_sources"]
    for name, expected in manifest["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    for name, expected in manifest["algorithm_sources"].items():
        assert hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() == expected
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


def test_r24_validation_rejects_changed_fixed_config_and_rehashed_training_failures_before_prices(tmp_path, monkeypatch):
    import hashlib
    import json
    import pandas as pd
    import pytest
    from gcn.backtest.signal_research_r24 import run_training, CHALLENGERS
    from gcn.backtest.signal_research_r24_validation import run_validation
    from gcn.backtest import signal_research_r7_validation as shared
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    training, output = tmp_path / "training", tmp_path / "validation"
    run_training(snapshot, training)
    manifest_path = training / "manifest.json"
    original = manifest_path.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("stop before validation prices")
    monkeypatch.setattr(shared, "load_snapshot", forbidden)
    changes = [("entry_joint_pressure_col", value) for value in (None, "", "OTHER", True)]
    changes += [("joint_pressure_reference_cost", value) for value in (None, False, .0025, "0.001", float("nan"))]
    changes += [("joint_pressure_factor_threshold", value) for value in (None, True, .5, 2., "1", float("inf"))]
    for key, value in changes:
        changed = json.loads(original)
        if value is None:
            changed.pop(key)
        else:
            changed[key] = value
        manifest_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="共同压力配置"):
            run_validation(snapshot, training, output)
        assert not output.exists()
    manifest_path.write_bytes(original)
    csv_path = training / "training.csv"
    raw_csv = csv_path.read_bytes(); table = pd.read_csv(csv_path)
    table.loc[table.rule.eq(CHALLENGERS[0]), "buy_covered"] = table.loc[table.rule.eq("v5"), "buy_covered"].iloc[0]-1
    table.to_csv(csv_path, index=False)
    changed = json.loads(original)
    changed["outputs"]["training.csv"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="唯一候选"):
        run_validation(snapshot, training, output)
    csv_path.write_bytes(raw_csv); manifest_path.write_bytes(original)
    with pytest.raises(AssertionError, match="stop before validation prices"):
        run_validation(snapshot, training, output)
    assert not output.exists()


def test_r24_shared_validation_preserves_old_r21_bytes_and_rejects_stray_joint_config_when_disabled(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import pytest
    from gcn.backtest.signal_research_r21 import run_training
    from gcn.backtest.signal_research_r21_validation import run_validation
    from gcn.backtest import signal_research_r7_validation as shared
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    training, output = tmp_path / "training", tmp_path / "validation"
    run_training(snapshot, training)
    run_validation(snapshot, training, output)
    for path, frozen in ((training, ROOT / "reports/gcn-historical-r21-20260905/results"),
                          (output, ROOT / "reports/gcn-historical-r21-20260905/validation")):
        manifest = json.loads((path / "manifest.json").read_bytes())
        assert not {"entry_joint_pressure_col", "joint_pressure_reference_cost", "joint_pressure_factor_threshold"} & set(manifest)
        assert not any(name.startswith("joint_pressure_") for name in pd.read_csv(path / "trades.csv").columns)
        for name in json.loads((frozen / "manifest.json").read_bytes())["outputs"]:
            assert (path / name).read_bytes() == (frozen / name).read_bytes(), name
    manifest_path = training / "manifest.json"
    original = manifest_path.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled config must fail before prices")
    monkeypatch.setattr(shared, "load_snapshot", forbidden)
    for key, value in (("entry_joint_pressure_col", "ENTRY_JOINT_PRESSURE"),
                       ("joint_pressure_reference_cost", .001), ("joint_pressure_factor_threshold", 1.)):
        changed = json.loads(original); changed[key] = value
        manifest_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="共同压力配置"):
            run_validation(snapshot, training, tmp_path / "disabled")
    assert not (tmp_path / "disabled").exists()
