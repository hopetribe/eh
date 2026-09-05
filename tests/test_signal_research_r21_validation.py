"""r21验证：固定1R/费用配置绑定、原退出门槛及归档来源。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_r21_validator_reuses_frozen_exit_gates_and_declares_checker_source_dependency(monkeypatch):
    from gcn.backtest import signal_research_r21_validation as validation
    from gcn.backtest.signal_research_r19_validation import validation_failures
    from gcn.backtest.signal_research_r17 import candidate_failures
    from gcn.backtest.signal_research_r21 import candidate_signals, RULES, CHALLENGERS
    assert validation.validation_failures is validation_failures
    captured = {}
    def inspect(*args, **kwargs):
        captured.update(kwargs)
        return {"checked": True}
    monkeypatch.setattr(validation, "_run_validation", inspect)
    assert validation.run_validation(Path("snapshot"), Path("training"), Path("output")) == {"checked": True}
    assert captured["entry_breakeven_base_col"] == "ENTRY_BE_BASE"
    assert captured["training_failure_checker"] is candidate_failures
    assert captured["validation_failure_checker"] is validation_failures
    assert captured["candidate_builder"] is candidate_signals
    assert captured["rules"] == RULES and captured["challengers"] == CHALLENGERS
    assert captured["extra_validator_sources"] == ("gcn/backtest/signal_research_r19_validation.py",)
    assert captured["validator_source"] == "gcn/backtest/signal_research_r21_validation.py"
    assert captured.get("entry_floor_col") is None and captured.get("profit_keeps") is None


def test_r21_validation_binds_training_and_audits_r_arm_dates_and_original_v5_orders(tmp_path):
    import hashlib
    import json
    import numpy as np
    import pandas as pd
    import pytest
    from gcn.backtest.signal_research_r21_validation import run_validation, validation_failures
    from gcn.backtest.signal_research_r21 import CHALLENGERS, run_training
    from gcn.backtest.historical_research import load_snapshot
    from gcn.recipes.gcn_main import compute_ehopt10
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    frozen_training = ROOT / "reports/gcn-historical-r21-20260905/results"
    frozen_before = (frozen_training / "manifest.json").read_bytes()
    training, output = tmp_path / "training", tmp_path / "validation"
    run_training(snapshot, training)
    for name in json.loads(frozen_before)["outputs"]:
        assert (training / name).read_bytes() == (frozen_training / name).read_bytes(), name
    before = (training / "manifest.json").read_bytes()
    decision = run_validation(snapshot, training, output)
    rows = pd.read_csv(output / "comparisons.csv").set_index("rule")
    assert decision["failures"] == validation_failures(rows.iloc[1].to_dict(), rows.iloc[0].to_dict())
    assert decision["status"] == ("rejected_keep_v5" if decision["failures"] else "passed_validation_pending_stress")
    assert not decision["production_changed"] and decision["recommended"] == "v5"
    old = ROOT / "reports/gcn-historical-r19-20260905/validation"
    expected = pd.read_csv(old / "comparisons.csv").set_index("rule")
    pd.testing.assert_series_equal(rows.loc["v5"], expected.loc["v5"], check_exact=True)
    trades = pd.read_csv(output / "trades.csv", float_precision="round_trip")
    old_trades = pd.read_csv(old / "trades.csv", float_precision="round_trip")
    columns = ["rule", "symbol", "entry_date", "exit_date", "return_pct", "hold_days", "exit_reason", "peak_close_pct"]
    pd.testing.assert_frame_equal(trades.loc[trades.rule.eq("v5"), columns].reset_index(drop=True),
                                  old_trades.loc[old_trades.rule.eq("v5"), columns].reset_index(drop=True), check_exact=True)
    frames, quality = load_snapshot(snapshot)
    for row in trades.itertuples():
        frame = compute_ehopt10(frames[row.symbol].loc[:"2025-08-26"], version="v5")
        s = frame.index.get_loc(pd.Timestamp(row.entry_signal_date))
        i = frame.index.get_loc(pd.Timestamp(row.entry_date))
        terminal = row.exit_reason == "terminal"
        j = frame.index.get_loc(pd.Timestamp(row.exit_date)) + int(terminal)
        signal = frame.iloc[s]; held = frame.CLOSE.iloc[i:j]; entry = frame.OPEN.iloc[i]
        enabled = row.rule == CHALLENGERS[0] and signal.ICON_JUEFAN and not signal.B_SIGNAL
        assert row.entry_b == bool(signal.B_SIGNAL) and row.entry_jf == bool(signal.ICON_JUEFAN)
        price = held.iloc[-1] if terminal else frame.OPEN.iloc[j]
        assert np.isclose(row.return_pct, (price/entry*.999**2-1)*100)
        if not enabled:
            assert pd.isna(row.breakeven_risk) and not row.breakeven_armed
            continue
        floor = frame.LOW.iloc[s-2:s+1].min(); risk = entry-floor
        assert row.breakeven_base_price == floor and row.breakeven_risk == risk
        armed = held.cummax().ge(entry+risk) if risk > 0 else pd.Series(False, index=held.index)
        assert row.breakeven_armed == armed.any() and pd.notna(row.breakeven_arm_date) == armed.any()
        if armed.any():
            assert row.breakeven_arm_date == held.index[np.flatnonzero(armed)[0]].date().isoformat()
        if row.exit_reason == "breakeven":
            assert np.flatnonzero(armed & held.le(entry/.999**2))[0] == len(held)-1
            assert not frame.S_SIGNAL.iloc[j-1] and held.iloc[-1] > held.max()*.8
    events = pd.read_csv(output / "events.csv")
    pd.testing.assert_frame_equal(events[events.rule.eq("v5")].drop(columns="rule").reset_index(drop=True),
                                  events[events.rule.eq(CHALLENGERS[0])].drop(columns="rule").reset_index(drop=True))
    assert events.outcome_date.max() <= "2025-08-26"
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["entry_breakeven_base_col"] == "ENTRY_BE_BASE"
    assert manifest["breakeven_arm_r"] == 1. and manifest["breakeven_reference_cost"] == .001
    assert "gcn/backtest/signal_research_r19_validation.py" in manifest["algorithm_sources"]
    assert manifest["training_manifest_sha256"] == hashlib.sha256(before).hexdigest()
    assert manifest["source_quality"] == quality and (training / "manifest.json").read_bytes() == before
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["algorithm_sources"].items():
        assert hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    frozen_validation = ROOT / "reports/gcn-historical-r21-20260905/validation"
    for name in json.loads((frozen_validation / "manifest.json").read_bytes())["outputs"]:
        assert (output / name).read_bytes() == (frozen_validation / name).read_bytes(), name
    assert (frozen_training / "manifest.json").read_bytes() == frozen_before
    with pytest.raises(ValueError, match="训练后算法源码变化"):
        run_validation(snapshot, frozen_training, tmp_path / "old-source")
    with pytest.raises(FileExistsError):
        run_validation(snapshot, training, output)


def test_r21_validation_rejects_tampered_breakeven_config_and_training_coverage_before_prices(tmp_path, monkeypatch):
    import hashlib
    import json
    import pandas as pd
    import pytest
    from gcn.backtest.signal_research_r21 import CHALLENGERS, run_training
    from gcn.backtest.signal_research_r21_validation import run_validation
    from gcn.backtest import signal_research_r7_validation as shared
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    training, output = tmp_path / "training", tmp_path / "validation"
    run_training(snapshot, training)
    manifest_path = training / "manifest.json"
    original = manifest_path.read_bytes()
    def stop_before_prices(*args, **kwargs):
        raise AssertionError("stop before validation prices")
    monkeypatch.setattr(shared, "load_snapshot", stop_before_prices)
    changes = [("entry_breakeven_base_col", value) for value in (None, "", "OTHER_BASE", True)]
    changes += [("breakeven_arm_r", value) for value in (None, True, .5, 2., "1", float("nan"))]
    changes += [("breakeven_reference_cost", value) for value in (None, False, .0025, "0.001", float("inf"))]
    for key, value in changes:
        changed = json.loads(original)
        if value is None:
            changed.pop(key)
        else:
            changed[key] = value
        manifest_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="净保本配置"):
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


def test_r21_shared_validator_preserves_legacy_r19_results_and_absent_breakeven_schema(tmp_path):
    import json
    import pandas as pd
    from gcn.backtest.signal_research_r19 import run_training
    from gcn.backtest.signal_research_r19_validation import run_validation
    snapshot = ROOT / "reports/signal-audit-v5-review-20260904"
    training, output = tmp_path / "training", tmp_path / "validation"
    run_training(snapshot, training)
    run_validation(snapshot, training, output)
    frozen = ROOT / "reports/gcn-historical-r19-20260905/validation"
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert not {"entry_breakeven_base_col", "breakeven_arm_r", "breakeven_reference_cost"} & set(manifest)
    assert not any(name.startswith("breakeven_") for name in pd.read_csv(output / "trades.csv").columns)
    for name in ("comparisons.csv", "trades.csv", "events.csv", "missed_turns.csv", "decision.json", "protocol.md"):
        assert (output / name).read_bytes() == (frozen / name).read_bytes(), name
