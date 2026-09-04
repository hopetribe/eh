# -*- coding: utf-8 -*-
"""v6 前向影子验证的预注册、完整性与成熟度测试。"""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from gcn.backtest.engine import PRESETS, V5_RECOMMENDED_PRESET
from gcn.backtest.shadow_validation import (
    canonical_bar_hash, canonical_spec_hash, load_spec, merge_accepted_bars,
    parse_spec_json, validate_spec,
)
from gcn.backtest.shadow_runner import (
    build_pre_ready_ledger, derive_shadow_boundaries,
    label_affected_events, maturity_gate_passes, reset_v5_confirmation_window,
    run_shadow_update, resolve_accrual_phase, summarize_shadow_window,
)
from gcn.recipes.gcn_main import VERSIONS


_SPEC_PATH = (
    Path(__file__).resolve().parents[1]
    / "gcn"
    / "backtest"
    / "shadow_specs"
    / "v6-profit-arm20-keep50-20260905.json"
)


def _minimal_spec():
    return {
        "schema_version": "gcn-forward-oos-prereg-v1",
        "experiment_id": "v6-profit-arm20-keep50-20260905",
        "immutable": True,
        "frozen_on": "2026-09-05",
        "parent_evidence": {},
        "candidate_selection_audit": {},
        "strategies": {},
        "universe": {},
        "boundaries": {},
        "maturity": {},
        "evaluation": {},
        "decision": {},
    }


def _frozen_spec():
    return parse_spec_json(_SPEC_PATH.read_text(encoding="utf-8"))


def _expect_value_error(message, callback):
    try:
        callback()
    except ValueError as error:
        assert message in str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def _set_path(value, path, replacement):
    current = value
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = replacement


def _bars(periods=4):
    index = pd.date_range("2026-09-01", periods=periods, freq="D")
    return pd.DataFrame({
        "open": [100.0 + value for value in range(periods)],
        "high": [102.0 + value for value in range(periods)],
        "low": [99.0 + value for value in range(periods)],
        "close": [101.0 + value for value in range(periods)],
        "volume": [1000.0 + value for value in range(periods)],
    }, index=index)


def test_canonical_spec_hash_ignores_key_order_and_whitespace():
    left = {"schema_version": 1, "nested": {"name": "盈利保护", "value": 20}}
    right = json.loads(
        '{\n  "nested": {"value": 20, "name": "盈利保护"},\n'
        '  "schema_version": 1\n}'
    )

    assert canonical_spec_hash(left) == canonical_spec_hash(right)


def test_parse_spec_json_rejects_duplicate_keys():
    try:
        parse_spec_json('{"schema_version": 1, "schema_version": 2}')
    except ValueError as error:
        assert "重复字段" in str(error)
    else:
        raise AssertionError("duplicate JSON keys must be rejected")


def test_parse_spec_json_rejects_nonfinite_constants():
    for value in ("NaN", "Infinity", "-Infinity"):
        try:
            parse_spec_json('{"threshold": ' + value + "}")
        except ValueError as error:
            assert "非有限" in str(error)
        else:
            raise AssertionError(f"{value} must be rejected")


def test_validate_spec_rejects_unknown_top_level_fields():
    spec = _minimal_spec()
    spec["threshold_override"] = 0.9

    _expect_value_error("未知字段", lambda: validate_spec(spec))


def test_frozen_forward_spec_registers_only_keep50_candidate():
    spec = _frozen_spec()
    validate_spec(spec)

    assert spec["candidate_selection_audit"]["selected_candidate_id"] == (
        "profit-arm20-keep50"
    )
    assert spec["candidate_selection_audit"]["excluded_candidate_ids"] == [
        "profit-arm20-keep20"
    ]
    assert spec["strategies"]["challenger"]["profit_keep_bps"] == 5000
    assert spec["boundaries"]["signal_cutoff_exclusive"] == "2026-09-05"
    assert spec["maturity"]["at_36_months"] == {
        "affected_exits_min": 15,
        "affected_symbols_min": 7,
    }
    assert spec["maturity"]["at_48_months"] == {
        "affected_exits_min": 20,
        "affected_symbols_min": 8,
    }


def test_frozen_spec_discards_embargo_orders_and_starts_flat():
    spec = _frozen_spec()

    assert spec["universe"]["initial_state"] == (
        "flat_cash_1_per_symbol_no_pending_orders"
    )
    assert spec["universe"]["carry_in_positions"] is False
    assert spec["universe"]["carry_in_pending_orders"] is False
    assert spec["boundaries"]["embargo_order_policy"] == (
        "discard_all_decisions_and_pending_orders"
    )
    assert spec["boundaries"]["first_eligible_decision"] == (
        "actual_accrual_start_CLOSE"
    )
    assert spec["boundaries"]["pre_accrual_setup_policy"] == (
        "discard_pre_accrual_B_SETUP_confirmation_state"
    )


def test_validate_spec_rejects_unknown_nested_fields():
    spec = _frozen_spec()
    spec["strategies"]["challenger"]["threshold_override"] = 0.9

    _expect_value_error("未知字段", lambda: validate_spec(spec))


def test_validate_spec_rejects_boolean_for_integer_field():
    spec = _frozen_spec()
    spec["decision"]["formal_evaluations_allowed"] = True

    _expect_value_error("formal_evaluations_allowed", lambda: validate_spec(spec))


def test_validate_spec_rejects_candidate_substitution():
    spec = _frozen_spec()
    spec["candidate_selection_audit"]["selected_candidate_id"] = (
        "profit-arm20-keep20"
    )

    _expect_value_error("selected_candidate_id", lambda: validate_spec(spec))


def test_validate_spec_rejects_frozen_protocol_drift():
    cases = [
        (("schema_version",), "gcn-forward-oos-prereg-v2"),
        (("immutable",), False),
        (("strategies", "challenger", "profit_keep_bps"), 2000),
        (("boundaries", "signal_cutoff_exclusive"), "2026-08-27"),
        (("boundaries", "intermediate_counts_can_lock_end"), True),
        (("maturity", "at_36_months", "affected_exits_min"), 14),
        (("evaluation", "stress_reuses_base_orders"), False),
        (("evaluation", "bootstrap", "seed"), 7),
        (("decision", "formal_evaluations_allowed"), 2),
    ]

    for path, replacement in cases:
        spec = _frozen_spec()
        _set_path(spec, path, replacement)
        _expect_value_error("必须固定", lambda: validate_spec(spec))


def test_validate_spec_rejects_any_registered_content_drift():
    cases = [
        (("strategies", "incumbent", "version"), "v3"),
        (("strategies", "challenger", "arm_sticky"), False),
        (("strategies", "challenger", "effective_floor_formula"), "min(...)"),
        (("strategies", "challenger", "rule_cost_bps_per_side"), -10),
        (("universe", "core_symbols"), ["NVDA"] * 10),
        (("boundaries", "maximum_accrual_months"), 12),
        (("maturity", "common", "forward_horizons_sessions"), []),
        (("maturity", "at_48_months", "affected_symbols_min"), 99),
        (("evaluation", "bootstrap", "replications"), 0),
        (("decision", "pre_ready_visible_fields"), ["return"]),
        (("decision", "states"), []),
    ]

    for path, replacement in cases:
        spec = _frozen_spec()
        _set_path(spec, path, replacement)
        _expect_value_error("注册哈希", lambda: validate_spec(spec))


def test_load_spec_rejects_mismatched_detached_hash():
    with TemporaryDirectory() as directory:
        spec_path = Path(directory) / _SPEC_PATH.name
        spec_path.write_text(_SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        spec_path.with_suffix(".sha256").write_text("0" * 64 + "\n", encoding="ascii")

        _expect_value_error("哈希不匹配", lambda: load_spec(spec_path))


def test_load_spec_binds_registered_experiment_to_filename():
    with TemporaryDirectory() as directory:
        spec_path = Path(directory) / "renamed-experiment.json"
        spec = _frozen_spec()
        spec_path.write_text(_SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        spec_path.with_suffix(".sha256").write_text(
            canonical_spec_hash(spec) + "\n", encoding="ascii",
        )

        _expect_value_error("文件名", lambda: load_spec(spec_path))


def test_frozen_spec_detached_hash_matches_content():
    spec = load_spec(_SPEC_PATH)

    assert canonical_spec_hash(spec) == _SPEC_PATH.with_suffix(".sha256").read_text(
        encoding="ascii"
    ).strip()


def test_merge_accepted_bars_allows_prefix_truncation_and_appends_new_rows():
    full = _bars()
    accepted = full.iloc[:3]
    incoming = full.iloc[1:]

    merged = merge_accepted_bars(accepted, incoming)

    pd.testing.assert_frame_equal(merged, full, check_freq=False)


def test_merge_accepted_bars_rejects_rewritten_overlap():
    accepted = _bars(3)
    incoming = _bars(4).iloc[1:].copy()
    incoming.loc[incoming.index[0], "open"] += 0.25

    _expect_value_error(
        "已接受K线被改写",
        lambda: merge_accepted_bars(accepted, incoming),
    )


def test_canonical_bar_hash_tracks_dates_and_exact_ohlcv_values():
    bars = _bars()
    same = bars.copy()
    changed = bars.copy()
    changed.loc[changed.index[-1], "close"] += 0.001

    assert canonical_bar_hash(bars) == canonical_bar_hash(same)
    assert canonical_bar_hash(bars) != canonical_bar_hash(changed)


def test_shadow_candidate_is_not_registered_in_production_v5_surfaces():
    assert VERSIONS == ("v3", "v4", "v4-exp", "v5")
    assert "profit_keep" not in V5_RECOMMENDED_PRESET
    assert all("profit_keep" not in preset for preset in PRESETS)
    assert "profit-arm20-keep50" not in json.dumps(
        [V5_RECOMMENDED_PRESET, *PRESETS], ensure_ascii=False,
    )


def test_shadow_accrual_starts_on_sixth_core_common_session_after_cutoff():
    spec = _frozen_spec()
    dates = pd.to_datetime([
        "2026-09-04", "2026-09-08", "2026-09-09", "2026-09-10",
        "2026-09-11", "2026-09-14", "2026-09-15",
    ])
    template = _bars(len(dates)).set_axis(dates)
    frames = {
        symbol: template.copy()
        for symbol in spec["universe"]["core_symbols"]
    }

    boundaries = derive_shadow_boundaries(spec, frames)

    assert boundaries["state"] == "ACCRUING_36M"
    assert boundaries["elapsed_common_sessions"] == 6
    assert boundaries["actual_accrual_start"] == "2026-09-15"


def test_signal_cutoff_date_itself_is_not_a_forward_session():
    spec = _frozen_spec()
    dates = pd.to_datetime([
        "2026-09-05", "2026-09-08", "2026-09-09", "2026-09-10",
        "2026-09-11", "2026-09-14",
    ])
    template = _bars(len(dates)).set_axis(dates)
    frames = {
        symbol: template.copy()
        for symbol in spec["universe"]["core_symbols"]
    }

    boundaries = derive_shadow_boundaries(spec, frames)

    assert boundaries["state"] == "INITIAL_EMBARGO"
    assert boundaries["elapsed_common_sessions"] == 5
    assert boundaries["actual_accrual_start"] is None


def test_shadow_blocks_when_core_forward_calendars_diverge():
    spec = _frozen_spec()
    dates = pd.to_datetime([
        "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11",
    ])
    template = _bars(len(dates)).set_axis(dates)
    frames = {
        symbol: template.copy()
        for symbol in spec["universe"]["core_symbols"]
    }
    frames[spec["universe"]["core_symbols"][-1]] = template.drop(dates[2])

    _expect_value_error(
        "前向交易日不一致",
        lambda: derive_shadow_boundaries(spec, frames),
    )


def test_initial_shadow_ledger_discloses_only_preregistered_fields():
    spec = _frozen_spec()
    template = _bars()
    frames = {
        symbol: template.copy()
        for symbol in spec["universe"]["core_symbols"]
    }

    ledger = build_pre_ready_ledger(
        spec,
        frames,
    )

    assert set(ledger) == set(spec["decision"]["pre_ready_visible_fields"])
    assert ledger["state"] == "INITIAL_EMBARGO"
    assert ledger["elapsed_common_sessions"] == 0
    assert ledger["accepted_bar_hashes"] == {
        symbol: canonical_bar_hash(template) for symbol in sorted(frames)
    }
    assert set(ledger["source_hashes"]) == {
        "gcn/backtest/engine.py",
        "gcn/backtest/shadow_runner.py",
        "gcn/core/tdx.py",
        "gcn/recipes/gcn_main.py",
    }
    assert all(
        len(digest) == 64 for digest in ledger["source_hashes"].values()
    )
    assert all(
        ledger[field] == 0
        for field in (
            "incumbent_reference_entries",
            "challenger_armed_cohorts",
            "challenger_armed_symbols",
            "incumbent_active_symbols",
            "incumbent_negative_20_session_blocks",
            "affected_exits",
            "affected_symbols",
            "pending_20_session_labels",
            "pending_60_session_labels",
        )
    )
    assert not set(spec["decision"]["pre_ready_forbidden_metrics"]) & set(ledger)


def test_shadow_ledger_never_reports_pass_for_a_tampered_spec():
    spec = _frozen_spec()
    spec["maturity"]["definitions"]["negative_block"] = "changed after freeze"
    template = _bars()
    frames = {
        symbol: template.copy()
        for symbol in spec["universe"]["core_symbols"]
    }

    _expect_value_error(
        "注册哈希",
        lambda: build_pre_ready_ledger(
            spec,
            frames,
        ),
    )


def test_shadow_update_persists_accepted_bars_and_rejects_later_rewrite():
    spec = _frozen_spec()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "incoming"
        state_root = root / "state"
        data_dir.mkdir()
        original = _bars()
        original.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            original.to_csv(data_dir / f"{symbol}_1d.csv")

        first = run_shadow_update(_SPEC_PATH, data_dir, state_root)
        assert first["state"] == "INITIAL_EMBARGO"

        rewritten = original.copy()
        rewritten.loc[rewritten.index[-1], "close"] += 0.01
        rewritten.to_csv(
            data_dir / f"{spec['universe']['core_symbols'][0]}_1d.csv"
        )

        _expect_value_error(
            "已接受K线被改写",
            lambda: run_shadow_update(_SPEC_PATH, data_dir, state_root),
        )


def test_blocked_calendar_update_does_not_commit_partial_bars():
    spec = _frozen_spec()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "incoming"
        state_root = root / "state"
        data_dir.mkdir()
        baseline = _bars()
        baseline.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            baseline.to_csv(data_dir / f"{symbol}_1d.csv")
        run_shadow_update(_SPEC_PATH, data_dir, state_root)

        future_dates = pd.to_datetime([
            "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11",
        ])
        extended = pd.concat([
            baseline,
            _bars(len(future_dates)).set_axis(future_dates),
        ])
        extended.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            frame = extended
            if symbol == spec["universe"]["core_symbols"][-1]:
                frame = extended.drop(pd.Timestamp("2026-09-10"))
            frame.to_csv(data_dir / f"{symbol}_1d.csv")

        _expect_value_error(
            "前向交易日不一致",
            lambda: run_shadow_update(_SPEC_PATH, data_dir, state_root),
        )

        extended.to_csv(
            data_dir / f"{spec['universe']['core_symbols'][-1]}_1d.csv"
        )
        recovered = run_shadow_update(_SPEC_PATH, data_dir, state_root)

        assert recovered["state"] == "INITIAL_EMBARGO"
        assert recovered["elapsed_common_sessions"] == 4


def test_v5_confirmation_state_is_reset_at_accrual_start():
    index = pd.date_range("2026-09-10", periods=8, freq="D")
    indicator = pd.DataFrame({
        "B_SETUP": False,
        "B_SIGNAL": False,
        "HIGH": 100.0,
        "CLOSE": 90.0,
        "MID": 80.0,
    }, index=index)
    indicator.loc[index[1], "B_SETUP"] = True
    indicator.loc[index[2], ["CLOSE", "B_SIGNAL"]] = [101.0, True]
    start = index[2]

    window = reset_v5_confirmation_window(indicator, start, index[-1])

    assert not bool(window.loc[start, "B_SIGNAL"])
    assert not bool(window["B_SIGNAL"].any())


def test_setup_on_accrual_start_can_confirm_only_on_a_later_close():
    index = pd.date_range("2026-09-15", periods=3, freq="D")
    indicator = pd.DataFrame({
        "B_SETUP": [True, False, False],
        "B_SIGNAL": [True, False, False],
        "HIGH": [100.0, 102.0, 103.0],
        "CLOSE": [90.0, 101.0, 102.0],
        "MID": [80.0, 80.0, 80.0],
    }, index=index)

    window = reset_v5_confirmation_window(indicator, index[0], index[-1])

    assert not bool(window.loc[index[0], "B_SIGNAL"])
    assert bool(window.loc[index[1], "B_SIGNAL"])


def test_shadow_summary_counts_executed_reference_armed_and_affected_positions():
    spec = _frozen_spec()
    index = pd.date_range("2026-09-15", periods=5, freq="D")
    prepared = pd.DataFrame({
        "OPEN": [10.0, 100.0, 110.0, 109.0, 109.0],
        "HIGH": [10.0, 121.0, 111.0, 110.0, 110.0],
        "LOW": [10.0, 99.0, 109.0, 108.0, 108.0],
        "CLOSE": [10.0, 120.0, 110.0, 109.0, 109.0],
        "B_SIGNAL": [True, False, False, False, False],
        "ICON_JUEFAN": False,
        "S_SIGNAL": False,
    }, index=index)
    prepared_by_symbol = {
        symbol: prepared.copy()
        for symbol in spec["universe"]["core_symbols"]
    }

    summary = summarize_shadow_window(spec, prepared_by_symbol)

    assert summary["incumbent_reference_entries"] == 10
    assert summary["incumbent_active_symbols"] == 10
    assert summary["challenger_armed_cohorts"] == 10
    assert summary["challenger_armed_symbols"] == 10
    assert summary["affected_exits"] == 10
    assert summary["affected_symbols"] == 10
    assert len(summary["affected_events"]) == 10
    assert all(event["exit_reason"] == "profit_lock" for event in summary["affected_events"])


def test_pending_profit_lock_is_armed_but_not_an_affected_exit():
    spec = _frozen_spec()
    index = pd.date_range("2026-09-15", periods=3, freq="D")
    prepared = pd.DataFrame({
        "OPEN": [10.0, 100.0, 110.0],
        "HIGH": [10.0, 121.0, 111.0],
        "LOW": [10.0, 99.0, 109.0],
        "CLOSE": [10.0, 120.0, 110.0],
        "B_SIGNAL": [True, False, False],
        "ICON_JUEFAN": False,
        "S_SIGNAL": False,
    }, index=index)

    summary = summarize_shadow_window(spec, {
        symbol: prepared.copy()
        for symbol in spec["universe"]["core_symbols"]
    })

    assert summary["challenger_armed_cohorts"] == 10
    assert summary["affected_exits"] == 0
    assert summary["affected_events"] == []


def test_ledger_enters_accruing_state_on_sixth_session_without_metrics():
    spec = _frozen_spec()
    dates = pd.to_datetime([
        "2026-09-04", "2026-09-08", "2026-09-09", "2026-09-10",
        "2026-09-11", "2026-09-14", "2026-09-15",
    ])
    template = _bars(len(dates)).set_axis(dates)
    frames = {
        symbol: template.copy()
        for symbol in spec["universe"]["core_symbols"]
    }

    ledger = build_pre_ready_ledger(spec, frames)

    assert ledger["state"] == "ACCRUING_36M"
    assert ledger["elapsed_common_sessions"] == 6
    assert set(ledger) == set(spec["decision"]["pre_ready_visible_fields"])
    assert not set(spec["decision"]["pre_ready_forbidden_metrics"]) & set(ledger)


def test_maturity_gate_uses_all_common_and_endpoint_thresholds_inclusively():
    spec = _frozen_spec()
    summary = {
        "incumbent_reference_entries": 45,
        "challenger_armed_cohorts": 24,
        "challenger_armed_symbols": 9,
        "incumbent_active_symbols": 9,
        "incumbent_negative_20_session_blocks": 6,
        "affected_exits": 15,
        "affected_symbols": 7,
    }

    assert maturity_gate_passes(spec, summary, months=36)
    summary["challenger_armed_cohorts"] = 23
    assert not maturity_gate_passes(spec, summary, months=36)


def test_accrual_phase_checks_36_before_48_and_never_locks_between_endpoints():
    spec = _frozen_spec()
    sessions = pd.DatetimeIndex(pd.to_datetime([
        "2026-09-15", "2029-09-14", "2030-01-02", "2030-09-13",
    ]))
    pass36 = {
        "incumbent_reference_entries": 45,
        "challenger_armed_cohorts": 24,
        "challenger_armed_symbols": 9,
        "incumbent_active_symbols": 9,
        "incumbent_negative_20_session_blocks": 6,
        "affected_exits": 15,
        "affected_symbols": 7,
    }
    fail36 = {**pass36, "affected_exits": 14}
    pass48 = {**pass36, "affected_exits": 20, "affected_symbols": 8}

    locked36 = resolve_accrual_phase(
        spec, sessions[:2], lambda _end: pass36,
    )
    assert locked36["state"] == "OUTCOME_EMBARGO_60"
    assert locked36["locked_end"] == "2029-09-14"

    between = resolve_accrual_phase(
        spec,
        sessions[:3],
        lambda end: fail36 if end == pd.Timestamp("2029-09-14") else pass48,
    )
    assert between["state"] == "ACCRUING_TO_48M"
    assert between["locked_end"] is None

    calls = []
    locked48 = resolve_accrual_phase(
        spec,
        sessions,
        lambda end: calls.append(end) or (
            fail36 if end == pd.Timestamp("2029-09-14") else pass48
        ),
    )
    assert calls[:2] == [pd.Timestamp("2029-09-14"), pd.Timestamp("2030-09-13")]
    assert locked48["state"] == "OUTCOME_EMBARGO_60"
    assert locked48["locked_end"] == "2030-09-13"


def test_affected_labels_use_fill_as_session_one_and_wait_full_embargo():
    spec = _frozen_spec()
    locked_end = pd.Timestamp("2029-09-14")
    dates = pd.bdate_range(locked_end, periods=61)
    template = pd.DataFrame({
        "open": 100.0,
        "high": [101.0 + value for value in range(61)],
        "low": [99.0 - value / 10 for value in range(61)],
        "close": [100.0 + value for value in range(61)],
        "volume": 1000.0,
    }, index=dates)
    frames = {
        symbol: template.copy()
        for symbol in spec["universe"]["core_symbols"]
    }
    event = {
        "event_id": "event-1",
        "symbol": "NVDA",
        "trigger_decision_date": "2029-09-13",
        "exit_fill_date": locked_end.date().isoformat(),
        "exit_fill_open": 100.0,
    }

    nineteen = label_affected_events(
        spec, [event], {symbol: frame.iloc[:19] for symbol, frame in frames.items()},
        locked_end,
    )
    twenty = label_affected_events(
        spec, [event], {symbol: frame.iloc[:20] for symbol, frame in frames.items()},
        locked_end,
    )
    before_ready = label_affected_events(
        spec, [event], {symbol: frame.iloc[:60] for symbol, frame in frames.items()},
        locked_end,
    )
    ready = label_affected_events(spec, [event], frames, locked_end)

    assert nineteen["pending_20_session_labels"] == 1
    assert nineteen["pending_60_session_labels"] == 1
    assert nineteen["events"][0]["labels"]["20"] is None
    assert twenty["pending_20_session_labels"] == 0
    assert twenty["pending_60_session_labels"] == 1
    assert before_ready["pending_20_session_labels"] == 0
    assert before_ready["pending_60_session_labels"] == 0
    assert before_ready["post_lock_common_sessions"] == 59
    assert before_ready["state"] == "OUTCOME_EMBARGO_60"
    label60 = before_ready["events"][0]["labels"]["60"]
    assert math.isclose(label60["return"], 0.59)
    assert math.isclose(label60["mfe"], 0.60)
    assert math.isclose(label60["mae"], -0.069)
    assert ready["post_lock_common_sessions"] == 60
    assert ready["state"] == "READY_ONCE"
