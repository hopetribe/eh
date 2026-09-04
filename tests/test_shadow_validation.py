# -*- coding: utf-8 -*-
"""v6 前向影子验证的预注册、完整性与成熟度测试。"""
import hashlib
import json
import math
import multiprocessing
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from gcn.backtest import shadow_runner, shadow_validation
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


def _run_shadow_update_in_process(
    spec_path, data_dir, state_root, start_barrier, result_queue,
):
    """Process target kept at module scope so flock is tested across processes."""
    try:
        start_barrier.wait(timeout=15)
        result_queue.put((
            "ok",
            run_shadow_update(
                Path(spec_path), Path(data_dir), Path(state_root),
            ),
        ))
    except BaseException as error:  # pragma: no cover - relayed to parent
        result_queue.put(("error", type(error).__name__, str(error)))


def _rewrite_generation_chain(experiment_dir, mutate_sequence, mutate):
    """Test-only attacker: mutate one generation and rehash its full suffix."""
    registration_hash = hashlib.sha256(
        (experiment_dir / "registration.json").read_bytes()
    ).hexdigest()
    previous_hash = registration_hash
    commits_dir = experiment_dir / "commits"
    generations_dir = experiment_dir / "generations"
    rewritten_head = None
    for sequence in range(
        json.loads((experiment_dir / "CURRENT").read_text(encoding="utf-8"))[
            "sequence"
        ] + 1
    ):
        old_commit = next(commits_dir.glob(f"{sequence:016d}-*.commit"))
        old_generation = next(
            generations_dir.glob(f"{sequence:016d}-*.json")
        )
        generation = json.loads(old_generation.read_text(encoding="utf-8"))
        generation["previous_hash"] = previous_hash
        if sequence == mutate_sequence:
            mutate(generation)
        payload = shadow_runner._canonical_json_bytes(generation)
        generation_hash = hashlib.sha256(payload).hexdigest()
        pointer = {"generation_hash": generation_hash, "sequence": sequence}
        reference = f"{sequence:016d}-{generation_hash}"
        old_commit.unlink()
        old_generation.unlink()
        (generations_dir / f"{reference}.json").write_bytes(payload)
        (commits_dir / f"{reference}.commit").write_bytes(
            shadow_runner._canonical_json_bytes(pointer)
        )
        previous_hash = generation_hash
        rewritten_head = pointer
    (experiment_dir / "CURRENT").write_bytes(
        shadow_runner._canonical_json_bytes(rewritten_head)
    )


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


def test_load_spec_verifies_referenced_power_calibration_artifact():
    with TemporaryDirectory() as directory:
        target_dir = Path(directory)
        spec_path = target_dir / _SPEC_PATH.name
        spec_path.write_bytes(_SPEC_PATH.read_bytes())
        spec_path.with_suffix(".sha256").write_bytes(
            _SPEC_PATH.with_suffix(".sha256").read_bytes()
        )
        spec = _frozen_spec()
        artifact_name = spec["candidate_selection_audit"][
            "power_calibration_artifact"
        ]
        artifact = json.loads(
            (_SPEC_PATH.parent / artifact_name).read_text(encoding="utf-8")
        )
        artifact_hash_path = (_SPEC_PATH.parent / artifact_name).with_suffix(
            ".sha256"
        )
        (target_dir / artifact_name).with_suffix(".sha256").write_bytes(
            artifact_hash_path.read_bytes()
        )
        artifact["tampered_after_registration"] = True
        (target_dir / artifact_name).write_text(
            json.dumps(
                artifact, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )

        _expect_value_error(
            "功效校准",
            lambda: load_spec(spec_path),
        )


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


def test_shadow_rejects_synchronized_weekend_as_common_session():
    spec = _frozen_spec()
    dates = pd.to_datetime([
        "2026-09-04", "2026-09-06", "2026-09-08",
    ])
    template = _bars(len(dates)).set_axis(dates)
    frames = {
        symbol: template.copy()
        for symbol in spec["universe"]["core_symbols"]
    }

    _expect_value_error(
        "冻结美股交易日历",
        lambda: derive_shadow_boundaries(spec, frames),
    )


def test_shadow_rejects_synchronized_labor_day_as_common_session():
    spec = _frozen_spec()
    dates = pd.to_datetime(["2026-09-04", "2026-09-07"])
    template = _bars(len(dates)).set_axis(dates)
    frames = {
        symbol: template.copy()
        for symbol in spec["universe"]["core_symbols"]
    }

    _expect_value_error(
        "冻结美股交易日历",
        lambda: derive_shadow_boundaries(spec, frames),
    )


def test_shadow_rejects_synchronized_missing_regular_market_session():
    spec = _frozen_spec()
    dates = pd.to_datetime([
        "2026-09-04", "2026-09-08", "2026-09-09", "2026-09-10",
        "2026-09-14", "2026-09-15",
    ])
    template = _bars(len(dates)).set_axis(dates)
    frames = {
        symbol: template.copy()
        for symbol in spec["universe"]["core_symbols"]
    }

    _expect_value_error(
        "冻结美股交易日历",
        lambda: derive_shadow_boundaries(spec, frames),
    )


def test_empty_forward_prefix_still_validates_frozen_calendar_hash():
    spec = _frozen_spec()
    template = _bars()
    frames = {
        symbol: template.copy()
        for symbol in spec["universe"]["core_symbols"]
    }
    original_hash = shadow_validation._OBSERVATION_CALENDAR_SHA256
    shadow_validation._OBSERVATION_CALENDAR_SHA256 = "0" * 64
    try:
        _expect_value_error(
            "冻结美股交易日历哈希不匹配",
            lambda: derive_shadow_boundaries(spec, frames),
        )
    finally:
        shadow_validation._OBSERVATION_CALENDAR_SHA256 = original_hash


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
        "gcn/backtest/shadow_specs/"
        "nyse-us-equities-sessions-20260906-20301231.json",
        "gcn/backtest/shadow_evaluation.py",
        "gcn/backtest/shadow_runner.py",
        "gcn/backtest/shadow_validation.py",
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


def test_algorithm_source_hashes_rejects_caller_selected_root():
    with TemporaryDirectory() as directory:
        try:
            shadow_runner.algorithm_source_hashes(Path(directory))
        except TypeError:
            pass
        else:
            raise AssertionError("源码完整性不得由调用方选择替代根目录")


def test_initial_shadow_snapshot_keeps_private_protocol_out_of_public_ledger():
    spec = _frozen_spec()
    template = _bars()
    frames = {
        symbol: template.copy()
        for symbol in spec["universe"]["core_symbols"]
    }

    ledger, protocol = shadow_runner._build_pre_ready_snapshot(spec, frames)

    assert set(ledger) == set(spec["decision"]["pre_ready_visible_fields"])
    assert protocol == {
        "actual_accrual_start": None,
        "affected_events": [],
        "checkpoint_36": None,
        "evaluation_result": None,
        "formal_evaluation_count": 0,
        "locked_end": None,
        "locked_months": None,
        "maturity_36_passed": None,
        "maturity_summary": None,
        "order_artifact_sha256": None,
        "pending_20_session_labels": 0,
        "pending_60_session_labels": 0,
        "post_lock_common_sessions": 0,
        "state": "INITIAL_EMBARGO",
    }
    assert not set(protocol) <= set(ledger)


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


def test_shadow_update_rejects_nonuniform_later_rewrite():
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
            "DATA_BLOCKED",
            lambda: run_shadow_update(_SPEC_PATH, data_dir, state_root),
        )


def test_shadow_update_rebases_uniform_provider_adjustment_before_append():
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

        incoming = baseline.copy()
        incoming.loc[:, ["open", "high", "low", "close"]] /= 2.0
        incoming.loc[:, "volume"] *= 2.0
        future = _bars(1).set_axis(pd.to_datetime(["2026-09-08"]))
        future.loc[:, ["open", "high", "low", "close"]] /= 2.0
        future.loc[:, "volume"] *= 2.0
        incoming = pd.concat([incoming, future])
        incoming.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            incoming.to_csv(data_dir / f"{symbol}_1d.csv")

        ledger = run_shadow_update(_SPEC_PATH, data_dir, state_root)

        expected = pd.concat([
            baseline,
            _bars(1).set_axis(pd.to_datetime(["2026-09-08"])),
        ])
        expected.index.name = "date"
        assert ledger["accepted_bar_hashes"] == {
            symbol: canonical_bar_hash(expected)
            for symbol in sorted(spec["universe"]["core_symbols"])
        }


def test_shadow_update_rejects_runtime_environment_drift():
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

        original_identity = shadow_runner.runtime_environment_identity
        shadow_runner.runtime_environment_identity = lambda: {
            **original_identity(),
            "pandas": "changed-after-registration",
        }
        try:
            _expect_value_error(
                "运行环境",
                lambda: run_shadow_update(_SPEC_PATH, data_dir, state_root),
            )
        finally:
            shadow_runner.runtime_environment_identity = original_identity


def test_shadow_update_rejects_tampered_frozen_prefix_even_if_source_matches():
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
        run_shadow_update(_SPEC_PATH, data_dir, state_root)

        symbol = spec["universe"]["core_symbols"][0]
        tampered = original.copy()
        tampered.loc[tampered.index[0], "close"] += 0.01
        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        base_path = experiment_dir / "base" / f"{symbol}.bars"
        base_payload = json.loads(base_path.read_text(encoding="utf-8"))
        base_payload["rows"][0][4] = float(
            float.fromhex(base_payload["rows"][0][4]) + 0.01
        ).hex()
        base_path.write_text(
            json.dumps(
                base_payload, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        tampered.to_csv(data_dir / f"{symbol}_1d.csv")

        _expect_value_error(
            "冻结前缀",
            lambda: run_shadow_update(_SPEC_PATH, data_dir, state_root),
        )


def test_initial_shadow_update_failure_never_publishes_partial_experiment():
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
        original_write_json = shadow_runner._write_json

        def fail_ledger_write(path, value):
            if path.name == "ledger.json":
                raise OSError("injected ledger write failure")
            return original_write_json(path, value)

        shadow_runner._write_json = fail_ledger_write
        try:
            try:
                run_shadow_update(_SPEC_PATH, data_dir, state_root)
            except OSError as error:
                assert "injected ledger" in str(error)
            else:
                raise AssertionError("injected write failure must surface")
        finally:
            shadow_runner._write_json = original_write_json

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        assert not experiment_dir.exists()


def test_concurrent_initial_shadow_updates_serialize_one_publication():
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

        original_publish = shadow_runner._publish_initial_experiment
        guard = threading.Lock()
        active = 0
        maximum_active = 0

        def delayed_publish(*args, **kwargs):
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                time.sleep(0.1)
                return original_publish(*args, **kwargs)
            finally:
                with guard:
                    active -= 1

        start = threading.Barrier(2)

        def update():
            start.wait()
            return run_shadow_update(_SPEC_PATH, data_dir, state_root)

        shadow_runner._publish_initial_experiment = delayed_publish
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: update(), range(2)))
        finally:
            shadow_runner._publish_initial_experiment = original_publish

        assert results[0] == results[1]
        assert maximum_active == 1


def test_two_processes_flock_the_same_increment_into_one_generation():
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

        addition = _bars(1).set_axis(pd.to_datetime(["2026-09-08"]))
        incoming = pd.concat([baseline, addition])
        incoming.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            incoming.to_csv(data_dir / f"{symbol}_1d.csv")

        context = multiprocessing.get_context("fork")
        start = context.Barrier(2)
        results = context.Queue()
        processes = [
            context.Process(
                target=_run_shadow_update_in_process,
                args=(
                    str(_SPEC_PATH), str(data_dir), str(state_root), start,
                    results,
                ),
            )
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        received = [results.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0

        assert [item[0] for item in received] == ["ok", "ok"]
        assert received[0][1] == received[1][1]
        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        current = json.loads(
            (experiment_dir / "CURRENT").read_text(encoding="utf-8")
        )
        assert current["sequence"] == 1
        assert len(list((experiment_dir / "commits").glob("*.commit"))) == 2
        assert len(list((experiment_dir / "generations").glob("*.json"))) == 2


def test_shadow_update_loads_frozen_spec_once_before_locking():
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

        original_load = shadow_runner.load_spec
        calls = []

        def counted_load(path):
            calls.append(Path(path))
            return original_load(path)

        shadow_runner.load_spec = counted_load
        try:
            run_shadow_update(_SPEC_PATH, data_dir, state_root)
        finally:
            shadow_runner.load_spec = original_load

        assert calls == [_SPEC_PATH]


def test_initial_shadow_publication_fsyncs_files_and_directories():
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

        original_fsync = shadow_runner.os.fsync
        synced_modes = []

        def record_fsync(file_descriptor):
            synced_modes.append(shadow_runner.os.fstat(file_descriptor).st_mode)
            return original_fsync(file_descriptor)

        shadow_runner.os.fsync = record_fsync
        try:
            run_shadow_update(_SPEC_PATH, data_dir, state_root)
        finally:
            shadow_runner.os.fsync = original_fsync

        assert any(stat.S_ISREG(mode) for mode in synced_modes)
        assert any(stat.S_ISDIR(mode) for mode in synced_modes)


def test_initial_shadow_update_publishes_one_immutable_generation():
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

        ledger = run_shadow_update(_SPEC_PATH, data_dir, state_root)
        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        current = json.loads(
            (experiment_dir / "CURRENT").read_text(encoding="utf-8")
        )
        current_ref = (
            f"{current['sequence']:016d}-{current['generation_hash']}"
        )
        generation_path = experiment_dir / "generations" / f"{current_ref}.json"
        commit_path = experiment_dir / "commits" / f"{current_ref}.commit"
        generation_bytes = generation_path.read_bytes()
        generation = json.loads(generation_bytes)
        registration_hash = hashlib.sha256(
            (experiment_dir / "registration.json").read_bytes()
        ).hexdigest()

        assert current == {"generation_hash": current["generation_hash"], "sequence": 0}
        assert json.loads(commit_path.read_text(encoding="utf-8")) == current
        assert hashlib.sha256(generation_bytes).hexdigest() == current["generation_hash"]
        assert generation["schema_version"] == "gcn-shadow-generation-v1"
        assert generation["sequence"] == 0
        assert generation["session"] is None
        assert generation["previous_hash"] == registration_hash
        assert generation["accepted"] == {
            symbol: {
                "canonical_bar_hash": digest,
                "last_date": original.index[-1].date().isoformat(),
                "row_count": len(original),
            }
            for symbol, digest in ledger["accepted_bar_hashes"].items()
        }
        assert json.loads(
            (experiment_dir / "ledger.json").read_text(encoding="utf-8")
        ) == generation["public_ledger"] == ledger
        assert set(path.name for path in (experiment_dir / "base").glob("*.bars")) == {
            f"{symbol}.bars" for symbol in spec["universe"]["core_symbols"]
        }


def test_initial_registration_uses_declared_canonical_json_encoding():
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

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        registration_path = experiment_dir / "registration.json"
        registration = json.loads(registration_path.read_text(encoding="utf-8"))

        assert registration_path.read_bytes() == (
            shadow_runner._canonical_json_bytes(registration)
        )


def test_initial_registration_has_one_strict_authoritative_schema():
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

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        registration = json.loads(
            (experiment_dir / "registration.json").read_text(encoding="utf-8")
        )

        assert set(registration) == {
            "base", "core_symbols", "experiment_id", "implementation",
            "schema_version", "serialization", "signal_cutoff_exclusive",
            "spec_hash",
        }
        assert set(registration["implementation"]) == {
            "runtime_environment", "source_hashes",
        }
        cutoff = pd.Timestamp(spec["boundaries"]["signal_cutoff_exclusive"])
        assert all(
            pd.Timestamp(metadata["last_date"]) <= cutoff
            for metadata in registration["base"].values()
        )


def test_shadow_update_rejects_noncanonical_registration_bytes():
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

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        registration_path = experiment_dir / "registration.json"
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        registration_path.write_text(
            json.dumps(registration, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )

        _expect_value_error(
            "canonical",
            lambda: run_shadow_update(_SPEC_PATH, data_dir, state_root),
        )


def test_authoritative_bar_payload_rejects_noncanonical_encoding():
    payload_value = shadow_runner._canonical_bar_payload(_bars())
    pretty_payload = (
        json.dumps(payload_value, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")

    _expect_value_error(
        "canonical",
        lambda: shadow_runner._frame_from_bar_bytes(pretty_payload, "test bars"),
    )


def test_authoritative_bar_payload_requires_canonical_float_hex():
    payload_value = shadow_runner._canonical_bar_payload(_bars())
    payload_value["rows"][0][1] = "0x1.900p+6"
    payload = shadow_runner._canonical_json_bytes(payload_value)

    _expect_value_error(
        "float.hex",
        lambda: shadow_runner._frame_from_bar_bytes(payload, "test bars"),
    )


def test_shadow_update_commits_each_new_common_session_as_one_generation():
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

        addition = _bars(1).set_axis(pd.to_datetime(["2026-09-08"]))
        extended = pd.concat([baseline, addition])
        extended.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            extended.to_csv(data_dir / f"{symbol}_1d.csv")
        ledger = run_shadow_update(_SPEC_PATH, data_dir, state_root)

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        current = json.loads(
            (experiment_dir / "CURRENT").read_text(encoding="utf-8")
        )
        reference = f"{current['sequence']:016d}-{current['generation_hash']}"
        generation = json.loads(
            (experiment_dir / "generations" / f"{reference}.json").read_text(
                encoding="utf-8"
            )
        )
        genesis_paths = list(
            (experiment_dir / "generations").glob("0000000000000000-*.json")
        )

        assert current["sequence"] == 1
        assert generation["sequence"] == 1
        assert generation["session"] == "2026-09-08"
        assert len(genesis_paths) == 1
        assert generation["previous_hash"] == genesis_paths[0].stem.split("-", 1)[1]
        assert set(generation["rows_by_symbol"]) == set(
            spec["universe"]["core_symbols"]
        )
        assert all(
            row[0] == "2026-09-08"
            for row in generation["rows_by_symbol"].values()
        )
        assert generation["public_ledger"] == ledger
        assert ledger["elapsed_common_sessions"] == 1


def test_late_first_registration_matches_baseline_then_batch_append_bytes():
    spec = _frozen_spec()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        baseline = _bars()
        baseline.index.name = "date"
        future_dates = pd.bdate_range("2026-09-08", periods=5)
        future = _bars(len(future_dates)).set_axis(future_dates)
        extended = pd.concat([baseline, future])
        extended.index.name = "date"

        state_roots = []
        for name, update_mode in (
            ("late", "initial-full"),
            ("batch", "batch"),
            ("daily", "daily"),
        ):
            data_dir = root / f"incoming-{name}"
            state_root = root / f"state-{name}"
            data_dir.mkdir()
            initial = extended if update_mode == "initial-full" else baseline
            for symbol in spec["universe"]["core_symbols"]:
                initial.to_csv(data_dir / f"{symbol}_1d.csv")
            run_shadow_update(_SPEC_PATH, data_dir, state_root)
            if update_mode == "batch":
                for symbol in spec["universe"]["core_symbols"]:
                    extended.to_csv(data_dir / f"{symbol}_1d.csv")
                run_shadow_update(_SPEC_PATH, data_dir, state_root)
            elif update_mode == "daily":
                for session in future_dates:
                    daily = extended.loc[:session]
                    for symbol in spec["universe"]["core_symbols"]:
                        daily.to_csv(data_dir / f"{symbol}_1d.csv")
                    run_shadow_update(_SPEC_PATH, data_dir, state_root)
            state_roots.append(
                state_root / spec["experiment_id"] / canonical_spec_hash(spec)
            )

        def authority_bytes(experiment_dir):
            relative_paths = [Path("registration.json"), Path("CURRENT")]
            relative_paths.extend(sorted(
                path.relative_to(experiment_dir)
                for directory in ("base", "generations", "commits")
                for path in (experiment_dir / directory).glob("*")
            ))
            return {
                str(relative): (experiment_dir / relative).read_bytes()
                for relative in relative_paths
            }

        expected = authority_bytes(state_roots[0])
        assert authority_bytes(state_roots[1]) == expected
        assert authority_bytes(state_roots[2]) == expected


def test_shadow_update_rejects_tampered_committed_generation():
    spec = _frozen_spec()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "incoming"
        state_root = root / "state"
        data_dir.mkdir()
        baseline = _bars()
        baseline.index.name = "date"
        addition = _bars(1).set_axis(pd.to_datetime(["2026-09-08"]))
        extended = pd.concat([baseline, addition])
        extended.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            baseline.to_csv(data_dir / f"{symbol}_1d.csv")
        run_shadow_update(_SPEC_PATH, data_dir, state_root)
        for symbol in spec["universe"]["core_symbols"]:
            extended.to_csv(data_dir / f"{symbol}_1d.csv")
        run_shadow_update(_SPEC_PATH, data_dir, state_root)

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        current = json.loads(
            (experiment_dir / "CURRENT").read_text(encoding="utf-8")
        )
        reference = f"{current['sequence']:016d}-{current['generation_hash']}"
        generation_path = experiment_dir / "generations" / f"{reference}.json"
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        generation["public_ledger"]["elapsed_common_sessions"] = 999
        generation_path.write_text(
            json.dumps(
                generation, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )

        _expect_value_error(
            "generation哈希",
            lambda: run_shadow_update(_SPEC_PATH, data_dir, state_root),
        )


def test_shadow_update_rejects_current_without_commit_marker():
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

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        current = json.loads(
            (experiment_dir / "CURRENT").read_text(encoding="utf-8")
        )
        reference = f"{current['sequence']:016d}-{current['generation_hash']}"
        (experiment_dir / "commits" / f"{reference}.commit").unlink()

        _expect_value_error(
            "commit",
            lambda: run_shadow_update(_SPEC_PATH, data_dir, state_root),
        )


def test_shadow_update_rejects_generation_with_broken_previous_hash():
    spec = _frozen_spec()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "incoming"
        state_root = root / "state"
        data_dir.mkdir()
        baseline = _bars()
        baseline.index.name = "date"
        addition = _bars(1).set_axis(pd.to_datetime(["2026-09-08"]))
        extended = pd.concat([baseline, addition])
        extended.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            baseline.to_csv(data_dir / f"{symbol}_1d.csv")
        run_shadow_update(_SPEC_PATH, data_dir, state_root)
        for symbol in spec["universe"]["core_symbols"]:
            extended.to_csv(data_dir / f"{symbol}_1d.csv")
        run_shadow_update(_SPEC_PATH, data_dir, state_root)

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        current_path = experiment_dir / "CURRENT"
        current = json.loads(current_path.read_text(encoding="utf-8"))
        old_reference = f"{current['sequence']:016d}-{current['generation_hash']}"
        old_generation_path = (
            experiment_dir / "generations" / f"{old_reference}.json"
        )
        generation = json.loads(old_generation_path.read_text(encoding="utf-8"))
        generation["previous_hash"] = "0" * 64
        generation_payload = shadow_runner._canonical_json_bytes(generation)
        generation_hash = hashlib.sha256(generation_payload).hexdigest()
        replacement = {"generation_hash": generation_hash, "sequence": 1}
        new_reference = f"{1:016d}-{generation_hash}"
        old_generation_path.unlink()
        (experiment_dir / "commits" / f"{old_reference}.commit").unlink()
        (experiment_dir / "generations" / f"{new_reference}.json").write_bytes(
            generation_payload
        )
        commit_payload = shadow_runner._canonical_json_bytes(replacement)
        (experiment_dir / "commits" / f"{new_reference}.commit").write_bytes(
            commit_payload
        )
        current_path.write_bytes(commit_payload)

        _expect_value_error(
            "previous_hash",
            lambda: run_shadow_update(_SPEC_PATH, data_dir, state_root),
        )


def test_shadow_update_replays_committed_rows_when_incoming_is_stale():
    spec = _frozen_spec()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "incoming"
        state_root = root / "state"
        data_dir.mkdir()
        baseline = _bars()
        baseline.index.name = "date"
        addition = _bars(1).set_axis(pd.to_datetime(["2026-09-08"]))
        extended = pd.concat([baseline, addition])
        extended.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            baseline.to_csv(data_dir / f"{symbol}_1d.csv")
        run_shadow_update(_SPEC_PATH, data_dir, state_root)
        for symbol in spec["universe"]["core_symbols"]:
            extended.to_csv(data_dir / f"{symbol}_1d.csv")
        committed = run_shadow_update(_SPEC_PATH, data_dir, state_root)

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        current_before = (experiment_dir / "CURRENT").read_bytes()
        for symbol in spec["universe"]["core_symbols"]:
            baseline.to_csv(data_dir / f"{symbol}_1d.csv")

        replayed = run_shadow_update(_SPEC_PATH, data_dir, state_root)

        assert replayed == committed
        assert replayed["elapsed_common_sessions"] == 1
        assert (experiment_dir / "CURRENT").read_bytes() == current_before


def test_shadow_update_repairs_derived_accepted_bar_cache_from_authority():
    spec = _frozen_spec()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "incoming"
        state_root = root / "state"
        data_dir.mkdir()
        baseline = _bars()
        addition = _bars(1).set_axis(pd.to_datetime(["2026-09-08"]))
        extended = pd.concat([baseline, addition])
        extended.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            extended.to_csv(data_dir / f"{symbol}_1d.csv")
        expected = run_shadow_update(_SPEC_PATH, data_dir, state_root)

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        symbol = spec["universe"]["core_symbols"][0]
        cache_path = experiment_dir / "accepted_bars" / f"{symbol}_1d.csv"
        cache_path.write_text("broken derived cache\n", encoding="utf-8")

        recovered = run_shadow_update(_SPEC_PATH, data_dir, state_root)

        assert recovered == expected
        pd.testing.assert_frame_equal(
            shadow_runner._read_bar_csv(cache_path), extended,
            check_exact=True,
        )


def test_shadow_update_repairs_missing_current_from_commits():
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
        expected = run_shadow_update(_SPEC_PATH, data_dir, state_root)

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        current_path = experiment_dir / "CURRENT"
        original_current = current_path.read_bytes()
        current_path.unlink()

        recovered = run_shadow_update(_SPEC_PATH, data_dir, state_root)

        assert recovered == expected
        assert current_path.read_bytes() == original_current


def test_shadow_update_rejects_commit_tail_loss_instead_of_silent_rollback():
    spec = _frozen_spec()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "incoming"
        state_root = root / "state"
        data_dir.mkdir()
        baseline = _bars()
        addition = _bars(1).set_axis(pd.to_datetime(["2026-09-08"]))
        extended = pd.concat([baseline, addition])
        extended.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            extended.to_csv(data_dir / f"{symbol}_1d.csv")
        run_shadow_update(_SPEC_PATH, data_dir, state_root)

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        current = json.loads(
            (experiment_dir / "CURRENT").read_text(encoding="utf-8")
        )
        reference = shadow_runner._generation_reference(current)
        (experiment_dir / "commits" / f"{reference}.commit").unlink()
        (experiment_dir / "generations" / f"{reference}.json").unlink()

        _expect_value_error(
            "权威头回退",
            lambda: run_shadow_update(_SPEC_PATH, data_dir, state_root),
        )


def test_uncommitted_orphan_generation_is_ignored_before_normal_append():
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

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        orphan_payload = b'{"uncommitted":true}\n'
        orphan_hash = hashlib.sha256(orphan_payload).hexdigest()
        orphan_path = (
            experiment_dir / "generations"
            / f"{1:016d}-{orphan_hash}.json"
        )
        orphan_path.write_bytes(orphan_payload)

        addition = _bars(1).set_axis(pd.to_datetime(["2026-09-08"]))
        extended = pd.concat([baseline, addition])
        extended.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            extended.to_csv(data_dir / f"{symbol}_1d.csv")
        ledger = run_shadow_update(_SPEC_PATH, data_dir, state_root)
        current = json.loads(
            (experiment_dir / "CURRENT").read_text(encoding="utf-8")
        )

        assert orphan_path.is_file()
        assert current["sequence"] == 1
        assert current["generation_hash"] != orphan_hash
        assert ledger["elapsed_common_sessions"] == 1


def test_shadow_update_rejects_tampered_non_head_generation():
    spec = _frozen_spec()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "incoming"
        state_root = root / "state"
        data_dir.mkdir()
        baseline = _bars()
        baseline.index.name = "date"
        additions = _bars(2).set_axis(pd.to_datetime([
            "2026-09-08", "2026-09-09",
        ]))
        extended = pd.concat([baseline, additions])
        extended.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            extended.to_csv(data_dir / f"{symbol}_1d.csv")
        run_shadow_update(_SPEC_PATH, data_dir, state_root)

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        non_head = next(
            (experiment_dir / "generations").glob("0000000000000001-*.json")
        )
        non_head.write_bytes(non_head.read_bytes() + b" ")

        _expect_value_error(
            "generation哈希",
            lambda: run_shadow_update(_SPEC_PATH, data_dir, state_root),
        )


def test_shadow_update_rejects_rehashed_non_head_semantic_count_tamper():
    spec = _frozen_spec()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "incoming"
        state_root = root / "state"
        data_dir.mkdir()
        baseline = _bars()
        baseline.index.name = "date"
        additions = _bars(2).set_axis(pd.to_datetime([
            "2026-09-08", "2026-09-09",
        ]))
        extended = pd.concat([baseline, additions])
        extended.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            extended.to_csv(data_dir / f"{symbol}_1d.csv")
        run_shadow_update(_SPEC_PATH, data_dir, state_root)

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        _rewrite_generation_chain(
            experiment_dir,
            1,
            lambda generation: generation["public_ledger"].update({
                "incumbent_reference_entries": 1,
            }),
        )

        _expect_value_error(
            "generation语义重算",
            lambda: run_shadow_update(_SPEC_PATH, data_dir, state_root),
        )


def test_shadow_update_rejects_rehashed_non_head_state_and_checkpoint_tamper():
    mutations = (
        lambda generation: (
            generation["public_ledger"].update({"state": "ACCRUING_36M"}),
            generation["protocol_state"].update({"state": "ACCRUING_36M"}),
        ),
        lambda generation: generation["protocol_state"].update({
            "checkpoint_36": {
                "endpoint": "2029-09-14",
                "maturity_summary": {
                    "incumbent_reference_entries": 0,
                    "challenger_armed_cohorts": 0,
                    "challenger_armed_symbols": 0,
                    "incumbent_active_symbols": 0,
                    "incumbent_negative_20_session_blocks": 0,
                    "affected_exits": 0,
                    "affected_symbols": 0,
                },
                "passed": False,
            },
        }),
    )
    spec = _frozen_spec()
    for mutate in mutations:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "incoming"
            state_root = root / "state"
            data_dir.mkdir()
            baseline = _bars()
            baseline.index.name = "date"
            additions = _bars(2).set_axis(pd.to_datetime([
                "2026-09-08", "2026-09-09",
            ]))
            extended = pd.concat([baseline, additions])
            extended.index.name = "date"
            for symbol in spec["universe"]["core_symbols"]:
                extended.to_csv(data_dir / f"{symbol}_1d.csv")
            run_shadow_update(_SPEC_PATH, data_dir, state_root)
            experiment_dir = (
                state_root / spec["experiment_id"] / canonical_spec_hash(spec)
            )
            _rewrite_generation_chain(experiment_dir, 1, mutate)

            _expect_value_error(
                "generation语义重算",
                lambda: run_shadow_update(_SPEC_PATH, data_dir, state_root),
            )


def test_linear_history_summary_lookup_matches_direct_prefix_replay():
    spec = _frozen_spec()
    history_dates = pd.bdate_range(end="2026-09-04", periods=260)
    forward_dates = pd.bdate_range(start="2026-09-08", periods=180)
    dates = history_dates.append(forward_dates)
    values = pd.Series([
        100 + 12 * math.sin(index / 7) + 5 * math.sin(index / 2.7)
        + index * 0.02
        for index in range(len(dates))
    ], index=dates)
    open_values = pd.Series([
        close * (1 + 0.004 * math.sin(index))
        for index, close in enumerate(values)
    ], index=dates)
    frame = pd.DataFrame({
        "open": open_values,
        "high": pd.concat([open_values, values], axis=1).max(axis=1) * 1.02,
        "low": pd.concat([open_values, values], axis=1).min(axis=1) * 0.98,
        "close": values,
        "volume": [1000 + 100 * math.cos(index) for index in range(len(dates))],
    }, index=dates)
    frame.index.name = "date"
    frames = {
        symbol: frame.copy() for symbol in spec["universe"]["core_symbols"]
    }
    start = pd.Timestamp(spec["boundaries"]["expected_accrual_start"])

    lookup = shadow_runner._precompute_shadow_summary_lookup(
        spec, frames, start,
    )

    for end in (start, forward_dates[70], forward_dates[-1]):
        direct = summarize_shadow_window(
            spec,
            shadow_runner.prepare_shadow_windows(spec, frames, start, end),
        )
        assert lookup[end.date().isoformat()] == direct
    assert lookup[forward_dates[-1].date().isoformat()][
        "incumbent_reference_entries"
    ] > 0
    first_events = lookup[start.date().isoformat()]["affected_events"]
    final_events = lookup[forward_dates[-1].date().isoformat()][
        "affected_events"
    ]
    assert not isinstance(first_events, list)
    assert first_events._events is final_events._events
    assert list(final_events) == direct["affected_events"]


def test_shadow_update_rejects_rehashed_generation_with_wrong_elapsed_count():
    spec = _frozen_spec()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "incoming"
        state_root = root / "state"
        data_dir.mkdir()
        baseline = _bars()
        baseline.index.name = "date"
        addition = _bars(1).set_axis(pd.to_datetime(["2026-09-08"]))
        extended = pd.concat([baseline, addition])
        extended.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            extended.to_csv(data_dir / f"{symbol}_1d.csv")
        run_shadow_update(_SPEC_PATH, data_dir, state_root)

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        current_path = experiment_dir / "CURRENT"
        current = json.loads(current_path.read_text(encoding="utf-8"))
        old_reference = f"{current['sequence']:016d}-{current['generation_hash']}"
        old_generation_path = (
            experiment_dir / "generations" / f"{old_reference}.json"
        )
        generation = json.loads(old_generation_path.read_text(encoding="utf-8"))
        generation["public_ledger"]["elapsed_common_sessions"] = 999
        generation_payload = shadow_runner._canonical_json_bytes(generation)
        generation_hash = hashlib.sha256(generation_payload).hexdigest()
        replacement = {"generation_hash": generation_hash, "sequence": 1}
        new_reference = f"{1:016d}-{generation_hash}"
        old_generation_path.unlink()
        (experiment_dir / "commits" / f"{old_reference}.commit").unlink()
        (experiment_dir / "generations" / f"{new_reference}.json").write_bytes(
            generation_payload
        )
        pointer_payload = shadow_runner._canonical_json_bytes(replacement)
        (experiment_dir / "commits" / f"{new_reference}.commit").write_bytes(
            pointer_payload
        )
        current_path.write_bytes(pointer_payload)

        _expect_value_error(
            "elapsed_common_sessions",
            lambda: run_shadow_update(_SPEC_PATH, data_dir, state_root),
        )


def test_shadow_update_rejects_rehashed_head_ledger_not_matching_recompute():
    spec = _frozen_spec()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data_dir = root / "incoming"
        state_root = root / "state"
        data_dir.mkdir()
        baseline = _bars()
        baseline.index.name = "date"
        addition = _bars(1).set_axis(pd.to_datetime(["2026-09-08"]))
        extended = pd.concat([baseline, addition])
        extended.index.name = "date"
        for symbol in spec["universe"]["core_symbols"]:
            extended.to_csv(data_dir / f"{symbol}_1d.csv")
        run_shadow_update(_SPEC_PATH, data_dir, state_root)

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        current_path = experiment_dir / "CURRENT"
        current = json.loads(current_path.read_text(encoding="utf-8"))
        old_reference = f"{current['sequence']:016d}-{current['generation_hash']}"
        old_generation_path = (
            experiment_dir / "generations" / f"{old_reference}.json"
        )
        generation = json.loads(old_generation_path.read_text(encoding="utf-8"))
        generation["public_ledger"]["incumbent_reference_entries"] = 1
        generation_payload = shadow_runner._canonical_json_bytes(generation)
        generation_hash = hashlib.sha256(generation_payload).hexdigest()
        replacement = {"generation_hash": generation_hash, "sequence": 1}
        new_reference = f"{1:016d}-{generation_hash}"
        old_generation_path.unlink()
        (experiment_dir / "commits" / f"{old_reference}.commit").unlink()
        (experiment_dir / "generations" / f"{new_reference}.json").write_bytes(
            generation_payload
        )
        pointer_payload = shadow_runner._canonical_json_bytes(replacement)
        (experiment_dir / "commits" / f"{new_reference}.commit").write_bytes(
            pointer_payload
        )
        current_path.write_bytes(pointer_payload)

        _expect_value_error(
            "重算",
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
    assert locked36["locked_months"] == 36
    assert locked36["checkpoint_36"]["passed"] is True

    between = resolve_accrual_phase(
        spec,
        sessions[:3],
        lambda end: fail36 if end == pd.Timestamp("2029-09-14") else pass48,
    )
    assert between["state"] == "ACCRUING_TO_48M"
    assert between["locked_end"] is None
    assert between["checkpoint_36"] == {
        "endpoint": "2029-09-14",
        "maturity_summary": fail36,
        "passed": False,
    }

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
    assert locked48["locked_months"] == 48
    assert locked48["checkpoint_36"] == between["checkpoint_36"]


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

    cache = shadow_runner._AffectedLabelCache(spec, frames)
    assert cache.state_at(event_tuple := [event], locked_end, dates[18]) == nineteen
    first_plan = next(iter(cache._plans.values()))
    assert cache.state_at(event_tuple, locked_end, dates[19]) == twenty
    assert cache.state_at(event_tuple, locked_end, dates[59]) == before_ready
    assert cache.state_at(event_tuple, locked_end, dates[60]) == ready
    assert next(iter(cache._plans.values())) is first_plan
    assert len(cache._plans) == 1


def test_persisted_labels_stay_blinded_until_the_formal_terminal_generation():
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
    event = {
        "event_id": "event-1",
        "symbol": "NVDA",
        "trigger_decision_date": "2029-09-13",
        "exit_fill_date": locked_end.date().isoformat(),
        "exit_fill_open": 100.0,
    }
    maturity_summary = {
        "incumbent_reference_entries": 45,
        "challenger_armed_cohorts": 24,
        "challenger_armed_symbols": 9,
        "incumbent_active_symbols": 9,
        "incumbent_negative_20_session_blocks": 6,
        "affected_exits": 15,
        "affected_symbols": 7,
        "affected_events": [event],
    }
    summary_lookup = {
        locked_end.date().isoformat(): maturity_summary,
    }

    def snapshot(frame):
        frames = {
            symbol: frame.copy()
            for symbol in spec["universe"]["core_symbols"]
        }
        forward_sessions = pd.DatetimeIndex([
            pd.Timestamp(spec["boundaries"]["expected_accrual_start"]),
            *frame.index,
        ])
        boundaries = {
            "state": "ACCRUING_36M",
            "elapsed_common_sessions": len(forward_sessions),
            "latest_common_session": frame.index[-1].date().isoformat(),
            "actual_accrual_start": spec["boundaries"][
                "expected_accrual_start"
            ],
            "common_sessions": forward_sessions,
            "forward_sessions": forward_sessions,
        }
        ledger, protocol = shadow_runner._build_pre_ready_snapshot(
            spec,
            frames,
            _boundaries=boundaries,
            _summary_lookup=summary_lookup,
        )
        return frames, ledger, protocol

    def metric_values(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in {"return", "mfe", "mae"}:
                    yield nested
                yield from metric_values(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from metric_values(nested)

    frames59, ledger59, protocol59 = snapshot(template.iloc[:60])
    assert protocol59["state"] == "OUTCOME_EMBARGO_60"
    assert protocol59["post_lock_common_sessions"] == 59
    leaked_values = list(metric_values(protocol59["affected_events"]))
    assert leaked_values and all(value is None for value in leaked_values)

    with TemporaryDirectory() as directory:
        experiment_dir = Path(directory)
        (experiment_dir / "generations").mkdir()
        (experiment_dir / "commits").mkdir()
        previous = {"sequence": 0, "generation_hash": "a" * 64}
        current = shadow_runner._append_generation(
            experiment_dir,
            previous,
            dates[59],
            frames59,
            ledger59,
            protocol59,
        )
        preterminal_path = next(
            (experiment_dir / "generations").glob("0000000000000001-*.json")
        )
        preterminal = json.loads(preterminal_path.read_text(encoding="utf-8"))
        persisted_values = list(metric_values(
            preterminal["protocol_state"]["affected_events"]
        ))
        assert persisted_values and all(
            value is None for value in persisted_values
        )

        frames60, ready_ledger, ready_protocol = snapshot(template)
        assert ready_protocol["state"] == "READY_ONCE"
        ready_values = list(metric_values(ready_protocol["affected_events"]))
        assert ready_values and all(value is None for value in ready_values)

        calls = []
        original_evaluate = shadow_runner.formal_evaluate

        def fake_evaluate(_spec, *_args, **kwargs):
            context = kwargs["readiness_context"]
            calls.append(context)
            return {
                "state": "REJECTED_KEEP_V5",
                "decision": "REJECTED_KEEP_V5",
                "eligible": False,
                "formal_evaluation_consumed": True,
                "cas_transition": {
                    "expected_state": "READY_ONCE",
                    "expected_formal_evaluation_count": 0,
                    "next_formal_evaluation_count": 1,
                    "next_state": "REJECTED_KEEP_V5",
                },
                "provenance": {
                    "order_artifact_sha256": context[
                        "order_artifact_sha256"
                    ],
                },
            }

        shadow_runner.formal_evaluate = fake_evaluate
        try:
            final_ledger, final_protocol = shadow_runner._consume_ready_once(
                spec, frames60, ready_ledger, ready_protocol,
            )
        finally:
            shadow_runner.formal_evaluate = original_evaluate

        assert len(calls) == 1
        assert calls[0]["state"] == "READY_ONCE"
        assert final_ledger["state"] == "REJECTED_KEEP_V5"
        assert final_protocol["formal_evaluation_count"] == 1
        assert final_protocol["affected_events"] == label_affected_events(
            spec, [event], frames60, locked_end,
        )["events"]
        assert all(
            value is not None
            for value in metric_values(final_protocol["affected_events"])
        )

        shadow_runner._append_generation(
            experiment_dir,
            current,
            dates[60],
            frames60,
            final_ledger,
            final_protocol,
        )
        terminal_path = next(
            (experiment_dir / "generations").glob("0000000000000002-*.json")
        )
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        assert terminal["protocol_state"]["affected_events"] == (
            final_protocol["affected_events"]
        )


def test_runner_consumes_ready_once_via_exact_cas_transition():
    spec = _frozen_spec()
    template = _bars()
    frames = {
        symbol: template.copy()
        for symbol in spec["universe"]["core_symbols"]
    }
    ledger, protocol = shadow_runner._build_pre_ready_snapshot(spec, frames)
    maturity_summary = {
        "incumbent_reference_entries": 45,
        "challenger_armed_cohorts": 24,
        "challenger_armed_symbols": 9,
        "incumbent_active_symbols": 9,
        "incumbent_negative_20_session_blocks": 6,
        "affected_exits": 15,
        "affected_symbols": 7,
    }
    ledger.update({**maturity_summary, "state": "READY_ONCE"})
    protocol.update({
        "actual_accrual_start": "2026-09-15",
        "checkpoint_36": {
            "endpoint": "2029-09-14",
            "maturity_summary": maturity_summary,
            "passed": True,
        },
        "locked_end": "2029-09-14",
        "locked_months": 36,
        "maturity_36_passed": True,
        "maturity_summary": maturity_summary,
        "post_lock_common_sessions": 60,
        "state": "READY_ONCE",
    })
    fake_orders = {
        "challenger_entry_fills": [[False]],
        "challenger_exit_fills": [[False]],
        "challenger_exit_reasons": [[""]],
        "close_prices": [[100.0]],
        "incumbent_entry_fills": [[False]],
        "incumbent_exit_fills": [[False]],
        "locked_bar_hashes": ledger["accepted_bar_hashes"],
        "next_common_session_after_locked_end": "2029-09-17",
        "open_prices": [[100.0]],
        "order_artifact_sha256": "a" * 64,
        "session_dates": ("2029-09-14",),
        "symbols": tuple(spec["universe"]["core_symbols"]),
    }
    expected_result = {
        "state": "REJECTED_KEEP_V5",
        "decision": "REJECTED_KEEP_V5",
        "eligible": False,
        "formal_evaluation_consumed": True,
        "cas_transition": {
            "expected_state": "READY_ONCE",
            "expected_formal_evaluation_count": 0,
            "next_formal_evaluation_count": 1,
            "next_state": "REJECTED_KEEP_V5",
        },
        "provenance": {"order_artifact_sha256": "a" * 64},
    }
    original_orders = shadow_runner._build_fixed_order_inputs
    original_evaluate = shadow_runner.formal_evaluate
    captured = {}

    def fake_evaluate(_spec, *args, **kwargs):
        captured["context"] = kwargs["readiness_context"]
        return expected_result

    shadow_runner._build_fixed_order_inputs = lambda *args, **kwargs: fake_orders
    shadow_runner.formal_evaluate = fake_evaluate
    try:
        final_ledger, final_protocol = shadow_runner._consume_ready_once(
            spec, frames, ledger, protocol,
        )
    finally:
        shadow_runner._build_fixed_order_inputs = original_orders
        shadow_runner.formal_evaluate = original_evaluate

    assert captured["context"]["state"] == "READY_ONCE"
    assert captured["context"]["formal_evaluation_count"] == 0
    assert captured["context"]["next_common_session_after_locked_end"] == (
        "2029-09-17"
    )
    assert final_ledger["state"] == "REJECTED_KEEP_V5"
    assert final_protocol["state"] == "REJECTED_KEEP_V5"
    assert final_protocol["formal_evaluation_count"] == 1
    assert final_protocol["order_artifact_sha256"] == "a" * 64
    assert final_protocol["evaluation_result"] == expected_result


def test_consumed_evaluation_state_can_only_overlay_a_ready_snapshot():
    ledger = {"state": "READY_ONCE"}
    protocol = {
        "state": "READY_ONCE",
        "formal_evaluation_count": 0,
        "evaluation_result": None,
        "order_artifact_sha256": None,
    }
    result = {
        "state": "REJECTED_KEEP_V5",
        "decision": "REJECTED_KEEP_V5",
        "eligible": False,
        "formal_evaluation_consumed": True,
        "cas_transition": {
            "expected_state": "READY_ONCE",
            "expected_formal_evaluation_count": 0,
            "next_formal_evaluation_count": 1,
            "next_state": "REJECTED_KEEP_V5",
        },
        "provenance": {"order_artifact_sha256": "a" * 64},
    }

    final_ledger, final_protocol = shadow_runner._carry_consumed_evaluation(
        _frozen_spec(), ledger, protocol, result,
    )

    assert final_ledger == {"state": "REJECTED_KEEP_V5"}
    assert final_protocol["state"] == "REJECTED_KEEP_V5"
    assert final_protocol["formal_evaluation_count"] == 1
    assert final_protocol["evaluation_result"] == result
    assert final_protocol["order_artifact_sha256"] == "a" * 64

    _expect_value_error(
        "READY_ONCE",
        lambda: shadow_runner._carry_consumed_evaluation(
            _frozen_spec(), {"state": "ACCRUING_36M"},
            {**protocol, "state": "ACCRUING_36M"}, result,
        ),
    )


def test_replay_rejects_consumed_result_that_differs_from_one_recomputation():
    spec = _frozen_spec()
    template = _bars()
    frames = {
        symbol: template.copy()
        for symbol in spec["universe"]["core_symbols"]
    }
    ledger, protocol = shadow_runner._build_pre_ready_snapshot(spec, frames)
    maturity_summary = {
        "incumbent_reference_entries": 45,
        "challenger_armed_cohorts": 24,
        "challenger_armed_symbols": 9,
        "incumbent_active_symbols": 9,
        "incumbent_negative_20_session_blocks": 6,
        "affected_exits": 15,
        "affected_symbols": 7,
    }
    ledger.update({**maturity_summary, "state": "READY_ONCE"})
    protocol.update({
        "actual_accrual_start": "2026-09-15",
        "checkpoint_36": {
            "endpoint": "2029-09-14",
            "maturity_summary": maturity_summary,
            "passed": True,
        },
        "locked_end": "2029-09-14",
        "locked_months": 36,
        "maturity_36_passed": True,
        "maturity_summary": maturity_summary,
        "post_lock_common_sessions": 60,
        "state": "READY_ONCE",
    })
    fake_orders = {
        "challenger_entry_fills": [[False]],
        "challenger_exit_fills": [[False]],
        "challenger_exit_reasons": [[""]],
        "close_prices": [[100.0]],
        "incumbent_entry_fills": [[False]],
        "incumbent_exit_fills": [[False]],
        "locked_bar_hashes": ledger["accepted_bar_hashes"],
        "next_common_session_after_locked_end": "2029-09-17",
        "open_prices": [[100.0]],
        "order_artifact_sha256": "a" * 64,
        "session_dates": ("2029-09-14",),
        "symbols": tuple(spec["universe"]["core_symbols"]),
    }
    expected_result = {
        "state": "REJECTED_KEEP_V5",
        "decision": "REJECTED_KEEP_V5",
        "eligible": False,
        "formal_evaluation_consumed": True,
        "cas_transition": {
            "expected_state": "READY_ONCE",
            "expected_formal_evaluation_count": 0,
            "next_formal_evaluation_count": 1,
            "next_state": "REJECTED_KEEP_V5",
        },
        "provenance": {"order_artifact_sha256": "a" * 64},
    }
    forged_result = {
        **expected_result,
        "state": "ELIGIBLE_FOR_V6_IMPLEMENTATION",
        "decision": spec["decision"]["promotion_result"],
        "eligible": True,
        "cas_transition": {
            **expected_result["cas_transition"],
            "next_state": "ELIGIBLE_FOR_V6_IMPLEMENTATION",
        },
    }
    original_orders = shadow_runner._build_fixed_order_inputs
    original_evaluate = shadow_runner.formal_evaluate
    calls = {"formal": 0}

    def fake_evaluate(*_args, **_kwargs):
        calls["formal"] += 1
        return expected_result

    shadow_runner._build_fixed_order_inputs = lambda *args, **kwargs: fake_orders
    shadow_runner.formal_evaluate = fake_evaluate
    try:
        _expect_value_error(
            "正式评估结果重算不匹配",
            lambda: shadow_runner._verify_consumed_evaluation_transition(
                spec, frames, ledger, protocol, forged_result,
            ),
        )
    finally:
        shadow_runner._build_fixed_order_inputs = original_orders
        shadow_runner.formal_evaluate = original_evaluate

    assert calls["formal"] == 1


def test_terminal_shadow_update_repairs_caches_without_reading_new_incoming():
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

        original_build = shadow_runner._build_pre_ready_snapshot
        original_consume = shadow_runner._consume_ready_once
        original_verify = shadow_runner._verify_consumed_evaluation_transition
        original_read_bar_csv = shadow_runner._read_bar_csv
        calls = {"formal": 0, "replay_verify": 0}
        maturity_summary = {
            "incumbent_reference_entries": 45,
            "challenger_armed_cohorts": 24,
            "challenger_armed_symbols": 9,
            "incumbent_active_symbols": 9,
            "incumbent_negative_20_session_blocks": 6,
            "affected_exits": 15,
            "affected_symbols": 7,
        }
        result = {
            "state": "REJECTED_KEEP_V5",
            "decision": "REJECTED_KEEP_V5",
            "eligible": False,
            "formal_evaluation_consumed": True,
            "cas_transition": {
                "expected_state": "READY_ONCE",
                "expected_formal_evaluation_count": 0,
                "next_formal_evaluation_count": 1,
                "next_state": "REJECTED_KEEP_V5",
            },
            "provenance": {"order_artifact_sha256": "b" * 64},
        }

        def fake_build(
            frozen_spec, frames, *, formal_evaluation_count=0,
            evaluation_result=None, **_internal,
        ):
            ledger, protocol = original_build(frozen_spec, frames)
            if max(frame.index.max() for frame in frames.values()) <= pd.Timestamp(
                "2026-09-05"
            ):
                return ledger, protocol
            ledger.update({**maturity_summary, "state": "READY_ONCE"})
            protocol.update({
                "actual_accrual_start": "2026-09-08",
                "checkpoint_36": {
                    "endpoint": "2029-09-14",
                    "maturity_summary": maturity_summary,
                    "passed": True,
                },
                "locked_end": "2029-09-14",
                "locked_months": 36,
                "maturity_36_passed": True,
                "maturity_summary": maturity_summary,
                "post_lock_common_sessions": 60,
                "state": "READY_ONCE",
            })
            if formal_evaluation_count == 1:
                assert evaluation_result == result
                return shadow_runner._carry_consumed_evaluation(
                    frozen_spec, ledger, protocol, evaluation_result,
                )
            return ledger, protocol

        def fake_consume(frozen_spec, frames, ledger, protocol):
            if protocol["state"] != "READY_ONCE":
                return ledger, protocol
            calls["formal"] += 1
            return shadow_runner._carry_consumed_evaluation(
                frozen_spec, ledger, protocol, result,
            )

        def fake_verify(_spec, _frames, _ledger, _protocol, persisted):
            assert persisted == result
            calls["replay_verify"] += 1

        shadow_runner._build_pre_ready_snapshot = fake_build
        shadow_runner._consume_ready_once = fake_consume
        shadow_runner._verify_consumed_evaluation_transition = fake_verify
        try:
            first_addition = _bars(1).set_axis(pd.to_datetime(["2026-09-08"]))
            extended = pd.concat([baseline, first_addition])
            extended.index.name = "date"
            for symbol in spec["universe"]["core_symbols"]:
                extended.to_csv(data_dir / f"{symbol}_1d.csv")
            first_final = run_shadow_update(_SPEC_PATH, data_dir, state_root)
            experiment_dir = (
                state_root / spec["experiment_id"] / canonical_spec_hash(spec)
            )
            current_bytes = (experiment_dir / "CURRENT").read_bytes()
            generation_names = sorted(
                path.name for path in (experiment_dir / "generations").iterdir()
            )
            commit_names = sorted(
                path.name for path in (experiment_dir / "commits").iterdir()
            )
            evaluation_bytes = (experiment_dir / "evaluation.json").read_bytes()
            ledger_bytes = (experiment_dir / "ledger.json").read_bytes()
            accepted_bytes = {
                symbol: (
                    experiment_dir / "accepted_bars" / f"{symbol}_1d.csv"
                ).read_bytes()
                for symbol in spec["universe"]["core_symbols"]
            }

            (experiment_dir / "ledger.json").write_text(
                "{}\n", encoding="utf-8",
            )
            (experiment_dir / "evaluation.json").unlink()
            for symbol in spec["universe"]["core_symbols"]:
                (
                    experiment_dir / "accepted_bars" / f"{symbol}_1d.csv"
                ).write_text("corrupt\n", encoding="utf-8")

            second_addition = _bars(1).set_axis(pd.to_datetime(["2026-09-09"]))
            extended_again = pd.concat([extended, second_addition])
            extended_again.index.name = "date"
            for symbol in spec["universe"]["core_symbols"]:
                extended_again.to_csv(data_dir / f"{symbol}_1d.csv")

            def fail_if_incoming_is_read(_path):
                raise AssertionError("terminal update must not read incoming bars")

            shadow_runner._read_bar_csv = fail_if_incoming_is_read
            appended_final = run_shadow_update(_SPEC_PATH, data_dir, state_root)
            current = json.loads(
                (experiment_dir / "CURRENT").read_text(encoding="utf-8")
            )
            reference = shadow_runner._generation_reference(current)
            final_generation = json.loads(
                (experiment_dir / "generations" / f"{reference}.json")
                .read_text(encoding="utf-8")
            )
        finally:
            shadow_runner._build_pre_ready_snapshot = original_build
            shadow_runner._consume_ready_once = original_consume
            shadow_runner._verify_consumed_evaluation_transition = original_verify
            shadow_runner._read_bar_csv = original_read_bar_csv

        assert calls["formal"] == 1
        assert calls["replay_verify"] == 1
        assert first_final["state"] == "REJECTED_KEEP_V5"
        assert appended_final == first_final
        assert (experiment_dir / "CURRENT").read_bytes() == current_bytes
        assert sorted(
            path.name for path in (experiment_dir / "generations").iterdir()
        ) == generation_names
        assert sorted(
            path.name for path in (experiment_dir / "commits").iterdir()
        ) == commit_names
        assert (experiment_dir / "ledger.json").read_bytes() == ledger_bytes
        assert (experiment_dir / "evaluation.json").read_bytes() == evaluation_bytes
        for symbol, expected in accepted_bytes.items():
            assert (
                experiment_dir / "accepted_bars" / f"{symbol}_1d.csv"
            ).read_bytes() == expected
        assert final_generation["protocol_state"]["formal_evaluation_count"] == 1
        assert final_generation["protocol_state"]["evaluation_result"] == result
        assert final_generation["protocol_state"]["order_artifact_sha256"] == (
            "b" * 64
        )


def test_batch_stops_at_first_terminal_session_before_calendar_coverage_end():
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

        original_build = shadow_runner._build_pre_ready_snapshot

        def fake_build(frozen_spec, frames, **kwargs):
            ledger, protocol = original_build(frozen_spec, frames, **kwargs)
            if max(frame.index.max() for frame in frames.values()) >= pd.Timestamp(
                "2026-09-09"
            ):
                ledger["state"] = "INCONCLUSIVE_COVERAGE_KEEP_V5"
                protocol["state"] = "INCONCLUSIVE_COVERAGE_KEEP_V5"
            return ledger, protocol

        shadow_runner._build_pre_ready_snapshot = fake_build
        try:
            first = _bars(1).set_axis(pd.to_datetime(["2026-09-08"]))
            prefix = pd.concat([baseline, first])
            prefix.index.name = "date"
            for symbol in spec["universe"]["core_symbols"]:
                prefix.to_csv(data_dir / f"{symbol}_1d.csv")
            run_shadow_update(_SPEC_PATH, data_dir, state_root)

            terminal = _bars(1).set_axis(pd.to_datetime(["2026-09-09"]))
            beyond_coverage = _bars(1).set_axis(
                pd.to_datetime(["2031-01-02"])
            )
            incoming = pd.concat([prefix, terminal, beyond_coverage])
            incoming.index.name = "date"
            for symbol in spec["universe"]["core_symbols"]:
                incoming.to_csv(data_dir / f"{symbol}_1d.csv")
            final = run_shadow_update(_SPEC_PATH, data_dir, state_root)
        finally:
            shadow_runner._build_pre_ready_snapshot = original_build

        experiment_dir = (
            state_root / spec["experiment_id"] / canonical_spec_hash(spec)
        )
        current = json.loads(
            (experiment_dir / "CURRENT").read_text(encoding="utf-8")
        )
        assert final["state"] == "INCONCLUSIVE_COVERAGE_KEEP_V5"
        assert current["sequence"] == 2
        assert len(list((experiment_dir / "generations").glob("*.json"))) == 3
        for symbol in spec["universe"]["core_symbols"]:
            accepted = pd.read_csv(
                experiment_dir / "accepted_bars" / f"{symbol}_1d.csv"
            )
            assert accepted["date"].iloc[-1] == "2026-09-09"


def test_ready_once_commit_recovers_from_every_publication_boundary():
    spec = _frozen_spec()
    maturity_summary = {
        "incumbent_reference_entries": 45,
        "challenger_armed_cohorts": 24,
        "challenger_armed_symbols": 9,
        "incumbent_active_symbols": 9,
        "incumbent_negative_20_session_blocks": 6,
        "affected_exits": 15,
        "affected_symbols": 7,
    }
    result = {
        "state": "REJECTED_KEEP_V5",
        "decision": "REJECTED_KEEP_V5",
        "eligible": False,
        "formal_evaluation_consumed": True,
        "cas_transition": {
            "expected_state": "READY_ONCE",
            "expected_formal_evaluation_count": 0,
            "next_formal_evaluation_count": 1,
            "next_state": "REJECTED_KEEP_V5",
        },
        "provenance": {"order_artifact_sha256": "c" * 64},
    }
    result_hash = hashlib.sha256(
        shadow_runner._canonical_json_bytes(result)
    ).hexdigest()

    for fault_stage in (
        "generation_before", "generation_after", "commit_before",
        "commit_after", "derived_cache_after_current",
    ):
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

            original_build = shadow_runner._build_pre_ready_snapshot
            original_consume = shadow_runner._consume_ready_once
            original_verify = (
                shadow_runner._verify_consumed_evaluation_transition
            )
            original_atomic = shadow_runner._atomic_write_bytes
            original_canonical = shadow_runner._write_canonical_json
            original_write_bar_csv = shadow_runner._write_bar_csv
            formal_result_hashes = []

            def fake_build(
                frozen_spec, frames, *, formal_evaluation_count=0,
                evaluation_result=None, **_internal,
            ):
                ledger, protocol = original_build(frozen_spec, frames)
                if max(frame.index.max() for frame in frames.values()) <= (
                    pd.Timestamp("2026-09-05")
                ):
                    return ledger, protocol
                ledger.update({**maturity_summary, "state": "READY_ONCE"})
                protocol.update({
                    "actual_accrual_start": "2026-09-08",
                    "checkpoint_36": {
                        "endpoint": "2029-09-14",
                        "maturity_summary": maturity_summary,
                        "passed": True,
                    },
                    "locked_end": "2029-09-14",
                    "locked_months": 36,
                    "maturity_36_passed": True,
                    "maturity_summary": maturity_summary,
                    "post_lock_common_sessions": 60,
                    "state": "READY_ONCE",
                })
                if formal_evaluation_count == 1:
                    assert evaluation_result == result
                    return shadow_runner._carry_consumed_evaluation(
                        frozen_spec, ledger, protocol, evaluation_result,
                    )
                return ledger, protocol

            def fake_consume(frozen_spec, frames, ledger, protocol):
                if protocol["state"] != "READY_ONCE":
                    return ledger, protocol
                formal_result_hashes.append(result_hash)
                return shadow_runner._carry_consumed_evaluation(
                    frozen_spec, ledger, protocol, result,
                )

            def fake_verify(_spec, _frames, _ledger, _protocol, persisted):
                assert persisted == result

            fault = {"raised": False}

            def faulting_atomic(path, payload):
                target = path.parent.name == "generations"
                if target and not fault["raised"] and fault_stage in {
                    "generation_before", "generation_after",
                }:
                    fault["raised"] = True
                    if fault_stage == "generation_before":
                        raise OSError("injected before generation")
                    original_atomic(path, payload)
                    raise OSError("injected after generation")
                return original_atomic(path, payload)

            def faulting_canonical(path, value):
                target = path.parent.name == "commits"
                if target and not fault["raised"] and fault_stage in {
                    "commit_before", "commit_after",
                }:
                    fault["raised"] = True
                    if fault_stage == "commit_before":
                        raise OSError("injected before commit")
                    original_canonical(path, value)
                    raise OSError("injected after commit")
                return original_canonical(path, value)

            def faulting_bar_csv(path, frame):
                if (not fault["raised"]
                        and fault_stage == "derived_cache_after_current"):
                    current = json.loads(
                        (path.parents[1] / "CURRENT").read_text(
                            encoding="utf-8"
                        )
                    )
                    assert current["sequence"] == 1
                    fault["raised"] = True
                    raise OSError("injected derived cache failure")
                return original_write_bar_csv(path, frame)

            shadow_runner._build_pre_ready_snapshot = fake_build
            shadow_runner._consume_ready_once = fake_consume
            shadow_runner._verify_consumed_evaluation_transition = fake_verify
            shadow_runner._atomic_write_bytes = faulting_atomic
            shadow_runner._write_canonical_json = faulting_canonical
            shadow_runner._write_bar_csv = faulting_bar_csv
            try:
                addition = _bars(1).set_axis(pd.to_datetime(["2026-09-08"]))
                incoming = pd.concat([baseline, addition])
                incoming.index.name = "date"
                for symbol in spec["universe"]["core_symbols"]:
                    incoming.to_csv(data_dir / f"{symbol}_1d.csv")
                try:
                    run_shadow_update(_SPEC_PATH, data_dir, state_root)
                except OSError as error:
                    assert "injected" in str(error)
                else:
                    raise AssertionError(f"{fault_stage} must fail once")
                assert fault["raised"] is True

                shadow_runner._atomic_write_bytes = original_atomic
                shadow_runner._write_canonical_json = original_canonical
                shadow_runner._write_bar_csv = original_write_bar_csv
                recovered = run_shadow_update(
                    _SPEC_PATH, data_dir, state_root,
                )
            finally:
                shadow_runner._build_pre_ready_snapshot = original_build
                shadow_runner._consume_ready_once = original_consume
                shadow_runner._verify_consumed_evaluation_transition = (
                    original_verify
                )
                shadow_runner._atomic_write_bytes = original_atomic
                shadow_runner._write_canonical_json = original_canonical
                shadow_runner._write_bar_csv = original_write_bar_csv

            experiment_dir = (
                state_root / spec["experiment_id"] / canonical_spec_hash(spec)
            )
            current = json.loads(
                (experiment_dir / "CURRENT").read_text(encoding="utf-8")
            )
            assert current["sequence"] == 1
            committed_generations = []
            for commit_path in sorted(
                (experiment_dir / "commits").glob("*.commit")
            ):
                reference = commit_path.stem
                committed_generations.append(json.loads(
                    (experiment_dir / "generations" / f"{reference}.json")
                    .read_text(encoding="utf-8")
                ))
            consumed = [
                generation for generation in committed_generations
                if generation["protocol_state"]["formal_evaluation_count"] == 1
            ]
            assert len(consumed) == 1
            final_protocol = consumed[0]["protocol_state"]
            assert recovered["state"] == "REJECTED_KEEP_V5"
            assert final_protocol["evaluation_result"] == result
            assert final_protocol["order_artifact_sha256"] == "c" * 64
            assert hashlib.sha256(shadow_runner._canonical_json_bytes(
                final_protocol["evaluation_result"]
            )).hexdigest() == result_hash
            assert set(formal_result_hashes) == {result_hash}
            assert json.loads(
                (experiment_dir / "evaluation.json").read_text(
                    encoding="utf-8"
                )
            ) == result
            for symbol in spec["universe"]["core_symbols"]:
                repaired = pd.read_csv(
                    experiment_dir / "accepted_bars" / f"{symbol}_1d.csv"
                )
                assert len(repaired) == len(incoming)


def test_runner_builds_fixed_order_artifact_from_locked_authoritative_window():
    spec = _frozen_spec()
    dates = pd.bdate_range("2026-08-03", "2029-09-21")
    template = pd.DataFrame({
        "open": [100.0 + position / 100 for position in range(len(dates))],
        "high": [102.0 + position / 100 for position in range(len(dates))],
        "low": [99.0 + position / 100 for position in range(len(dates))],
        "close": [101.0 + position / 100 for position in range(len(dates))],
        "volume": [1000.0 + position for position in range(len(dates))],
    }, index=dates)
    frames = {
        symbol: template.copy()
        for symbol in spec["universe"]["core_symbols"]
    }
    ledger = {
        "accepted_bar_hashes": {
            symbol: canonical_bar_hash(frame)
            for symbol, frame in sorted(frames.items())
        },
        "source_hashes": shadow_runner.algorithm_source_hashes(),
        "spec_hash": canonical_spec_hash(spec),
    }
    protocol = {
        "actual_accrual_start": "2026-09-15",
        "locked_end": "2029-09-14",
    }

    orders = shadow_runner._build_fixed_order_inputs(
        spec, frames, ledger, protocol,
    )

    assert orders["symbols"] == tuple(spec["universe"]["core_symbols"])
    assert orders["session_dates"][0] == "2026-09-15"
    assert orders["session_dates"][-1] == "2029-09-14"
    assert orders["locked_bar_hashes"] != ledger["accepted_bar_hashes"]
    assert orders["open_prices"].shape == (
        len(spec["universe"]["core_symbols"]), len(orders["session_dates"]),
    )
    for field in (
        "incumbent_entry_fills", "incumbent_exit_fills",
        "challenger_entry_fills", "challenger_exit_fills",
    ):
        assert not orders[field][:, 0].any()
    assert orders["order_artifact_sha256"] == (
        shadow_runner.fixed_order_artifact_sha256(
            orders["open_prices"],
            orders["close_prices"],
            orders["incumbent_entry_fills"],
            orders["incumbent_exit_fills"],
            orders["challenger_entry_fills"],
            orders["challenger_exit_fills"],
            orders["challenger_exit_reasons"],
            spec_hash=ledger["spec_hash"],
            symbols=orders["symbols"],
            session_dates=orders["session_dates"],
            locked_end=protocol["locked_end"],
            source_hashes=ledger["source_hashes"],
            accepted_bar_hashes=orders["locked_bar_hashes"],
        )
    )
