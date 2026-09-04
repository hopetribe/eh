# -*- coding: utf-8 -*-
"""GCN 下一版本的前向、只追加影子验证。

该模块与正式 ``VERSIONS``、生产 preset 和 Web/API 入口完全隔离。候选只有在
预注册样本成熟且一次性晋升门槛全部通过后，才有资格另行实现为正式版本。
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_BAR_COLUMNS = ("open", "high", "low", "close", "volume")
# 12位有效数字远细于1ppm策略容差，但足以抹平等价比例运算的float末位。
_REBASE_SIGNIFICANT_DIGITS = 12
# 允许已规范值再次作为overlap时产生的最大舍入扩散；远小于1ppm。
_REBASE_NUMERIC_NOISE_PPM = float(
    2 * 10 ** (1 - _REBASE_SIGNIFICANT_DIGITS) * 1_000_000
)

_OBSERVATION_CALENDAR_FILE = (
    "shadow_specs/nyse-us-equities-sessions-20260906-20301231.json"
)
_OBSERVATION_CALENDAR_SHA256 = (
    "bbd5dad9dae12c34afd65adf61e63b44fde84b5e6d2ab7271fab00f4f296f398"
)
_OBSERVATION_CALENDAR_FIELDS = frozenset({
    "schema_version", "calendar_id", "market", "coverage_start",
    "coverage_end", "session_rule", "full_day_closures", "sources",
    "projection_policy", "unscheduled_closure_policy",
    "required_observation_end",
})

_TOP_LEVEL_FIELDS = frozenset({
    "schema_version",
    "experiment_id",
    "immutable",
    "frozen_on",
    "parent_evidence",
    "candidate_selection_audit",
    "strategies",
    "universe",
    "boundaries",
    "maturity",
    "evaluation",
    "decision",
})

_NESTED_FIELDS = {
    ("parent_evidence",): frozenset({
        "audit_run_id", "window_end", "manifest_sha256", "purpose",
    }),
    ("candidate_selection_audit",): frozenset({
        "stage", "criterion", "allowed_historical_fields",
        "forbidden_historical_fields", "selected_candidate_id",
        "excluded_candidate_ids", "power_calibration_artifact",
        "power_calibration_hash_method", "power_calibration_sha256",
    }),
    ("strategies",): frozenset({"incumbent", "challenger"}),
    ("strategies", "incumbent"): frozenset({
        "version", "strategy_id", "entry_signal_columns",
        "exit_signal_columns", "trail_bps", "hard_stop_bps",
        "max_hold_bars", "terminal_policy",
    }),
    ("strategies", "challenger"): frozenset({
        "strategy_id", "inherits", "trail_bps", "profit_keep_bps",
        "rule_cost_bps_per_side", "arm_peak_gain_bps", "peak_price",
        "arm_sticky", "breakeven_formula", "profit_floor_formula",
        "effective_floor_formula", "incremental_trigger", "confirmation",
        "execution", "exit_reason_priority", "terminal_policy",
    }),
    ("universe",): frozenset({
        "core_symbols", "external_only_symbols", "portfolio_weighting",
        "common_session_rule", "missing_policy", "initial_state",
        "input_adjustment", "accepted_price_basis",
        "historical_revision_policy", "revision_tolerance_ppm",
        "price_rebase_formula", "volume_rebase_formula",
        "nonuniform_revision_policy", "carry_in_positions",
        "carry_in_pending_orders",
    }),
    ("boundaries",): frozenset({
        "signal_cutoff_exclusive", "initial_embargo_common_sessions",
        "embargo_order_policy", "accrual_start_rule",
        "expected_accrual_start", "actual_accrual_start_must_equal_expected",
        "first_eligible_decision",
        "pre_accrual_setup_policy",
        "minimum_accrual_months", "expected_minimum_accrual_end",
        "maximum_accrual_months", "expected_maximum_accrual_end",
        "endpoint_rule", "accrual_end_selection",
        "intermediate_counts_can_lock_end",
        "outcome_embargo_common_sessions", "post_accrual_price_usage",
    }),
    ("maturity",): frozenset({
        "common", "at_36_months", "at_48_months", "definitions", "labels",
    }),
    ("maturity", "common"): frozenset({
        "incumbent_reference_entries_min", "challenger_armed_cohorts_min",
        "challenger_armed_symbols_min", "incumbent_active_symbols_min",
        "incumbent_negative_20_session_blocks_min",
        "forward_horizons_sessions",
    }),
    ("maturity", "at_36_months"): frozenset({
        "affected_exits_min", "affected_symbols_min",
    }),
    ("maturity", "at_48_months"): frozenset({
        "affected_exits_min", "affected_symbols_min",
    }),
    ("maturity", "definitions"): frozenset({
        "reference_entry", "armed_cohort", "affected_exit",
        "negative_block", "active_symbol", "label_completion",
    }),
    ("maturity", "labels"): frozenset({
        "event_population", "event_id_anchor", "price_anchor",
        "fill_session_ordinal", "horizon_end_position_formula",
        "return_formula", "mfe_formula", "mae_formula",
        "partial_label_policy", "pending_count_formula", "post_lock_policy",
        "promotion_use", "ready_policy",
    }),
    ("evaluation",): frozenset({
        "annual_sessions", "base_cost_bps_per_side",
        "stress_cost_bps_per_side", "stress_reuses_base_orders",
        "risk_free_rate_bps", "portfolio_return", "metric_deltas",
        "metric_formulas", "downside", "bootstrap", "stress_policy",
        "robustness", "gates",
    }),
    ("evaluation", "metric_formulas"): frozenset({
        "symbol_daily_return_first", "symbol_daily_return_later",
        "portfolio_daily_return", "cagr", "sharpe", "mdd", "exposure",
        "entry_count_ratio", "cross_symbol_total_return",
        "positive_contribution", "invalid_or_zero_denominator",
    }),
    ("evaluation", "downside"): frozenset({
        "block_sessions", "partition", "terminal_incomplete_block",
        "block_return_formula", "loss_formula", "bootstrap_statistic",
        "ratio_formula", "zero_incumbent_downside",
    }),
    ("evaluation", "bootstrap"): frozenset({
        "method", "replications", "bit_generator", "seed",
        "time_block_sessions", "time_block_scheme", "time_start_indices",
        "blocks_per_replication", "assembly", "symbol_resample_count",
        "symbol_sampling", "duplicate_symbol_weighting", "point_estimate",
        "confidence_side", "alpha_bps", "quantile_method", "studentized",
        "multiplicity", "gate_comparison",
    }),
    ("evaluation", "stress_policy"): frozenset({
        "order_source", "rule_cost_for_trigger_bps_per_side",
        "revalue_cost_bps_per_side", "path_recalculation",
        "terminal_open_position",
    }),
    ("evaluation", "robustness"): frozenset({
        "leave_one_out", "cross_symbol_positive",
        "contribution_denominator", "empty_or_nonfinite",
    }),
    ("evaluation", "gates"): frozenset({
        "base", "stress", "cross_symbol", "leave_one_out",
    }),
    ("evaluation", "gates", "base"): frozenset({
        "annualized_log_return_delta_q05_gt_bps",
        "point_cagr_delta_min_bps", "downside_improvement_q05_gt_bps",
        "downside_loss_ratio_max_bps", "mdd_delta_max_bps",
        "sharpe_delta_min_milli", "entry_count_ratio_min_bps",
        "exposure_ratio_min_bps",
    }),
    ("evaluation", "gates", "stress"): frozenset({
        "annualized_log_return_delta_min_bps", "mdd_delta_max_bps",
        "sharpe_delta_min_milli",
    }),
    ("evaluation", "gates", "cross_symbol"): frozenset({
        "median_total_return_delta_gt_bps", "positive_symbols_min",
        "positive_contribution_max_bps",
    }),
    ("evaluation", "gates", "leave_one_out"): frozenset({
        "required_passes", "annualized_log_return_delta_min_bps",
        "mdd_delta_max_bps", "sharpe_delta_min_milli",
    }),
    ("decision",): frozenset({
        "pre_ready_visible_fields", "pre_ready_forbidden_metrics",
        "formal_evaluations_allowed", "promotion_result", "ready_failure",
        "maximum_count_failure", "semantic_change", "states",
    }),
}

_INTEGER_PATHS = frozenset({
    ("universe", "revision_tolerance_ppm"),
    ("strategies", "incumbent", "trail_bps"),
    ("strategies", "challenger", "trail_bps"),
    ("strategies", "challenger", "profit_keep_bps"),
    ("strategies", "challenger", "rule_cost_bps_per_side"),
    ("strategies", "challenger", "arm_peak_gain_bps"),
    ("boundaries", "initial_embargo_common_sessions"),
    ("boundaries", "minimum_accrual_months"),
    ("boundaries", "maximum_accrual_months"),
    ("boundaries", "outcome_embargo_common_sessions"),
    ("maturity", "common", "incumbent_reference_entries_min"),
    ("maturity", "common", "challenger_armed_cohorts_min"),
    ("maturity", "common", "challenger_armed_symbols_min"),
    ("maturity", "common", "incumbent_active_symbols_min"),
    ("maturity", "common", "incumbent_negative_20_session_blocks_min"),
    ("maturity", "at_36_months", "affected_exits_min"),
    ("maturity", "at_36_months", "affected_symbols_min"),
    ("maturity", "at_48_months", "affected_exits_min"),
    ("maturity", "at_48_months", "affected_symbols_min"),
    ("maturity", "labels", "fill_session_ordinal"),
    ("evaluation", "annual_sessions"),
    ("evaluation", "base_cost_bps_per_side"),
    ("evaluation", "stress_cost_bps_per_side"),
    ("evaluation", "risk_free_rate_bps"),
    ("evaluation", "downside", "block_sessions"),
    ("evaluation", "bootstrap", "replications"),
    ("evaluation", "bootstrap", "seed"),
    ("evaluation", "bootstrap", "time_block_sessions"),
    ("evaluation", "bootstrap", "symbol_resample_count"),
    ("evaluation", "bootstrap", "alpha_bps"),
    ("evaluation", "stress_policy", "rule_cost_for_trigger_bps_per_side"),
    ("evaluation", "stress_policy", "revalue_cost_bps_per_side"),
    ("evaluation", "gates", "base", "annualized_log_return_delta_q05_gt_bps"),
    ("evaluation", "gates", "base", "point_cagr_delta_min_bps"),
    ("evaluation", "gates", "base", "downside_improvement_q05_gt_bps"),
    ("evaluation", "gates", "base", "downside_loss_ratio_max_bps"),
    ("evaluation", "gates", "base", "mdd_delta_max_bps"),
    ("evaluation", "gates", "base", "sharpe_delta_min_milli"),
    ("evaluation", "gates", "base", "entry_count_ratio_min_bps"),
    ("evaluation", "gates", "base", "exposure_ratio_min_bps"),
    ("evaluation", "gates", "stress", "annualized_log_return_delta_min_bps"),
    ("evaluation", "gates", "stress", "mdd_delta_max_bps"),
    ("evaluation", "gates", "stress", "sharpe_delta_min_milli"),
    ("evaluation", "gates", "cross_symbol", "median_total_return_delta_gt_bps"),
    ("evaluation", "gates", "cross_symbol", "positive_symbols_min"),
    ("evaluation", "gates", "cross_symbol", "positive_contribution_max_bps"),
    ("evaluation", "gates", "leave_one_out", "required_passes"),
    ("evaluation", "gates", "leave_one_out", "annualized_log_return_delta_min_bps"),
    ("evaluation", "gates", "leave_one_out", "mdd_delta_max_bps"),
    ("evaluation", "gates", "leave_one_out", "sharpe_delta_min_milli"),
    ("decision", "formal_evaluations_allowed"),
})

_FROZEN_VALUES = {
    ("schema_version",): "gcn-forward-oos-prereg-v1",
    ("immutable",): True,
    ("candidate_selection_audit", "selected_candidate_id"):
        "profit-arm20-keep50",
    ("candidate_selection_audit", "excluded_candidate_ids"):
        ["profit-arm20-keep20"],
    ("strategies", "challenger", "strategy_id"):
        "profit-arm20-keep50",
    ("strategies", "challenger", "profit_keep_bps"): 5000,
    ("boundaries", "signal_cutoff_exclusive"): "2026-09-05",
    ("boundaries", "intermediate_counts_can_lock_end"): False,
    ("maturity", "at_36_months", "affected_exits_min"): 15,
    ("maturity", "at_48_months", "affected_exits_min"): 20,
    ("evaluation", "stress_reuses_base_orders"): True,
    ("evaluation", "bootstrap", "seed"): 20260905,
    ("decision", "formal_evaluations_allowed"): 1,
}

_REGISTERED_SPEC_HASHES = {
    "v6-profit-arm20-keep50-20260905":
        "c12f7c4932072b9fa2352bbca733481c849121afa420f3cc53ce5002e80cb57f",
}


def _require_fields(value: Any, expected: frozenset[str], path: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须是对象")
    unknown = sorted(set(value) - expected)
    if unknown:
        raise ValueError(f"{path} 包含未知字段: {', '.join(unknown)}")
    missing = sorted(expected - set(value))
    if missing:
        raise ValueError(f"{path} 缺少字段: {', '.join(missing)}")


def _value_at_path(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for part in path:
        current = current[part]
    return current


def _require_equal(
    spec: dict[str, Any], path: tuple[str, ...], expected: Any,
) -> None:
    actual = _value_at_path(spec, path)
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(
            f"spec.{'.'.join(path)} 必须固定为 {expected!r}"
        )


def validate_spec(spec: dict[str, Any]) -> None:
    """严格校验预注册对象结构，拒绝遗漏或悄然生效的新字段。"""
    _require_fields(spec, _TOP_LEVEL_FIELDS, "spec")
    for path, expected in _NESTED_FIELDS.items():
        value = _value_at_path(spec, path)
        _require_fields(value, expected, "spec." + ".".join(path))
    for path in _INTEGER_PATHS:
        if type(_value_at_path(spec, path)) is not int:
            raise ValueError(f"spec.{'.'.join(path)} 必须是整数")
    for path, expected in _FROZEN_VALUES.items():
        _require_equal(spec, path, expected)
    experiment_id = spec["experiment_id"]
    registered_hash = _REGISTERED_SPEC_HASHES.get(experiment_id)
    if registered_hash is None:
        raise ValueError(f"未注册的前向实验: {experiment_id!r}")
    actual_hash = canonical_spec_hash(spec)
    if not hmac.compare_digest(actual_hash, registered_hash):
        raise ValueError(
            f"spec内容偏离注册哈希: expected={registered_hash}, actual={actual_hash}"
        )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON包含重复字段: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"JSON包含非有限数: {value}")


def parse_spec_json(text: str) -> dict[str, Any]:
    """严格解析预注册JSON，拒绝会掩盖配置的重复字段。"""
    parsed = json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_nonfinite,
    )
    if not isinstance(parsed, dict):
        raise ValueError("预注册spec根节点必须是对象")
    return parsed


def canonical_spec_hash(spec: dict[str, Any]) -> str:
    """返回与对象键顺序、JSON空白无关的规范化SHA-256。"""
    payload = json.dumps(
        spec,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_spec(path: str | Path) -> dict[str, Any]:
    """加载并校验带独立规范化SHA-256的冻结预注册spec。"""
    spec_path = Path(path)
    spec = parse_spec_json(spec_path.read_text(encoding="utf-8"))
    experiment_id = spec.get("experiment_id")
    if spec_path.stem != experiment_id:
        raise ValueError(
            f"spec文件名必须等于experiment_id: {experiment_id!r}"
        )
    hash_path = spec_path.with_suffix(".sha256")
    expected_hash = hash_path.read_text(encoding="ascii").strip()
    if (
        len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise ValueError(f"spec哈希文件格式无效: {hash_path}")
    actual_hash = canonical_spec_hash(spec)
    if not hmac.compare_digest(actual_hash, expected_hash):
        raise ValueError(
            f"spec哈希不匹配: expected={expected_hash}, actual={actual_hash}"
        )
    validate_spec(spec)
    calibration = spec["candidate_selection_audit"]
    artifact_name = calibration["power_calibration_artifact"]
    if (not isinstance(artifact_name, str)
            or Path(artifact_name).name != artifact_name):
        raise ValueError("功效校准工件名无效")
    artifact_path = spec_path.parent / artifact_name
    try:
        artifact = parse_spec_json(artifact_path.read_text(encoding="utf-8"))
        detached_hash = artifact_path.with_suffix(".sha256").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("功效校准工件缺失或格式无效") from error
    expected_artifact_hash = calibration["power_calibration_sha256"]
    actual_artifact_hash = canonical_spec_hash(artifact)
    if (calibration["power_calibration_hash_method"] != "canonical_json_sha256"
            or detached_hash != expected_artifact_hash
            or not hmac.compare_digest(
                actual_artifact_hash, expected_artifact_hash,
            )):
        raise ValueError(
            "功效校准工件哈希不匹配: "
            f"expected={expected_artifact_hash}, actual={actual_artifact_hash}"
        )
    return spec


def _iso_date(value: Any, name: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise ValueError(f"冻结美股交易日历{name}必须是ISO日期")
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"冻结美股交易日历{name}必须是ISO日期") from error
    if (parsed.tz is not None or parsed != parsed.normalize()
            or parsed.date().isoformat() != value):
        raise ValueError(f"冻结美股交易日历{name}必须使用YYYY-MM-DD")
    return parsed


def load_observation_calendar() -> dict[str, Any]:
    """加载并验证覆盖完整v6观察窗的离线NYSE会话工件。"""
    path = Path(__file__).resolve().parent / _OBSERVATION_CALENDAR_FILE
    try:
        calendar = parse_spec_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("冻结美股交易日历缺失或格式无效") from error
    actual_hash = canonical_spec_hash(calendar)
    if not hmac.compare_digest(actual_hash, _OBSERVATION_CALENDAR_SHA256):
        raise ValueError(
            "冻结美股交易日历哈希不匹配: "
            f"expected={_OBSERVATION_CALENDAR_SHA256}, actual={actual_hash}"
        )
    _require_fields(calendar, _OBSERVATION_CALENDAR_FIELDS, "calendar")
    expected_scalars = {
        "schema_version": "gcn-frozen-us-equities-calendar-v1",
        "calendar_id": "nyse-us-equities-sessions-20260906-20301231",
        "market": "NYSE_US_CASH_EQUITIES",
        "coverage_start": "2026-09-06",
        "coverage_end": "2030-12-31",
        "session_rule":
            "monday_through_friday_excluding_full_day_closures",
        "projection_policy":
            "2029_through_2030_dates_are_deterministic_rule_7_2_projections",
        "unscheduled_closure_policy":
            "fail_closed_until_a_new_registered_calendar_is_adopted",
        "required_observation_end": "2030-12-09",
    }
    for field, expected in expected_scalars.items():
        if calendar.get(field) != expected:
            raise ValueError(f"冻结美股交易日历{field}不匹配")
    sources = calendar.get("sources")
    if (not isinstance(sources, list) or len(sources) != 2
            or any(not isinstance(source, dict) or set(source) != {
                "authority", "scope", "url",
            } for source in sources)):
        raise ValueError("冻结美股交易日历来源元数据无效")
    coverage_start = _iso_date(calendar["coverage_start"], "coverage_start")
    coverage_end = _iso_date(calendar["coverage_end"], "coverage_end")
    required_end = _iso_date(
        calendar["required_observation_end"], "required_observation_end",
    )
    if not coverage_start <= required_end <= coverage_end:
        raise ValueError("冻结美股交易日历未覆盖完整观察窗")
    closures_value = calendar.get("full_day_closures")
    if not isinstance(closures_value, list):
        raise ValueError("冻结美股交易日历休市日必须是列表")
    closures = pd.DatetimeIndex([
        _iso_date(value, "full_day_closures") for value in closures_value
    ])
    if (closures.has_duplicates or not closures.is_monotonic_increasing
            or (closures.dayofweek >= 5).any()
            or (closures < coverage_start).any()
            or (closures > coverage_end).any()):
        raise ValueError("冻结美股交易日历休市日无效")
    return calendar


def validate_observation_sessions(
    sessions: pd.DatetimeIndex, *, cutoff: pd.Timestamp,
    maximum_accrual_end: pd.Timestamp,
    outcome_embargo_sessions: int,
) -> None:
    """要求cutoff后的共同会话是冻结NYSE日历从起点起的严格前缀。"""
    if not isinstance(sessions, pd.DatetimeIndex):
        raise ValueError("前向会话必须是DatetimeIndex")
    calendar = load_observation_calendar()
    cutoff = pd.Timestamp(cutoff)
    coverage_start = _iso_date(calendar["coverage_start"], "coverage_start")
    coverage_end = _iso_date(calendar["coverage_end"], "coverage_end")
    expected_start = cutoff + pd.Timedelta(days=1)
    if expected_start != coverage_start:
        raise ValueError("冻结美股交易日历与signal cutoff不连续")
    maximum_accrual_end = pd.Timestamp(maximum_accrual_end)
    if (type(outcome_embargo_sessions) is not int
            or outcome_embargo_sessions <= 0):
        raise ValueError("结果禁运共同交易日数必须是正整数")
    closures = pd.DatetimeIndex(pd.to_datetime(calendar["full_day_closures"]))
    post_lock_sessions = pd.bdate_range(
        maximum_accrual_end + pd.Timedelta(days=1), coverage_end,
    ).difference(closures)
    required_end = _iso_date(
        calendar["required_observation_end"], "required_observation_end",
    )
    if (len(post_lock_sessions) < outcome_embargo_sessions
            or post_lock_sessions[outcome_embargo_sessions - 1] != required_end):
        raise ValueError("冻结美股交易日历未覆盖完整观察窗")
    if sessions.empty:
        return
    if sessions[-1] > coverage_end:
        raise ValueError("前向会话超过冻结美股交易日历覆盖终点，DATA_BLOCKED")
    expected = pd.bdate_range(coverage_start, sessions[-1]).difference(closures)
    if not sessions.equals(expected):
        unexpected = sessions.difference(expected)
        missing = expected.difference(sessions)
        details = []
        if len(unexpected):
            details.append(
                "unexpected=" + ",".join(
                    value.date().isoformat() for value in unexpected[:5]
                )
            )
        if len(missing):
            details.append(
                "missing=" + ",".join(
                    value.date().isoformat() for value in missing[:5]
                )
            )
        raise ValueError(
            "前向会话不构成冻结美股交易日历的严格前缀，DATA_BLOCKED: "
            + "; ".join(details)
        )


def _validate_bar_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label}必须是DataFrame")
    if tuple(frame.columns) != _BAR_COLUMNS:
        raise ValueError(f"{label}字段必须严格为 {', '.join(_BAR_COLUMNS)}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{label}索引必须是DatetimeIndex")
    if (frame.index.tz is not None
            or not frame.index.equals(frame.index.normalize())):
        raise ValueError(f"{label}日期必须是无时区的自然日")
    if frame.empty:
        raise ValueError(f"{label}不能为空")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label}日期必须严格递增且不重复")
    values = frame.loc[:, _BAR_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label}包含非有限OHLCV")
    prices = frame.loc[:, ("open", "high", "low", "close")]
    if (prices <= 0).any(axis=None) or (frame["volume"] < 0).any():
        raise ValueError(f"{label}包含无效OHLCV")
    if ((frame["high"] < prices.max(axis=1)).any()
            or (frame["low"] > prices.min(axis=1)).any()):
        raise ValueError(f"{label}包含无效OHLC关系")
    return frame.astype(float).copy()


def canonical_bar_hash(frame: pd.DataFrame) -> str:
    """以日期和IEEE-754精确值生成跨CSV格式稳定的K线摘要。"""
    validated = _validate_bar_frame(frame, "bars")
    rows = [
        [date.date().isoformat(), *(float(value).hex() for value in values)]
        for date, values in zip(
            validated.index,
            validated.loc[:, _BAR_COLUMNS].to_numpy(dtype=float),
        )
    ]
    payload = json.dumps(
        {"columns": list(_BAR_COLUMNS), "rows": rows},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonicalize_rebased_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """消除等价缩放路径造成的IEEE-754末位差异。"""
    canonical = frame.copy()
    for column in _BAR_COLUMNS:
        canonical[column] = [
            0.0 if value == 0 else float(
                format(float(value), f".{_REBASE_SIGNIFICANT_DIGITS}g")
            )
            for value in canonical[column]
        ]
    return _validate_bar_frame(canonical, "rebased incoming bars")


def rebase_adjusted_incoming(
    accepted: pd.DataFrame | None,
    incoming: pd.DataFrame,
    tolerance_ppm: int = 1,
) -> tuple[pd.DataFrame, dict[str, int | float | bool]]:
    """把 provider-adjusted 输入统一缩放到已接受K线的价格基准。

    价格因子是重叠 OHLC 的 ``accepted / incoming`` 中位数；成交量因子
    独立取正值对的同一中位数。每个比例都必须在 ``tolerance_ppm`` 内，
    成交量零值必须成对。有新增行时，比例扩散还必须只来自12位规范化的
    数值噪声；否则没有唯一可冻结基准并按 ``DATA_BLOCKED`` 失败关闭。
    返回帧保留既有重叠行的精确值，并把全部新值规范为12位有效数字；
    第二个返回值是可直接写入 JSON generation 的因子审计元数据。
    ``accepted is None`` 表示首次注册，规范化输入且两个因子均为 1。
    """
    if type(tolerance_ppm) is not int or tolerance_ppm < 0:
        raise ValueError("tolerance_ppm必须是非负整数")
    current = _validate_bar_frame(incoming, "incoming bars")
    if accepted is None:
        return _canonicalize_rebased_frame(current), {
            "overlap_rows": 0,
            "price_factor": 1.0,
            "volume_factor": 1.0,
            "price_rebased": False,
            "volume_rebased": False,
            "rebase_applied": False,
        }

    previous = _validate_bar_frame(accepted, "accepted bars")
    overlap = previous.index.intersection(current.index)
    if overlap.empty:
        raise ValueError(
            "incoming bars与既有区间没有可验证的重叠日期，DATA_BLOCKED"
        )
    has_additions = bool((current.index > previous.index[-1]).any())
    old_prices = previous.loc[
        overlap, ("open", "high", "low", "close")
    ].to_numpy(dtype=float)
    new_prices = current.loc[
        overlap, ("open", "high", "low", "close")
    ].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        price_ratios = old_prices / new_prices
        price_factor = float(np.median(price_ratios))
    if (not np.isfinite(price_ratios).all()
            or not np.isfinite(price_factor) or price_factor <= 0):
        raise ValueError("价格缩放因子无效，DATA_BLOCKED")
    price_deviation_ppm = np.abs(price_ratios / price_factor - 1.0) * 1_000_000
    worst_price_position = np.unravel_index(
        int(np.argmax(price_deviation_ppm)), price_deviation_ppm.shape,
    )
    worst_price_date = overlap[worst_price_position[0]].date().isoformat()
    worst_price_field = ("open", "high", "low", "close")[
        worst_price_position[1]
    ]
    max_price_deviation_ppm = float(price_deviation_ppm[worst_price_position])
    price_diagnostic = (
        f"{worst_price_date} {worst_price_field}, "
        f"max_deviation_ppm={max_price_deviation_ppm:.12g}, "
        f"tolerance_ppm={tolerance_ppm}"
    )
    if (price_deviation_ppm > tolerance_ppm).any():
        raise ValueError(
            f"价格修订非统一缩放，DATA_BLOCKED: {price_diagnostic}"
        )
    if (has_additions
            and (price_deviation_ppm > _REBASE_NUMERIC_NOISE_PPM).any()):
        raise ValueError(
            "价格缩放在容差内但新增行基准不唯一，DATA_BLOCKED: "
            + price_diagnostic
        )

    old_volume = previous.loc[overlap, "volume"].to_numpy(dtype=float)
    new_volume = current.loc[overlap, "volume"].to_numpy(dtype=float)
    mismatched_zero = (old_volume == 0) != (new_volume == 0)
    if mismatched_zero.any():
        mismatch_date = overlap[int(np.flatnonzero(mismatched_zero)[0])]
        raise ValueError(
            "成交量零值未成对，DATA_BLOCKED: "
            + mismatch_date.date().isoformat()
        )
    positive_pairs = (old_volume > 0) & (new_volume > 0)
    if positive_pairs.any():
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            volume_ratios = (
                old_volume[positive_pairs] / new_volume[positive_pairs]
            )
            volume_factor = float(np.median(volume_ratios))
        if (not np.isfinite(volume_ratios).all()
                or not np.isfinite(volume_factor) or volume_factor <= 0):
            raise ValueError("成交量缩放因子无效，DATA_BLOCKED")
        volume_deviation_ppm = np.abs(
            volume_ratios / volume_factor - 1.0
        ) * 1_000_000
        worst_volume_position = int(np.argmax(volume_deviation_ppm))
        worst_volume_date = overlap[positive_pairs][
            worst_volume_position
        ].date().isoformat()
        max_volume_deviation_ppm = float(
            volume_deviation_ppm[worst_volume_position]
        )
        volume_diagnostic = (
            f"{worst_volume_date} volume, "
            f"max_deviation_ppm={max_volume_deviation_ppm:.12g}, "
            f"tolerance_ppm={tolerance_ppm}"
        )
        if (volume_deviation_ppm > tolerance_ppm).any():
            raise ValueError(
                f"成交量修订非统一缩放，DATA_BLOCKED: {volume_diagnostic}"
            )
        if (has_additions
                and (volume_deviation_ppm > _REBASE_NUMERIC_NOISE_PPM).any()):
            raise ValueError(
                "成交量缩放在容差内但新增行基准不唯一，DATA_BLOCKED: "
                + volume_diagnostic
            )
    else:
        if (current["volume"] != 0).any():
            raise ValueError("无正成交量对时必须全零，DATA_BLOCKED")
        volume_factor = 1.0

    rebased = current.copy()
    rebased.loc[:, ("open", "high", "low", "close")] *= price_factor
    rebased.loc[:, "volume"] *= volume_factor
    rebased = _canonicalize_rebased_frame(rebased)
    rebased.loc[overlap, _BAR_COLUMNS] = previous.loc[overlap, _BAR_COLUMNS]
    price_rebased = price_factor != 1.0
    volume_rebased = volume_factor != 1.0
    return rebased, {
        "overlap_rows": int(len(overlap)),
        "price_factor": price_factor,
        "volume_factor": volume_factor,
        "price_rebased": price_rebased,
        "volume_rebased": volume_rebased,
        "rebase_applied": price_rebased or volume_rebased,
    }


def merge_accepted_bars(
    accepted: pd.DataFrame | None, incoming: pd.DataFrame,
) -> pd.DataFrame:
    """按日期/数值验证重叠区，只追加新K线并容忍源缓存裁掉旧前缀。"""
    current = _validate_bar_frame(incoming, "incoming bars")
    if accepted is None:
        return current
    previous = _validate_bar_frame(accepted, "accepted bars")
    last_accepted = previous.index[-1]
    overlap = previous.index.intersection(current.index)
    if current.index[0] <= last_accepted and overlap.empty:
        raise ValueError("incoming bars与既有区间没有可验证的重叠日期")
    if len(overlap):
        old_values = previous.loc[overlap, _BAR_COLUMNS].to_numpy(dtype=float)
        new_values = current.loc[overlap, _BAR_COLUMNS].to_numpy(dtype=float)
        if not np.array_equal(old_values, new_values):
            changed_dates = [
                date.date().isoformat()
                for date, old, new in zip(overlap, old_values, new_values)
                if not np.array_equal(old, new)
            ]
            raise ValueError(
                "已接受K线被改写: " + ", ".join(changed_dates[:5])
            )
    historical = current.loc[current.index <= last_accepted]
    unexpected = historical.index.difference(previous.index)
    if len(unexpected):
        raise ValueError(
            "既有区间出现新历史日期: "
            + ", ".join(date.date().isoformat() for date in unexpected[:5])
        )
    overlap_start = current.index[0]
    expected_visible = previous.loc[
        (previous.index >= overlap_start) & (previous.index <= current.index[-1])
    ].index
    missing_visible = expected_visible.difference(current.index)
    if len(missing_visible):
        raise ValueError(
            "incoming bars在可见区间丢失既有日期: "
            + ", ".join(date.date().isoformat() for date in missing_visible[:5])
        )
    additions = current.loc[current.index > last_accepted]
    if additions.empty:
        return previous
    return pd.concat([previous, additions]).loc[:, _BAR_COLUMNS]
