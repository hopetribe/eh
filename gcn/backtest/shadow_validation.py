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
    return spec


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
