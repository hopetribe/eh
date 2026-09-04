# -*- coding: utf-8 -*-
"""v6 前向影子实验的纯函数正式评估器。

该模块不读写状态、不生成交易决策，只消费已经固定的成交路径。
因此压力成本情景仅重估同一组成交，不会重新计算触发规则。
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from numbers import Real
from typing import Any

import numpy as np

from gcn.backtest.shadow_validation import canonical_spec_hash, validate_spec


def _positive_float_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind == "b":
        raise ValueError(f"{name}必须是数值矩阵")
    try:
        array = array.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}必须是数值矩阵") from error
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name}必须是非空二维矩阵")
    if not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError(f"{name}必须全部为有限正数")
    return array


def _bool_matrix(value: Any, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "b" or array.shape != shape:
        raise ValueError(f"{name}必须是形状为 {shape} 的布尔矩阵")
    return array.astype(bool, copy=False)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name}必须是正整数")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name}必须是正整数")
    return value


def _finite_scalar(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    """仅在分子有限且分母严格大于零时返回比率。"""
    numerator = _finite_scalar(numerator)
    denominator = _finite_scalar(denominator)
    if numerator is None or denominator is None or denominator <= 0:
        return None
    ratio = numerator / denominator
    return float(ratio) if math.isfinite(ratio) else None


def replay_fixed_orders(
    open_prices: Any,
    close_prices: Any,
    entry_fills: Any,
    exit_fills: Any,
    *,
    cost_bps: Real,
) -> dict[str, np.ndarray]:
    """以指定单边成本重放已固定的 OPEN 成交，不重算任何信号。

    每个标的从现金 1 开始，不允许加仓、空仓卖出或同日买卖。
    末日尚未平仓时按 CLOSE 盯市，不扣除虚构的退出成本。
    """
    opens = _positive_float_matrix(open_prices, "open_prices")
    closes = _positive_float_matrix(close_prices, "close_prices")
    if closes.shape != opens.shape:
        raise ValueError("open_prices与close_prices形状必须一致")
    entries = _bool_matrix(entry_fills, "entry_fills", opens.shape)
    exits = _bool_matrix(exit_fills, "exit_fills", opens.shape)
    if np.logical_and(entries, exits).any():
        raise ValueError("同一标的同一日不能同时入场和退出成交")
    if isinstance(cost_bps, (bool, np.bool_)) or not isinstance(cost_bps, Real):
        raise ValueError("cost_bps必须是 [0, 10000) 内的有限数")
    cost_bps = float(cost_bps)
    if not math.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("cost_bps必须是 [0, 10000) 内的有限数")
    retained = 1.0 - cost_bps / 10_000.0

    symbol_count, session_count = opens.shape
    equity = np.empty_like(opens, dtype=float)
    held = np.zeros(opens.shape, dtype=bool)
    for symbol_pos in range(symbol_count):
        cash = 1.0
        shares = 0.0
        position_open = False
        for session_pos in range(session_count):
            if exits[symbol_pos, session_pos]:
                if not position_open:
                    raise ValueError(
                        f"空仓时不能执行退出成交: "
                        f"symbol={symbol_pos}, session={session_pos}"
                    )
                cash = shares * opens[symbol_pos, session_pos] * retained
                shares = 0.0
                position_open = False
            elif entries[symbol_pos, session_pos]:
                if position_open:
                    raise ValueError(
                        f"已持仓时不能再次执行入场成交: "
                        f"symbol={symbol_pos}, session={session_pos}"
                    )
                shares = cash * retained / opens[symbol_pos, session_pos]
                cash = 0.0
                position_open = True
            held[symbol_pos, session_pos] = position_open
            equity[symbol_pos, session_pos] = (
                shares * closes[symbol_pos, session_pos]
                if position_open else cash
            )
    return {
        "equity": equity,
        "held_at_close": held,
        "entry_fills": entries.copy(),
        "exit_fills": exits.copy(),
    }


def symbol_daily_returns(equity_close: Any) -> np.ndarray:
    """按冻结公式计算逐标的日收益，每个标的初始净值均为 1。"""
    equity = _positive_float_matrix(equity_close, "equity_close")
    previous = np.concatenate(
        [np.ones((equity.shape[0], 1), dtype=float), equity[:, :-1]], axis=1,
    )
    returns = equity / previous - 1.0
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("逐标的日收益必须有限且严格大于 -1")
    return returns


def metrics_from_symbol_daily_returns(
    symbol_returns: Any,
    *,
    annual_sessions: int = 252,
) -> dict[str, Any]:
    """由逐标的日收益直接计算固定等权组合指标。"""
    returns = _return_matrix(symbol_returns, "symbol_returns")
    annual_sessions = _positive_integer(annual_sessions, "annual_sessions")
    portfolio = returns.mean(axis=0)
    session_count = portfolio.size
    annualized_log = float(
        annual_sessions / session_count * np.log1p(portfolio).sum()
    )
    try:
        cagr_value = math.expm1(annualized_log)
    except OverflowError:
        cagr_value = math.inf
    cagr = float(cagr_value) if math.isfinite(cagr_value) else None

    sharpe = None
    if session_count >= 2:
        volatility = float(portfolio.std(ddof=1))
        if math.isfinite(volatility) and volatility > 0:
            value = math.sqrt(annual_sessions) * float(portfolio.mean()) / volatility
            if math.isfinite(value):
                sharpe = float(value)

    portfolio_equity = np.cumprod(1.0 + portfolio)
    curve_with_initial = np.concatenate([[1.0], portfolio_equity])
    running_max = np.maximum.accumulate(curve_with_initial)
    mdd = float(np.max(1.0 - curve_with_initial / running_max))
    return {
        "portfolio_daily_returns": portfolio,
        "annualized_log_return": annualized_log,
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
    }


def path_metrics(
    equity_close: Any,
    held_at_close: Any,
    *,
    annual_sessions: int = 252,
) -> dict[str, Any]:
    """计算等权固定标的路径指标。"""
    equity = _positive_float_matrix(equity_close, "equity_close")
    held = _bool_matrix(held_at_close, "held_at_close", equity.shape)
    annual_sessions = _positive_integer(annual_sessions, "annual_sessions")
    returns = symbol_daily_returns(equity)
    metrics = metrics_from_symbol_daily_returns(
        returns, annual_sessions=annual_sessions,
    )
    metrics.update({
        "symbol_daily_returns": returns,
        "symbol_total_returns": equity[:, -1] - 1.0,
        "exposure": float(held.mean()),
    })
    return metrics


def nonoverlapping_downside(
    portfolio_daily_returns: Any,
    *,
    block_sessions: int = 20,
) -> dict[str, Any]:
    """从累积起点锚定非重叠分块，丢弃末尾不完整块。"""
    daily = np.asarray(portfolio_daily_returns)
    if daily.dtype.kind == "b":
        raise ValueError("portfolio_daily_returns必须是一维数值序列")
    try:
        daily = daily.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "portfolio_daily_returns必须是一维数值序列"
        ) from error
    if daily.ndim != 1 or daily.size == 0:
        raise ValueError("portfolio_daily_returns必须是非空一维序列")
    if not np.isfinite(daily).all() or (daily <= -1).any():
        raise ValueError("日收益必须有限且严格大于 -1")
    block_sessions = _positive_integer(block_sessions, "block_sessions")
    full_blocks = daily.size // block_sessions
    complete_size = full_blocks * block_sessions
    if full_blocks:
        blocks = daily[:complete_size].reshape(full_blocks, block_sessions)
        block_returns = np.prod(1.0 + blocks, axis=1) - 1.0
    else:
        block_returns = np.empty(0, dtype=float)
    losses = np.maximum(-block_returns, 0.0)
    return {
        "block_returns": block_returns,
        "losses": losses,
        "mean_loss": float(losses.mean()) if losses.size else None,
        "full_blocks": int(full_blocks),
        "excluded_sessions": int(daily.size - complete_size),
    }


def _return_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind == "b":
        raise ValueError(f"{name}必须是数值矩阵")
    try:
        array = array.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}必须是数值矩阵") from error
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name}必须是非空二维矩阵")
    if not np.isfinite(array).all() or (array <= -1).any():
        raise ValueError(f"{name}必须全部有限且严格大于 -1")
    return array


def paired_two_axis_bootstrap(
    incumbent_symbol_daily_returns: Any,
    challenger_symbol_daily_returns: Any,
    *,
    annual_sessions: int,
    replications: int,
    seed: int,
    time_block_sessions: int,
    downside_block_sessions: int,
    symbol_resample_count: int,
    alpha: float = 0.05,
    batch_size: int = 128,
) -> dict[str, Any]:
    """执行 PCG64 配对双轴普通非循环 moving-block bootstrap。

    为使结果不受内存批大小影响，先一次性从同一个 PCG64 流中
    抽取全部时间块起点，再抽取全部标的序号；批处理只负责组装。
    每次复制对两策略使用同一组时间块和标的样本。
    """
    incumbent = _return_matrix(
        incumbent_symbol_daily_returns, "incumbent_symbol_daily_returns",
    )
    challenger = _return_matrix(
        challenger_symbol_daily_returns, "challenger_symbol_daily_returns",
    )
    if challenger.shape != incumbent.shape:
        raise ValueError("两策略逐标的日收益矩阵形状必须一致")
    annual_sessions = _positive_integer(annual_sessions, "annual_sessions")
    replications = _positive_integer(replications, "replications")
    time_block_sessions = _positive_integer(
        time_block_sessions, "time_block_sessions",
    )
    downside_block_sessions = _positive_integer(
        downside_block_sessions, "downside_block_sessions",
    )
    symbol_resample_count = _positive_integer(
        symbol_resample_count, "symbol_resample_count",
    )
    batch_size = _positive_integer(batch_size, "batch_size")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed必须是非负整数")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed必须是非负整数")
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise ValueError("alpha必须是 (0, 1) 内的有限数")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha必须是 (0, 1) 内的有限数")

    symbol_count, session_count = incumbent.shape
    if session_count < time_block_sessions:
        raise ValueError("session_count不得小于time_block_sessions")
    if session_count < downside_block_sessions:
        raise ValueError("session_count不得小于downside_block_sessions")
    blocks_per_replication = math.ceil(session_count / time_block_sessions)
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = rng.integers(
        0,
        session_count - time_block_sessions + 1,
        size=(replications, blocks_per_replication),
    )
    sampled_symbols = rng.integers(
        0,
        symbol_count,
        size=(replications, symbol_resample_count),
    )

    annualized_samples = np.empty(replications, dtype=float)
    downside_samples = np.empty(replications, dtype=float)
    time_offsets = np.arange(time_block_sessions, dtype=np.int64)
    full_downside_sessions = (
        session_count // downside_block_sessions * downside_block_sessions
    )
    full_downside_blocks = full_downside_sessions // downside_block_sessions
    for batch_start in range(0, replications, batch_size):
        batch_end = min(batch_start + batch_size, replications)
        time_positions = (
            starts[batch_start:batch_end, :, None] + time_offsets
        ).reshape(batch_end - batch_start, -1)[:, :session_count]
        symbol_positions = sampled_symbols[batch_start:batch_end]
        incumbent_portfolio = incumbent[
            symbol_positions[:, :, None], time_positions[:, None, :]
        ].mean(axis=1)
        challenger_portfolio = challenger[
            symbol_positions[:, :, None], time_positions[:, None, :]
        ].mean(axis=1)
        annualized_samples[batch_start:batch_end] = (
            annual_sessions / session_count
            * (
                np.log1p(challenger_portfolio).sum(axis=1)
                - np.log1p(incumbent_portfolio).sum(axis=1)
            )
        )
        incumbent_blocks = incumbent_portfolio[:, :full_downside_sessions].reshape(
            batch_end - batch_start,
            full_downside_blocks,
            downside_block_sessions,
        )
        challenger_blocks = challenger_portfolio[:, :full_downside_sessions].reshape(
            batch_end - batch_start,
            full_downside_blocks,
            downside_block_sessions,
        )
        incumbent_losses = np.maximum(
            -(np.prod(1.0 + incumbent_blocks, axis=2) - 1.0), 0.0,
        )
        challenger_losses = np.maximum(
            -(np.prod(1.0 + challenger_blocks, axis=2) - 1.0), 0.0,
        )
        downside_samples[batch_start:batch_end] = (
            incumbent_losses - challenger_losses
        ).mean(axis=1)

    return {
        "annualized_log_return_delta_samples": annualized_samples,
        "downside_improvement_samples": downside_samples,
        "annualized_log_return_delta_q05": float(
            np.quantile(annualized_samples, alpha, method="linear")
        ),
        "downside_improvement_q05": float(
            np.quantile(downside_samples, alpha, method="linear")
        ),
        "replications": replications,
        "seed": seed,
        "bit_generator": "PCG64",
        "blocks_per_replication": blocks_per_replication,
    }


def _symbol_names(symbols: Any, symbol_count: int) -> tuple[str, ...]:
    if symbols is None:
        return tuple(str(position) for position in range(symbol_count))
    try:
        names = tuple(symbols)
    except TypeError as error:
        raise ValueError("symbols必须是与矩阵行数一致的唯一名称序列") from error
    if (
        len(names) != symbol_count
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("symbols必须是与矩阵行数一致的唯一名称序列")
    return names


def cross_symbol_robustness(
    incumbent_equity_close: Any,
    challenger_equity_close: Any,
    *,
    symbols: Any = None,
) -> dict[str, Any]:
    """按各标的末日盯市净值计算横截面收益差与正贡献集中度。"""
    incumbent = _positive_float_matrix(
        incumbent_equity_close, "incumbent_equity_close",
    )
    challenger = _positive_float_matrix(
        challenger_equity_close, "challenger_equity_close",
    )
    if challenger.shape != incumbent.shape:
        raise ValueError("两策略的净值矩阵形状必须一致")
    names = _symbol_names(symbols, incumbent.shape[0])
    deltas = (challenger[:, -1] - 1.0) - (incumbent[:, -1] - 1.0)
    positive = deltas[deltas > 0]
    contribution = (
        float(positive.max() / positive.sum()) if positive.size else None
    )
    return {
        "total_return_deltas": deltas,
        "by_symbol": {
            symbol: float(delta) for symbol, delta in zip(names, deltas)
        },
        "median_total_return_delta": float(np.median(deltas)),
        "positive_symbols": int(positive.size),
        "positive_contribution": contribution,
    }


def _finite_threshold(gates: dict[str, Any], key: str, divisor: float) -> float:
    if not isinstance(gates, dict) or key not in gates:
        raise ValueError(f"缺少门槛: {key}")
    value = _finite_scalar(gates[key])
    if value is None:
        raise ValueError(f"门槛必须为有限数: {key}")
    return value / divisor


def _finite_difference(left: Any, right: Any) -> float | None:
    left = _finite_scalar(left)
    right = _finite_scalar(right)
    if left is None or right is None:
        return None
    value = left - right
    return float(value) if math.isfinite(value) else None


def leave_one_out_robustness(
    incumbent_symbol_daily_returns: Any,
    challenger_symbol_daily_returns: Any,
    *,
    symbols: Any = None,
    annual_sessions: int = 252,
    gates: dict[str, Any],
) -> dict[str, Any]:
    """每次丢弃一个标的，其余标的重新等权，不重做 bootstrap。"""
    incumbent = _return_matrix(
        incumbent_symbol_daily_returns, "incumbent_symbol_daily_returns",
    )
    challenger = _return_matrix(
        challenger_symbol_daily_returns, "challenger_symbol_daily_returns",
    )
    if challenger.shape != incumbent.shape:
        raise ValueError("两策略逐标的日收益矩阵形状必须一致")
    if incumbent.shape[0] < 2:
        raise ValueError("留一法至少需要2个标的")
    names = _symbol_names(symbols, incumbent.shape[0])
    annual_sessions = _positive_integer(annual_sessions, "annual_sessions")
    log_min = _finite_threshold(
        gates, "annualized_log_return_delta_min_bps", 10_000.0,
    )
    mdd_max = _finite_threshold(gates, "mdd_delta_max_bps", 10_000.0)
    sharpe_min = _finite_threshold(
        gates, "sharpe_delta_min_milli", 1_000.0,
    )
    rows = []
    for dropped_pos, dropped_symbol in enumerate(names):
        keep = np.arange(incumbent.shape[0]) != dropped_pos
        incumbent_metrics = metrics_from_symbol_daily_returns(
            incumbent[keep], annual_sessions=annual_sessions,
        )
        challenger_metrics = metrics_from_symbol_daily_returns(
            challenger[keep], annual_sessions=annual_sessions,
        )
        annualized_delta = _finite_difference(
            challenger_metrics["annualized_log_return"],
            incumbent_metrics["annualized_log_return"],
        )
        mdd_delta = _finite_difference(
            challenger_metrics["mdd"], incumbent_metrics["mdd"],
        )
        sharpe_delta = _finite_difference(
            challenger_metrics["sharpe"], incumbent_metrics["sharpe"],
        )
        passed = (
            annualized_delta is not None
            and annualized_delta >= log_min
            and mdd_delta is not None
            and mdd_delta <= mdd_max
            and sharpe_delta is not None
            and sharpe_delta >= sharpe_min
        )
        rows.append({
            "dropped_symbol": dropped_symbol,
            "annualized_log_return_delta": annualized_delta,
            "mdd_delta": mdd_delta,
            "sharpe_delta": sharpe_delta,
            "passed": bool(passed),
        })
    return {
        "rows": rows,
        "passes": int(sum(row["passed"] for row in rows)),
    }


def _evidence_value(
    evidence: Any, section: str, field: str,
) -> float | None:
    if not isinstance(evidence, dict):
        return None
    section_value = evidence.get(section)
    if not isinstance(section_value, dict):
        return None
    return _finite_scalar(section_value.get(field))


def _gate_section(gates: Any, section: str) -> dict[str, Any]:
    if not isinstance(gates, dict) or not isinstance(gates.get(section), dict):
        raise ValueError(f"缺少门槛分组: {section}")
    return gates[section]


def evaluate_promotion_gates(
    gates: dict[str, Any], evidence: dict[str, Any],
) -> dict[str, Any]:
    """逐项执行冻结门槛；任一空值或非有限证据均判该项失败。"""
    # evidence字段, spec门槛字段, 单位除数, 比较符。
    rules = {
        "base": (
            ("annualized_log_return_delta_q05",
             "annualized_log_return_delta_q05_gt_bps", 10_000.0, ">"),
            ("point_cagr_delta", "point_cagr_delta_min_bps", 10_000.0, ">="),
            ("downside_improvement_q05",
             "downside_improvement_q05_gt_bps", 10_000.0, ">"),
            ("downside_loss_ratio", "downside_loss_ratio_max_bps", 10_000.0, "<="),
            ("mdd_delta", "mdd_delta_max_bps", 10_000.0, "<="),
            ("sharpe_delta", "sharpe_delta_min_milli", 1_000.0, ">="),
            ("entry_count_ratio", "entry_count_ratio_min_bps", 10_000.0, ">="),
            ("exposure_ratio", "exposure_ratio_min_bps", 10_000.0, ">="),
        ),
        "stress": (
            ("annualized_log_return_delta",
             "annualized_log_return_delta_min_bps", 10_000.0, ">="),
            ("mdd_delta", "mdd_delta_max_bps", 10_000.0, "<="),
            ("sharpe_delta", "sharpe_delta_min_milli", 1_000.0, ">="),
        ),
        "cross_symbol": (
            ("median_total_return_delta",
             "median_total_return_delta_gt_bps", 10_000.0, ">"),
            ("positive_symbols", "positive_symbols_min", 1.0, ">="),
            ("positive_contribution",
             "positive_contribution_max_bps", 10_000.0, "<="),
        ),
        "leave_one_out": (
            ("passes", "required_passes", 1.0, ">="),
        ),
    }
    result: dict[str, Any] = {}
    for section, section_rules in rules.items():
        section_gates = _gate_section(gates, section)
        checks = {}
        for field, threshold_field, divisor, comparison in section_rules:
            value = _evidence_value(evidence, section, field)
            threshold = _finite_threshold(
                section_gates, threshold_field, divisor,
            )
            if value is None:
                passed = False
            elif comparison == ">":
                passed = value > threshold
            elif comparison == ">=":
                passed = value >= threshold
            else:
                passed = value <= threshold
            checks[field] = bool(passed)
        result[section] = checks
    result["all"] = bool(all(
        passed
        for section in rules
        for passed in result[section].values()
    ))
    return result


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name}必须是对象")
    return value


def _required(value: dict[str, Any], key: str, name: str) -> Any:
    if key not in value:
        raise ValueError(f"缺少协议字段: {name}.{key}")
    return value[key]


def _require_equal(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise ValueError(
            f"不支持的冻结协议: {name}={value!r}, expected={expected!r}"
        )


def _formal_protocol(spec: Any) -> dict[str, Any]:
    spec = _mapping(spec, "spec")
    universe = _mapping(_required(spec, "universe", "spec"), "spec.universe")
    core_symbols = _symbol_names(
        _required(universe, "core_symbols", "spec.universe"),
        len(_required(universe, "core_symbols", "spec.universe")),
    )
    evaluation = _mapping(
        _required(spec, "evaluation", "spec"), "spec.evaluation",
    )
    downside = _mapping(
        _required(evaluation, "downside", "spec.evaluation"),
        "spec.evaluation.downside",
    )
    bootstrap = _mapping(
        _required(evaluation, "bootstrap", "spec.evaluation"),
        "spec.evaluation.bootstrap",
    )
    stress = _mapping(
        _required(evaluation, "stress_policy", "spec.evaluation"),
        "spec.evaluation.stress_policy",
    )
    strategies = _mapping(
        _required(spec, "strategies", "spec"), "spec.strategies",
    )
    incumbent_strategy = _mapping(
        _required(strategies, "incumbent", "spec.strategies"),
        "spec.strategies.incumbent",
    )
    challenger_strategy = _mapping(
        _required(strategies, "challenger", "spec.strategies"),
        "spec.strategies.challenger",
    )
    decision = _mapping(
        _required(spec, "decision", "spec"), "spec.decision",
    )

    fixed_values = (
        (_required(evaluation, "stress_reuses_base_orders", "spec.evaluation"),
         True, "evaluation.stress_reuses_base_orders"),
        (_required(evaluation, "risk_free_rate_bps", "spec.evaluation"),
         0, "evaluation.risk_free_rate_bps"),
        (_required(evaluation, "portfolio_return", "spec.evaluation"),
         "daily_equal_weight_fixed_core_mark_to_market",
         "evaluation.portfolio_return"),
        (_required(downside, "partition", "spec.evaluation.downside"),
         "nonoverlapping_anchored_at_accrual_start", "downside.partition"),
        (_required(downside, "terminal_incomplete_block", "spec.evaluation.downside"),
         "exclude", "downside.terminal_incomplete_block"),
        (_required(bootstrap, "method", "spec.evaluation.bootstrap"),
         "paired_two_axis_moving_block", "bootstrap.method"),
        (_required(bootstrap, "bit_generator", "spec.evaluation.bootstrap"),
         "PCG64", "bootstrap.bit_generator"),
        (_required(bootstrap, "time_block_scheme", "spec.evaluation.bootstrap"),
         "ordinary_non_circular_moving_block", "bootstrap.time_block_scheme"),
        (_required(bootstrap, "quantile_method", "spec.evaluation.bootstrap"),
         "linear", "bootstrap.quantile_method"),
        (_required(bootstrap, "studentized", "spec.evaluation.bootstrap"),
         False, "bootstrap.studentized"),
        (_required(bootstrap, "point_estimate", "spec.evaluation.bootstrap"),
         "unsampled_original_series", "bootstrap.point_estimate"),
        (_required(stress, "order_source", "spec.evaluation.stress_policy"),
         "reuse_exact_base_rule_orders_and_fills", "stress_policy.order_source"),
        (_required(stress, "path_recalculation", "spec.evaluation.stress_policy"),
         False, "stress_policy.path_recalculation"),
        (_required(stress, "terminal_open_position", "spec.evaluation.stress_policy"),
         "mark_without_exit_cost", "stress_policy.terminal_open_position"),
        (_required(incumbent_strategy, "terminal_policy", "spec.strategies.incumbent"),
         "mark", "strategies.incumbent.terminal_policy"),
        (_required(challenger_strategy, "terminal_policy", "spec.strategies.challenger"),
         "mark", "strategies.challenger.terminal_policy"),
    )
    for actual, expected, name in fixed_values:
        _require_equal(actual, expected, name)

    annual_sessions = _positive_integer(
        _required(evaluation, "annual_sessions", "spec.evaluation"),
        "evaluation.annual_sessions",
    )
    base_cost = _finite_scalar(
        _required(evaluation, "base_cost_bps_per_side", "spec.evaluation"),
    )
    stress_cost = _finite_scalar(
        _required(evaluation, "stress_cost_bps_per_side", "spec.evaluation"),
    )
    rule_cost = _finite_scalar(
        _required(challenger_strategy, "rule_cost_bps_per_side",
                  "spec.strategies.challenger"),
    )
    trigger_cost = _finite_scalar(
        _required(stress, "rule_cost_for_trigger_bps_per_side",
                  "spec.evaluation.stress_policy"),
    )
    revalue_cost = _finite_scalar(
        _required(stress, "revalue_cost_bps_per_side",
                  "spec.evaluation.stress_policy"),
    )
    if None in (base_cost, stress_cost, rule_cost, trigger_cost, revalue_cost):
        raise ValueError("成本协议必须全部为有限数")
    _require_equal(rule_cost, base_cost, "challenger.rule_cost_bps_per_side")
    _require_equal(trigger_cost, base_cost,
                   "stress_policy.rule_cost_for_trigger_bps_per_side")
    _require_equal(revalue_cost, stress_cost,
                   "stress_policy.revalue_cost_bps_per_side")

    symbol_resample_count = _positive_integer(
        _required(bootstrap, "symbol_resample_count", "spec.evaluation.bootstrap"),
        "bootstrap.symbol_resample_count",
    )
    _require_equal(
        symbol_resample_count, len(core_symbols), "bootstrap.symbol_resample_count",
    )
    alpha_bps = _finite_scalar(
        _required(bootstrap, "alpha_bps", "spec.evaluation.bootstrap"),
    )
    if alpha_bps is None or not 0 < alpha_bps < 10_000:
        raise ValueError("bootstrap.alpha_bps必须在 (0, 10000) 内")
    return {
        "core_symbols": core_symbols,
        "annual_sessions": annual_sessions,
        "base_cost_bps": base_cost,
        "stress_cost_bps": stress_cost,
        "downside_block_sessions": _positive_integer(
            _required(downside, "block_sessions", "spec.evaluation.downside"),
            "downside.block_sessions",
        ),
        "bootstrap_replications": _positive_integer(
            _required(bootstrap, "replications", "spec.evaluation.bootstrap"),
            "bootstrap.replications",
        ),
        "bootstrap_seed": _positive_integer(
            _required(bootstrap, "seed", "spec.evaluation.bootstrap"),
            "bootstrap.seed",
        ),
        "time_block_sessions": _positive_integer(
            _required(bootstrap, "time_block_sessions", "spec.evaluation.bootstrap"),
            "bootstrap.time_block_sessions",
        ),
        "symbol_resample_count": symbol_resample_count,
        "alpha": alpha_bps / 10_000.0,
        "gates": _mapping(
            _required(evaluation, "gates", "spec.evaluation"),
            "spec.evaluation.gates",
        ),
        "decision": decision,
    }


def _public_path_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "annualized_log_return": metrics["annualized_log_return"],
        "cagr": metrics["cagr"],
        "sharpe": metrics["sharpe"],
        "mdd": metrics["mdd"],
        "exposure": metrics["exposure"],
        "symbol_total_returns": [
            float(value) for value in metrics["symbol_total_returns"]
        ],
    }


def _public_downside(downside: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_returns": [float(value) for value in downside["block_returns"]],
        "losses": [float(value) for value in downside["losses"]],
        "mean_loss": downside["mean_loss"],
        "full_blocks": downside["full_blocks"],
        "excluded_sessions": downside["excluded_sessions"],
    }


def calculate_evaluation(
    spec: dict[str, Any],
    open_prices: Any,
    close_prices: Any,
    incumbent_entry_fills: Any,
    incumbent_exit_fills: Any,
    challenger_entry_fills: Any,
    challenger_exit_fills: Any,
) -> dict[str, Any]:
    """计算固定成交路径的全部指标，但不产生正式状态转换。

    这是可用于合成测试的低层纯计算 API。它可报告指标门槛是否全部通过，
    但返回值刻意不包含 ``state``、正式决策或次数消耗。只有
    :func:`formal_evaluate` 在验证完整注册 spec 和 READY 快照后才能产生这些字段。
    """
    protocol = _formal_protocol(spec)
    opens = _positive_float_matrix(open_prices, "open_prices")
    if opens.shape[0] != len(protocol["core_symbols"]):
        raise ValueError("open_prices行数必须与冻结核心标的数一致")
    closes = _positive_float_matrix(close_prices, "close_prices")
    if closes.shape != opens.shape:
        raise ValueError("open_prices与close_prices形状必须一致")

    base_incumbent = replay_fixed_orders(
        opens, closes, incumbent_entry_fills, incumbent_exit_fills,
        cost_bps=protocol["base_cost_bps"],
    )
    base_challenger = replay_fixed_orders(
        opens, closes, challenger_entry_fills, challenger_exit_fills,
        cost_bps=protocol["base_cost_bps"],
    )
    # 压力情景重用上面已验证的同一组成交 mask。
    stress_incumbent = replay_fixed_orders(
        opens, closes,
        base_incumbent["entry_fills"], base_incumbent["exit_fills"],
        cost_bps=protocol["stress_cost_bps"],
    )
    stress_challenger = replay_fixed_orders(
        opens, closes,
        base_challenger["entry_fills"], base_challenger["exit_fills"],
        cost_bps=protocol["stress_cost_bps"],
    )

    annual_sessions = protocol["annual_sessions"]
    base_incumbent_metrics = path_metrics(
        base_incumbent["equity"], base_incumbent["held_at_close"],
        annual_sessions=annual_sessions,
    )
    base_challenger_metrics = path_metrics(
        base_challenger["equity"], base_challenger["held_at_close"],
        annual_sessions=annual_sessions,
    )
    stress_incumbent_metrics = path_metrics(
        stress_incumbent["equity"], stress_incumbent["held_at_close"],
        annual_sessions=annual_sessions,
    )
    stress_challenger_metrics = path_metrics(
        stress_challenger["equity"], stress_challenger["held_at_close"],
        annual_sessions=annual_sessions,
    )

    block_sessions = protocol["downside_block_sessions"]
    incumbent_downside = nonoverlapping_downside(
        base_incumbent_metrics["portfolio_daily_returns"],
        block_sessions=block_sessions,
    )
    challenger_downside = nonoverlapping_downside(
        base_challenger_metrics["portfolio_daily_returns"],
        block_sessions=block_sessions,
    )
    incumbent_mean_loss = incumbent_downside["mean_loss"]
    challenger_mean_loss = challenger_downside["mean_loss"]

    bootstrap_result = None
    not_mature_state = None
    if incumbent_mean_loss is None:
        not_mature_state = "NOT_MATURE_INSUFFICIENT_DOWNSIDE_BLOCKS"
    elif incumbent_mean_loss <= 0:
        not_mature_state = "NOT_MATURE_ZERO_INCUMBENT_DOWNSIDE"
    else:
        bootstrap_result = paired_two_axis_bootstrap(
            base_incumbent_metrics["symbol_daily_returns"],
            base_challenger_metrics["symbol_daily_returns"],
            annual_sessions=annual_sessions,
            replications=protocol["bootstrap_replications"],
            seed=protocol["bootstrap_seed"],
            time_block_sessions=protocol["time_block_sessions"],
            downside_block_sessions=block_sessions,
            symbol_resample_count=protocol["symbol_resample_count"],
            alpha=protocol["alpha"],
        )

    cross_symbol = cross_symbol_robustness(
        base_incumbent["equity"], base_challenger["equity"],
        symbols=protocol["core_symbols"],
    )
    leave_one_out = leave_one_out_robustness(
        base_incumbent_metrics["symbol_daily_returns"],
        base_challenger_metrics["symbol_daily_returns"],
        symbols=protocol["core_symbols"],
        annual_sessions=annual_sessions,
        gates=protocol["gates"]["leave_one_out"],
    )

    incumbent_entries = int(base_incumbent["entry_fills"].sum())
    challenger_entries = int(base_challenger["entry_fills"].sum())
    evidence = {
        "base": {
            "annualized_log_return_delta_q05": (
                bootstrap_result["annualized_log_return_delta_q05"]
                if bootstrap_result is not None else None
            ),
            "point_cagr_delta": _finite_difference(
                base_challenger_metrics["cagr"], base_incumbent_metrics["cagr"],
            ),
            "downside_improvement_q05": (
                bootstrap_result["downside_improvement_q05"]
                if bootstrap_result is not None else None
            ),
            "downside_loss_ratio": safe_ratio(
                challenger_mean_loss, incumbent_mean_loss,
            ),
            "mdd_delta": _finite_difference(
                base_challenger_metrics["mdd"], base_incumbent_metrics["mdd"],
            ),
            "sharpe_delta": _finite_difference(
                base_challenger_metrics["sharpe"],
                base_incumbent_metrics["sharpe"],
            ),
            "entry_count_ratio": safe_ratio(
                challenger_entries, incumbent_entries,
            ),
            "exposure_ratio": safe_ratio(
                base_challenger_metrics["exposure"],
                base_incumbent_metrics["exposure"],
            ),
        },
        "stress": {
            "annualized_log_return_delta": _finite_difference(
                stress_challenger_metrics["annualized_log_return"],
                stress_incumbent_metrics["annualized_log_return"],
            ),
            "mdd_delta": _finite_difference(
                stress_challenger_metrics["mdd"],
                stress_incumbent_metrics["mdd"],
            ),
            "sharpe_delta": _finite_difference(
                stress_challenger_metrics["sharpe"],
                stress_incumbent_metrics["sharpe"],
            ),
        },
        "cross_symbol": {
            "median_total_return_delta": cross_symbol[
                "median_total_return_delta"
            ],
            "positive_symbols": cross_symbol["positive_symbols"],
            "positive_contribution": cross_symbol["positive_contribution"],
        },
        "leave_one_out": {"passes": leave_one_out["passes"]},
    }
    gates = evaluate_promotion_gates(protocol["gates"], evidence)

    if not_mature_state is not None:
        eligible_by_metrics = False
    else:
        eligible_by_metrics = bool(gates["all"])

    public_bootstrap = None
    if bootstrap_result is not None:
        public_bootstrap = {
            "annualized_log_return_delta_q05": bootstrap_result[
                "annualized_log_return_delta_q05"
            ],
            "downside_improvement_q05": bootstrap_result[
                "downside_improvement_q05"
            ],
            "replications": bootstrap_result["replications"],
            "seed": bootstrap_result["seed"],
            "bit_generator": bootstrap_result["bit_generator"],
            "blocks_per_replication": bootstrap_result[
                "blocks_per_replication"
            ],
            "quantile": protocol["alpha"],
        }
    return {
        "calculation_status": not_mature_state or "METRIC_GATES_EVALUATED",
        "eligible_by_metrics": eligible_by_metrics,
        "order_replay": {
            "base_cost_bps_per_side": protocol["base_cost_bps"],
            "stress_cost_bps_per_side": protocol["stress_cost_bps"],
            "stress_reuses_base_orders": True,
            "incumbent_entry_fills": incumbent_entries,
            "incumbent_exit_fills": int(base_incumbent["exit_fills"].sum()),
            "challenger_entry_fills": challenger_entries,
            "challenger_exit_fills": int(base_challenger["exit_fills"].sum()),
        },
        "metrics": {
            "base": {
                "incumbent": _public_path_metrics(base_incumbent_metrics),
                "challenger": _public_path_metrics(base_challenger_metrics),
            },
            "downside": {
                "incumbent": _public_downside(incumbent_downside),
                "challenger": _public_downside(challenger_downside),
                "point_estimate": "unsampled_original_series",
            },
            "bootstrap": public_bootstrap,
            "stress": {
                "incumbent": _public_path_metrics(stress_incumbent_metrics),
                "challenger": _public_path_metrics(stress_challenger_metrics),
            },
            "cross_symbol": {
                "by_symbol": cross_symbol["by_symbol"],
                "total_return_deltas": [
                    float(value) for value in cross_symbol["total_return_deltas"]
                ],
                "median_total_return_delta": cross_symbol[
                    "median_total_return_delta"
                ],
                "positive_symbols": cross_symbol["positive_symbols"],
                "positive_contribution": cross_symbol["positive_contribution"],
            },
            "leave_one_out": leave_one_out,
        },
        "evidence": evidence,
        "gates": gates,
    }


def _sha256_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name}必须是64位SHA-256十六进制字符串")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{name}必须是64位SHA-256十六进制字符串"
        ) from error
    return value.lower()


def _hash_mapping(
    value: Any,
    name: str,
    *,
    expected_keys: tuple[str, ...] | None = None,
) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name}必须是非空SHA-256映射")
    if (
        any(not isinstance(key, str) or not key for key in value)
        or (expected_keys is not None and set(value) != set(expected_keys))
    ):
        raise ValueError(f"{name}的键集与冻结协议不一致")
    return {
        key: _sha256_text(hash_value, f"{name}.{key}")
        for key, hash_value in sorted(value.items())
    }


def _session_date_strings(
    session_dates: Any, expected_count: int,
) -> tuple[str, ...]:
    try:
        values = tuple(session_dates)
    except TypeError as error:
        raise ValueError("session_dates必须是严格递增的ISO日期序列") from error
    if len(values) != expected_count:
        raise ValueError("session_dates数量必须与净值列数一致")
    parsed = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("session_dates必须是严格递增的ISO日期序列")
        try:
            parsed_value = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("session_dates必须是严格递增的ISO日期序列") from error
        if parsed_value.isoformat() != value:
            raise ValueError("session_dates必须使用YYYY-MM-DD格式")
        parsed.append(parsed_value)
    if any(left >= right for left, right in zip(parsed, parsed[1:])):
        raise ValueError("session_dates必须严格递增")
    return values


def _exit_reason_matrix(
    value: Any,
    exit_fills: np.ndarray,
    *,
    allowed_reasons: set[str] | None = None,
) -> np.ndarray:
    reasons = np.asarray(value, dtype=object)
    if reasons.shape != exit_fills.shape:
        raise ValueError("challenger_exit_reasons形状必须与退出成交矩阵一致")
    normalized = np.empty(reasons.shape, dtype=object)
    for position in np.ndindex(reasons.shape):
        reason = reasons[position]
        if exit_fills[position]:
            if not isinstance(reason, str) or not reason:
                raise ValueError("challenger每笔退出成交都必须有退出原因")
            if allowed_reasons is not None and reason not in allowed_reasons:
                raise ValueError(f"非冻结退出原因: {reason}")
            normalized[position] = reason
        else:
            if reason not in (None, ""):
                raise ValueError("无退出成交的会话不得填写退出原因")
            normalized[position] = ""
    return normalized


def derive_challenger_cohorts(
    spec: dict[str, Any],
    open_prices: Any,
    close_prices: Any,
    incumbent_exit_fills: Any,
    challenger_entry_fills: Any,
    challenger_exit_fills: Any,
    challenger_exit_reasons: Any,
    *,
    symbols: Any,
    session_dates: Any,
) -> dict[str, Any]:
    """从固定路径推导已武装cohort和差异profit-lock退出。"""
    protocol = _formal_protocol(spec)
    opens = _positive_float_matrix(open_prices, "open_prices")
    closes = _positive_float_matrix(close_prices, "close_prices")
    if closes.shape != opens.shape:
        raise ValueError("open_prices与close_prices形状必须一致")
    names = _symbol_names(symbols, opens.shape[0])
    _require_equal(names, protocol["core_symbols"], "symbols")
    dates = _session_date_strings(session_dates, opens.shape[1])
    incumbent_exits = _bool_matrix(
        incumbent_exit_fills, "incumbent_exit_fills", opens.shape,
    )
    challenger_entries = _bool_matrix(
        challenger_entry_fills, "challenger_entry_fills", opens.shape,
    )
    challenger_exits = _bool_matrix(
        challenger_exit_fills, "challenger_exit_fills", opens.shape,
    )
    if np.logical_and(challenger_entries, challenger_exits).any():
        raise ValueError("challenger同一会话不得同时入场与退出")
    allowed_reasons = set(spec["strategies"]["challenger"]["exit_reason_priority"])
    reasons = _exit_reason_matrix(
        challenger_exit_reasons,
        challenger_exits,
        allowed_reasons=allowed_reasons,
    )
    challenger = spec["strategies"]["challenger"]
    arm_gain = challenger["arm_peak_gain_bps"] / 10_000.0
    trail = challenger["trail_bps"] / 10_000.0
    profit_keep = challenger["profit_keep_bps"] / 10_000.0
    rule_cost = challenger["rule_cost_bps_per_side"] / 10_000.0
    spec_hash = canonical_spec_hash(spec)
    rows = []
    for symbol_pos, symbol in enumerate(names):
        entry_pos = None
        ordinal = 0
        for session_pos in range(opens.shape[1]):
            if challenger_entries[symbol_pos, session_pos]:
                if entry_pos is not None:
                    raise ValueError("challenger已持仓时不得再次入场")
                entry_pos = session_pos
                ordinal += 1
            if not challenger_exits[symbol_pos, session_pos]:
                continue
            if entry_pos is None:
                raise ValueError("challenger空仓时不得退出")
            rows.append(_cohort_evidence_row(
                spec_hash=spec_hash,
                strategy_id=challenger["strategy_id"],
                symbol=symbol,
                ordinal=ordinal,
                dates=dates,
                opens=opens[symbol_pos],
                closes=closes[symbol_pos],
                entry_pos=entry_pos,
                exit_pos=session_pos,
                exit_reason=str(reasons[symbol_pos, session_pos]),
                incumbent_same_exit=bool(
                    incumbent_exits[symbol_pos, session_pos]
                ),
                arm_gain=arm_gain,
                trail=trail,
                profit_keep=profit_keep,
                rule_cost=rule_cost,
            ))
            entry_pos = None
        if entry_pos is not None:
            rows.append(_cohort_evidence_row(
                spec_hash=spec_hash,
                strategy_id=challenger["strategy_id"],
                symbol=symbol,
                ordinal=ordinal,
                dates=dates,
                opens=opens[symbol_pos],
                closes=closes[symbol_pos],
                entry_pos=entry_pos,
                exit_pos=None,
                exit_reason=None,
                incumbent_same_exit=None,
                arm_gain=arm_gain,
                trail=trail,
                profit_keep=profit_keep,
                rule_cost=rule_cost,
            ))
    armed = [row for row in rows if row["arm_date"] is not None]
    affected = [row for row in rows if row["affected"]]
    serialized = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "challenger_armed_cohorts": len(armed),
        "challenger_armed_symbols": len({row["symbol"] for row in armed}),
        "affected_exits": len(affected),
        "affected_symbols": len({row["symbol"] for row in affected}),
        "cohort_ledger_sha256": hashlib.sha256(serialized).hexdigest(),
        "rows": rows,
    }


def _cohort_evidence_row(
    *,
    spec_hash: str,
    strategy_id: str,
    symbol: str,
    ordinal: int,
    dates: tuple[str, ...],
    opens: np.ndarray,
    closes: np.ndarray,
    entry_pos: int,
    exit_pos: int | None,
    exit_reason: str | None,
    incumbent_same_exit: bool | None,
    arm_gain: float,
    trail: float,
    profit_keep: float,
    rule_cost: float,
) -> dict[str, Any]:
    path_end = exit_pos if exit_pos is not None else len(dates)
    held_closes = closes[entry_pos:path_end]
    threshold = opens[entry_pos] * (1.0 + arm_gain)
    arm_hits = np.flatnonzero(held_closes >= threshold)
    arm_pos = entry_pos + int(arm_hits[0]) if len(arm_hits) else None
    cohort_payload = (
        f"{spec_hash}|{strategy_id}|{symbol}|{dates[entry_pos]}|{ordinal}"
    ).encode("utf-8")
    cohort_id = hashlib.sha256(cohort_payload).hexdigest()
    affected = False
    if exit_pos is not None and exit_reason == "profit_lock":
        if arm_pos is None:
            raise ValueError("profit_lock退出的cohort必须已达到武装阈值")
        decision_pos = exit_pos - 1
        if decision_pos < entry_pos:
            raise ValueError("profit_lock退出必须在入场后的CLOSE决策")
        peak_close = float(closes[entry_pos:exit_pos].max())
        entry_open = float(opens[entry_pos])
        break_even = entry_open / (1.0 - rule_cost) ** 2
        trail_floor = peak_close * (1.0 - trail)
        profit_floor = break_even + profit_keep * (peak_close - break_even)
        decision_close = float(closes[decision_pos])
        if not trail_floor < decision_close <= profit_floor:
            raise ValueError("profit_lock退出原因与冻结价格路径不一致")
        affected = not bool(incumbent_same_exit)
    return {
        "cohort_id": cohort_id,
        "symbol": symbol,
        "entry_fill_date": dates[entry_pos],
        "arm_date": dates[arm_pos] if arm_pos is not None else None,
        "exit_fill_date": dates[exit_pos] if exit_pos is not None else None,
        "exit_reason": exit_reason,
        "incumbent_same_session_exit_order": incumbent_same_exit,
        "affected": affected,
    }


def fixed_order_artifact_sha256(
    open_prices: Any,
    close_prices: Any,
    incumbent_entry_fills: Any,
    incumbent_exit_fills: Any,
    challenger_entry_fills: Any,
    challenger_exit_fills: Any,
    challenger_exit_reasons: Any,
    *,
    spec_hash: str,
    symbols: Any,
    session_dates: Any,
    locked_end: str,
    source_hashes: Any,
    accepted_bar_hashes: Any,
) -> str:
    """对锁窗价格、成交mask、行序与来源证据生成内容哈希。"""
    spec_hash = _sha256_text(spec_hash, "spec_hash")
    opens = _positive_float_matrix(open_prices, "open_prices")
    closes = _positive_float_matrix(close_prices, "close_prices")
    if closes.shape != opens.shape:
        raise ValueError("open_prices与close_prices形状必须一致")
    names = _symbol_names(symbols, opens.shape[0])
    dates = _session_date_strings(session_dates, opens.shape[1])
    entries_i = _bool_matrix(
        incumbent_entry_fills, "incumbent_entry_fills", opens.shape,
    )
    exits_i = _bool_matrix(
        incumbent_exit_fills, "incumbent_exit_fills", opens.shape,
    )
    entries_c = _bool_matrix(
        challenger_entry_fills, "challenger_entry_fills", opens.shape,
    )
    exits_c = _bool_matrix(
        challenger_exit_fills, "challenger_exit_fills", opens.shape,
    )
    reasons_c = _exit_reason_matrix(challenger_exit_reasons, exits_c)
    if not isinstance(locked_end, str) or dates[-1] != locked_end:
        raise ValueError("locked_end必须与锁窗价格的最后一个交易日一致")
    sources = _hash_mapping(source_hashes, "source_hashes")
    bars = _hash_mapping(
        accepted_bar_hashes, "accepted_bar_hashes", expected_keys=names,
    )
    header = {
        "schema_version": "gcn-shadow-fixed-orders-v1",
        "spec_hash": spec_hash,
        "symbols": names,
        "session_dates": dates,
        "locked_end": locked_end,
        "source_hashes": sources,
        "accepted_bar_hashes": bars,
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(
        header, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8"))
    digest.update(b"\n")
    for symbol_pos in range(opens.shape[0]):
        for session_pos in range(opens.shape[1]):
            row = "|".join((
                names[symbol_pos],
                dates[session_pos],
                float(opens[symbol_pos, session_pos]).hex(),
                float(closes[symbol_pos, session_pos]).hex(),
                str(int(entries_i[symbol_pos, session_pos])),
                str(int(exits_i[symbol_pos, session_pos])),
                str(int(entries_c[symbol_pos, session_pos])),
                str(int(exits_c[symbol_pos, session_pos])),
                str(reasons_c[symbol_pos, session_pos]),
            ))
            digest.update(row.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


_READY_CONTEXT_FIELDS = frozenset({
    "state", "spec_hash", "actual_accrual_start", "locked_end",
    "next_common_session_after_locked_end",
    "performance_end", "locked_months", "maturity_36_passed",
    "maturity_summary", "post_lock_common_sessions",
    "pending_20_session_labels", "pending_60_session_labels",
    "formal_evaluation_count", "source_hashes", "accepted_bar_hashes",
    "order_artifact_sha256",
})

_MATURITY_COUNT_FIELDS = (
    "incumbent_reference_entries",
    "challenger_armed_cohorts",
    "challenger_armed_symbols",
    "incumbent_active_symbols",
    "incumbent_negative_20_session_blocks",
    "affected_exits",
    "affected_symbols",
)


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name}必须是非负整数")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name}必须是非负整数")
    return value


def _validate_ready_context(
    spec: dict[str, Any],
    context: Any,
    *,
    symbols: tuple[str, ...],
    session_dates: tuple[str, ...],
    shape: tuple[int, int],
    incumbent_entry_fills: np.ndarray,
    challenger_entry_fills: np.ndarray,
    challenger_exit_fills: np.ndarray,
) -> dict[str, Any]:
    context = _mapping(context, "readiness_context")
    if set(context) != _READY_CONTEXT_FIELDS:
        missing = sorted(_READY_CONTEXT_FIELDS - set(context))
        unknown = sorted(set(context) - _READY_CONTEXT_FIELDS)
        raise ValueError(
            f"readiness_context字段不一致: missing={missing}, unknown={unknown}"
        )
    _require_equal(context["state"], "READY_ONCE", "readiness_context.state")
    spec_hash = canonical_spec_hash(spec)
    _require_equal(
        _sha256_text(context["spec_hash"], "readiness_context.spec_hash"),
        spec_hash,
        "readiness_context.spec_hash",
    )
    boundaries = spec["boundaries"]
    actual_start = context["actual_accrual_start"]
    _require_equal(
        actual_start, boundaries["expected_accrual_start"],
        "readiness_context.actual_accrual_start",
    )
    _require_equal(session_dates[0], actual_start, "session_dates[0]")
    locked_end = context["locked_end"]
    if not isinstance(locked_end, str):
        raise ValueError("readiness_context.locked_end必须是ISO日期")
    try:
        locked_date = date.fromisoformat(locked_end)
    except ValueError as error:
        raise ValueError("readiness_context.locked_end必须是ISO日期") from error
    _require_equal(session_dates[-1], locked_end, "session_dates[-1]")
    _require_equal(
        context["performance_end"], locked_end,
        "readiness_context.performance_end",
    )
    locked_months = context["locked_months"]
    if locked_months not in (36, 48) or isinstance(locked_months, bool):
        raise ValueError("readiness_context.locked_months必须是36或48")
    endpoint = date.fromisoformat(
        boundaries[
            "expected_minimum_accrual_end"
            if locked_months == 36 else "expected_maximum_accrual_end"
        ]
    )
    next_common = context["next_common_session_after_locked_end"]
    if not isinstance(next_common, str):
        raise ValueError("next_common_session_after_locked_end必须是ISO日期")
    try:
        next_common_date = date.fromisoformat(next_common)
    except ValueError as error:
        raise ValueError(
            "next_common_session_after_locked_end必须是ISO日期"
        ) from error
    if next_common_date.isoformat() != next_common:
        raise ValueError("next_common_session_after_locked_end必须使用YYYY-MM-DD格式")
    if locked_date > endpoint:
        raise ValueError("locked_end不得晚于对应的冻结端点")
    if next_common_date <= locked_date or next_common_date <= endpoint:
        raise ValueError("locked_end不是冻结端点当日或之前最后共同交易日")
    expected_36_flag = locked_months == 36
    _require_equal(
        context["maturity_36_passed"], expected_36_flag,
        "readiness_context.maturity_36_passed",
    )

    maturity_summary = _mapping(
        context["maturity_summary"], "readiness_context.maturity_summary",
    )
    if set(maturity_summary) != set(_MATURITY_COUNT_FIELDS):
        raise ValueError("readiness_context.maturity_summary字段不完整")
    counts = {
        field: _nonnegative_integer(
            maturity_summary[field], f"maturity_summary.{field}",
        )
        for field in _MATURITY_COUNT_FIELDS
    }
    common = spec["maturity"]["common"]
    endpoint_maturity = spec["maturity"][f"at_{locked_months}_months"]
    minimums = {
        "incumbent_reference_entries": common[
            "incumbent_reference_entries_min"
        ],
        "challenger_armed_cohorts": common["challenger_armed_cohorts_min"],
        "challenger_armed_symbols": common["challenger_armed_symbols_min"],
        "incumbent_active_symbols": common["incumbent_active_symbols_min"],
        "incumbent_negative_20_session_blocks": common[
            "incumbent_negative_20_session_blocks_min"
        ],
        "affected_exits": endpoint_maturity["affected_exits_min"],
        "affected_symbols": endpoint_maturity["affected_symbols_min"],
    }
    failed = [field for field, minimum in minimums.items()
              if counts[field] < minimum]
    if failed:
        raise ValueError("READY_ONCE成熟计数不足: " + ", ".join(failed))
    core_count = len(symbols)
    for field in (
        "challenger_armed_symbols", "incumbent_active_symbols",
        "affected_symbols",
    ):
        if counts[field] > core_count:
            raise ValueError(f"maturity_summary.{field}不得超过核心标的数")
    if counts["affected_symbols"] > counts["affected_exits"]:
        raise ValueError("affected_symbols不得超过affected_exits")
    if counts["incumbent_reference_entries"] != int(incumbent_entry_fills.sum()):
        raise ValueError("在职参考入场计数与固定成交路径不一致")
    if counts["challenger_armed_cohorts"] > int(challenger_entry_fills.sum()):
        raise ValueError("challenger已武装cohort不得超过成交入场数")
    if counts["affected_exits"] > int(challenger_exit_fills.sum()):
        raise ValueError("affected退出不得超过challenger成交退出数")

    post_lock = _nonnegative_integer(
        context["post_lock_common_sessions"], "post_lock_common_sessions",
    )
    if post_lock < boundaries["outcome_embargo_common_sessions"]:
        raise ValueError("READY_ONCE尚未完成结果禁运60共同交易日")
    for field in ("pending_20_session_labels", "pending_60_session_labels"):
        if _nonnegative_integer(context[field], field) != 0:
            raise ValueError(f"READY_ONCE仍有待完成标签: {field}")
    if _nonnegative_integer(
        context["formal_evaluation_count"], "formal_evaluation_count",
    ) != 0:
        raise ValueError("正式评估次数已消耗")
    if shape[1] != len(session_dates):
        raise ValueError("价格矩阵与会话日期数不一致")
    return {
        "spec_hash": spec_hash,
        "locked_end": locked_end,
        "next_common_session_after_locked_end": next_common,
        "source_hashes": _hash_mapping(
            context["source_hashes"], "readiness_context.source_hashes",
        ),
        "accepted_bar_hashes": _hash_mapping(
            context["accepted_bar_hashes"],
            "readiness_context.accepted_bar_hashes",
            expected_keys=symbols,
        ),
        "order_artifact_sha256": _sha256_text(
            context["order_artifact_sha256"],
            "readiness_context.order_artifact_sha256",
        ),
        "locked_months": locked_months,
        "maturity_summary": counts,
    }


def formal_evaluate(
    spec: dict[str, Any],
    open_prices: Any,
    close_prices: Any,
    incumbent_entry_fills: Any,
    incumbent_exit_fills: Any,
    challenger_entry_fills: Any,
    challenger_exit_fills: Any,
    challenger_exit_reasons: Any,
    *,
    symbols: Any,
    session_dates: Any,
    readiness_context: dict[str, Any],
) -> dict[str, Any]:
    """验证冻结 spec 和 READY 快照后，生成可供 CAS 提交的正式决策。

    本函数是无状态的受控计算边界：它校验输入快照中的旧计数为0，
    并返回 ``0 -> 1`` 的状态转换意图。防重与并发原子性必须由调用方
    在持久化锁/CAS内完成，不能仅依赖这个纯函数。
    """
    validate_spec(spec)
    protocol = _formal_protocol(spec)
    _require_equal(
        spec["decision"]["formal_evaluations_allowed"], 1,
        "decision.formal_evaluations_allowed",
    )
    opens = _positive_float_matrix(open_prices, "open_prices")
    closes = _positive_float_matrix(close_prices, "close_prices")
    if closes.shape != opens.shape:
        raise ValueError("open_prices与close_prices形状必须一致")
    names = _symbol_names(symbols, opens.shape[0])
    _require_equal(names, protocol["core_symbols"], "symbols")
    dates = _session_date_strings(session_dates, opens.shape[1])
    entries_i = _bool_matrix(
        incumbent_entry_fills, "incumbent_entry_fills", opens.shape,
    )
    exits_i = _bool_matrix(
        incumbent_exit_fills, "incumbent_exit_fills", opens.shape,
    )
    entries_c = _bool_matrix(
        challenger_entry_fills, "challenger_entry_fills", opens.shape,
    )
    exits_c = _bool_matrix(
        challenger_exit_fills, "challenger_exit_fills", opens.shape,
    )
    reasons_c = _exit_reason_matrix(
        challenger_exit_reasons,
        exits_c,
        allowed_reasons=set(
            spec["strategies"]["challenger"]["exit_reason_priority"]
        ),
    )
    if any(mask[:, 0].any() for mask in (entries_i, exits_i, entries_c, exits_c)):
        raise ValueError("累积起点CLOSE才能首次决策，首列不得有OPEN成交")
    ready = _validate_ready_context(
        spec,
        readiness_context,
        symbols=names,
        session_dates=dates,
        shape=opens.shape,
        incumbent_entry_fills=entries_i,
        challenger_entry_fills=entries_c,
        challenger_exit_fills=exits_c,
    )
    actual_order_hash = fixed_order_artifact_sha256(
        opens,
        closes,
        entries_i,
        exits_i,
        entries_c,
        exits_c,
        reasons_c,
        spec_hash=ready["spec_hash"],
        symbols=names,
        session_dates=dates,
        locked_end=ready["locked_end"],
        source_hashes=ready["source_hashes"],
        accepted_bar_hashes=ready["accepted_bar_hashes"],
    )
    _require_equal(
        actual_order_hash,
        ready["order_artifact_sha256"],
        "readiness_context.order_artifact_sha256",
    )
    calculation = calculate_evaluation(
        spec,
        opens,
        closes,
        entries_i,
        exits_i,
        entries_c,
        exits_c,
    )
    observed_active_symbols = int(np.count_nonzero(entries_i.any(axis=1)))
    if observed_active_symbols != ready["maturity_summary"]["incumbent_active_symbols"]:
        raise ValueError("在职活跃标的数与固定成交路径不一致")
    observed_negative_blocks = sum(
        value < 0
        for value in calculation["metrics"]["downside"]["incumbent"][
            "block_returns"
        ]
    )
    if (
        observed_negative_blocks
        != ready["maturity_summary"]["incumbent_negative_20_session_blocks"]
    ):
        raise ValueError("在职负20日块计数与锁窗评估路径不一致")
    cohort_evidence = derive_challenger_cohorts(
        spec,
        opens,
        closes,
        exits_i,
        entries_c,
        exits_c,
        reasons_c,
        symbols=names,
        session_dates=dates,
    )
    for field in (
        "challenger_armed_cohorts", "challenger_armed_symbols",
        "affected_exits", "affected_symbols",
    ):
        if cohort_evidence[field] != ready["maturity_summary"][field]:
            raise ValueError(f"{field}与锁窗cohort事件证据不一致")
    if calculation["calculation_status"] != "METRIC_GATES_EVALUATED":
        raise ValueError(
            "READY_ONCE与评估数据矛盾: "
            + calculation["calculation_status"]
        )
    eligible = bool(calculation["eligible_by_metrics"])
    state = (
        "ELIGIBLE_FOR_V6_IMPLEMENTATION"
        if eligible else spec["decision"]["ready_failure"]
    )
    decision = (
        spec["decision"]["promotion_result"] if eligible else state
    )
    return {
        **calculation,
        "state": state,
        "decision": decision,
        "eligible": eligible,
        "formal_evaluation_consumed": True,
        "cas_transition": {
            "expected_state": "READY_ONCE",
            "expected_formal_evaluation_count": 0,
            "next_formal_evaluation_count": 1,
            "next_state": state,
        },
        "provenance": {
            "spec_hash": ready["spec_hash"],
            "locked_end": ready["locked_end"],
            "next_common_session_after_locked_end": ready[
                "next_common_session_after_locked_end"
            ],
            "locked_months": ready["locked_months"],
            "order_artifact_sha256": actual_order_hash,
            "cohort_ledger_sha256": cohort_evidence["cohort_ledger_sha256"],
        },
    }
