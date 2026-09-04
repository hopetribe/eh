# -*- coding: utf-8 -*-
"""隔离于生产入口的 v6 前向影子账本重放器。"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gcn.backtest.engine import _one_strategy
from gcn.backtest.shadow_evaluation import (
    fixed_order_artifact_sha256, formal_evaluate,
)
from gcn.backtest.shadow_validation import (
    canonical_bar_hash, canonical_spec_hash, load_spec, merge_accepted_bars,
    rebase_adjusted_incoming, validate_observation_sessions, validate_spec,
)
from gcn.recipes.gcn_main import _stage_confirmation, compute_ehopt10


_COUNT_FIELDS = (
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

_TERMINAL_STATES = frozenset({
    "ELIGIBLE_FOR_V6_IMPLEMENTATION",
    "INCONCLUSIVE_COVERAGE_KEEP_V5",
    "REJECTED_KEEP_V5",
})

_REGISTRATION_FIELDS = {
    "base", "core_symbols", "experiment_id", "implementation",
    "schema_version", "serialization", "signal_cutoff_exclusive",
    "spec_hash",
}

_SERIALIZATION_PROTOCOL = {
    "adjusted_rebase_addition_uniqueness":
        "required_fail_closed_data_blocked",
    "adjusted_rebase_canonical_significant_digits": 12,
    "bar_hash": "sha256-v1",
    "canonical_json": "sorted_compact_utf8_lf",
    "float": "python_float_hex",
    "generation": "one_core_common_session_v1",
}

ALGORITHM_SOURCE_PATHS = (
    "gcn/backtest/engine.py",
    "gcn/backtest/shadow_specs/nyse-us-equities-sessions-20260906-20301231.json",
    "gcn/backtest/shadow_evaluation.py",
    "gcn/backtest/shadow_operations.py",
    "gcn/backtest/shadow_runner.py",
    "gcn/backtest/shadow_validation.py",
    "gcn/core/tdx.py",
    "gcn/data/service.py",
    "gcn/recipes/gcn_main.py",
)

_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}


class ShadowRunnerLifecycleError(ValueError):
    """官方调用模式与实验生命周期不匹配。"""


class ShadowRunnerDataBlockedError(ValueError):
    """官方内存行情不满足追加或交易日约束。"""


@contextmanager
def _experiment_lock(
    state_root: Path, experiment_id: str, spec_hash: str, *,
    create: bool = True, shared: bool = False,
):
    """以进程内互斥+相邻flock串行化首次发布和全部增量提交。"""
    lock_dir = state_root / experiment_id
    if create:
        lock_dir.mkdir(parents=True, exist_ok=True)
    elif not lock_dir.is_dir():
        raise ShadowRunnerLifecycleError("影子实验尚未初始化")
    lock_path = lock_dir / f".{spec_hash}.lock"
    if not create and not lock_path.is_file():
        raise ShadowRunnerLifecycleError("影子实验锁缺失")
    lock_key = str(lock_path.resolve())
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(lock_key, threading.Lock())
    with thread_lock:
        with lock_path.open("a+b" if create else "rb") as lock_file:
            lock_mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(lock_file.fileno(), lock_mode)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def runtime_environment_identity() -> dict[str, str]:
    """冻结会影响数值重放的解释器、数值库与平台身份。"""
    return {
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "system": platform.system(),
    }


def algorithm_source_hashes() -> dict[str, str]:
    """从实际算法文件计算哈希；调用方不能自报完整性。"""
    root = Path(__file__).resolve().parents[2]
    hashes = {}
    for relative in ALGORITHM_SOURCE_PATHS:
        source = root / relative
        if not source.is_file():
            raise ValueError(f"缺少影子算法源码: {relative}")
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def derive_shadow_boundaries(
    spec: dict[str, Any], frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """从核心池共同交易日确定隔离期与真实样本累积起点。"""
    core_symbols = tuple(spec["universe"]["core_symbols"])
    missing = [symbol for symbol in core_symbols if symbol not in frames]
    if missing:
        raise ValueError("缺少核心标的行情: " + ", ".join(missing))
    validated = {
        symbol: merge_accepted_bars(None, frames[symbol])
        for symbol in core_symbols
    }
    common_sessions = validated[core_symbols[0]].index
    for symbol in core_symbols[1:]:
        common_sessions = common_sessions.intersection(validated[symbol].index)
    common_sessions = common_sessions.sort_values()
    cutoff = pd.Timestamp(spec["boundaries"]["signal_cutoff_exclusive"])
    forward_sessions = validated[core_symbols[0]].index[
        validated[core_symbols[0]].index > cutoff
    ]
    for symbol in core_symbols[1:]:
        symbol_forward = validated[symbol].index[validated[symbol].index > cutoff]
        if not symbol_forward.equals(forward_sessions):
            raise ValueError(
                "核心池前向交易日不一致，按DATA_BLOCKED处理: " + symbol
            )
    validate_observation_sessions(
        forward_sessions,
        cutoff=cutoff,
        maximum_accrual_end=pd.Timestamp(
            spec["boundaries"]["expected_maximum_accrual_end"]
        ),
        outcome_embargo_sessions=spec["boundaries"][
            "outcome_embargo_common_sessions"
        ],
    )
    embargo = spec["boundaries"]["initial_embargo_common_sessions"]
    accrual_start = forward_sessions[embargo] if len(forward_sessions) > embargo else None
    return {
        "state": "ACCRUING_36M" if accrual_start is not None else "INITIAL_EMBARGO",
        "elapsed_common_sessions": int(len(forward_sessions)),
        "latest_common_session": (
            forward_sessions[-1].date().isoformat() if len(forward_sessions) else None
        ),
        "actual_accrual_start": (
            accrual_start.date().isoformat() if accrual_start is not None else None
        ),
        "common_sessions": common_sessions,
        "forward_sessions": forward_sessions,
    }


def reset_v5_confirmation_window(
    indicator: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    """保留全历史指标预热，但丢弃累积起点前的B Setup挂起状态。"""
    required = {"B_SETUP", "B_SIGNAL", "HIGH", "CLOSE", "MID"}
    missing = sorted(required - set(indicator.columns))
    if missing:
        raise ValueError("v5确认重放缺少字段: " + ", ".join(missing))
    window = indicator.loc[pd.Timestamp(start):pd.Timestamp(end)].copy()
    confirmed, expired = _stage_confirmation(
        window["B_SETUP"].fillna(False).astype(bool),
        window["HIGH"],
        window["CLOSE"],
        window["MID"],
        window=5,
    )
    window["B_SIGNAL"] = confirmed
    if "B_SETUP_EXPIRED" in window:
        window["B_SETUP_EXPIRED"] = expired
    return window


def _position_rows(
    result: dict[str, Any], frame: pd.DataFrame, strategy_id: str,
    symbol: str, spec_hash: str,
) -> list[dict[str, Any]]:
    rows = []
    positions: list[tuple[int, int | None, dict[str, Any] | None]] = [
        (int(trade["i"]), int(trade["j"]), trade)
        for trade in result["trades"]
    ]
    state = result["state"]
    if state["position"] == "open":
        positions.append((int(state["entry_i"]), None, None))
    for ordinal, (entry_i, exit_j, trade) in enumerate(positions, start=1):
        entry_date = frame.index[entry_i].date().isoformat()
        cohort_payload = (
            f"{spec_hash}|{strategy_id}|{symbol}|{entry_date}|{ordinal}"
        ).encode("utf-8")
        held_closes = frame["CLOSE"].iloc[
            entry_i:exit_j if exit_j is not None else len(frame)
        ]
        rows.append({
            "cohort_id": hashlib.sha256(cohort_payload).hexdigest(),
            "entry_ordinal": ordinal,
            "entry_i": entry_i,
            "entry_fill_date": entry_date,
            "entry_open": float(frame["OPEN"].iloc[entry_i]),
            "peak_close": float(held_closes.max()),
            "exit_j": exit_j,
            "trade": trade,
        })
    return rows


def summarize_shadow_window(
    spec: dict[str, Any], prepared_by_symbol: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """重放冻结双策略并只汇总READY前允许使用的成熟计数。"""
    validate_spec(spec)
    core_symbols = tuple(spec["universe"]["core_symbols"])
    if set(prepared_by_symbol) != set(core_symbols):
        raise ValueError("prepared核心标的集合与spec不一致")
    cost = spec["evaluation"]["base_cost_bps_per_side"] / 10_000
    trail = spec["strategies"]["incumbent"]["trail_bps"] / 10_000
    profit_keep = spec["strategies"]["challenger"]["profit_keep_bps"] / 10_000
    arm_gain = spec["strategies"]["challenger"]["arm_peak_gain_bps"] / 10_000
    spec_hash = canonical_spec_hash(spec)
    reference_entries = 0
    active_symbols = set()
    armed_cohorts = 0
    armed_symbols = set()
    affected_events = []
    incumbent_curves = []

    for symbol in core_symbols:
        frame = prepared_by_symbol[symbol]
        incumbent = _one_strategy(
            frame, ["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"],
            cost=cost, max_hold=None, trail=trail, terminal_policy="mark",
        )
        challenger = _one_strategy(
            frame, ["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"],
            cost=cost, max_hold=None, trail=trail, profit_keep=profit_keep,
            terminal_policy="mark",
        )
        incumbent_positions = _position_rows(
            incumbent, frame, spec["strategies"]["incumbent"]["strategy_id"],
            symbol, spec_hash,
        )
        challenger_positions = _position_rows(
            challenger, frame, spec["strategies"]["challenger"]["strategy_id"],
            symbol, spec_hash,
        )
        reference_entries += len(incumbent_positions)
        if incumbent_positions:
            active_symbols.add(symbol)
        incumbent_exit_orders = {
            frame.index[int(trade["j"]) - 1]
            for trade in incumbent["trades"]
        }
        if incumbent["state"]["status"] == "pending_exit":
            incumbent_exit_orders.add(frame.index[-1])

        for position in challenger_positions:
            armed = position["peak_close"] >= position["entry_open"] * (1 + arm_gain)
            if armed:
                armed_cohorts += 1
                armed_symbols.add(symbol)
            trade = position["trade"]
            if trade is None or trade["exit_reason"] != "profit_lock":
                continue
            exit_j = int(trade["j"])
            decision_date = frame.index[exit_j - 1]
            if decision_date in incumbent_exit_orders:
                continue
            affected_events.append({
                "event_id": position["cohort_id"],
                "symbol": symbol,
                "entry_fill_date": position["entry_fill_date"],
                "trigger_decision_date": decision_date.date().isoformat(),
                "exit_fill_date": frame.index[exit_j].date().isoformat(),
                "exit_fill_open": float(frame["OPEN"].iloc[exit_j]),
                "exit_reason": "profit_lock",
            })
        incumbent_curves.append(np.asarray(incumbent["equity"], dtype=float))

    curve_matrix = np.vstack(incumbent_curves)
    previous = np.concatenate(
        [np.ones((len(core_symbols), 1)), curve_matrix[:, :-1]], axis=1,
    )
    symbol_daily_returns = curve_matrix / previous - 1
    portfolio_daily_returns = symbol_daily_returns.mean(axis=0)
    block_size = spec["evaluation"]["downside"]["block_sessions"]
    full_blocks = len(portfolio_daily_returns) // block_size
    negative_blocks = sum(
        np.prod(1 + portfolio_daily_returns[pos:pos + block_size]) - 1 < 0
        for pos in range(0, full_blocks * block_size, block_size)
    )
    affected_symbols = {event["symbol"] for event in affected_events}
    return {
        "incumbent_reference_entries": int(reference_entries),
        "challenger_armed_cohorts": int(armed_cohorts),
        "challenger_armed_symbols": int(len(armed_symbols)),
        "incumbent_active_symbols": int(len(active_symbols)),
        "incumbent_negative_20_session_blocks": int(negative_blocks),
        "affected_exits": int(len(affected_events)),
        "affected_symbols": int(len(affected_symbols)),
        "affected_events": affected_events,
    }


def prepare_shadow_windows(
    spec: dict[str, Any], frames: dict[str, pd.DataFrame],
    start: pd.Timestamp, end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """在end截断的完整历史上算指标，再以清空确认状态的窗口重放。"""
    prepared = {}
    for symbol in spec["universe"]["core_symbols"]:
        history = frames[symbol].loc[:pd.Timestamp(end)]
        indicator = compute_ehopt10(history, version="v5")
        prepared[symbol] = reset_v5_confirmation_window(
            indicator, pd.Timestamp(start), pd.Timestamp(end),
        )
    return prepared


class _EventPrefix:
    """O(1)事件前缀视图；保持按核心标的生成的确定性事件顺序。"""

    __slots__ = ("_events", "_exit_positions", "_end", "_length")

    def __init__(
        self, events: tuple[dict[str, Any], ...],
        exit_positions: tuple[int, ...], end: int, length: int,
    ):
        self._events = events
        self._exit_positions = exit_positions
        self._end = end
        self._length = length

    def __iter__(self):
        return (
            event for exit_pos, event in zip(
                self._exit_positions, self._events,
            ) if exit_pos <= self._end
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index):
        return tuple(iter(self))[index]

    def __eq__(self, other: object) -> bool:
        try:
            return list(self) == list(other)  # type: ignore[arg-type]
        except TypeError:
            return False

    def __repr__(self) -> str:
        return repr(list(self))


def _precompute_shadow_summary_lookup(
    spec: dict[str, Any], frames: dict[str, pd.DataFrame], start: pd.Timestamp,
) -> dict[str, dict[str, Any]]:
    """一次策略重放生成全部前缀成熟度摘要，供历史代际线性校验。"""
    prepared = prepare_shadow_windows(
        spec, frames, pd.Timestamp(start), frames[next(iter(frames))].index[-1],
    )
    symbols = tuple(spec["universe"]["core_symbols"])
    sessions = prepared[symbols[0]].index
    if any(not prepared[symbol].index.equals(sessions) for symbol in symbols[1:]):
        raise ValueError("历史语义重放的核心池交易日不一致")
    size = len(sessions)
    increments = {
        field: np.zeros(size, dtype=np.int64)
        for field in _COUNT_FIELDS[:-2]
    }
    affected_events: list[tuple[int, dict[str, Any]]] = []
    incumbent_curves = []
    cost = spec["evaluation"]["base_cost_bps_per_side"] / 10_000
    trail = spec["strategies"]["incumbent"]["trail_bps"] / 10_000
    profit_keep = spec["strategies"]["challenger"]["profit_keep_bps"] / 10_000
    arm_gain = spec["strategies"]["challenger"]["arm_peak_gain_bps"] / 10_000
    spec_hash = canonical_spec_hash(spec)

    for symbol in symbols:
        frame = prepared[symbol]
        incumbent = _one_strategy(
            frame, ["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"],
            cost=cost, max_hold=None, trail=trail, terminal_policy="mark",
        )
        challenger = _one_strategy(
            frame, ["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"],
            cost=cost, max_hold=None, trail=trail, profit_keep=profit_keep,
            terminal_policy="mark",
        )
        incumbent_positions = _position_rows(
            incumbent, frame, spec["strategies"]["incumbent"]["strategy_id"],
            symbol, spec_hash,
        )
        challenger_positions = _position_rows(
            challenger, frame,
            spec["strategies"]["challenger"]["strategy_id"], symbol, spec_hash,
        )
        for position in incumbent_positions:
            increments["incumbent_reference_entries"][position["entry_i"]] += 1
        if incumbent_positions:
            increments["incumbent_active_symbols"][min(
                position["entry_i"] for position in incumbent_positions
            )] += 1

        incumbent_exit_orders = {
            int(trade["j"]) - 1 for trade in incumbent["trades"]
        }
        if incumbent["state"]["status"] == "pending_exit":
            incumbent_exit_orders.add(size - 1)
        symbol_armed_at: list[int] = []
        for position in challenger_positions:
            exit_pos = (
                int(position["exit_j"])
                if position["exit_j"] is not None else size
            )
            held = frame["CLOSE"].iloc[position["entry_i"]:exit_pos]
            crossings = np.flatnonzero(
                held.to_numpy(dtype=float)
                >= position["entry_open"] * (1 + arm_gain)
            )
            if len(crossings):
                armed_at = int(position["entry_i"] + crossings[0])
                increments["challenger_armed_cohorts"][armed_at] += 1
                symbol_armed_at.append(armed_at)
            trade = position["trade"]
            if trade is None or trade["exit_reason"] != "profit_lock":
                continue
            exit_j = int(trade["j"])
            decision_i = exit_j - 1
            if decision_i in incumbent_exit_orders:
                continue
            event = {
                "event_id": position["cohort_id"],
                "symbol": symbol,
                "entry_fill_date": position["entry_fill_date"],
                "trigger_decision_date": sessions[decision_i].date().isoformat(),
                "exit_fill_date": sessions[exit_j].date().isoformat(),
                "exit_fill_open": float(frame["OPEN"].iloc[exit_j]),
                "exit_reason": "profit_lock",
            }
            affected_events.append((exit_j, event))
            increments["affected_exits"][exit_j] += 1
        if symbol_armed_at:
            increments["challenger_armed_symbols"][min(symbol_armed_at)] += 1
        symbol_event_positions = [exit_i for exit_i, event in affected_events
                                  if event["symbol"] == symbol]
        if symbol_event_positions:
            increments["affected_symbols"][min(symbol_event_positions)] += 1
        incumbent_curves.append(np.asarray(incumbent["equity"], dtype=float))

    curve_matrix = np.vstack(incumbent_curves)
    previous = np.concatenate(
        [np.ones((len(symbols), 1)), curve_matrix[:, :-1]], axis=1,
    )
    portfolio_returns = (curve_matrix / previous - 1).mean(axis=0)
    block_size = spec["evaluation"]["downside"]["block_sessions"]
    for block_start in range(0, size - block_size + 1, block_size):
        block_end = block_start + block_size
        if np.prod(1 + portfolio_returns[block_start:block_end]) - 1 < 0:
            increments["incumbent_negative_20_session_blocks"][
                block_end - 1
            ] += 1

    counters = {
        field: np.cumsum(values) for field, values in increments.items()
    }
    event_positions = tuple(exit_i for exit_i, _event in affected_events)
    events = tuple(event for _exit_i, event in affected_events)

    lookup = {}
    for index, session in enumerate(sessions):
        lookup[session.date().isoformat()] = {
            **{field: int(values[index]) for field, values in counters.items()},
            "affected_events": _EventPrefix(
                events, event_positions, index,
                int(counters["affected_exits"][index]),
            ),
        }
    return lookup


def _strategy_fill_masks(
    result: dict[str, Any], session_count: int, *, include_reasons: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    entries = np.zeros(session_count, dtype=bool)
    exits = np.zeros(session_count, dtype=bool)
    reasons = np.full(session_count, "", dtype=object) if include_reasons else None
    for trade in result["trades"]:
        entry_pos = int(trade["i"])
        exit_pos = int(trade["j"])
        if (not 0 <= entry_pos < session_count
                or not 0 <= exit_pos < session_count
                or entries[entry_pos] or exits[exit_pos]):
            raise ValueError("固定成交路径包含越界或重复成交")
        entries[entry_pos] = True
        exits[exit_pos] = True
        if reasons is not None:
            reason = str(trade["exit_reason"])
            reasons[exit_pos] = "S_SIGNAL" if reason == "signal" else reason
    state = result["state"]
    if state["position"] == "open":
        entry_pos = int(state["entry_i"])
        if not 0 <= entry_pos < session_count or entries[entry_pos]:
            raise ValueError("固定成交路径的开放仓位入场无效")
        entries[entry_pos] = True
    if np.logical_and(entries, exits).any():
        raise ValueError("固定成交路径同日不能同时买卖")
    return entries, exits, reasons


def _build_fixed_order_inputs(
    spec: dict[str, Any], frames: dict[str, pd.DataFrame],
    ledger: dict[str, Any], protocol: dict[str, Any],
) -> dict[str, Any]:
    """从权威锁窗K线重建正式评估唯一允许使用的成交工件。"""
    start = protocol.get("actual_accrual_start")
    locked_end = protocol.get("locked_end")
    if not isinstance(start, str) or not isinstance(locked_end, str):
        raise ValueError("READY_ONCE缺少锁窗起止日")
    prepared = prepare_shadow_windows(
        spec, frames, pd.Timestamp(start), pd.Timestamp(locked_end),
    )
    symbols = tuple(spec["universe"]["core_symbols"])
    sessions = prepared[symbols[0]].index
    if sessions.empty:
        raise ValueError("正式评估锁窗不能为空")
    for symbol in symbols[1:]:
        if not prepared[symbol].index.equals(sessions):
            raise ValueError("正式评估核心标的锁窗交易日不一致")
    post_lock_sessions = frames[symbols[0]].index[
        frames[symbols[0]].index > pd.Timestamp(locked_end)
    ]
    if post_lock_sessions.empty:
        raise ValueError("正式评估缺少锁窗后的共同交易日证明")
    for symbol in symbols[1:]:
        symbol_post_lock = frames[symbol].index[
            frames[symbol].index > pd.Timestamp(locked_end)
        ]
        if not symbol_post_lock.equals(post_lock_sessions):
            raise ValueError("正式评估锁窗后核心标的交易日不一致")
    next_common_session = post_lock_sessions[0].date().isoformat()
    locked_bar_hashes = {
        symbol: canonical_bar_hash(
            frames[symbol].loc[:pd.Timestamp(locked_end)]
        )
        for symbol in sorted(symbols)
    }

    open_prices = np.vstack([
        prepared[symbol]["OPEN"].to_numpy(dtype=float) for symbol in symbols
    ])
    close_prices = np.vstack([
        prepared[symbol]["CLOSE"].to_numpy(dtype=float) for symbol in symbols
    ])
    shape = open_prices.shape
    incumbent_entries = np.zeros(shape, dtype=bool)
    incumbent_exits = np.zeros(shape, dtype=bool)
    challenger_entries = np.zeros(shape, dtype=bool)
    challenger_exits = np.zeros(shape, dtype=bool)
    challenger_reasons = np.full(shape, "", dtype=object)
    cost = spec["evaluation"]["base_cost_bps_per_side"] / 10_000
    trail = spec["strategies"]["incumbent"]["trail_bps"] / 10_000
    profit_keep = (
        spec["strategies"]["challenger"]["profit_keep_bps"] / 10_000
    )
    for symbol_pos, symbol in enumerate(symbols):
        frame = prepared[symbol]
        incumbent = _one_strategy(
            frame, ["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"],
            cost=cost, max_hold=None, trail=trail, terminal_policy="mark",
        )
        challenger = _one_strategy(
            frame, ["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"],
            cost=cost, max_hold=None, trail=trail, profit_keep=profit_keep,
            terminal_policy="mark",
        )
        entries, exits, _unused = _strategy_fill_masks(
            incumbent, len(sessions), include_reasons=False,
        )
        incumbent_entries[symbol_pos] = entries
        incumbent_exits[symbol_pos] = exits
        entries, exits, reasons = _strategy_fill_masks(
            challenger, len(sessions), include_reasons=True,
        )
        challenger_entries[symbol_pos] = entries
        challenger_exits[symbol_pos] = exits
        challenger_reasons[symbol_pos] = reasons

    session_dates = tuple(date.date().isoformat() for date in sessions)
    order_hash = fixed_order_artifact_sha256(
        open_prices,
        close_prices,
        incumbent_entries,
        incumbent_exits,
        challenger_entries,
        challenger_exits,
        challenger_reasons,
        spec_hash=ledger["spec_hash"],
        symbols=symbols,
        session_dates=session_dates,
        locked_end=locked_end,
        source_hashes=ledger["source_hashes"],
        accepted_bar_hashes=locked_bar_hashes,
    )
    return {
        "challenger_entry_fills": challenger_entries,
        "challenger_exit_fills": challenger_exits,
        "challenger_exit_reasons": challenger_reasons,
        "close_prices": close_prices,
        "incumbent_entry_fills": incumbent_entries,
        "incumbent_exit_fills": incumbent_exits,
        "open_prices": open_prices,
        "locked_bar_hashes": locked_bar_hashes,
        "next_common_session_after_locked_end": next_common_session,
        "order_artifact_sha256": order_hash,
        "session_dates": session_dates,
        "symbols": symbols,
    }


def _evaluate_ready_snapshot(
    spec: dict[str, Any], frames: dict[str, pd.DataFrame],
    ledger: dict[str, Any], protocol: dict[str, Any],
) -> dict[str, Any]:
    """从未消费的READY快照重建订单并计算唯一正式结果。"""
    if protocol.get("state") != "READY_ONCE":
        raise ValueError("只有READY_ONCE快照可以执行正式评估")
    if protocol.get("formal_evaluation_count") != 0:
        raise ValueError("READY_ONCE正式评估次数已消耗")
    checkpoint = protocol.get("checkpoint_36")
    if (not isinstance(checkpoint, dict)
            or checkpoint.get("passed") is not (
                protocol.get("locked_months") == 36
            )):
        raise ValueError("READY_ONCE缺少一致的36月不可变检查点")
    orders = _build_fixed_order_inputs(spec, frames, ledger, protocol)
    readiness_context = {
        "state": "READY_ONCE",
        "spec_hash": ledger["spec_hash"],
        "actual_accrual_start": protocol["actual_accrual_start"],
        "locked_end": protocol["locked_end"],
        "next_common_session_after_locked_end": orders[
            "next_common_session_after_locked_end"
        ],
        "performance_end": protocol["locked_end"],
        "locked_months": protocol["locked_months"],
        "maturity_36_passed": protocol["maturity_36_passed"],
        "maturity_summary": protocol["maturity_summary"],
        "post_lock_common_sessions": protocol["post_lock_common_sessions"],
        "pending_20_session_labels": protocol[
            "pending_20_session_labels"
        ],
        "pending_60_session_labels": protocol[
            "pending_60_session_labels"
        ],
        "formal_evaluation_count": 0,
        "source_hashes": ledger["source_hashes"],
        "accepted_bar_hashes": orders["locked_bar_hashes"],
        "order_artifact_sha256": orders["order_artifact_sha256"],
    }
    result = formal_evaluate(
        spec,
        orders["open_prices"],
        orders["close_prices"],
        orders["incumbent_entry_fills"],
        orders["incumbent_exit_fills"],
        orders["challenger_entry_fills"],
        orders["challenger_exit_fills"],
        orders["challenger_exit_reasons"],
        symbols=orders["symbols"],
        session_dates=orders["session_dates"],
        readiness_context=readiness_context,
    )
    next_state = result.get("state") if isinstance(result, dict) else None
    expected_transition = {
        "expected_state": "READY_ONCE",
        "expected_formal_evaluation_count": 0,
        "next_formal_evaluation_count": 1,
        "next_state": next_state,
    }
    if (next_state not in {
            "ELIGIBLE_FOR_V6_IMPLEMENTATION", "REJECTED_KEEP_V5",
        }
            or result.get("cas_transition") != expected_transition
            or result.get("formal_evaluation_consumed") is not True
            or result.get("provenance", {}).get("order_artifact_sha256")
            != orders["order_artifact_sha256"]):
        raise ValueError("正式评估器返回无效CAS意图")
    _canonical_json_bytes(result)
    return result


def _consume_ready_once(
    spec: dict[str, Any], frames: dict[str, pd.DataFrame],
    ledger: dict[str, Any], protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """在调用方持有实验锁时计算并应用唯一一次正式评估CAS。"""
    if protocol.get("state") != "READY_ONCE":
        return ledger, protocol
    revealed_protocol = _reveal_ready_affected_event_labels(
        spec, frames, protocol,
    )
    result = _evaluate_ready_snapshot(spec, frames, ledger, protocol)
    return _carry_consumed_evaluation(
        spec, ledger, revealed_protocol, result,
    )


def _verify_consumed_evaluation_transition(
    spec: dict[str, Any], frames: dict[str, pd.DataFrame],
    ready_ledger: dict[str, Any], ready_protocol: dict[str, Any],
    persisted_result: dict[str, Any],
) -> None:
    """重算一次0→1转换，拒绝可重哈希但语义伪造的正式结果。"""
    expected_result = _evaluate_ready_snapshot(
        spec, frames, ready_ledger, ready_protocol,
    )
    if _canonical_json_bytes(persisted_result) != _canonical_json_bytes(
        expected_result
    ):
        raise ValueError("正式评估结果重算不匹配")


def _carry_consumed_evaluation(
    spec: dict[str, Any], ledger: dict[str, Any], protocol: dict[str, Any],
    result: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """校验并携带已消费的唯一正式结果，不再次运行评估器。"""
    if (ledger.get("state") != "READY_ONCE"
            or protocol.get("state") != "READY_ONCE"
            or protocol.get("formal_evaluation_count") != 0
            or protocol.get("evaluation_result") is not None
            or protocol.get("order_artifact_sha256") is not None):
        raise ValueError("只有未消费的READY_ONCE快照可以承接正式结果")
    if not isinstance(result, dict):
        raise ValueError("正式评估结果必须是对象")
    next_state = result.get("state")
    allowed_states = {
        "ELIGIBLE_FOR_V6_IMPLEMENTATION", spec["decision"]["ready_failure"],
    }
    if next_state not in allowed_states:
        raise ValueError("正式评估结果状态无效")
    eligible = result.get("eligible")
    if type(eligible) is not bool or eligible != (
        next_state == "ELIGIBLE_FOR_V6_IMPLEMENTATION"
    ):
        raise ValueError("正式评估结果eligible与状态不一致")
    expected_decision = (
        spec["decision"]["promotion_result"]
        if eligible else spec["decision"]["ready_failure"]
    )
    if result.get("decision") != expected_decision:
        raise ValueError("正式评估结果decision与状态不一致")
    expected_transition = {
        "expected_state": "READY_ONCE",
        "expected_formal_evaluation_count": 0,
        "next_formal_evaluation_count": 1,
        "next_state": next_state,
    }
    if (result.get("formal_evaluation_consumed") is not True
            or result.get("cas_transition") != expected_transition):
        raise ValueError("正式评估结果CAS意图无效")
    provenance = result.get("provenance")
    order_hash = (
        provenance.get("order_artifact_sha256")
        if isinstance(provenance, dict) else None
    )
    if (not isinstance(order_hash, str) or len(order_hash) != 64
            or any(character not in "0123456789abcdef" for character in order_hash)):
        raise ValueError("正式评估结果订单工件摘要无效")
    _canonical_json_bytes(result)
    final_ledger = dict(ledger)
    final_ledger["state"] = next_state
    final_protocol = dict(protocol)
    final_protocol.update({
        "evaluation_result": result,
        "formal_evaluation_count": 1,
        "order_artifact_sha256": order_hash,
        "state": next_state,
    })
    return final_ledger, final_protocol


def maturity_gate_passes(
    spec: dict[str, Any], summary: dict[str, Any], *, months: int,
) -> bool:
    """按36/48月固定端点执行全部AND成熟度门槛。"""
    if months not in (36, 48):
        raise ValueError("months必须是36或48")
    common = spec["maturity"]["common"]
    required = {
        "incumbent_reference_entries": common["incumbent_reference_entries_min"],
        "challenger_armed_cohorts": common["challenger_armed_cohorts_min"],
        "challenger_armed_symbols": common["challenger_armed_symbols_min"],
        "incumbent_active_symbols": common["incumbent_active_symbols_min"],
        "incumbent_negative_20_session_blocks":
            common["incumbent_negative_20_session_blocks_min"],
    }
    endpoint = spec["maturity"][f"at_{months}_months"]
    required.update({
        "affected_exits": endpoint["affected_exits_min"],
        "affected_symbols": endpoint["affected_symbols_min"],
    })
    return all(int(summary.get(field, -1)) >= minimum
               for field, minimum in required.items())


def resolve_accrual_phase(
    spec: dict[str, Any],
    forward_sessions: pd.DatetimeIndex,
    summary_at: Any,
) -> dict[str, Any]:
    """只在固定36/48月端点检查一次成熟度，并先重建36月决定。"""
    if len(forward_sessions) == 0:
        raise ValueError("forward_sessions不能为空")
    latest = forward_sessions[-1]
    expected36 = pd.Timestamp(spec["boundaries"]["expected_minimum_accrual_end"])
    expected48 = pd.Timestamp(spec["boundaries"]["expected_maximum_accrual_end"])
    maturity_fields = (
        "incumbent_reference_entries", "challenger_armed_cohorts",
        "challenger_armed_symbols", "incumbent_active_symbols",
        "incumbent_negative_20_session_blocks", "affected_exits",
        "affected_symbols",
    )
    if latest < expected36:
        return {
            "state": "ACCRUING_36M",
            "locked_end": None,
            "locked_months": None,
            "checkpoint_36": None,
            "summary": summary_at(latest),
        }

    eligible36 = forward_sessions[forward_sessions <= expected36]
    if len(eligible36) == 0:
        raise ValueError("36月端点前没有共同交易日")
    endpoint36 = eligible36[-1]
    summary36 = summary_at(endpoint36)
    passed36 = maturity_gate_passes(spec, summary36, months=36)
    checkpoint36 = {
        "endpoint": endpoint36.date().isoformat(),
        "maturity_summary": {
            field: int(summary36[field]) for field in maturity_fields
        },
        "passed": passed36,
    }
    if passed36:
        return {
            "state": "OUTCOME_EMBARGO_60",
            "locked_end": endpoint36.date().isoformat(),
            "locked_months": 36,
            "checkpoint_36": checkpoint36,
            "summary": summary36,
        }
    if latest < expected48:
        return {
            "state": "ACCRUING_TO_48M",
            "locked_end": None,
            "locked_months": None,
            "checkpoint_36": checkpoint36,
            "summary": summary_at(latest),
        }

    eligible48 = forward_sessions[forward_sessions <= expected48]
    if len(eligible48) == 0:
        raise ValueError("48月端点前没有共同交易日")
    endpoint48 = eligible48[-1]
    summary48 = summary_at(endpoint48)
    if maturity_gate_passes(spec, summary48, months=48):
        return {
            "state": "OUTCOME_EMBARGO_60",
            "locked_end": endpoint48.date().isoformat(),
            "locked_months": 48,
            "checkpoint_36": checkpoint36,
            "summary": summary48,
        }
    return {
        "state": "INCONCLUSIVE_COVERAGE_KEEP_V5",
        "locked_end": None,
        "locked_months": None,
        "checkpoint_36": checkpoint36,
        "summary": summary48,
    }


def label_affected_events(
    spec: dict[str, Any],
    events: list[dict[str, Any]],
    frames: dict[str, pd.DataFrame],
    locked_end: pd.Timestamp,
) -> dict[str, Any]:
    """成熟锁窗时固定的差异退出标签；标签数值不参与晋升。"""
    locked_end = pd.Timestamp(locked_end)
    core_symbols = tuple(spec["universe"]["core_symbols"])
    missing = [symbol for symbol in core_symbols if symbol not in frames]
    if missing:
        raise ValueError("标签行情缺少核心标的: " + ", ".join(missing))
    common = frames[core_symbols[0]].index[
        frames[core_symbols[0]].index > locked_end
    ]
    for symbol in core_symbols[1:]:
        symbol_post = frames[symbol].index[frames[symbol].index > locked_end]
        if not symbol_post.equals(common):
            raise ValueError("标签期核心交易日不一致，按DATA_BLOCKED处理")
    post_lock_common = int((common > locked_end).sum())
    horizons = tuple(spec["maturity"]["common"]["forward_horizons_sessions"])
    pending = {horizon: 0 for horizon in horizons}
    labeled_events = []
    seen_ids = set()
    for event in events:
        event_id = str(event["event_id"])
        if event_id in seen_ids:
            raise ValueError(f"重复affected event_id: {event_id}")
        seen_ids.add(event_id)
        symbol = str(event["symbol"])
        frame = frames[symbol]
        fill_date = pd.Timestamp(event["exit_fill_date"])
        if fill_date > locked_end:
            raise ValueError("affected事件成交日在锁窗终点之后")
        locations = np.flatnonzero(frame.index == fill_date)
        if len(locations) != 1:
            raise ValueError(f"affected成交日缺失或重复: {symbol} {fill_date.date()}")
        fill_pos = int(locations[0])
        fill_open = float(event["exit_fill_open"])
        labels = {}
        for horizon in horizons:
            outcome_pos = fill_pos + int(horizon) - 1
            if outcome_pos >= len(frame):
                labels[str(horizon)] = None
                pending[horizon] += 1
                continue
            path = frame.iloc[fill_pos:outcome_pos + 1]
            labels[str(horizon)] = {
                "return": float(frame["close"].iloc[outcome_pos] / fill_open - 1),
                "mfe": float(path["high"].max() / fill_open - 1),
                "mae": float(path["low"].min() / fill_open - 1),
            }
        labeled_events.append({**event, "labels": labels})
    pending20 = pending.get(20, 0)
    pending60 = pending.get(60, 0)
    embargo = spec["boundaries"]["outcome_embargo_common_sessions"]
    ready = post_lock_common >= embargo and pending20 == 0 and pending60 == 0
    return {
        "state": "READY_ONCE" if ready else "OUTCOME_EMBARGO_60",
        "post_lock_common_sessions": post_lock_common,
        "pending_20_session_labels": int(pending20),
        "pending_60_session_labels": int(pending60),
        "events": labeled_events,
    }


def _blind_affected_event_labels(
    spec: dict[str, Any], events: Any,
) -> list[dict[str, Any]]:
    """保留冻结事件身份，但在正式消费前不持久化任何标签数值。"""
    horizons = spec["maturity"]["common"]["forward_horizons_sessions"]
    hidden = {metric: None for metric in ("return", "mfe", "mae")}
    return [
        {
            **event,
            "labels": {
                str(horizon): dict(hidden) for horizon in horizons
            },
        }
        for event in events
    ]


def _reveal_ready_affected_event_labels(
    spec: dict[str, Any], frames: dict[str, pd.DataFrame],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """在唯一READY消费事务内重算并揭示完整20/60日标签。"""
    events = protocol.get("affected_events")
    if not events:
        return protocol
    locked_end = protocol.get("locked_end")
    if not isinstance(locked_end, str):
        raise ValueError("READY_ONCE缺少标签锁窗终点")
    label_state = label_affected_events(
        spec, events, frames, pd.Timestamp(locked_end),
    )
    expected = {
        "state": "READY_ONCE",
        "post_lock_common_sessions": protocol.get(
            "post_lock_common_sessions"
        ),
        "pending_20_session_labels": 0,
        "pending_60_session_labels": 0,
    }
    if any(label_state.get(field) != value for field, value in expected.items()):
        raise ValueError("READY_ONCE标签尚未完整成熟")
    revealed = dict(protocol)
    revealed["affected_events"] = label_state["events"]
    return revealed


class _AffectedLabelCache:
    """在完整权威历史上只计算一次事件标签路径，按前缀投影状态。"""

    def __init__(
        self, spec: dict[str, Any], frames: dict[str, pd.DataFrame],
    ):
        self._spec = spec
        self._frames = frames
        self._plans: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    def _plan(
        self, events: Any, locked_end: pd.Timestamp,
    ) -> dict[str, Any]:
        locked_end = pd.Timestamp(locked_end)
        event_tuple = tuple(events)
        key = (
            locked_end.date().isoformat(),
            tuple(str(event["event_id"]) for event in event_tuple),
        )
        cached = self._plans.get(key)
        if cached is not None:
            return cached
        core_symbols = tuple(self._spec["universe"]["core_symbols"])
        missing = [symbol for symbol in core_symbols if symbol not in self._frames]
        if missing:
            raise ValueError("标签行情缺少核心标的: " + ", ".join(missing))
        common = self._frames[core_symbols[0]].index[
            self._frames[core_symbols[0]].index > locked_end
        ]
        for symbol in core_symbols[1:]:
            symbol_post = self._frames[symbol].index[
                self._frames[symbol].index > locked_end
            ]
            if not symbol_post.equals(common):
                raise ValueError("标签期核心交易日不一致，按DATA_BLOCKED处理")
        horizons = tuple(
            self._spec["maturity"]["common"]["forward_horizons_sessions"]
        )
        seen_ids = set()
        event_plans = []
        for event in event_tuple:
            event_id = str(event["event_id"])
            if event_id in seen_ids:
                raise ValueError(f"重复affected event_id: {event_id}")
            seen_ids.add(event_id)
            symbol = str(event["symbol"])
            frame = self._frames[symbol]
            fill_date = pd.Timestamp(event["exit_fill_date"])
            if fill_date > locked_end:
                raise ValueError("affected事件成交日在锁窗终点之后")
            locations = np.flatnonzero(frame.index == fill_date)
            if len(locations) != 1:
                raise ValueError(
                    f"affected成交日缺失或重复: {symbol} {fill_date.date()}"
                )
            fill_pos = int(locations[0])
            fill_open = float(event["exit_fill_open"])
            outcomes = {}
            for horizon in horizons:
                outcome_pos = fill_pos + int(horizon) - 1
                if outcome_pos >= len(frame):
                    outcomes[horizon] = (None, None)
                    continue
                path = frame.iloc[fill_pos:outcome_pos + 1]
                outcomes[horizon] = (
                    frame.index[outcome_pos],
                    {
                        "return": float(
                            frame["close"].iloc[outcome_pos] / fill_open - 1
                        ),
                        "mfe": float(path["high"].max() / fill_open - 1),
                        "mae": float(path["low"].min() / fill_open - 1),
                    },
                )
            event_plans.append((event, outcomes))
        cached = {
            "common": common,
            "event_plans": tuple(event_plans),
            "horizons": horizons,
        }
        self._plans[key] = cached
        return cached

    def state_at(
        self, events: Any, locked_end: pd.Timestamp, latest: pd.Timestamp,
    ) -> dict[str, Any]:
        plan = self._plan(events, locked_end)
        latest = pd.Timestamp(latest)
        pending = {horizon: 0 for horizon in plan["horizons"]}
        labeled_events = []
        for event, outcomes in plan["event_plans"]:
            labels = {}
            for horizon, (outcome_date, label) in outcomes.items():
                if outcome_date is None or outcome_date > latest:
                    labels[str(horizon)] = None
                    pending[horizon] += 1
                else:
                    labels[str(horizon)] = label
            labeled_events.append({**event, "labels": labels})
        common = plan["common"]
        post_lock_common = int(common.searchsorted(latest, side="right"))
        pending20 = pending.get(20, 0)
        pending60 = pending.get(60, 0)
        embargo = self._spec["boundaries"]["outcome_embargo_common_sessions"]
        ready = (
            post_lock_common >= embargo and pending20 == 0 and pending60 == 0
        )
        return {
            "state": "READY_ONCE" if ready else "OUTCOME_EMBARGO_60",
            "post_lock_common_sessions": post_lock_common,
            "pending_20_session_labels": int(pending20),
            "pending_60_session_labels": int(pending60),
            "events": labeled_events,
        }


def _build_pre_ready_snapshot(
    spec: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    *,
    formal_evaluation_count: int = 0,
    evaluation_result: dict[str, Any] | None = None,
    _boundaries: dict[str, Any] | None = None,
    _summary_lookup: dict[str, dict[str, Any]] | None = None,
    _label_cache: _AffectedLabelCache | None = None,
    _accepted_bar_hashes: dict[str, str] | None = None,
    _source_hashes: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """同时构造公开白名单账本与不可公开的协议状态快照。"""
    validate_spec(spec)
    if (type(formal_evaluation_count) is not int
            or formal_evaluation_count < 0
            or formal_evaluation_count > 1):
        raise ValueError("formal_evaluation_count必须是0或1")
    if formal_evaluation_count == 0 and evaluation_result is not None:
        raise ValueError("未消费正式评估时不得携带evaluation_result")
    if formal_evaluation_count == 1 and not isinstance(
        evaluation_result, dict
    ):
        raise ValueError("已消费正式评估必须携带evaluation_result")
    boundaries = _boundaries or derive_shadow_boundaries(spec, frames)
    counts = {field: 0 for field in _COUNT_FIELDS}
    state = boundaries["state"]
    protocol: dict[str, Any] = {
        "actual_accrual_start": boundaries["actual_accrual_start"],
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
        "state": state,
    }
    if boundaries["state"] != "INITIAL_EMBARGO":
        actual_start = boundaries["actual_accrual_start"]
        expected_start = spec["boundaries"]["expected_accrual_start"]
        if actual_start != expected_start:
            raise ValueError(
                "真实accrual start偏离预注册日期，按DATA_BLOCKED处理: "
                f"expected={expected_start}, actual={actual_start}"
            )
        def summary_at(end: pd.Timestamp) -> dict[str, Any]:
            if _summary_lookup is not None:
                try:
                    return _summary_lookup[pd.Timestamp(end).date().isoformat()]
                except KeyError as error:
                    raise ValueError("历史generation缺少确定性摘要") from error
            prepared = prepare_shadow_windows(
                spec, frames, pd.Timestamp(actual_start), end,
            )
            return summarize_shadow_window(spec, prepared)

        phase = resolve_accrual_phase(
            spec, boundaries["forward_sessions"], summary_at,
        )
        state = phase["state"]
        summary = phase["summary"]
        counts.update({field: summary[field] for field in _COUNT_FIELDS
                       if field in summary})
        protocol["checkpoint_36"] = phase.get("checkpoint_36")
        if phase["locked_end"] is not None:
            locked_end = phase["locked_end"]
            locked_months = phase["locked_months"]
            if _label_cache is None:
                label_state = label_affected_events(
                    spec,
                    summary["affected_events"],
                    frames,
                    pd.Timestamp(locked_end),
                )
            else:
                latest = boundaries.get("latest_common_session")
                if latest is None:
                    raise ValueError("锁窗标签前缀缺少最新共同交易日")
                label_state = _label_cache.state_at(
                    summary["affected_events"], pd.Timestamp(locked_end),
                    pd.Timestamp(latest),
                )
            state = label_state["state"]
            counts["pending_20_session_labels"] = label_state[
                "pending_20_session_labels"
            ]
            counts["pending_60_session_labels"] = label_state[
                "pending_60_session_labels"
            ]
            maturity_fields = (
                "incumbent_reference_entries",
                "challenger_armed_cohorts", "challenger_armed_symbols",
                "incumbent_active_symbols",
                "incumbent_negative_20_session_blocks",
                "affected_exits", "affected_symbols",
            )
            protocol.update({
                "affected_events": (
                    label_state["events"]
                    if formal_evaluation_count == 1
                    else _blind_affected_event_labels(
                        spec, label_state["events"],
                    )
                ),
                "locked_end": locked_end,
                "locked_months": locked_months,
                "maturity_36_passed": locked_months == 36,
                "maturity_summary": {
                    field: int(summary[field]) for field in maturity_fields
                },
                "pending_20_session_labels": label_state[
                    "pending_20_session_labels"
                ],
                "pending_60_session_labels": label_state[
                    "pending_60_session_labels"
                ],
                "post_lock_common_sessions": label_state[
                    "post_lock_common_sessions"
                ],
            })
    core_symbols = tuple(spec["universe"]["core_symbols"])
    ledger: dict[str, Any] = {
        "spec_hash": canonical_spec_hash(spec),
        "source_hashes": dict(sorted(
            (_source_hashes or algorithm_source_hashes()).items()
        )),
        "accepted_bar_hashes": (
            dict(sorted(_accepted_bar_hashes.items()))
            if _accepted_bar_hashes is not None else {
                symbol: canonical_bar_hash(frames[symbol])
                for symbol in sorted(core_symbols)
            }
        ),
        "integrity_status": "PASS",
        "state": state,
        "elapsed_common_sessions": boundaries["elapsed_common_sessions"],
    }
    ledger.update(counts)
    allowed = set(spec["decision"]["pre_ready_visible_fields"])
    if set(ledger) != allowed:
        raise ValueError("pre_ready_visible_fields与账本字段不一致")
    protocol["state"] = state
    if formal_evaluation_count == 1:
        ledger, protocol = _carry_consumed_evaluation(
            spec, ledger, protocol, evaluation_result,
        )
    return ledger, protocol


def build_pre_ready_ledger(
    spec: dict[str, Any],
    frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """构造READY前的白名单账本；绝不返回收益或风险指标。"""
    ledger, _protocol = _build_pre_ready_snapshot(
        spec, frames,
    )
    return ledger


def _read_bar_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"缺少行情文件: {path.name}")
    raw = pd.read_csv(path)
    expected = ["date", "open", "high", "low", "close", "volume"]
    if list(raw.columns) != expected:
        raise ValueError(f"{path.name}字段必须严格为 {', '.join(expected)}")
    dates = pd.to_datetime(raw.pop("date"), errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{path.name}包含无效日期")
    raw.index = pd.DatetimeIndex(dates)
    raw.index.name = "date"
    return merge_accepted_bars(None, raw)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _write_bar_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=True,
        index_label="date",
        float_format="%.17g",
    )
    with temporary.open("rb") as persisted:
        os.fsync(persisted.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
        ) + "\n").encode("utf-8"),
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _write_canonical_json(path: Path, value: Any) -> bytes:
    payload = _canonical_json_bytes(value)
    _atomic_write_bytes(path, payload)
    return payload


def _canonical_bar_payload(frame: pd.DataFrame) -> dict[str, Any]:
    validated = merge_accepted_bars(None, frame)
    return {
        "columns": ["open", "high", "low", "close", "volume"],
        "rows": [
            [
                date.date().isoformat(),
                *(float(value).hex() for value in values),
            ]
            for date, values in zip(
                validated.index,
                validated.loc[:, ["open", "high", "low", "close", "volume"]]
                .to_numpy(dtype=float),
            )
        ],
    }


def _bar_file_bytes(frame: pd.DataFrame) -> bytes:
    return _canonical_json_bytes(_canonical_bar_payload(frame))


def _frame_from_bar_bytes(payload: bytes, label: str) -> pd.DataFrame:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label}格式无效") from error
    try:
        canonical_payload = _canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}格式无效") from error
    if payload != canonical_payload:
        raise ValueError(f"{label}不是canonical编码")
    if (not isinstance(value, dict)
            or set(value) != {"columns", "rows"}
            or value["columns"] != ["open", "high", "low", "close", "volume"]
            or not isinstance(value["rows"], list)
            or not value["rows"]):
        raise ValueError(f"{label}格式无效")
    dates = []
    rows = []
    for row in value["rows"]:
        if not isinstance(row, list) or len(row) != 6:
            raise ValueError(f"{label}格式无效")
        try:
            parsed_date = pd.Timestamp(row[0])
            parsed = [float.fromhex(item) for item in row[1:]]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}格式无效") from error
        if (not isinstance(row[0], str) or parsed_date.tz is not None
                or parsed_date != parsed_date.normalize()
                or parsed_date.date().isoformat() != row[0]):
            raise ValueError(f"{label}日期必须使用YYYY-MM-DD")
        dates.append(parsed_date)
        if any(not isinstance(item, str) or item != number.hex()
               for item, number in zip(row[1:], parsed)):
            raise ValueError(f"{label}浮点必须使用canonical float.hex")
        rows.append(parsed)
    frame = pd.DataFrame(
        rows,
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return merge_accepted_bars(None, frame)


def _read_registered_base(
    experiment_dir: Path, registration: dict[str, Any], symbol: str,
) -> pd.DataFrame:
    try:
        expected = registration["base"][symbol]
    except (KeyError, TypeError) as error:
        raise ValueError(f"registration缺少冻结前缀: {symbol}") from error
    path = experiment_dir / "base" / f"{symbol}.bars"
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"已接受K线冻结前缀缺失: {symbol}") from error
    if hashlib.sha256(payload).hexdigest() != expected.get("sha256"):
        raise ValueError(f"已接受K线冻结前缀被改写: {symbol}")
    frame = _frame_from_bar_bytes(payload, f"base {symbol}")
    actual = {
        "canonical_bar_hash": canonical_bar_hash(frame),
        "row_count": len(frame),
        "first_date": frame.index[0].date().isoformat(),
        "last_date": frame.index[-1].date().isoformat(),
    }
    if any(expected.get(key) != value for key, value in actual.items()):
        raise ValueError(f"已接受K线冻结前缀元数据不匹配: {symbol}")
    return frame


def _accepted_metadata(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return {
        symbol: {
            "canonical_bar_hash": canonical_bar_hash(frame),
            "row_count": len(frame),
            "last_date": frame.index[-1].date().isoformat(),
        }
        for symbol, frame in sorted(frames.items())
    }


def _bar_prefix_hasher(frame: pd.DataFrame) -> tuple[Any, int]:
    """构造可复制的canonical bars前缀哈希，避免逐代重哈希全部历史。"""
    payload = _canonical_bar_payload(frame)
    prefix = json.dumps(
        {"columns": payload["columns"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )[:-1].encode("utf-8") + b',"rows":['
    digest = hashlib.sha256(prefix)
    for index, row in enumerate(payload["rows"]):
        if index:
            digest.update(b",")
        digest.update(json.dumps(
            row, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8"))
    return digest, len(payload["rows"])


def _extend_bar_prefix_hash(
    digest: Any, row_count: int, row: list[str],
) -> tuple[int, str]:
    if row_count:
        digest.update(b",")
    digest.update(json.dumps(
        row, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8"))
    row_count += 1
    finalized = digest.copy()
    finalized.update(b"]}")
    return row_count, finalized.hexdigest()


def _verify_frozen_prefix(
    registration: dict[str, Any], symbol: str, accepted: pd.DataFrame,
) -> None:
    """确认首次注册的行情前缀仍逐字节等价，防止双边改写绕过。"""
    if registration.get("schema_version") != "gcn-shadow-registration-v2":
        raise ValueError("registration版本不受支持")
    try:
        expected = registration["base"][symbol]
        count = expected["row_count"]
        expected_end = expected["last_date"]
        expected_hash = expected["canonical_bar_hash"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"registration缺少冻结前缀: {symbol}") from error
    if type(count) is not int or count <= 0 or len(accepted) < count:
        raise ValueError(f"已接受K线冻结前缀长度不匹配: {symbol}")
    prefix = accepted.iloc[:count]
    actual_end = prefix.index[-1].date().isoformat()
    actual_hash = canonical_bar_hash(prefix)
    if actual_end != expected_end or actual_hash != expected_hash:
        raise ValueError(f"已接受K线冻结前缀被改写: {symbol}")


def _validate_registration(
    registration: dict[str, Any], spec: dict[str, Any], spec_hash: str,
    source_hashes: dict[str, str], runtime_environment: dict[str, str],
) -> None:
    if not isinstance(registration, dict) or set(registration) != (
        _REGISTRATION_FIELDS
    ):
        raise ValueError("registration字段不匹配")
    expected_scalars = {
        "schema_version": "gcn-shadow-registration-v2",
        "experiment_id": spec["experiment_id"],
        "spec_hash": spec_hash,
        "signal_cutoff_exclusive":
            spec["boundaries"]["signal_cutoff_exclusive"],
        "core_symbols": list(spec["universe"]["core_symbols"]),
        "serialization": _SERIALIZATION_PROTOCOL,
        "implementation": {
            "runtime_environment": runtime_environment,
            "source_hashes": source_hashes,
        },
    }
    for field, expected in expected_scalars.items():
        if registration.get(field) != expected:
            if field == "implementation":
                raise ValueError("影子源码或运行环境偏离首次注册身份")
            raise ValueError(f"registration {field}不匹配")
    base = registration.get("base")
    if not isinstance(base, dict) or set(base) != set(
        spec["universe"]["core_symbols"]
    ):
        raise ValueError("registration base标的集合不匹配")
    for symbol, metadata in base.items():
        if not isinstance(metadata, dict) or set(metadata) != {
            "canonical_bar_hash", "first_date", "last_date", "row_count",
            "sha256",
        }:
            raise ValueError(f"registration base元数据字段不匹配: {symbol}")
        if (type(metadata["row_count"]) is not int
                or metadata["row_count"] <= 0):
            raise ValueError(f"registration base行数无效: {symbol}")
        for hash_field in ("canonical_bar_hash", "sha256"):
            digest = metadata[hash_field]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(
                    f"registration base哈希无效: {symbol} {hash_field}"
                )


def _initial_registration(
    spec: dict[str, Any], spec_hash: str,
    source_hashes: dict[str, str], runtime_environment: dict[str, str],
    frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    symbols = sorted(frames)
    base = {}
    for symbol in symbols:
        frame = frames[symbol]
        payload = _bar_file_bytes(frame)
        base[symbol] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "canonical_bar_hash": canonical_bar_hash(frame),
            "row_count": len(frame),
            "first_date": frame.index[0].date().isoformat(),
            "last_date": frame.index[-1].date().isoformat(),
        }
    return {
        "schema_version": "gcn-shadow-registration-v2",
        "experiment_id": spec["experiment_id"],
        "spec_hash": spec_hash,
        "signal_cutoff_exclusive": spec["boundaries"]["signal_cutoff_exclusive"],
        "core_symbols": list(spec["universe"]["core_symbols"]),
        "serialization": _SERIALIZATION_PROTOCOL,
        "implementation": {
            "runtime_environment": runtime_environment,
            "source_hashes": source_hashes,
        },
        "base": base,
    }


def _publish_initial_experiment(
    experiment_dir: Path, registration: dict[str, Any],
    frames: dict[str, pd.DataFrame], ledger: dict[str, Any],
    protocol_state: dict[str, Any],
) -> None:
    """先在同一父目录写完整快照，再以单次rename发布首次状态。"""
    parent = experiment_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{experiment_dir.name}.tmp-", dir=parent,
    ))
    try:
        base_dir = staging / "base"
        base_dir.mkdir()
        staging_bars = staging / "accepted_bars"
        staging_bars.mkdir()
        for symbol in sorted(frames):
            _atomic_write_bytes(
                base_dir / f"{symbol}.bars", _bar_file_bytes(frames[symbol]),
            )
            _write_bar_csv(staging_bars / f"{symbol}_1d.csv", frames[symbol])
        _write_canonical_json(staging / "registration.json", registration)
        registration_hash = hashlib.sha256(
            (staging / "registration.json").read_bytes()
        ).hexdigest()
        generation = {
            "schema_version": "gcn-shadow-generation-v1",
            "sequence": 0,
            "previous_hash": registration_hash,
            "session": None,
            "rows_by_symbol": {},
            "accepted": _accepted_metadata(frames),
            "protocol_state": protocol_state,
            "public_ledger": ledger,
        }
        generation_payload = _canonical_json_bytes(generation)
        generation_hash = hashlib.sha256(generation_payload).hexdigest()
        current = {"generation_hash": generation_hash, "sequence": 0}
        reference = f"{0:016d}-{generation_hash}"
        generations_dir = staging / "generations"
        commits_dir = staging / "commits"
        generations_dir.mkdir()
        commits_dir.mkdir()
        _atomic_write_bytes(
            generations_dir / f"{reference}.json", generation_payload,
        )
        _write_canonical_json(commits_dir / f"{reference}.commit", current)
        _write_canonical_json(staging / "CURRENT", current)
        _write_json(staging / "ledger.json", ledger)
        for directory in (
            base_dir, staging_bars, generations_dir, commits_dir, staging,
        ):
            _fsync_directory(directory)
        staging.replace(experiment_dir)
        _fsync_directory(parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _generation_reference(current: dict[str, Any]) -> str:
    sequence = current.get("sequence")
    generation_hash = current.get("generation_hash")
    if (type(sequence) is not int or sequence < 0
            or not isinstance(generation_hash, str)
            or len(generation_hash) != 64
            or any(character not in "0123456789abcdef"
                   for character in generation_hash)):
        raise ValueError("CURRENT格式无效")
    return f"{sequence:016d}-{generation_hash}"


def _read_current(
    experiment_dir: Path, *, repair_cache: bool = True,
) -> dict[str, Any]:
    """以连续commit链定位权威头；写入口可选择修复CURRENT缓存。"""
    commits_dir = experiment_dir / "commits"
    try:
        commit_paths = sorted(commits_dir.glob("*.commit"))
    except OSError as error:
        raise ValueError("commit目录缺失或不可读") from error
    if not commit_paths:
        raise ValueError("commit链为空")

    pointers: dict[int, dict[str, Any]] = {}
    for path in commit_paths:
        parts = path.stem.split("-", 1)
        if (len(parts) != 2 or len(parts[0]) != 16 or not parts[0].isdigit()):
            raise ValueError("commit文件名无效")
        sequence = int(parts[0])
        pointer = {"generation_hash": parts[1], "sequence": sequence}
        if _generation_reference(pointer) != path.stem:
            raise ValueError("commit文件名无效")
        if sequence in pointers:
            raise ValueError(f"commit序号{sequence}发生分叉")
        try:
            commit_payload = path.read_bytes()
        except OSError as error:
            raise ValueError("commit标记不可读") from error
        if commit_payload != _canonical_json_bytes(pointer):
            raise ValueError("commit标记内容不匹配")
        generation_path = (
            experiment_dir / "generations" / f"{path.stem}.json"
        )
        try:
            generation_payload = generation_path.read_bytes()
        except OSError as error:
            raise ValueError("commit对应generation缺失") from error
        if hashlib.sha256(generation_payload).hexdigest() != parts[1]:
            raise ValueError("generation哈希不匹配")
        pointers[sequence] = pointer

    expected_sequences = list(range(max(pointers) + 1))
    if sorted(pointers) != expected_sequences:
        raise ValueError("commit序号不连续")
    current = pointers[expected_sequences[-1]]
    current_path = experiment_dir / "CURRENT"
    current_payload = _canonical_json_bytes(current)
    try:
        cached_payload = current_path.read_bytes()
    except OSError:
        cached_payload = None
    if cached_payload is not None:
        try:
            cached_pointer = json.loads(cached_payload.decode("utf-8"))
            cached_reference = _generation_reference(cached_pointer)
            cached_is_canonical = (
                cached_payload == _canonical_json_bytes(cached_pointer)
                and cached_reference
                == f"{cached_pointer['sequence']:016d}-{cached_pointer['generation_hash']}"
            )
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError,
                TypeError, ValueError):
            cached_pointer = None
            cached_is_canonical = False
        if (repair_cache and cached_is_canonical
                and cached_pointer["sequence"] > current["sequence"]):
            raise ValueError("CURRENT显示权威头回退，commit尾部可能丢失")
    if repair_cache and cached_payload != current_payload:
        _write_canonical_json(current_path, current)
    return current


def _replay_committed_frames(
    experiment_dir: Path,
    registration: dict[str, Any],
    current: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    """从冻结base与已提交哈希链重建权威K线，不信任派生CSV缓存。"""
    symbols = tuple(registration.get("core_symbols", ()))
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("registration核心标的无效")
    frames = {
        symbol: _read_registered_base(experiment_dir, registration, symbol)
        for symbol in symbols
    }
    bar_hashers = {}
    row_counts = {}
    bar_hashes = {}
    added_rows: dict[str, list[tuple[pd.Timestamp, list[float]]]] = {
        symbol: [] for symbol in symbols
    }
    for symbol, frame in frames.items():
        digest, row_count = _bar_prefix_hasher(frame)
        finalized = digest.copy()
        finalized.update(b"]}")
        bar_hashers[symbol] = digest
        row_counts[symbol] = row_count
        bar_hashes[symbol] = finalized.hexdigest()
    cutoff = pd.Timestamp(registration["signal_cutoff_exclusive"])
    if any(frame.index[-1] > cutoff for frame in frames.values()):
        raise ValueError("registration base包含cutoff后的K线")
    try:
        registration_payload = (experiment_dir / "registration.json").read_bytes()
    except OSError as error:
        raise ValueError("registration.json缺失") from error
    previous_hash = hashlib.sha256(registration_payload).hexdigest()
    previous_session: pd.Timestamp | None = None
    head_ledger: dict[str, Any] | None = None
    head_protocol: dict[str, Any] | None = None
    generation_references: list[str] = []
    generation_sessions: list[pd.Timestamp] = []
    base_lengths = {symbol: len(frame) for symbol, frame in frames.items()}
    generation_fields = {
        "schema_version", "sequence", "previous_hash", "session",
        "rows_by_symbol", "accepted", "protocol_state", "public_ledger",
    }
    commit_paths = sorted((experiment_dir / "commits").glob("*.commit"))
    if len(commit_paths) != current["sequence"] + 1:
        raise ValueError("commit数量与权威头序号不一致")

    for sequence, commit_path in enumerate(commit_paths):
        reference = commit_path.stem
        try:
            reference_sequence, generation_hash = reference.split("-", 1)
            if int(reference_sequence) != sequence:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError("commit文件名无效") from error
        pointer = {"generation_hash": generation_hash, "sequence": sequence}
        if commit_path.read_bytes() != _canonical_json_bytes(pointer):
            raise ValueError("commit标记内容不匹配")

        generation_path = experiment_dir / "generations" / f"{reference}.json"
        try:
            payload = generation_path.read_bytes()
        except OSError as error:
            raise ValueError("commit对应generation缺失") from error
        if hashlib.sha256(payload).hexdigest() != generation_hash:
            raise ValueError("generation哈希不匹配")
        try:
            generation = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("generation格式无效") from error
        if payload != _canonical_json_bytes(generation):
            raise ValueError("generation不是canonical JSON")
        if not isinstance(generation, dict) or set(generation) != generation_fields:
            raise ValueError("generation字段无效")
        if generation.get("schema_version") != "gcn-shadow-generation-v1":
            raise ValueError("generation版本不受支持")
        if generation.get("sequence") != sequence:
            raise ValueError("generation序号不匹配")
        if generation.get("previous_hash") != previous_hash:
            raise ValueError("generation previous_hash不匹配")

        if sequence == 0:
            if generation.get("session") is not None or generation.get(
                "rows_by_symbol"
            ) != {}:
                raise ValueError("genesis generation内容无效")
        else:
            session_text = generation.get("session")
            rows_by_symbol = generation.get("rows_by_symbol")
            if not isinstance(session_text, str) or set(rows_by_symbol or {}) != set(
                symbols
            ):
                raise ValueError("generation新增K线字段无效")
            try:
                session = pd.Timestamp(session_text)
            except (TypeError, ValueError) as error:
                raise ValueError("generation交易日无效") from error
            if (session.tz is not None or session != session.normalize()
                    or session.date().isoformat() != session_text
                    or session <= cutoff
                    or (previous_session is not None and session <= previous_session)):
                raise ValueError("generation交易日无效或未递增")
            for symbol in symbols:
                row = rows_by_symbol[symbol]
                if (not isinstance(row, list) or len(row) != 6
                        or row[0] != session_text):
                    raise ValueError(f"generation K线行无效: {symbol}")
                try:
                    values = [float.fromhex(value) for value in row[1:]]
                except (TypeError, ValueError) as error:
                    raise ValueError(f"generation K线行无效: {symbol}") from error
                if any(not isinstance(item, str) or item != value.hex()
                       for item, value in zip(row[1:], values)):
                    raise ValueError(
                        f"generation K线必须使用canonical float.hex: {symbol}"
                    )
                open_, high, low, close, volume = values
                if (not np.isfinite(values).all() or min(
                        open_, high, low, close) <= 0 or volume < 0
                        or high < max(open_, high, low, close)
                        or low > min(open_, high, low, close)):
                    raise ValueError(f"generation K线OHLCV无效: {symbol}")
                added_rows[symbol].append((session, values))
                row_counts[symbol], bar_hashes[symbol] = _extend_bar_prefix_hash(
                    bar_hashers[symbol], row_counts[symbol], row,
                )
            previous_session = session
            generation_sessions.append(session)

        accepted = {
            symbol: {
                "canonical_bar_hash": bar_hashes[symbol],
                "row_count": row_counts[symbol],
                "last_date": (
                    generation_sessions[-1].date().isoformat()
                    if generation_sessions else frames[symbol].index[-1]
                    .date().isoformat()
                ),
            }
            for symbol in sorted(symbols)
        }
        if generation.get("accepted") != accepted:
            raise ValueError("generation accepted元数据不匹配")
        ledger = generation.get("public_ledger")
        expected_bar_hashes = {
            symbol: metadata["canonical_bar_hash"]
            for symbol, metadata in accepted.items()
        }
        if (not isinstance(ledger, dict)
                or ledger.get("spec_hash") != registration.get("spec_hash")
                or ledger.get("accepted_bar_hashes") != expected_bar_hashes):
            raise ValueError("generation公开账本与K线不匹配")
        if set(ledger) != set(spec["decision"]["pre_ready_visible_fields"]):
            raise ValueError("generation公开账本字段不匹配")
        if ledger.get("source_hashes") != registration.get(
            "implementation", {}
        ).get("source_hashes"):
            raise ValueError("generation公开账本源码哈希不匹配")
        if ledger.get("integrity_status") != "PASS":
            raise ValueError("generation完整性状态无效")
        if ledger.get("elapsed_common_sessions") != sequence:
            raise ValueError("generation elapsed_common_sessions与序号不匹配")
        for field in _COUNT_FIELDS:
            value = ledger.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"generation计数字段无效: {field}")
        protocol_state = generation.get("protocol_state")
        if (not isinstance(protocol_state, dict)
                or set(protocol_state) != {
                    "actual_accrual_start", "affected_events",
                    "checkpoint_36", "evaluation_result",
                    "formal_evaluation_count", "locked_end", "locked_months",
                    "maturity_36_passed", "maturity_summary",
                    "order_artifact_sha256",
                    "pending_20_session_labels",
                    "pending_60_session_labels", "post_lock_common_sessions",
                    "state",
                }
                or not isinstance(protocol_state["affected_events"], list)
                or type(protocol_state["formal_evaluation_count"]) is not int
                or protocol_state["formal_evaluation_count"] < 0
                or protocol_state["formal_evaluation_count"] > 1
                or protocol_state["state"] != ledger.get("state")):
            raise ValueError("generation协议状态与公开账本不匹配")
        head_ledger = ledger
        head_protocol = protocol_state
        generation_references.append(reference)
        previous_hash = generation_hash

    if head_ledger is None or head_protocol is None:
        raise ValueError("commit链缺少genesis")

    for symbol in symbols:
        if added_rows[symbol]:
            additions = pd.DataFrame(
                [values for _session, values in added_rows[symbol]],
                columns=["open", "high", "low", "close", "volume"],
                index=pd.DatetimeIndex(
                    [session for session, _values in added_rows[symbol]],
                    name="date",
                ),
            )
            frames[symbol] = merge_accepted_bars(frames[symbol], additions)

    forward_sessions = pd.DatetimeIndex(generation_sessions)
    embargo = spec["boundaries"]["initial_embargo_common_sessions"]
    actual_start = (
        forward_sessions[embargo] if len(forward_sessions) > embargo else None
    )
    summary_lookup = (
        _precompute_shadow_summary_lookup(spec, frames, actual_start)
        if actual_start is not None else {}
    )
    label_cache = _AffectedLabelCache(spec, frames)
    previous_protocol: dict[str, Any] | None = None
    formal_transition_verified = False
    for sequence, reference in enumerate(generation_references):
        generation_path = experiment_dir / "generations" / f"{reference}.json"
        try:
            payload = generation_path.read_bytes()
        except OSError as error:
            raise ValueError("语义重放时generation缺失") from error
        generation_hash = reference.split("-", 1)[1]
        if hashlib.sha256(payload).hexdigest() != generation_hash:
            raise ValueError("语义重放时generation哈希不匹配")
        try:
            generation = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("语义重放时generation格式无效") from error
        accepted = generation["accepted"]
        ledger = generation["public_ledger"]
        protocol = generation["protocol_state"]
        prefix_forward = forward_sessions[:sequence]
        prefix_end = (
            cutoff if sequence == 0 else prefix_forward[-1]
        )
        prefix_frames = {
            symbol: frame.iloc[:base_lengths[symbol] + sequence]
            for symbol, frame in frames.items()
        }
        prefix_start = (
            prefix_forward[embargo]
            if len(prefix_forward) > embargo else None
        )
        boundaries = {
            "state": (
                "ACCRUING_36M" if prefix_start is not None
                else "INITIAL_EMBARGO"
            ),
            "elapsed_common_sessions": sequence,
            "latest_common_session": (
                prefix_end.date().isoformat() if sequence else None
            ),
            "actual_accrual_start": (
                prefix_start.date().isoformat()
                if prefix_start is not None else None
            ),
            "common_sessions": prefix_frames[symbols[0]].index,
            "forward_sessions": prefix_forward,
        }
        snapshot_arguments = {
            "formal_evaluation_count": protocol["formal_evaluation_count"],
            "evaluation_result": protocol["evaluation_result"],
        }
        expected_ledger, expected_protocol = _build_pre_ready_snapshot(
            spec,
            prefix_frames,
            **snapshot_arguments,
            _boundaries=boundaries,
            _summary_lookup=summary_lookup,
            _label_cache=label_cache,
            _accepted_bar_hashes={
                symbol: accepted[symbol]["canonical_bar_hash"]
                for symbol in symbols
            },
            _source_hashes=registration["implementation"]["source_hashes"],
        )
        if ledger != expected_ledger or protocol != expected_protocol:
            raise ValueError(
                f"generation语义重算不匹配: sequence={sequence}"
            )
        previous_count = (
            previous_protocol["formal_evaluation_count"]
            if previous_protocol is not None else 0
        )
        current_count = protocol["formal_evaluation_count"]
        if previous_count == 0 and current_count == 1:
            if formal_transition_verified:
                raise ValueError("正式评估0→1转换只能出现一次")
            ready_ledger, ready_protocol = _build_pre_ready_snapshot(
                spec,
                prefix_frames,
                _boundaries=boundaries,
                _summary_lookup=summary_lookup,
                _label_cache=label_cache,
                _accepted_bar_hashes={
                    symbol: accepted[symbol]["canonical_bar_hash"]
                    for symbol in symbols
                },
                _source_hashes=registration["implementation"][
                    "source_hashes"
                ],
            )
            _verify_consumed_evaluation_transition(
                spec, prefix_frames, ready_ledger, ready_protocol,
                protocol["evaluation_result"],
            )
            formal_transition_verified = True
        if previous_protocol is not None:
            if current_count < previous_count or current_count - previous_count > 1:
                raise ValueError("formal_evaluation_count必须单调且仅消费一次")
            previous_checkpoint = previous_protocol["checkpoint_36"]
            if (previous_checkpoint is not None
                    and protocol["checkpoint_36"] != previous_checkpoint):
                raise ValueError("checkpoint_36一旦形成不可改写")
            if previous_protocol["locked_end"] is not None:
                for field in (
                    "locked_end", "locked_months", "maturity_36_passed",
                    "maturity_summary",
                ):
                    if protocol[field] != previous_protocol[field]:
                        raise ValueError("成熟度锁窗一旦形成不可改写")
            if previous_count == 1:
                for field in (
                    "evaluation_result", "formal_evaluation_count",
                    "order_artifact_sha256", "state",
                ):
                    if protocol[field] != previous_protocol[field]:
                        raise ValueError("正式评估结果一旦消费不可改写")
        previous_protocol = protocol
    if head_protocol["formal_evaluation_count"] == 1 and not (
        formal_transition_verified
    ):
        raise ValueError("已消费正式评估缺少唯一0→1转换代")
    return frames, head_ledger, head_protocol


def _append_generation(
    experiment_dir: Path, previous: dict[str, Any],
    session: pd.Timestamp, frames: dict[str, pd.DataFrame],
    ledger: dict[str, Any], protocol_state: dict[str, Any],
) -> dict[str, Any]:
    sequence = int(previous["sequence"]) + 1
    day = pd.Timestamp(session)
    rows_by_symbol = {}
    for symbol, frame in sorted(frames.items()):
        values = frame.loc[day, ["open", "high", "low", "close", "volume"]]
        rows_by_symbol[symbol] = [
            day.date().isoformat(),
            *(float(value).hex() for value in values),
        ]
    generation = {
        "schema_version": "gcn-shadow-generation-v1",
        "sequence": sequence,
        "previous_hash": previous["generation_hash"],
        "session": day.date().isoformat(),
        "rows_by_symbol": rows_by_symbol,
        "accepted": _accepted_metadata(frames),
        "protocol_state": protocol_state,
        "public_ledger": ledger,
    }
    payload = _canonical_json_bytes(generation)
    generation_hash = hashlib.sha256(payload).hexdigest()
    current = {"generation_hash": generation_hash, "sequence": sequence}
    reference = _generation_reference(current)
    generations_dir = experiment_dir / "generations"
    commits_dir = experiment_dir / "commits"
    generation_path = generations_dir / f"{reference}.json"
    if generation_path.exists():
        if generation_path.read_bytes() != payload:
            raise ValueError("同序号generation发生内容分叉")
    else:
        _atomic_write_bytes(generation_path, payload)
    _write_canonical_json(
        commits_dir / f"{reference}.commit", current,
    )
    _write_canonical_json(experiment_dir / "CURRENT", current)
    return current


def _run_shadow_update_locked(
    spec: dict[str, Any],
    data_dir: Path | None,
    state_root: Path,
    *,
    captured_frames: Mapping[str, pd.DataFrame] | None = None,
    operation: str = "legacy",
) -> dict[str, Any]:
    """验证并追加本地日K，随后原子更新READY前白名单账本。"""
    if operation not in {"legacy", "initialize", "update", "repair"}:
        raise ValueError("未知shadow操作模式")
    if operation == "legacy":
        if data_dir is None or captured_frames is not None:
            raise ValueError("legacy模式必须且只能使用data_dir")
    elif operation == "repair":
        if data_dir is not None or captured_frames is not None:
            raise ValueError("repair模式不得读取外部行情")
    elif data_dir is not None or captured_frames is None:
        raise ValueError("官方shadow模式必须且只能使用内存行情快照")
    normalized_captured: dict[str, pd.DataFrame] | None = None
    if captured_frames is not None:
        expected_symbols = set(spec["universe"]["core_symbols"])
        if set(captured_frames) != expected_symbols:
            raise ShadowRunnerDataBlockedError(
                "内存行情快照标的集合不匹配"
            )
        try:
            normalized_captured = {
                symbol: merge_accepted_bars(
                    None, captured_frames[symbol].copy(deep=True),
                )
                for symbol in spec["universe"]["core_symbols"]
            }
        except ValueError as error:
            raise ShadowRunnerDataBlockedError(str(error)) from error
    spec_hash = canonical_spec_hash(spec)
    experiment_dir = state_root / spec["experiment_id"] / spec_hash
    bars_dir = experiment_dir / "accepted_bars"
    registration_path = experiment_dir / "registration.json"
    ledger_path = experiment_dir / "ledger.json"
    current_source_hashes = algorithm_source_hashes()
    current_runtime_environment = runtime_environment_identity()

    registration = None
    current = None
    accepted_frames = None
    if registration_path.exists():
        if operation == "initialize":
            raise ShadowRunnerLifecycleError("影子实验已经初始化")
        try:
            registration_payload = registration_path.read_bytes()
            registration = json.loads(registration_payload.decode("utf-8"))
            canonical_registration = _canonical_json_bytes(registration)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError,
                TypeError, ValueError) as error:
            raise ValueError("registration不是有效canonical JSON") from error
        if registration_payload != canonical_registration:
            raise ValueError("registration不是canonical JSON")
        _validate_registration(
            registration, spec, spec_hash, current_source_hashes,
            current_runtime_environment,
        )
        current = _read_current(experiment_dir)
        (
            accepted_frames,
            authoritative_ledger,
            authoritative_protocol,
        ) = _replay_committed_frames(
            experiment_dir, registration, current, spec,
        )
        recomputed_ledger, recomputed_protocol = _build_pre_ready_snapshot(
            spec, accepted_frames,
            formal_evaluation_count=authoritative_protocol[
                "formal_evaluation_count"
            ],
            evaluation_result=authoritative_protocol["evaluation_result"],
        )
        if authoritative_ledger != recomputed_ledger:
            raise ValueError("头部公开账本与权威K线重算结果不匹配")
        if authoritative_protocol != recomputed_protocol:
            raise ValueError("头部协议状态与权威K线重算结果不匹配")
        if authoritative_protocol["state"] in _TERMINAL_STATES:
            for symbol in spec["universe"]["core_symbols"]:
                _write_bar_csv(
                    bars_dir / f"{symbol}_1d.csv", accepted_frames[symbol],
                )
            _write_json(ledger_path, authoritative_ledger)
            if authoritative_protocol.get("evaluation_result") is not None:
                _write_json(
                    experiment_dir / "evaluation.json",
                    authoritative_protocol["evaluation_result"],
                )
            return authoritative_ledger
        if operation == "repair":
            raise ShadowRunnerLifecycleError(
                "非终态shadow缓存修复必须通过带行情的update执行"
            )
    elif operation in {"update", "repair"}:
        raise ShadowRunnerLifecycleError("影子实验尚未初始化")
    elif experiment_dir.exists():
        raise ValueError("影子状态缺少registration.json，拒绝接续")

    merged_frames: dict[str, pd.DataFrame] = {}
    for symbol in spec["universe"]["core_symbols"]:
        incoming = (
            normalized_captured[symbol]
            if normalized_captured is not None else
            _read_bar_csv(data_dir / f"{symbol}_1d.csv")
        )
        accepted = (
            accepted_frames[symbol]
            if registration is not None else None
        )
        if registration is not None:
            _verify_frozen_prefix(registration, symbol, accepted)
        try:
            rebased_incoming, _revision_metadata = rebase_adjusted_incoming(
                accepted,
                incoming,
                tolerance_ppm=spec["universe"]["revision_tolerance_ppm"],
            )
            merged = merge_accepted_bars(accepted, rebased_incoming)
        except ValueError as error:
            if operation == "legacy":
                raise
            raise ShadowRunnerDataBlockedError(str(error)) from error
        merged_frames[symbol] = merged

    if registration is not None:
        ledger = authoritative_ledger
        protocol_state = authoritative_protocol
        committed_frames = accepted_frames
    if registration is None:
        if operation == "initialize":
            try:
                boundaries = derive_shadow_boundaries(spec, merged_frames)
            except ValueError as error:
                raise ShadowRunnerDataBlockedError(str(error)) from error
            if boundaries["elapsed_common_sessions"] < 1:
                raise ShadowRunnerLifecycleError(
                    "首次建账仍无cutoff后的核心池共同交易日"
                )
        cutoff = pd.Timestamp(spec["boundaries"]["signal_cutoff_exclusive"])
        base_frames = {
            symbol: frame.loc[frame.index <= cutoff]
            for symbol, frame in merged_frames.items()
        }
        if any(frame.empty for frame in base_frames.values()):
            if operation == "legacy":
                raise ValueError("首次注册在cutoff前缺少核心标的基线行情")
            raise ShadowRunnerDataBlockedError(
                "首次注册在cutoff前缺少核心标的基线行情"
            )
        base_ledger, base_protocol = _build_pre_ready_snapshot(
            spec, base_frames,
        )
        registration = _initial_registration(
            spec, spec_hash, current_source_hashes,
            current_runtime_environment, base_frames,
        )
        _publish_initial_experiment(
            experiment_dir, registration, base_frames, base_ledger,
            base_protocol,
        )
        current = _read_current(experiment_dir)
        if all(len(base_frames[symbol]) == len(merged_frames[symbol])
               for symbol in base_frames):
            return base_ledger
        ledger = base_ledger
        protocol_state = base_protocol
        committed_frames = base_frames

    cutoff = pd.Timestamp(spec["boundaries"]["signal_cutoff_exclusive"])
    previous_session = cutoff
    if current["sequence"] > 0:
        reference = _generation_reference(current)
        generation = json.loads(
            (experiment_dir / "generations" / f"{reference}.json")
            .read_text(encoding="utf-8")
        )
        previous_session = pd.Timestamp(generation["session"])
    new_sessions = merged_frames[spec["universe"]["core_symbols"][0]].index
    new_sessions = new_sessions[new_sessions > previous_session]
    for symbol in spec["universe"]["core_symbols"][1:]:
        symbol_sessions = merged_frames[symbol].index[
            merged_frames[symbol].index > previous_session
        ]
        if not symbol_sessions.equals(new_sessions):
            message = "核心池前向交易日不一致，按DATA_BLOCKED处理"
            if operation == "legacy":
                raise ValueError(message)
            raise ShadowRunnerDataBlockedError(message)
    for session in new_sessions:
        prefix_frames = {
            symbol: frame.loc[:session]
            for symbol, frame in merged_frames.items()
        }
        prefix_ledger, prefix_protocol = _build_pre_ready_snapshot(
            spec, prefix_frames,
            formal_evaluation_count=protocol_state[
                "formal_evaluation_count"
            ],
            evaluation_result=protocol_state["evaluation_result"],
        )
        prefix_ledger, prefix_protocol = _consume_ready_once(
            spec, prefix_frames, prefix_ledger, prefix_protocol,
        )
        current = _append_generation(
            experiment_dir, current, session, prefix_frames, prefix_ledger,
            prefix_protocol,
        )
        ledger = prefix_ledger
        protocol_state = prefix_protocol
        committed_frames = prefix_frames
        if protocol_state["state"] in _TERMINAL_STATES:
            break
    if registration is not None and len(new_sessions) == 0:
        ledger = authoritative_ledger
        protocol_state = authoritative_protocol
    for symbol in spec["universe"]["core_symbols"]:
        _write_bar_csv(
            bars_dir / f"{symbol}_1d.csv", committed_frames[symbol],
        )
    _write_json(ledger_path, ledger)
    evaluation_path = experiment_dir / "evaluation.json"
    if protocol_state.get("evaluation_result") is not None:
        _write_json(
            evaluation_path,
            protocol_state["evaluation_result"],
        )
    elif evaluation_path.exists():
        evaluation_path.unlink()
        _fsync_directory(experiment_dir)
    return ledger


def run_shadow_update(
    spec_path: Path,
    data_dir: Path,
    state_root: Path,
) -> dict[str, Any]:
    """在实验级锁内验证、重放并提交一次前向影子更新。"""
    spec = load_spec(spec_path)
    spec_hash = canonical_spec_hash(spec)
    with _experiment_lock(state_root, spec["experiment_id"], spec_hash):
        return _run_shadow_update_locked(
            spec, data_dir, state_root,
        )


def run_shadow_snapshot(
    spec: dict[str, Any], frames: Mapping[str, pd.DataFrame],
    state_root: Path, *, operation: str,
) -> dict[str, Any]:
    """只消费已捕获内存行情的官方初始化/更新入口。"""
    if operation not in {"initialize", "update"}:
        raise ValueError("官方shadow操作必须是initialize或update")
    if operation == "initialize":
        try:
            boundaries = derive_shadow_boundaries(spec, dict(frames))
        except ValueError as error:
            raise ShadowRunnerDataBlockedError(str(error)) from error
        if boundaries["elapsed_common_sessions"] < 1:
            raise ShadowRunnerLifecycleError(
                "首次建账仍无cutoff后的核心池共同交易日"
            )
    spec_hash = canonical_spec_hash(spec)
    with _experiment_lock(
        Path(state_root), spec["experiment_id"], spec_hash,
        create=operation == "initialize",
    ):
        return _run_shadow_update_locked(
            spec, None, Path(state_root), captured_frames=frames,
            operation=operation,
        )


def repair_shadow_caches(
    spec: dict[str, Any], state_root: Path,
) -> dict[str, Any]:
    """不读取外部行情，只修复已经封口终态的派生缓存。"""
    spec_hash = canonical_spec_hash(spec)
    with _experiment_lock(
        Path(state_root), spec["experiment_id"], spec_hash, create=False,
    ):
        return _run_shadow_update_locked(
            spec, None, Path(state_root), operation="repair",
        )
