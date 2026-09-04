# -*- coding: utf-8 -*-
"""隔离于生产入口的 v6 前向影子账本重放器。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gcn.backtest.engine import _one_strategy
from gcn.backtest.shadow_validation import (
    canonical_bar_hash, canonical_spec_hash, load_spec, merge_accepted_bars,
    validate_spec,
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

ALGORITHM_SOURCE_PATHS = (
    "gcn/backtest/engine.py",
    "gcn/backtest/shadow_runner.py",
    "gcn/core/tdx.py",
    "gcn/recipes/gcn_main.py",
)


def algorithm_source_hashes(source_root: Path | None = None) -> dict[str, str]:
    """从实际算法文件计算哈希；调用方不能自报完整性。"""
    root = source_root or Path(__file__).resolve().parents[2]
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
    if latest < expected36:
        return {
            "state": "ACCRUING_36M",
            "locked_end": None,
            "summary": summary_at(latest),
        }

    eligible36 = forward_sessions[forward_sessions <= expected36]
    if len(eligible36) == 0:
        raise ValueError("36月端点前没有共同交易日")
    endpoint36 = eligible36[-1]
    summary36 = summary_at(endpoint36)
    if maturity_gate_passes(spec, summary36, months=36):
        return {
            "state": "OUTCOME_EMBARGO_60",
            "locked_end": endpoint36.date().isoformat(),
            "summary": summary36,
        }
    if latest < expected48:
        return {
            "state": "ACCRUING_TO_48M",
            "locked_end": None,
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
            "summary": summary48,
        }
    return {
        "state": "INCONCLUSIVE_COVERAGE_KEEP_V5",
        "locked_end": None,
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


def build_pre_ready_ledger(
    spec: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """构造READY前的白名单账本；绝不返回收益或风险指标。"""
    validate_spec(spec)
    boundaries = derive_shadow_boundaries(spec, frames)
    counts = {field: 0 for field in _COUNT_FIELDS}
    state = boundaries["state"]
    if boundaries["state"] != "INITIAL_EMBARGO":
        actual_start = boundaries["actual_accrual_start"]
        expected_start = spec["boundaries"]["expected_accrual_start"]
        if actual_start != expected_start:
            raise ValueError(
                "真实accrual start偏离预注册日期，按DATA_BLOCKED处理: "
                f"expected={expected_start}, actual={actual_start}"
            )
        def summary_at(end: pd.Timestamp) -> dict[str, Any]:
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
        if phase["locked_end"] is not None:
            label_state = label_affected_events(
                spec,
                summary["affected_events"],
                frames,
                pd.Timestamp(phase["locked_end"]),
            )
            state = label_state["state"]
            counts["pending_20_session_labels"] = label_state[
                "pending_20_session_labels"
            ]
            counts["pending_60_session_labels"] = label_state[
                "pending_60_session_labels"
            ]
    core_symbols = tuple(spec["universe"]["core_symbols"])
    ledger: dict[str, Any] = {
        "spec_hash": canonical_spec_hash(spec),
        "source_hashes": dict(sorted(algorithm_source_hashes(source_root).items())),
        "accepted_bar_hashes": {
            symbol: canonical_bar_hash(frames[symbol])
            for symbol in sorted(core_symbols)
        },
        "integrity_status": "PASS",
        "state": state,
        "elapsed_common_sessions": boundaries["elapsed_common_sessions"],
    }
    ledger.update(counts)
    allowed = set(spec["decision"]["pre_ready_visible_fields"])
    if set(ledger) != allowed:
        raise ValueError("pre_ready_visible_fields与账本字段不一致")
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


def _write_bar_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=True,
        index_label="date",
        float_format="%.17g",
    )
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_shadow_update(
    spec_path: Path,
    data_dir: Path,
    state_root: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """验证并追加本地日K，随后原子更新READY前白名单账本。"""
    spec = load_spec(spec_path)
    spec_hash = canonical_spec_hash(spec)
    experiment_dir = state_root / spec["experiment_id"] / spec_hash
    bars_dir = experiment_dir / "accepted_bars"
    registration_path = experiment_dir / "registration.json"
    ledger_path = experiment_dir / "ledger.json"
    current_source_hashes = algorithm_source_hashes(source_root)

    registration = None
    if registration_path.exists():
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        if registration.get("spec_hash") != spec_hash:
            raise ValueError("registration spec哈希不匹配")
        if registration.get("source_hashes") != current_source_hashes:
            raise ValueError("影子算法源码偏离首次注册哈希")
    elif experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise ValueError("影子状态缺少registration.json，拒绝接续")

    merged_frames: dict[str, pd.DataFrame] = {}
    changed_symbols = []
    for symbol in spec["universe"]["core_symbols"]:
        incoming = _read_bar_csv(data_dir / f"{symbol}_1d.csv")
        accepted_path = bars_dir / f"{symbol}_1d.csv"
        if registration is not None and not accepted_path.is_file():
            raise ValueError(f"影子状态缺少已接受K线: {symbol}")
        accepted = _read_bar_csv(accepted_path) if accepted_path.exists() else None
        merged = merge_accepted_bars(accepted, incoming)
        merged_frames[symbol] = merged
        if accepted is None or len(merged) != len(accepted):
            changed_symbols.append(symbol)

    ledger = build_pre_ready_ledger(
        spec, merged_frames, source_root=source_root,
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    bars_dir.mkdir(parents=True, exist_ok=True)
    for symbol in changed_symbols:
        _write_bar_csv(bars_dir / f"{symbol}_1d.csv", merged_frames[symbol])

    if registration is None:
        registration = {
            "schema_version": "gcn-shadow-registration-v1",
            "experiment_id": spec["experiment_id"],
            "spec_hash": spec_hash,
            "source_hashes": current_source_hashes,
            "initial_accepted_bar_hashes": {
                symbol: canonical_bar_hash(merged_frames[symbol])
                for symbol in sorted(merged_frames)
            },
        }
        _write_json(registration_path, registration)

    _write_json(ledger_path, ledger)
    return ledger
