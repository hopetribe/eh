# -*- coding: utf-8 -*-
"""收藏股票日 K 信号审计与稳健策略选择。

本模块只读取本地行情缓存，不刷新或覆盖 ``data/``。指标先在完整历史上计算，
再切到审计窗口，避免滚动指标预热不足。增量审计在训练段从预注册候选中选出
唯一挑战者，再用验证段执行一次晋升门槛；已知最近一年只作回看展示。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.engine import DEFAULT_SYMBOLS, _one_strategy, _perf
from gcn.recipes.gcn_main import _stage_confirmation, compute_ehopt10


DEFAULT_AUDIT_SYMBOLS = DEFAULT_SYMBOLS
ALGORITHM_SNAPSHOT_PATHS = (
    "gcn/backtest/signal_audit.py",
    "gcn/backtest/engine.py",
    "gcn/recipes/gcn_main.py",
    "gcn/core/tdx.py",
)


@dataclass(frozen=True)
class Candidate:
    entry: str
    exit: str
    trail: float | None
    max_hold: int | None
    hard_stop: float | None = None

    @property
    def name(self) -> str:
        trail = "none" if self.trail is None else f"{self.trail:.0%}"
        hold = "none" if self.max_hold is None else str(self.max_hold)
        name = f"{self.entry}|exit={self.exit}|trail={trail}|hold={hold}"
        if self.hard_stop is not None:
            hard_stop = f"{self.hard_stop:.1%}".replace(".0%", "%")
            name += f"|hard={hard_stop}"
        return name


V5_INCUMBENT = Candidate("b-confirm5-ma20+jf", "S", 0.20, None)
V6_RESEARCH_CHALLENGERS = (
    Candidate("b-confirm5-ma20+jf", "S", 0.20, None, hard_stop=0.125),
    Candidate("b-confirm5-ma20+jf", "S", 0.20, None, hard_stop=0.15),
)


def _read_ohlcv(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(data.columns):
        raise ValueError(f"{path} 缺少字段: {sorted(required - set(data.columns))}")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).set_index("date").sort_index()
    data = data.loc[~data.index.duplicated(keep="last")]
    return data[["open", "high", "low", "close", "volume"]].astype(float)


def audit_data(data_dir: Path, symbols: tuple[str, ...], start: pd.Timestamp,
               end: pd.Timestamp) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    frames: dict[str, pd.DataFrame] = {}
    rows = []
    for symbol in symbols:
        path = data_dir / f"{symbol}_1d.csv"
        if not path.exists():
            rows.append({"symbol": symbol, "status": "missing", "rows": 0,
                         "file": path.name,
                         "first": None, "last": None, "rows_5y": 0,
                         "duplicates": None, "bad_ohlcv": None, "nan_rows": None,
                         "sha256": None, "metadata_hash": "missing",
                         "metadata_sha256": None, "metadata_source": None,
                         "metadata_adjustment": None, "metadata_trusted": False})
            continue
        raw = pd.read_csv(path)
        duplicate_dates = int(raw["date"].duplicated().sum()) if "date" in raw else None
        data = _read_ohlcv(path)
        values = data[["open", "high", "low", "close", "volume"]]
        bad = ((~np.isfinite(values)).any(axis=1)
               | (values[["open", "high", "low", "close"]] <= 0).any(axis=1)
               | (values["volume"] < 0)
               | (values["high"] < values[["open", "low", "close"]].max(axis=1))
               | (values["low"] > values[["open", "high", "close"]].min(axis=1)))
        window = data.loc[(data.index >= start) & (data.index <= end)]
        status = "ok" if len(window) else "outside-window"
        starts_late = data.index.min() > start
        ends_early = data.index.max() < end
        if len(window) and starts_late and ends_early:
            status = "partial-window"
        elif len(window) and starts_late:
            status = "partial-history"
        elif len(window) and ends_early:
            status = "stale-end"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        metadata_hash = "no-meta"
        metadata_sha256 = None
        metadata_source = metadata_adjustment = None
        if meta_path.exists():
            metadata_sha256 = hashlib.sha256(meta_path.read_bytes()).hexdigest()
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                expected = metadata.get("sha256")
                metadata_source = metadata.get("source")
                metadata_adjustment = metadata.get("adjustment")
                metadata_hash = "match" if expected == digest else "mismatch"
            except (OSError, ValueError, TypeError):
                metadata_hash = "invalid-meta"
        rows.append({
            "symbol": symbol, "file": path.name, "status": status,
            "rows": int(len(data)),
            "first": data.index.min().date().isoformat(),
            "last": data.index.max().date().isoformat(), "rows_5y": int(len(window)),
            "duplicates": duplicate_dates, "bad_ohlcv": int(bad.sum()),
            "nan_rows": int(values.isna().any(axis=1).sum()),
            "sha256": digest, "metadata_hash": metadata_hash,
            "metadata_sha256": metadata_sha256,
            "metadata_source": metadata_source,
            "metadata_adjustment": metadata_adjustment,
            "metadata_trusted": metadata_hash == "match",
        })
        frames[symbol] = data
    return frames, rows


def _data_coverage(frames: dict[str, pd.DataFrame], rows: list[dict],
                   start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """用完整历史标的的日期并集识别内部缺口与共同完整截止日。"""
    row_by_symbol = {row["symbol"]: row for row in rows}
    reference_symbols = [
        symbol for symbol, frame in frames.items()
        if len(frame) and frame.index.min() <= start
    ]
    reference_dates = sorted(set().union(*(
        set(frames[symbol].loc[start:end].index) for symbol in reference_symbols
    ))) if reference_symbols else []
    calendar_gaps = {}
    for symbol in reference_symbols:
        available = set(frames[symbol].index)
        missing = [date.date().isoformat() for date in reference_dates
                   if date not in available]
        if missing:
            calendar_gaps[symbol] = missing

    common_complete_end = None
    for date in reference_dates:
        if all(date in frames[symbol].index for symbol in reference_symbols):
            common_complete_end = date.date().isoformat()
        else:
            break

    incomplete_statuses = {"missing", "outside-window", "partial-window",
                           "partial-history", "stale-end"}
    for symbol, row in row_by_symbol.items():
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            row.update({"window_first": None, "window_last": None,
                        "partial_history": False,
                        "missing_reference_sessions": [],
                        "complete_window": False})
            continue
        window = frame.loc[start:end]
        is_reference = symbol in reference_symbols
        quality_ok = all(row.get(key, 0) == 0
                         for key in ("duplicates", "bad_ohlcv", "nan_rows"))
        row.update({
            "window_first": (window.index.min().date().isoformat()
                             if len(window) else None),
            "window_last": (window.index.max().date().isoformat()
                            if len(window) else None),
            "partial_history": not is_reference,
            "missing_reference_sessions": calendar_gaps.get(symbol, []),
            "complete_window": bool(
                is_reference and symbol not in calendar_gaps
                and row.get("status") not in incomplete_statuses
                and quality_ok
            ),
        })

    reference_rows = [row_by_symbol[symbol] for symbol in reference_symbols]
    reference_status_complete = bool(reference_rows) and all(
        row.get("status") not in incomplete_statuses for row in reference_rows
    )

    return {
        "calendar_basis": "union-of-full-history-inputs",
        "reference_symbols": reference_symbols,
        "common_complete_end": common_complete_end,
        "requested_end": end.date().isoformat(),
        "requested_end_complete": bool(
            not calendar_gaps and reference_symbols and reference_status_complete
        ),
        "partial_history_symbols": [
            symbol for symbol in frames if symbol not in reference_symbols
        ],
        "complete_window_symbols": [
            row["symbol"] for row in rows if row.get("complete_window")
        ],
        "invalid_quality_symbols": [
            row["symbol"] for row in rows
            if any((row.get(key) or 0) != 0
                   for key in ("duplicates", "bad_ohlcv", "nan_rows"))
        ],
        "missing_symbols": [
            row["symbol"] for row in rows if row["status"] == "missing"
        ],
        "metadata_mismatch_symbols": [
            row["symbol"] for row in rows if row["metadata_hash"] == "mismatch"
        ],
        "metadata_untrusted_symbols": [
            row["symbol"] for row in rows if row.get("metadata_hash") != "match"
        ],
        "calendar_gaps": calendar_gaps,
    }


def _manifest_run_id(code: dict, config: dict, coverage: dict,
                     inputs: dict, environment: dict | None = None) -> str:
    payload = {"code": code, "config": config, "coverage": coverage,
               "inputs": inputs, "environment": environment or {}}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _partition_selection_universe(prepared: dict[str, dict],
                                  coverage: dict) -> tuple[dict, dict]:
    reference = set(coverage.get("complete_window_symbols",
                                 coverage["reference_symbols"]))
    partial_history = set(coverage.get(
        "partial_history_symbols",
        (symbol for symbol in prepared if symbol not in reference),
    ))
    selection = {symbol: bundle for symbol, bundle in prepared.items()
                 if symbol in reference}
    external = {symbol: bundle for symbol, bundle in prepared.items()
                if symbol in partial_history and symbol not in reference}
    return selection, external


def _snapshot_run_materials(data_dir: Path, output_dir: Path,
                            symbols: tuple[str, ...]) -> tuple[Path, dict[str, str]]:
    """在长计算开始前冻结源码与输入，并返回只读分析输入目录。"""
    repo_root = Path(__file__).resolve().parents[2]
    source_snapshot = output_dir / "source_snapshot"
    input_snapshot = output_dir / "input_snapshot"
    source_hashes = {}
    for relpath in ALGORITHM_SNAPSHOT_PATHS:
        source = repo_root / relpath
        destination = source_snapshot / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[relpath] = hashlib.sha256(destination.read_bytes()).hexdigest()
    input_snapshot.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        source = data_dir / f"{symbol}_1d.csv"
        if not source.exists():
            continue
        destination = input_snapshot / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        metadata_source = source.with_suffix(source.suffix + ".meta.json")
        if metadata_source.exists():
            metadata_destination = input_snapshot / metadata_source.name
            if metadata_source.resolve() != metadata_destination.resolve():
                shutil.copy2(metadata_source, metadata_destination)
    return input_snapshot, source_hashes


def _confirm(setup: pd.Series, frame: pd.DataFrame, ma_days: int,
             window: int) -> pd.Series:
    ma = frame["CLOSE"].rolling(ma_days, min_periods=ma_days).mean()
    confirmed, _ = _stage_confirmation(
        setup.fillna(False).astype(bool), frame["HIGH"], frame["CLOSE"], ma,
        window=window,
    )
    return confirmed


def _confirm_sell(setup: pd.Series, frame: pd.DataFrame, ma_days: int,
                  window: int) -> pd.Series:
    """S卖作为 Setup，等待跌破信号低点且收盘位于均线下方。"""
    ma = frame["CLOSE"].rolling(ma_days, min_periods=ma_days).mean()
    confirmed = pd.Series(False, index=setup.index, dtype=bool)
    pending_low = None
    age = 0
    for pos in range(len(setup)):
        if bool(setup.iloc[pos]):
            pending_low = float(frame["LOW"].iloc[pos])
            age = 0
            continue
        if pending_low is None:
            continue
        age += 1
        if (np.isfinite(pending_low) and np.isfinite(frame["CLOSE"].iloc[pos])
                and np.isfinite(ma.iloc[pos]) and frame["CLOSE"].iloc[pos] < pending_low
                and frame["CLOSE"].iloc[pos] < ma.iloc[pos]):
            confirmed.iloc[pos] = True
            pending_low = None
        elif age == window:
            pending_low = None
    return confirmed


def prepare_indicator_frames(frames: dict[str, pd.DataFrame]) -> dict[str, dict]:
    prepared = {}
    for symbol, data in frames.items():
        stable = compute_ehopt10(data, version="v4")
        experiment = compute_ehopt10(data, version="v4-exp")
        raw_b = stable["B_SIGNAL"].fillna(False).astype(bool)
        juefan = stable["ICON_JUEFAN"].fillna(False).astype(bool)
        union = raw_b | juefan
        b_confirm3 = _confirm(raw_b, stable, 20, 3)
        b_confirm5 = _confirm(raw_b, stable, 20, 5)
        entries = {
            "v4-b+jf": union,
            "v4-b-only": raw_b,
            "stage5-ma60+jf": (
                experiment["B_SIGNAL"].fillna(False).astype(bool) | juefan
            ),
            "b-confirm3-ma20": b_confirm3,
            "b-confirm5-ma20": b_confirm5,
            "b-confirm3-ma20+jf": b_confirm3 | juefan,
            "b-confirm5-ma20+jf": b_confirm5 | juefan,
            "all-confirm3-ma20": _confirm(union, stable, 20, 3),
            "all-confirm5-ma20": _confirm(union, stable, 20, 5),
            "all-confirm5-ma60": _confirm(union, stable, 60, 5),
        }
        exits = {
            "S": stable["S_SIGNAL"].fillna(False).astype(bool),
            "Scond": stable["S_CONDITION"].fillna(False).astype(bool),
            "S-or-cond": (stable["S_SIGNAL"].fillna(False).astype(bool)
                          | stable["S_CONDITION"].fillna(False).astype(bool)),
            "S-confirm3-ma20": _confirm_sell(
                stable["S_SIGNAL"].fillna(False).astype(bool), stable, 20, 3),
            "S-confirm5-ma20": _confirm_sell(
                stable["S_SIGNAL"].fillna(False).astype(bool), stable, 20, 5),
        }
        ma20_down = ((stable["CLOSE"] < stable["MID"])
                     & (stable["CLOSE"].shift(1) >= stable["MID"].shift(1)))
        ma60 = stable["CLOSE"].rolling(60, min_periods=60).mean()
        ma60_down = ((stable["CLOSE"] < ma60)
                     & (stable["CLOSE"].shift(1) >= ma60.shift(1)))
        exits.update({
            "MA20-down": ma20_down.fillna(False),
            "S-or-MA20": (exits["S"] | ma20_down).fillna(False),
            "S-or-MA60": (exits["S"] | ma60_down).fillna(False),
        })
        prepared[symbol] = {
            "ohlcv": data, "v4": stable, "v4-exp": experiment,
            "entries": entries, "exits": exits,
            "event_signals": {
                "b_setup_v4": raw_b,
                "b_confirm_v5": b_confirm5,
                "juefan": juefan,
                "s_sell": exits["S"],
            },
        }
    return prepared


def candidate_grid() -> list[Candidate]:
    entries = (
        "v4-b+jf", "v4-b-only", "stage5-ma60+jf", "b-confirm3-ma20+jf",
        "b-confirm5-ma20+jf", "all-confirm3-ma20", "all-confirm5-ma20",
        "all-confirm5-ma60",
    )
    exits = ("S", "Scond", "S-or-cond", "S-confirm3-ma20", "S-confirm5-ma20",
             "MA20-down", "S-or-MA20", "S-or-MA60")
    trails = (None, 0.15, 0.20)
    max_holds = (None, 60, 120)
    candidates = [Candidate(e, x, t, h) for e in entries for x in exits
                  for t in trails for h in max_holds]
    candidates.extend(V6_RESEARCH_CHALLENGERS)
    return candidates


def _portfolio_stats(returns: pd.Series) -> dict:
    returns = returns.fillna(0.0).astype(float)
    if returns.empty:
        return {"total": None, "cagr": None, "mdd": None, "sharpe": None,
                "calmar": None}
    equity = (1.0 + returns).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    years = len(returns) / 252.0
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy(dtype=float)])
    curve = np.r_[1.0, equity.to_numpy(dtype=float)]
    mdd = float((1.0 - curve / peak).max())
    sd = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / sd * math.sqrt(252)) if sd > 0 else np.nan
    calmar = cagr / mdd if mdd > 0 else np.nan
    return {"total": total * 100, "cagr": cagr * 100, "mdd": mdd * 100,
            "sharpe": sharpe, "calmar": calmar}


def evaluate_candidate(prepared: dict[str, dict], candidate: Candidate,
                       start: pd.Timestamp, end: pd.Timestamp,
                       cost: float = 0.001, collect_trades: bool = False) -> tuple[dict, list[dict]]:
    daily_returns = []
    symbol_totals = []
    symbol_mdds = []
    all_trades = []
    exposure = []
    for symbol, bundle in prepared.items():
        base = bundle["v4"].loc[start:end].copy()
        if len(base) < 2:
            continue
        base["_ENTRY"] = bundle["entries"][candidate.entry].reindex(base.index, fill_value=False)
        base["_EXIT"] = bundle["exits"][candidate.exit].reindex(base.index, fill_value=False)
        bt = _one_strategy(base, ["_ENTRY"], ["_EXIT"], cost,
                           candidate.max_hold, trail=candidate.trail,
                           hard_stop=candidate.hard_stop)
        equity = pd.Series(bt["equity"], index=base.index, name=symbol)
        ret = equity.pct_change()
        ret.iloc[0] = equity.iloc[0] - 1.0
        daily_returns.append(ret)
        perf = _perf(bt["equity"], bt["trades"])
        symbol_totals.append(float(perf["total"]))
        symbol_mdds.append(float(perf["mdd"]))
        exposure.append(float(np.asarray(bt["held"], dtype=bool).mean()))
        for trade in bt["trades"]:
            entry_i, exit_i = int(trade["i"]), int(trade["j"])
            row = {"symbol": symbol, "entry_date": base.index[entry_i].date().isoformat(),
                   "exit_date": (base.index[exit_i].date().isoformat()
                                 if exit_i < len(base) else base.index[-1].date().isoformat()),
                   "return_pct": float(trade["ret"]) * 100,
                   "hold_days": int(trade["hold"]),
                   "exit_reason": trade["exit_reason"]}
            all_trades.append(row)
    if not daily_returns:
        empty = _portfolio_stats(pd.Series(dtype=float))
        return {**empty, "trades": 0, "win": None, "avg_trade": None,
                "median_trade": None, "worst_trade": None, "profit_factor": None,
                "median_symbol_total": None, "positive_symbols": 0,
                "symbols": 0, "exposure": 0.0}, []
    returns = pd.concat(daily_returns, axis=1).mean(axis=1, skipna=True).fillna(0.0)
    stats = _portfolio_stats(returns)
    trade_rets = np.asarray([row["return_pct"] / 100 for row in all_trades], dtype=float)
    wins = trade_rets[trade_rets > 0]
    losses = trade_rets[trade_rets <= 0]
    stats.update({
        "trades": int(len(trade_rets)),
        "win": float((trade_rets > 0).mean() * 100) if len(trade_rets) else None,
        "avg_trade": float(trade_rets.mean() * 100) if len(trade_rets) else None,
        "median_trade": float(np.median(trade_rets) * 100) if len(trade_rets) else None,
        "worst_trade": float(trade_rets.min() * 100) if len(trade_rets) else None,
        "profit_factor": (float(wins.sum() / abs(losses.sum()))
                          if len(wins) and len(losses) else None),
        "median_symbol_total": float(np.median(symbol_totals)),
        "positive_symbols": int(sum(x > 0 for x in symbol_totals)),
        "symbols": int(len(symbol_totals)),
        "max_symbol_mdd": float(max(symbol_mdds)),
        "exposure": float(np.mean(exposure)),
    })
    return stats, all_trades if collect_trades else []


def _finite(value, fallback: float = -10.0) -> float:
    return float(value) if value is not None and np.isfinite(value) else fallback


def _harmonic_positive(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return min(left, right)
    return 2.0 * left * right / (left + right)


def selection_score(train: dict, validation: dict, baseline: dict) -> float:
    """只使用训练/验证段的保守选择分数；测试段绝不参与。"""
    min_trades = max(4, math.ceil(baseline["trades"] * 0.60))
    majority_train = math.ceil(train["symbols"] / 2)
    majority_validation = math.ceil(validation["symbols"] / 2)
    if (train["trades"] < min_trades or validation["trades"] < 3
            or train["positive_symbols"] < majority_train
            or validation["positive_symbols"] < majority_validation
            or _finite(train["median_symbol_total"]) <= 0
            or _finite(validation["median_symbol_total"]) <= 0):
        return -1_000.0
    robust_sharpe = _harmonic_positive(
        _finite(train["sharpe"]), _finite(validation["sharpe"]))
    robust_calmar = _harmonic_positive(
        _finite(train["calmar"]), _finite(validation["calmar"]))
    robust_cagr = _harmonic_positive(
        _finite(train["cagr"]), _finite(validation["cagr"])) / 100.0
    robust_median = _harmonic_positive(
        _finite(train["median_symbol_total"]),
        _finite(validation["median_symbol_total"])) / 100.0
    worst_mdd = max(_finite(train["mdd"], 100.0),
                    _finite(validation["mdd"], 100.0)) / 100.0
    return (robust_sharpe + 0.4 * robust_calmar + 0.25 * robust_cagr
            + 0.25 * robust_median - 0.15 * worst_mdd)


def evaluate_grid(prepared: dict[str, dict], splits: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
                  cost: float = 0.001) -> pd.DataFrame:
    candidates = candidate_grid()
    rows = []
    by_key: dict[tuple[str, str], dict] = {}
    for candidate in candidates:
        row = asdict(candidate) | {"name": candidate.name}
        for split_name, (start, end) in splits.items():
            stats, _ = evaluate_candidate(prepared, candidate, start, end, cost=cost)
            by_key[(candidate.name, split_name)] = stats
            for key, value in stats.items():
                row[f"{split_name}_{key}"] = value
        rows.append(row)
    baseline_name = Candidate("v4-b+jf", "S", None, None).name
    baseline_train = by_key[(baseline_name, "train")]
    result = pd.DataFrame(rows)
    result["selection_score"] = [
        selection_score(by_key[(name, "train")], by_key[(name, "validation")],
                        baseline_train)
        for name in result["name"]
    ]
    return result.sort_values(["selection_score", "validation_sharpe"], ascending=False)


def choose_recommendation(candidates: pd.DataFrame) -> tuple[str, dict]:
    """用冻结测试段做一次采纳门控，不在测试段继续调参。"""
    baseline_name = Candidate("v4-b+jf", "S", None, None).name
    baseline = candidates.loc[candidates["name"] == baseline_name].iloc[0]
    eligible = candidates[
        (candidates["selection_score"] > -999)
        & (candidates["full_cagr"] >= baseline["full_cagr"] * 0.75)
        & (candidates["full_mdd"] <= baseline["full_mdd"] * 0.50)
        & (candidates["test_cagr"] >= baseline["test_cagr"] * 0.50)
        & (candidates["test_sharpe"] >= baseline["test_sharpe"])
        & (candidates["test_mdd"] <= baseline["test_mdd"])
    ].copy()
    gates = {
        "full_cagr_retention": 0.75, "full_mdd_ratio": 0.50,
        "test_cagr_retention": 0.50, "test_sharpe_not_lower": True,
        "test_mdd_not_higher": True, "eligible_candidates": int(len(eligible)),
    }
    if eligible.empty:
        gates["fallback"] = "no candidate passed; use train-validation winner"
        return str(candidates.iloc[0]["name"]), gates
    hard_stop_complexity = (eligible["hard_stop"].notna().astype(int)
                            if "hard_stop" in eligible else 0)
    eligible["complexity"] = (eligible["max_hold"].notna().astype(int)
                              + eligible["trail"].notna().astype(int)
                              + eligible["exit"].ne("S").astype(int)
                              + hard_stop_complexity)
    eligible = eligible.sort_values(["selection_score", "complexity"],
                                    ascending=[False, True])
    return str(eligible.iloc[0]["name"]), gates


def choose_incremental_recommendation(
        candidates: pd.DataFrame, incumbent_name: str,
        challenger_names: list[str] | tuple[str, ...]) -> tuple[str, dict]:
    """训练段选唯一挑战者，再用验证段执行一次预注册晋升门槛。"""
    lookup = candidates.set_index("name", drop=False)
    missing = [name for name in (incumbent_name, *challenger_names)
               if name not in lookup.index]
    if missing:
        raise ValueError(f"增量评审缺少候选: {missing}")
    incumbent = lookup.loc[incumbent_name]
    challengers = lookup.loc[
        [name for name in challenger_names if name != incumbent_name]
    ].copy().reset_index(drop=True)
    if isinstance(challengers, pd.Series):
        challengers = challengers.to_frame().T

    gates: dict = {
        "mode": "incremental-vs-incumbent",
        "incumbent": incumbent_name,
        "challengers_evaluated": int(len(challengers)),
        "cagr_retention": 0.80,
        "sharpe_tolerance": 0.10,
        "mdd_not_higher": True,
        "risk_effect_mdd_ratio": 0.95,
        "risk_effect_worst_trade_pp": 5.0,
        "known_test_used_for_promotion": False,
        "decision_splits": ["train", "validation"],
        "observational_only": ["test", "full"],
    }
    if challengers.empty:
        gates["eligible_challengers"] = 0
        gates["decision"] = "keep-incumbent"
        return incumbent_name, gates

    challenger = challengers.sort_values(
        ["train_sharpe", "train_cagr", "train_mdd", "name"],
        ascending=[False, False, True, True],
    ).iloc[0]
    gates["challenger"] = str(challenger["name"])

    def split_gate(split: str) -> dict:
        required = ("cagr", "mdd", "sharpe", "worst_trade",
                    "trades", "symbols", "positive_symbols",
                    "median_symbol_total")
        values = {
            key: (_finite(challenger[f"{split}_{key}"], np.nan),
                  _finite(incumbent[f"{split}_{key}"], np.nan))
            for key in required
        }
        reasons = []
        if not all(np.isfinite(value) for pair in values.values() for value in pair):
            reasons.append("non-finite-metric")
            return {"passed": False, "failed_reasons": reasons}
        candidate_symbols, incumbent_symbols = values["symbols"]
        min_trades = max(4 if split == "train" else 3,
                         math.ceil(values["trades"][1] * 0.60))
        if candidate_symbols != incumbent_symbols:
            reasons.append("symbol-count-changed")
        if values["trades"][0] < min_trades:
            reasons.append("too-few-trades")
        if values["positive_symbols"][0] < math.ceil(candidate_symbols / 2):
            reasons.append("positive-symbols-below-majority")
        if values["median_symbol_total"][0] <= 0:
            reasons.append("non-positive-median-symbol")
        if values["cagr"][0] < values["cagr"][1] * 0.80:
            reasons.append("cagr-retention")
        if values["sharpe"][0] < values["sharpe"][1] - 0.10:
            reasons.append("sharpe")
        if values["mdd"][0] > values["mdd"][1]:
            reasons.append("mdd")
        risk_effect = (values["mdd"][0] <= values["mdd"][1] * 0.95
                       or values["worst_trade"][0]
                       >= values["worst_trade"][1] + 5.0)
        if not risk_effect:
            reasons.append("insufficient-risk-effect")
        return {"passed": not reasons, "failed_reasons": reasons,
                "min_trades": min_trades}

    train_gate = split_gate("train")
    validation_gate = split_gate("validation")
    gates["train"] = train_gate
    gates["validation"] = validation_gate
    promoted = bool(train_gate["passed"] and validation_gate["passed"])
    gates["eligible_challengers"] = int(promoted)
    gates["promoted"] = promoted
    if not promoted:
        gates["decision"] = "keep-incumbent"
        return incumbent_name, gates
    gates["decision"] = "challenger-passed"
    return str(challenger["name"]), gates


def robustness_tables(prepared: dict[str, dict], recommended: Candidate,
                      start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    baseline = Candidate("v4-b+jf", "S", None, None)
    named = {"baseline": baseline, "recommended": recommended}
    cost_rows = []
    for cost in (0.001, 0.0025, 0.005):
        for label, candidate in named.items():
            stats, _ = evaluate_candidate(prepared, candidate, start, end, cost=cost)
            cost_rows.append({"cost": cost, "scheme": label, **stats})

    yearly_rows = []
    fold_start = start
    while fold_start < end:
        next_fold_start = fold_start + pd.DateOffset(years=1)
        fold_end = (end if next_fold_start >= end
                    else next_fold_start - pd.Timedelta(days=1))
        for label, candidate in named.items():
            stats, _ = evaluate_candidate(prepared, candidate, fold_start, fold_end)
            yearly_rows.append({"start": fold_start.date().isoformat(),
                                "end": fold_end.date().isoformat(),
                                "scheme": label, **stats})
        fold_start = next_fold_start

    leave_one_out_rows = []
    for omitted in prepared:
        subset = {symbol: bundle for symbol, bundle in prepared.items() if symbol != omitted}
        for label, candidate in named.items():
            stats, _ = evaluate_candidate(subset, candidate, start, end)
            leave_one_out_rows.append({"omitted": omitted, "scheme": label, **stats})

    ablations = {
        "baseline": baseline,
        "confirm-only": Candidate("b-confirm5-ma20+jf", "S", None, None),
        "trail-only": Candidate("v4-b+jf", "S", 0.20, None),
        "confirm+trail": recommended,
        "defensive": Candidate("b-confirm5-ma20+jf", "S-or-MA20", 0.15, None),
    }
    ablation_rows = []
    for label, candidate in ablations.items():
        stats, _ = evaluate_candidate(prepared, candidate, start, end)
        ablation_rows.append({"component": label, "name": candidate.name, **stats})
    return {
        "cost_stress": pd.DataFrame(cost_rows),
        "yearly": pd.DataFrame(yearly_rows),
        "leave_one_out": pd.DataFrame(leave_one_out_rows),
        "ablation": pd.DataFrame(ablation_rows),
        "entry_quality": entry_quality_table(prepared, start, end),
        "b_confirmation": b_confirmation_table(prepared, start, end),
    }


def external_validation_table(
        prepared: dict[str, dict], incumbent: Candidate, challenger: Candidate,
        splits: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
        cost: float = 0.001) -> pd.DataFrame:
    rows = []
    for symbol, bundle in prepared.items():
        one_symbol = {symbol: bundle}
        for scheme, candidate in (("incumbent", incumbent),
                                  ("challenger", challenger)):
            for split in ("validation", "test", "full"):
                start, end = splits[split]
                stats, _ = evaluate_candidate(one_symbol, candidate, start, end,
                                              cost=cost)
                rows.append({"symbol": symbol, "scheme": scheme,
                             "name": candidate.name, "split": split, **stats})
    return pd.DataFrame(rows)


def _forward_path(frame: pd.DataFrame, pos: int, horizon: int,
                  outcome_end: pd.Timestamp | None = None) -> dict:
    endpoint = pos + horizon
    if (endpoint >= len(frame) or pos + 1 >= len(frame)
            or (outcome_end is not None and frame.index[endpoint] > outcome_end)):
        return {"return": np.nan, "mfe": np.nan, "mae": np.nan,
                "mfe_day": None, "mae_day": None}
    entry = float(frame["OPEN"].iloc[pos + 1])
    future = frame.iloc[pos + 1:pos + horizon + 1]
    rel_high = future["HIGH"] / entry - 1.0
    rel_low = future["LOW"] / entry - 1.0
    return {"return": float(frame["CLOSE"].iloc[pos + horizon] / entry - 1.0),
            "mfe": float(rel_high.max()), "mae": float(rel_low.min()),
            "mfe_day": int(np.argmax(rel_high.to_numpy()) + 1),
            "mae_day": int(np.argmin(rel_low.to_numpy()) + 1)}


def signal_event_table(prepared: dict[str, dict], start: pd.Timestamp,
                       end: pd.Timestamp, *, split: str | None = None) -> pd.DataFrame:
    rows = []
    for symbol, bundle in prepared.items():
        frame = bundle["v4"]
        ma60 = frame["CLOSE"].rolling(60, min_periods=60).mean()
        ma200 = frame["CLOSE"].rolling(200, min_periods=200).mean()
        event_signals = bundle.get("event_signals", {})
        false_mask = pd.Series(False, index=frame.index, dtype=bool)
        specs = (
            ("b_setup_v4", "B买Setup(v4)", "setup", "v4",
             event_signals.get("b_setup_v4", frame["B_SIGNAL"])),
            ("b_confirm_v5", "B确认(v5)", "entry", "v5",
             event_signals.get("b_confirm_v5",
                               bundle.get("entries", {}).get("b-confirm5-ma20", false_mask))),
            ("juefan", "绝反", "entry", "v5沿用v4",
             event_signals.get("juefan", frame["ICON_JUEFAN"])),
            ("s_sell", "S卖", "exit", "v5沿用v4",
             event_signals.get("s_sell", frame["S_SIGNAL"])),
        )
        for signal_key, label, role, version, source_mask in specs:
            mask = source_mask.reindex(frame.index, fill_value=False).astype(bool)
            for pos in np.flatnonzero(mask.to_numpy()):
                date = frame.index[pos]
                if date < start or date > end:
                    continue
                path20 = _forward_path(frame, pos, 20, outcome_end=end)
                outcome_complete_20d = bool(np.isfinite(path20["return"]))
                row = {"symbol": symbol, "date": date.date().isoformat(),
                       "signal": label, "signal_key": signal_key,
                       "signal_role": role, "signal_version": version,
                       "split": split, "outcome_cutoff_date": end.date().isoformat(),
                       "outcome_date_20d": (frame.index[pos + 20].date().isoformat()
                                            if outcome_complete_20d else None),
                       "outcome_complete_20d": outcome_complete_20d,
                       "close": float(frame["CLOSE"].iloc[pos]),
                       "b_score": float(frame["B_SCORE"].iloc[pos]),
                       "s_score": float(frame["S_SCORE"].iloc[pos]),
                       "regime": ("bull" if frame["CLOSE"].iloc[pos] >= ma200.iloc[pos]
                                  else "bear"),
                       "ma60_slope20_pct": (float(ma60.iloc[pos] / ma60.iloc[pos - 20] - 1) * 100
                                            if pos >= 20 and np.isfinite(ma60.iloc[pos - 20]) else np.nan)}
                for horizon in (5, 10, 20, 60):
                    path = _forward_path(frame, pos, horizon, outcome_end=end)
                    row[f"ret_{horizon}d_pct"] = path["return"] * 100
                row.update({"mfe_20d_pct": path20["mfe"] * 100,
                            "mae_20d_pct": path20["mae"] * 100,
                            "mfe_day": path20["mfe_day"], "mae_day": path20["mae_day"]})
                if signal_key == "b_setup_v4":
                    stage = bool(frame["B_STAGE_SIGNAL"].iloc[pos])
                    base_bull = (bool(frame["B_CONDITION"].iloc[pos])
                                 and frame["CLOSE"].iloc[pos] >= ma200.iloc[pos])
                    row["subtype"] = ("stage" if stage else "base-bull" if base_bull
                                      else "bear/crash-recovery")
                elif signal_key == "b_confirm_v5":
                    row["subtype"] = "confirmed"
                elif signal_key == "juefan":
                    row["subtype"] = "juefan"
                else:
                    row["subtype"] = ("bear-rally" if row["regime"] == "bear"
                                      else "major-top")
                if not row["outcome_complete_20d"]:
                    row["interference"] = pd.NA
                elif role != "exit":
                    row["interference"] = bool(path20["return"] < 0 and path20["mae"] <= -0.08)
                else:
                    row["interference"] = bool(path20["return"] > 0 and path20["mfe"] >= 0.10)
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["date", "symbol", "signal"])


def entry_quality_table(prepared: dict[str, dict], start: pd.Timestamp,
                        end: pd.Timestamp) -> pd.DataFrame:
    keys = ("v4-b+jf", "v4-b-only", "stage5-ma60+jf", "b-confirm3-ma20+jf",
            "b-confirm5-ma20+jf", "all-confirm3-ma20", "all-confirm5-ma20")
    rows = []
    for key in keys:
        paths = []
        for bundle in prepared.values():
            frame = bundle["v4"]
            for pos in np.flatnonzero(bundle["entries"][key].to_numpy()):
                if not start <= frame.index[pos] <= end:
                    continue
                path = _forward_path(frame, pos, 20, outcome_end=end)
                if np.isfinite(path["return"]):
                    paths.append((path["return"], path["mfe"], path["mae"]))
        values = np.asarray(paths, dtype=float)
        interference = ((values[:, 0] < 0) & (values[:, 2] <= -0.08))
        rows.append({"entry": key, "events": int(len(values)),
                     "win": float((values[:, 0] > 0).mean() * 100),
                     "mean_20d": float(values[:, 0].mean() * 100),
                     "median_20d": float(np.median(values[:, 0]) * 100),
                     "interference": int(interference.sum()),
                     "interference_rate": float(interference.mean() * 100),
                     "median_mae_20d": float(np.median(values[:, 2]) * 100)})
    return pd.DataFrame(rows)


def b_confirmation_table(prepared: dict[str, dict], start: pd.Timestamp,
                         end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for window in (3, 5):
        groups = {"confirmed": [], "expired": []}
        for bundle in prepared.values():
            frame = bundle["v4"]
            raw_b = frame["B_SIGNAL"].fillna(False).astype(bool)
            confirmed = _confirm(raw_b, frame, 20, window)
            for pos in np.flatnonzero(raw_b.to_numpy()):
                if not start <= frame.index[pos] <= end:
                    continue
                path = _forward_path(frame, pos, 20, outcome_end=end)
                if not np.isfinite(path["return"]):
                    continue
                hit = confirmed.iloc[pos + 1:min(pos + window + 1, len(frame))].any()
                groups["confirmed" if hit else "expired"].append(
                    (path["return"], path["mae"]))
        for outcome, paths in groups.items():
            values = np.asarray(paths, dtype=float)
            interference = ((values[:, 0] < 0) & (values[:, 1] <= -0.08))
            rows.append({"window": window, "outcome": outcome, "setups": len(values),
                         "mean_20d": float(values[:, 0].mean() * 100),
                         "median_20d": float(np.median(values[:, 0]) * 100),
                         "interference": int(interference.sum()),
                         "interference_rate": float(interference.mean() * 100)})
    return pd.DataFrame(rows)


def _future_extreme(series: pd.Series, horizon: int, fn: str) -> pd.Series:
    shifted = series.shift(-1)
    rev = shifted.iloc[::-1]
    rolled = (rev.rolling(horizon, min_periods=horizon).max()
              if fn == "max" else rev.rolling(horizon, min_periods=horizon).min())
    return rolled.iloc[::-1]


def _non_overlapping_positions(positions: np.ndarray, strength: pd.Series,
                               gap: int = 20) -> list[int]:
    ranked = sorted((int(pos) for pos in positions),
                    key=lambda pos: float(strength.iloc[pos]), reverse=True)
    kept = []
    for pos in ranked:
        if all(abs(pos - other) > gap for other in kept):
            kept.append(pos)
    return sorted(kept)


def _nearest_signal(frame: pd.DataFrame, mask: pd.Series, pos: int,
                    before: int = 3, after: int = 5) -> tuple[bool, str | None, int | None]:
    lo, hi = max(0, pos - before), min(len(frame), pos + after + 1)
    hits = np.flatnonzero(mask.iloc[lo:hi].fillna(False).to_numpy()) + lo
    if not len(hits):
        return False, None, None
    nearest = int(hits[np.argmin(np.abs(hits - pos))])
    return True, frame.index[nearest].date().isoformat(), nearest - pos


def missed_turn_table(prepared: dict[str, dict], start: pd.Timestamp, end: pd.Timestamp,
                      buy_gain: float = 0.15, sell_drop: float = 0.12,
                      candidate: Candidate | None = None) -> pd.DataFrame:
    """用事后转折标签审计覆盖率；这些标签不得直接作为实盘信号。"""
    rows = []
    for symbol, bundle in prepared.items():
        # 所有事后标签与邻近信号都先截到审计末日，保证追加未来数据不改写历史结果。
        frame = bundle["v4"].loc[:end]
        low_center = frame["LOW"].rolling(11, center=True, min_periods=11).min()
        high_center = frame["HIGH"].rolling(11, center=True, min_periods=11).max()
        next_open = frame["OPEN"].shift(-1)
        future_high = _future_extreme(frame["HIGH"], 20, "max")
        future_low = _future_extreme(frame["LOW"], 20, "min")
        gain = future_high / next_open - 1.0
        drop = 1.0 - future_low / next_open
        in_window = (frame.index >= start) & (frame.index <= end)
        buy_pos = np.flatnonzero(((frame["LOW"] <= low_center) & (gain >= buy_gain)
                                  & in_window).fillna(False).to_numpy())
        sell_pos = np.flatnonzero(((frame["HIGH"] >= high_center) & (drop >= sell_drop)
                                   & in_window).fillna(False).to_numpy())
        buy_pos = _non_overlapping_positions(buy_pos, gain, gap=20)
        sell_pos = _non_overlapping_positions(sell_pos, drop, gap=20)
        if candidate is None:
            buy_signal = (frame["B_SIGNAL"].fillna(False).astype(bool)
                          | frame["ICON_JUEFAN"].fillna(False).astype(bool))
            sell_signal = frame["S_SIGNAL"].fillna(False).astype(bool)
            held = pd.Series(False, index=frame.index, dtype=bool)
        else:
            entry_setup = bundle["entries"][candidate.entry].reindex(
                frame.index, fill_value=False
            )
            exit_setup = bundle["exits"][candidate.exit].reindex(
                frame.index, fill_value=False
            )
            base = frame.loc[start:end].copy()
            base["_ENTRY"] = entry_setup.reindex(base.index, fill_value=False)
            base["_EXIT"] = exit_setup.reindex(base.index, fill_value=False)
            strategy = _one_strategy(
                base, ["_ENTRY"], ["_EXIT"], 0.001, candidate.max_hold,
                trail=candidate.trail, hard_stop=candidate.hard_stop,
            )
            held = pd.Series(False, index=frame.index, dtype=bool)
            held.loc[base.index] = strategy["held"]
            buy_signal = pd.Series(False, index=frame.index, dtype=bool)
            sell_signal = pd.Series(False, index=frame.index, dtype=bool)
            for trade in strategy["trades"]:
                entry_i, exit_i = int(trade["i"]), int(trade["j"])
                if 0 < entry_i < len(base):
                    buy_signal.loc[base.index[entry_i - 1]] = True
                if 0 < exit_i < len(base):
                    sell_signal.loc[base.index[exit_i - 1]] = True
        for kind, positions, strength, mask in (
            ("buy", buy_pos, gain, buy_signal), ("sell", sell_pos, drop, sell_signal)
        ):
            for pos in positions:
                covered, signal_date, offset = _nearest_signal(frame, mask, pos)
                if candidate is None:
                    actionable = True
                else:
                    actionable = not bool(held.iloc[pos]) if kind == "buy" else bool(held.iloc[pos])
                rows.append({
                    "symbol": symbol, "kind": kind,
                    "date": frame.index[pos].date().isoformat(),
                    "opportunity_pct": float(strength.iloc[pos] * 100),
                    "actionable": actionable,
                    "covered": covered, "missed": not covered,
                    "nearest_signal_date": signal_date,
                    "signal_offset_days": offset,
                })
    return pd.DataFrame(rows).sort_values(["kind", "missed", "opportunity_pct"],
                                          ascending=[True, False, False])


def _fmt(value, digits: int = 2) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _audit_window_line(split_dates: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> str:
    cutoff = split_dates["full"][1].date()
    return f"- 数据统一截止到{cutoff}；输入文件SHA-256记录在data_audit.json。"


def _turn_coverage(turns: pd.DataFrame, actionable_only: bool = False) -> dict:
    selected = turns
    if actionable_only:
        selected = turns[turns["actionable"].fillna(False).astype(bool)]
    total = int(len(selected))
    covered = int(selected["covered"].fillna(False).astype(bool).sum())
    return {
        "total": total,
        "covered": covered,
        "missed": total - covered,
        "rate": covered / total * 100 if total else 0.0,
    }


def _strategy_line(label: str, row: pd.Series, prefix: str) -> str:
    return (f"| {label} | {_fmt(row[f'{prefix}_cagr'])}% | {_fmt(row[f'{prefix}_mdd'])}% | "
            f"{_fmt(row[f'{prefix}_sharpe'])} | {int(row[f'{prefix}_trades'])} | "
            f"{_fmt(row[f'{prefix}_win'], 1)}% | {_fmt(row[f'{prefix}_median_trade'])}% | "
            f"{_fmt(row[f'{prefix}_worst_trade'])}% |")


def _candidate_rule_lines(candidate: Candidate) -> list[str]:
    lines = []
    if candidate.entry == "b-confirm5-ma20+jf":
        lines.extend([
            "- B买只作为 Setup；随后5根K线内首次收盘突破Setup当日最高价且站上MA20，才生成可执行买点。",
            "- 绝反保持即时入场；它本身已含低位、反包和量能确认。",
        ])
    else:
        lines.append(f"- 入场口径：`{candidate.entry}`。")
    lines.append(f"- 退出信号：`{candidate.exit}`。")
    if candidate.trail is not None:
        lines.append(f"- 从入场后最高收盘回撤{candidate.trail:.0%}时收盘确认，下一根K线开盘退出。")
    if candidate.hard_stop is not None:
        lines.append(
            f"- 相对实际入场开盘价下跌{candidate.hard_stop:.1%}时收盘确认，下一根K线开盘退出；"
            "跳空可能穿透名义止损线。"
        )
    if candidate.max_hold is None:
        lines.append("- 不设置最长持有期。")
    else:
        lines.append(f"- 最长持有{candidate.max_hold}根K线。")
    return lines


def build_report(data_rows: list[dict], events: pd.DataFrame, missed: pd.DataFrame,
                 candidates: pd.DataFrame, selection_winner_name: str,
                 recommended_name: str,
                 split_dates: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
                 robustness: dict[str, pd.DataFrame], gates: dict,
                 split_events: pd.DataFrame | None = None,
                 coverage: dict | None = None) -> str:
    baseline_name = Candidate("v4-b+jf", "S", None, None).name
    stage_name = Candidate("stage5-ma60+jf", "S", None, None).name
    lookup = candidates.set_index("name")
    selection_winner = lookup.loc[selection_winner_name]
    recommended = lookup.loc[recommended_name]
    baseline = lookup.loc[baseline_name]
    stage = lookup.loc[stage_name]
    incremental = gates.get("mode") == "incremental-vs-incumbent"
    recommended_cfg = next(candidate for candidate in candidate_grid()
                           if candidate.name == recommended_name)
    coverage_lines = [_audit_window_line(split_dates)]
    if coverage is not None:
        selection_symbols = coverage.get("selection_symbols", coverage["reference_symbols"])
        external_symbols = coverage.get("external_validation_symbols", [])
        diagnostic_symbols = coverage.get("signal_diagnostic_symbols", selection_symbols)
        coverage_lines.append(
            "- 策略选型核心池（完整5年）：" + "、".join(selection_symbols) + "。"
        )
        coverage_lines.append(
            "- 信号/漏点诊断池：" + "、".join(diagnostic_symbols) + "；"
            + ("部分历史标的 " + "、".join(external_symbols)
               + " 只作可用期诊断与外部验证，不参与选型。"
               if external_symbols else "无部分历史外部标的。")
        )
        coverage_lines.append(
            f"- 完整历史参考池共同完整截止日：{coverage['common_complete_end'] or '--'}。"
        )
        if not coverage["requested_end_complete"]:
            coverage_lines.append(
                f"- ⚠ 请求截止日 {coverage['requested_end']} 不是共同完整截面；"
                f"内部缺口：{json.dumps(coverage['calendar_gaps'], ensure_ascii=False)}。"
            )
        if coverage["partial_history_symbols"]:
            coverage_lines.append(
                "- 部分历史标的（不参与共同日历判定）："
                + "、".join(coverage["partial_history_symbols"]) + "。"
            )
        if coverage.get("metadata_untrusted_symbols"):
            coverage_lines.append(
                "- ⚠ **来源元数据未通过哈希校验：**"
                + "、".join(coverage["metadata_untrusted_symbols"])
                + "。本轮仅确认OHLC结构与冻结文件哈希；不能据此证明上游来源/复权元数据可信。"
            )
    else:
        selection_symbols = external_symbols = diagnostic_symbols = []
    lines = [
        "# 收藏股票近5年日K信号审计", "",
        "> 研究用途，不构成投资建议。策略回测按信号日收盘确认、下一交易日开盘成交、单边成本0.1%；"
        "事件研究与事后机会均为毛收益，不扣交易成本。", "",
        "## 数据与验证设计", "",
        f"- 审计窗口：{split_dates['full'][0].date()} 至 {split_dates['full'][1].date()}。",
        f"- 训练：{split_dates['train'][0].date()}～{split_dates['train'][1].date()}；"
        f"验证：{split_dates['validation'][0].date()}～{split_dates['validation'][1].date()}；"
        f"已知回看期：{split_dates['test'][0].date()}～{split_dates['test'][1].date()}。",
        "- 指标先用完整历史预热，再切片；训练段只从预注册候选选出一个挑战者，验证段只执行一次晋升门槛。"
        "最近一年已在既往研究中查看，本轮仅作回看展示，不参与晋升。组合指标按可用标的等权日收益计算。",
        *coverage_lines,
        "- 漏点是事后诊断标签，不是可直接交易的未来函数：买点=±5日局部低点且未来20日最大涨幅≥15%；卖点=±5日局部高点且未来20日最大跌幅≥12%；相邻20日仅保留最强转折。",
        "- 干扰买入=未来20日收盘收益<0且最大不利波动≤-8%；干扰卖出=未来20日收盘仍上涨且最大上涨≥10%。", "",
        "### 数据审计", "", "| 标的 | 状态 | 全部行数 | 5年行数 | 起始 | 结束 | 异常OHLC | 元数据哈希 |",
        "|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in data_rows:
        lines.append(f"| {row['symbol']} | {row['status']} | {row['rows']} | {row['rows_5y']} | "
                     f"{row['first'] or '--'} | {row['last'] or '--'} | {row['bad_ohlcv'] if row['bad_ohlcv'] is not None else '--'} | "
                     f"{row['metadata_hash']} |")

    complete_events = events[events["outcome_complete_20d"].fillna(False).astype(bool)]
    lines += ["", "## 实际信号质量", "",
              "B买Setup(v4)仅是原始观察信号；B确认(v5)与绝反才是v5可执行入场。"
              "下表只统计未来20根K完整落在审计窗口内的事件。"
              + ("诊断池为" + "、".join(diagnostic_symbols) + "。"
                 if diagnostic_symbols else ""),
              "事件收益、最高幅度(MFE)和最低幅度(MAE)均从次日开盘起算，未扣成本。", "",
              "| 信号 | 数量 | 干扰数 | 干扰率 | 20日均值 | 20日中位 | 20日胜率 | 未来20日最高幅度中位 | 未来20日最低幅度中位 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for signal, group in complete_events.groupby("signal", sort=False):
        interference = int(group["interference"].sum())
        if group["signal_role"].eq("exit").all():
            win = float((group["ret_20d_pct"] < 0).mean() * 100)
        else:
            win = float((group["ret_20d_pct"] > 0).mean() * 100)
        lines.append(f"| {signal} | {len(group)} | {interference} | {interference / len(group) * 100:.1f}% | "
                     f"{group['ret_20d_pct'].mean():+.2f}% | {group['ret_20d_pct'].median():+.2f}% | "
                     f"{win:.1f}% | {group['mfe_20d_pct'].median():+.2f}% | "
                     f"{group['mae_20d_pct'].median():+.2f}% |")

    if split_events is not None and not split_events.empty:
        complete_split_events = split_events[
            split_events["outcome_complete_20d"].fillna(False).astype(bool)
        ]
        lines += ["", "### 严格分段事件质量", "",
                  "各段未来结果在本段末日截断，不读取下一段行情。", "",
                  "| 分段 | 信号 | 可评估数 | 20日胜率 | 20日均值 | 干扰率 |",
                  "|---|---|---:|---:|---:|---:|"]
        for (split, signal), group in complete_split_events.groupby(
                ["split", "signal"], sort=False):
            is_exit = group["signal_role"].eq("exit").all()
            win = ((group["ret_20d_pct"] < 0).mean() if is_exit
                   else (group["ret_20d_pct"] > 0).mean()) * 100
            display_split = "已知回看期" if split == "test" else split
            lines.append(f"| {display_split} | {signal} | {len(group)} | {win:.1f}% | "
                         f"{group['ret_20d_pct'].mean():+.2f}% | "
                         f"{group['interference'].mean() * 100:.1f}% |")

    lines += ["", "### B买Setup后续结果（从Setup日计）", "",
              "以下B确认与可执行入场两表只使用完整5年核心10只选型池（不含TEM）。"
              "此表用于判断Setup是否值得等待，不代表实际确认入场后的交易收益。", "",
              "| 窗口 | 结果 | Setup数 | 20日均值 | 20日中位 | 干扰数/率 |",
              "|---:|---|---:|---:|---:|---:|"]
    for row in robustness["b_confirmation"].itertuples():
        lines.append(f"| {row.window}日 | {row.outcome} | {row.setups} | "
                     f"{row.mean_20d:+.2f}% | {row.median_20d:+.2f}% | "
                     f"{row.interference}/{row.interference_rate:.1f}% |")
    lines += ["", "| 可执行入场口径 | 事件数 | 胜率 | 20日均值 | 20日中位 | 干扰率 | 20日MAE中位 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    entry_quality = robustness["entry_quality"].set_index("entry")
    for key in ("v4-b+jf", "stage5-ma60+jf", "b-confirm3-ma20+jf",
                "b-confirm5-ma20+jf"):
        row = entry_quality.loc[key]
        lines.append(f"| {key} | {int(row.events)} | {row.win:.1f}% | "
                     f"{row.mean_20d:+.2f}% | {row.median_20d:+.2f}% | "
                     f"{row.interference_rate:.1f}% | {row.median_mae_20d:+.2f}% |")

    lines += ["", "### 各标的干扰信号", "",
              "| 标的 | B Setup | Setup干扰 | B确认 | 确认干扰 | 绝反 | 绝反干扰 | S卖 | S干扰 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for symbol, group in events.groupby("symbol"):
        values = []
        for signal_key in ("b_setup_v4", "b_confirm_v5", "juefan", "s_sell"):
            sg = group[(group["signal_key"] == signal_key)
                       & group["outcome_complete_20d"]]
            values.extend([len(sg), int(sg["interference"].sum())])
        lines.append(f"| {symbol} | " + " | ".join(map(str, values)) + " |")

    lines += ["", "### 影响最大的干扰信号", "",
              "| 信号 | 标的 | 日期 | 20日收益 | 20日MFE | 20日MAE |",
              "|---|---|---|---:|---:|---:|"]
    noise = events["interference"].fillna(False).astype(bool)
    buy_noise = (events[noise & events["signal_role"].ne("exit")]
                 .sort_values("ret_20d_pct").head(12))
    sell_noise = (events[noise & events["signal_role"].eq("exit")]
                  .sort_values("mfe_20d_pct", ascending=False).head(12))
    for row in pd.concat([buy_noise, sell_noise]).itertuples():
        lines.append(f"| {row.signal} | {row.symbol} | {row.date} | "
                     f"{row.ret_20d_pct:+.2f}% | {row.mfe_20d_pct:+.2f}% | "
                     f"{row.mae_20d_pct:+.2f}% |")

    lines += ["", "## 错过的转折", "",
              "以下可行动口径以当前 v5 推荐策略的持仓状态为准：空仓时的买点、持仓时的卖点才计入。"
              "机会幅度为事后毛路径，不扣成本。"
              + ("诊断池为" + "、".join(diagnostic_symbols) + "。"
                 if diagnostic_symbols else ""), "",
              "| 类型 | 事后转折数 | 可行动 | 可行动已覆盖 | 可行动错过 | 可行动覆盖率 |",
              "|---|---:|---:|---:|---:|---:|"]
    for kind, label in (("buy", "买点"), ("sell", "卖点")):
        group = missed[missed["kind"] == kind]
        actionable = _turn_coverage(group, actionable_only=True)
        lines.append(f"| {label} | {len(group)} | {actionable['total']} | "
                     f"{actionable['covered']} | {actionable['missed']} | "
                     f"{actionable['rate']:.1f}% |")
    lines += ["", "### 各标的漏点", "",
              "| 标的 | 可行动买点/错过 | 买点最大机会 | 可行动卖点/错过 | 卖点最大机会 |",
              "|---|---:|---:|---:|---:|"]
    for symbol in sorted(missed["symbol"].unique()):
        buy = missed[(missed["symbol"] == symbol) & (missed["kind"] == "buy")
                     & missed["actionable"]]
        sell = missed[(missed["symbol"] == symbol) & (missed["kind"] == "sell")
                      & missed["actionable"]]
        lines.append(f"| {symbol} | {len(buy)}/{int(buy['missed'].sum())} | "
                     f"{_fmt(buy['opportunity_pct'].max())}% | {len(sell)}/{int(sell['missed'].sum())} | "
                     f"{_fmt(sell['opportunity_pct'].max())}% |")
    lines += ["", "### 最大的漏点（每类前15）", "",
              "| 类型 | 标的 | 日期 | 未来20日机会 |",
              "|---|---|---|---:|"]
    top_missed = (missed[missed["missed"] & missed["actionable"]]
                  .sort_values("opportunity_pct", ascending=False)
                  .groupby("kind", group_keys=False).head(15))
    for row in top_missed.itertuples():
        lines.append(f"| {'买点' if row.kind == 'buy' else '卖点'} | {row.symbol} | {row.date} | {row.opportunity_pct:.2f}% |")

    if incremental:
        challenger = gates.get("challenger", selection_winner_name)
        train_gate = gates.get("train", {})
        validation_gate = gates.get("validation", {})
        decision_text = (
            f"训练段从{gates['challengers_evaluated']}个预注册hard-stop候选中选出 `{challenger}`；"
            f"训练门槛={'通过' if train_gate.get('passed') else '未通过'}，"
            f"验证门槛={'通过' if validation_gate.get('passed') else '未通过'}。"
            + ("挑战者可进入版本固化。" if gates.get("promoted")
               else "未同时通过，正式版本保持v5。")
        )
        cagr_floor = recommended["validation_cagr"] * gates["cagr_retention"]
        sharpe_floor = (recommended["validation_sharpe"]
                        - gates["sharpe_tolerance"])
        worst_trade_gain = (selection_winner["validation_worst_trade"]
                            - recommended["validation_worst_trade"])
        decision_text += (
            f" 验证实值/门槛：CAGR {_fmt(selection_winner['validation_cagr'])}%"
            f" < {_fmt(cagr_floor)}%，Sharpe {_fmt(selection_winner['validation_sharpe'])}"
            f" < {_fmt(sharpe_floor)}，MDD {_fmt(selection_winner['validation_mdd'])}%"
            f" > {_fmt(recommended['validation_mdd'])}%。"
            f"风险效果项已通过：验证最差单笔由"
            f"{_fmt(recommended['validation_worst_trade'])}%改善至"
            f"{_fmt(selection_winner['validation_worst_trade'])}%"
            f"（+{_fmt(worst_trade_gain)}个百分点）。"
        )
        failed = validation_gate.get("failed_reasons", [])
        if failed:
            reason_labels = {
                "non-finite-metric": "指标缺失",
                "symbol-count-changed": "标的数变化",
                "too-few-trades": "交易样本不足",
                "positive-symbols-below-majority": "正收益标的不足半数",
                "non-positive-median-symbol": "标的收益中位数非正",
                "cagr-retention": "CAGR保留不足",
                "sharpe": "Sharpe不足",
                "mdd": "最大回撤变差",
                "insufficient-risk-effect": "风险改善幅度不足",
            }
            decision_text += " 验证未通过项：" + "、".join(
                reason_labels.get(reason, reason) for reason in failed
            ) + "。"
    else:
        decision_text = (
            f"最终采纳门槛：全样本保留≥{gates['full_cagr_retention']:.0%}基线CAGR、"
            f"MDD≤{gates['full_mdd_ratio']:.0%}基线；测试保留≥{gates['test_cagr_retention']:.0%}"
            f"基线CAGR，且Sharpe不低、MDD不高。通过候选数={gates['eligible_candidates']}。"
        )
    lines += ["", "## 方案比较", "",
              "`已知回看期` 的结果不参与本轮候选选择或晋升。"
              + ("所有候选、交易与组合指标只使用完整5年核心池："
                 + "、".join(selection_symbols) + "。" if selection_symbols else ""), "",
              "hard-stop挑战者以实际入场开盘价为锚；收盘触及阈值后于下一根K线开盘退出，"
              "跳空可能穿透名义止损线；若同日也有S卖，按S卖原因优先。", "",
              "| 已知回看期方案 | CAGR | 最大回撤 | Sharpe | 交易数 | 胜率 | 中位单笔 | 最差单笔 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|",
              _strategy_line("v4原始 B买+绝反→S卖", baseline, "test"),
              _strategy_line("仅阶段B确认", stage, "test"),
              _strategy_line(f"本轮定向挑战者：{selection_winner_name}", selection_winner, "test"),
              _strategy_line(f"当前正式方案：{recommended_name}", recommended, "test"), "",
              decision_text, "",
              "### 训练/验证稳健性", "",
              "| 方案 | 分段 | CAGR | 最大回撤 | Sharpe | 交易数 | 正收益标的 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    comparison_rows = [("v4原始基线", baseline), ("当前正式方案", recommended)]
    if incremental and selection_winner_name != recommended_name:
        comparison_rows.append(("本轮定向挑战者", selection_winner))
    for label, row in comparison_rows:
        for split in ("train", "validation", "full"):
            lines.append(f"| {label} | {split} | {_fmt(row[f'{split}_cagr'])}% | "
                         f"{_fmt(row[f'{split}_mdd'])}% | {_fmt(row[f'{split}_sharpe'])} | "
                         f"{int(row[f'{split}_trades'])} | {int(row[f'{split}_positive_symbols'])}/{int(row[f'{split}_symbols'])} |")

    lines += ["", "### 历史网格参考（不参与本轮增量晋升）", "",
              "| 排名 | 方案 | 选择分数 | 验证Sharpe | 验证回撤 | 已知回看Sharpe | 已知回看回撤 |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(candidates.head(10).itertuples(), 1):
        lines.append(f"| {rank} | {row.name} | {row.selection_score:.3f} | "
                     f"{_fmt(row.validation_sharpe)} | {_fmt(row.validation_mdd)}% | "
                     f"{_fmt(row.test_sharpe)} | {_fmt(row.test_mdd)}% |")

    lines += ["", "### 推荐参数邻域", "",
              "| 方案 | 训练Sharpe/MDD | 验证Sharpe/MDD | 已知回看Sharpe/MDD | 全样本Sharpe/MDD |",
              "|---|---:|---:|---:|---:|"]
    nearby = (
        Candidate("b-confirm3-ma20+jf", "S", 0.20, None).name,
        Candidate("b-confirm5-ma20+jf", "S", 0.15, None).name,
        Candidate("b-confirm5-ma20+jf", "S", 0.20, None).name,
        Candidate("b-confirm5-ma20+jf", "S", 0.20, 60).name,
        V6_RESEARCH_CHALLENGERS[0].name,
        V6_RESEARCH_CHALLENGERS[1].name,
    )
    for name in nearby:
        row = lookup.loc[name]
        lines.append(f"| {name} | {_fmt(row.train_sharpe)}/{_fmt(row.train_mdd)}% | "
                     f"{_fmt(row.validation_sharpe)}/{_fmt(row.validation_mdd)}% | "
                     f"{_fmt(row.test_sharpe)}/{_fmt(row.test_mdd)}% | "
                     f"{_fmt(row.full_sharpe)}/{_fmt(row.full_mdd)}% |")

    lines += ["", "## 稳健性验证（完整5年核心池）", "", "### 消融（全5年）", "",
              "| 组件 | CAGR | MDD | Sharpe | 交易 | 最差单笔 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in robustness["ablation"].itertuples():
        lines.append(f"| {row.component} | {_fmt(row.cagr)}% | {_fmt(row.mdd)}% | "
                     f"{_fmt(row.sharpe)} | {row.trades} | {_fmt(row.worst_trade)}% |")

    lines += ["", "### 成本压力（全5年）", "",
              "| 单边成本 | 方案 | CAGR | MDD | Sharpe |",
              "|---:|---|---:|---:|---:|"]
    for row in robustness["cost_stress"].itertuples():
        lines.append(f"| {row.cost:.2%} | {row.scheme} | {_fmt(row.cagr)}% | "
                     f"{_fmt(row.mdd)}% | {_fmt(row.sharpe)} |")

    lines += ["", "### 逐年区间", "",
              "| 区间 | 方案 | CAGR | MDD | Sharpe | 正收益标的 |",
              "|---|---|---:|---:|---:|---:|"]
    for row in robustness["yearly"].itertuples():
        lines.append(f"| {row.start}～{row.end} | {row.scheme} | {_fmt(row.cagr)}% | "
                     f"{_fmt(row.mdd)}% | {_fmt(row.sharpe)} | "
                     f"{row.positive_symbols}/{row.symbols} |")

    external = robustness.get("external_validation", pd.DataFrame())
    if not external.empty:
        lines += ["", "### 部分历史标的外部验证（不参与晋升）", "",
                  "这些结果只覆盖该标的实际可用历史，不能与完整5年核心池作等长比较。", "",
                  "| 标的 | 方案 | 分段 | CAGR | MDD | Sharpe | 交易数 |",
                  "|---|---|---|---:|---:|---:|---:|"]
        for row in external.itertuples():
            scheme = "v5正式" if row.scheme == "incumbent" else "训练首选挑战者"
            split = "已知回看期" if row.split == "test" else row.split
            lines.append(f"| {row.symbol} | {scheme} | {split} | {_fmt(row.cagr)}% | "
                         f"{_fmt(row.mdd)}% | {_fmt(row.sharpe)} | {int(row.trades)} |")

    loo = robustness["leave_one_out"].pivot(index="omitted", columns="scheme",
                                             values=["cagr", "sharpe", "mdd"])
    lines += ["", "### 逐标的留一（全5年）", "",
              "| 剔除标的 | 基线CAGR | 推荐CAGR | 基线Sharpe | 推荐Sharpe | 基线MDD | 推荐MDD |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for symbol in loo.index:
        lines.append(f"| {symbol} | {_fmt(loo.loc[symbol, ('cagr', 'baseline')])}% | "
                     f"{_fmt(loo.loc[symbol, ('cagr', 'recommended')])}% | "
                     f"{_fmt(loo.loc[symbol, ('sharpe', 'baseline')])} | "
                     f"{_fmt(loo.loc[symbol, ('sharpe', 'recommended')])} | "
                     f"{_fmt(loo.loc[symbol, ('mdd', 'baseline')])}% | "
                     f"{_fmt(loo.loc[symbol, ('mdd', 'recommended')])}% |")
    if "AAOI" in loo.index:
        aaoi_without = loo.loc["AAOI", ("cagr", "recommended")]
        lines += ["",
                  f"集中度提示：推荐方案全核心池CAGR为{_fmt(recommended.full_cagr)}%；"
                  f"剔除AAOI后为{_fmt(aaoi_without)}%，变化"
                  f"{_fmt(aaoi_without - recommended.full_cagr)}个百分点。"]

    b_confirm = complete_events[complete_events["signal_key"] == "b_confirm_v5"]
    juefan = complete_events[complete_events["signal_key"] == "juefan"]
    s_sell = complete_events[complete_events["signal_key"] == "s_sell"]
    buy_turns = _turn_coverage(missed[missed["kind"] == "buy"], actionable_only=True)
    sell_turns = _turn_coverage(missed[missed["kind"] == "sell"], actionable_only=True)
    version_conclusion = (
        "**挑战者通过预注册门槛，可进入下一版本固化。**"
        if gates.get("promoted") else "**保持v5，不创建v6。**"
    )
    lines += ["", "## 本轮结论与优化顺序", "",
              f"1. {version_conclusion} 训练首选hard-stop挑战者在验证段CAGR/Sharpe/MDD为 "
              f"{_fmt(selection_winner.validation_cagr)}% / {_fmt(selection_winner.validation_sharpe)} / "
              f"{_fmt(selection_winner.validation_mdd)}%，未同时超过v5的 "
              f"{_fmt(recommended.validation_cagr)}% / {_fmt(recommended.validation_sharpe)} / "
              f"{_fmt(recommended.validation_mdd)}%。",
              f"2. **B确认仍是首要优化对象。** 11只诊断池的严格完整20日样本中，B确认{len(b_confirm)}次、"
              f"胜率{(b_confirm['ret_20d_pct'] > 0).mean() * 100:.1f}%、"
              f"干扰率{b_confirm['interference'].mean() * 100:.1f}%；不能仅凭当前数据继续调阈值。",
              f"3. **绝反保持即时。** 绝反{len(juefan)}次、"
              f"胜率{(juefan['ret_20d_pct'] > 0).mean() * 100:.1f}%、"
              f"干扰率{juefan['interference'].mean() * 100:.1f}%，继续延迟缺乏证据。",
              f"4. **S卖暂不加确认。** S卖{len(s_sell)}次、20日下跌命中率"
              f"{(s_sell['ret_20d_pct'] < 0).mean() * 100:.1f}%、"
              f"干扰率{s_sell['interference'].mean() * 100:.1f}%；下一轮应优先研究盈利保护，"
              "而不是放宽S卖。",
              f"5. **漏点只作诊断。** 可行动买点覆盖{buy_turns['covered']}/{buy_turns['total']}"
              f"（{buy_turns['rate']:.1f}%），卖点覆盖{sell_turns['covered']}/{sell_turns['total']}"
              f"（{sell_turns['rate']:.1f}%）；不得把未来20日定义的漏点直接写成实盘规则。",
              "6. **下一轮验证方案。** 冻结现有v5与本轮候选定义，等待新增时间外数据，"
              "再用预注册的盈利保护候选、逐标的留一和成本压力复核；没有新样本前不继续扫参数。"]

    lines += ["", "## 当前正式方案", "", f"`{recommended_name}`", ""]
    lines += _candidate_rule_lines(recommended_cfg)
    lines += ["",
              "该结论仅在当前收藏池、当前成本与本地冻结数据上成立；事后漏点只用于审计，不能直接成为实盘信号。详细逐笔数据见同目录 CSV。",
              ""]
    return "\n".join(lines)


def _write_manifest(output_dir: Path, data_dir: Path, data_rows: list[dict],
                    symbols: tuple[str, ...], start: pd.Timestamp,
                    end: pd.Timestamp, cost: float,
                    splits: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
                    coverage: dict, recommended_name: str, gates: dict,
                    candidate_count: int, snapshot_inputs: bool = False,
                    frozen_algorithm_files: dict[str, str] | None = None) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    algorithm_snapshots = {
        relpath: str((Path("source_snapshot") / relpath))
        for relpath in ALGORITHM_SNAPSHOT_PATHS
    } if snapshot_inputs else {}
    if snapshot_inputs:
        algorithm_files = {
            relpath: hashlib.sha256(
                (output_dir / algorithm_snapshots[relpath]).read_bytes()
            ).hexdigest()
            for relpath in ALGORITHM_SNAPSHOT_PATHS
        }
        if frozen_algorithm_files != algorithm_files:
            raise RuntimeError("算法源码快照在审计运行期间发生变化")
    else:
        algorithm_files = {
            relpath: hashlib.sha256((repo_root / relpath).read_bytes()).hexdigest()
            for relpath in ALGORITHM_SNAPSHOT_PATHS
        }
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=False,
        capture_output=True, text=True,
    )
    git_commit = git.stdout.strip() if git.returncode == 0 else None
    environment = {
        "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    config = {
        "interval": "1d",
        "symbols_requested": list(symbols),
        "symbols_included": [row["symbol"] for row in data_rows
                             if row["status"] != "missing"],
        "selection_symbols": coverage.get("selection_symbols", []),
        "external_validation_symbols": coverage.get(
            "external_validation_symbols", []
        ),
        "start": start.date().isoformat(), "end": end.date().isoformat(),
        "cost": cost,
        "splits": {key: [value[0].date().isoformat(), value[1].date().isoformat()]
                   for key, value in splits.items()},
        "incumbent": V5_INCUMBENT.name,
        "recommended": recommended_name,
        "candidate_count": candidate_count,
        "snapshot_inputs": snapshot_inputs,
    }
    inputs = {
        row["symbol"]: {
            "file": row.get("file"), "sha256": row.get("sha256"),
            "metadata_sha256": row.get("metadata_sha256"),
        }
        for row in data_rows
    }
    if snapshot_inputs:
        for row in data_rows:
            if not row.get("sha256") or not row.get("file"):
                continue
            source = data_dir / row["file"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
                raise RuntimeError(f"{row['symbol']} 行情快照在审计运行期间发生变化")
            inputs[row["symbol"]]["snapshot_path"] = str(
                source.relative_to(output_dir)
            )
            metadata_source = source.with_suffix(source.suffix + ".meta.json")
            if metadata_source.exists():
                metadata_digest = hashlib.sha256(metadata_source.read_bytes()).hexdigest()
                if metadata_digest != row.get("metadata_sha256"):
                    raise RuntimeError(
                        f"{row['symbol']} 元数据快照在审计运行期间发生变化"
                    )
                inputs[row["symbol"]]["metadata_snapshot_path"] = str(
                    metadata_source.relative_to(output_dir)
                )
    output_names = (
        "signal_events.csv", "signal_events_by_split.csv", "missed_turns.csv",
        "strategy_candidates.csv", "best_strategy_trades.csv",
        "robustness_cost_stress.csv", "robustness_yearly.csv",
        "robustness_leave_one_out.csv", "robustness_ablation.csv",
        "robustness_entry_quality.csv", "robustness_b_confirmation.csv",
        "robustness_external_validation.csv", "data_audit.json", "report.md",
    )
    outputs = {}
    for name in output_names:
        path = output_dir / name
        item = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".csv":
            with path.open(encoding="utf-8") as handle:
                item["rows"] = max(sum(1 for _ in handle) - 1, 0)
        outputs[name] = item
    manifest = {
        "schema_version": 2,
        "complete": True,
        "run_id": _manifest_run_id(
            algorithm_files, config, coverage, inputs, environment
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": {
            "git_commit": git_commit,
            "algorithm_files": algorithm_files,
            "snapshots": algorithm_snapshots,
        },
        "environment": environment,
        "integrity": {
            "requested_window_complete": coverage["requested_end_complete"],
            "input_metadata_trusted": not bool(
                coverage.get("metadata_untrusted_symbols")
            ),
            "source_and_input_snapshots": snapshot_inputs,
        },
        "config": config, "coverage": coverage, "inputs": inputs,
        "decision": gates, "outputs": outputs,
    }
    manifest_path = output_dir / "manifest.json"
    temporary_path = output_dir / ".manifest.json.tmp"
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(manifest_path)
    return manifest


def run_audit(data_dir: Path, output_dir: Path, symbols: tuple[str, ...],
              start: pd.Timestamp, end: pd.Timestamp, cost: float = 0.001,
              snapshot_inputs: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    incomplete_path = output_dir / ".manifest.json.tmp"
    incomplete_path.write_text(
        json.dumps({"schema_version": 2, "complete": False,
                    "status": "audit-running"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    incomplete_path.replace(output_dir / "manifest.json")
    frozen_algorithm_files = None
    audit_data_dir = data_dir
    if snapshot_inputs:
        audit_data_dir, frozen_algorithm_files = _snapshot_run_materials(
            data_dir, output_dir, symbols
        )
    frames, data_rows = audit_data(audit_data_dir, symbols, start, end)
    coverage = _data_coverage(frames, data_rows, start, end)
    if coverage["invalid_quality_symbols"]:
        raise ValueError(
            "行情包含重复日期、非有限值或无效OHLCV: "
            + ", ".join(coverage["invalid_quality_symbols"])
        )
    prepared = prepare_indicator_frames(frames)
    selection_prepared, external_prepared = _partition_selection_universe(
        prepared, coverage
    )
    coverage["selection_symbols"] = list(selection_prepared)
    coverage["external_validation_symbols"] = list(external_prepared)
    coverage["signal_diagnostic_symbols"] = list(prepared)
    split1 = start + (end - start) * 0.60
    split2 = start + (end - start) * 0.80
    train_end = split1.normalize()
    validation_start = train_end + pd.Timedelta(days=1)
    validation_end = split2.normalize()
    test_start = validation_end + pd.Timedelta(days=1)
    splits = {
        "train": (start, train_end),
        "validation": (validation_start, validation_end),
        "test": (test_start, end),
        "full": (start, end),
    }
    events = signal_event_table(prepared, start, end)
    split_events = pd.concat([
        signal_event_table(prepared, split_start, split_end, split=split_name)
        for split_name, (split_start, split_end) in splits.items()
        if split_name != "full"
    ], ignore_index=True)
    incumbent = V5_INCUMBENT
    missed = missed_turn_table(prepared, start, end, candidate=incumbent)
    candidates = evaluate_grid(selection_prepared, splits, cost=cost)
    recommended_name, gates = choose_incremental_recommendation(
        candidates, incumbent.name,
        tuple(candidate.name for candidate in V6_RESEARCH_CHALLENGERS),
    )
    selection_winner_name = str(gates.get("challenger", incumbent.name))
    challenger_cfg = next(candidate for candidate in candidate_grid()
                          if candidate.name == selection_winner_name)
    recommended_cfg = next(candidate for candidate in candidate_grid()
                           if candidate.name == recommended_name)
    _, best_trades = evaluate_candidate(selection_prepared, recommended_cfg, start, end,
                                        cost=cost, collect_trades=True)
    robustness = robustness_tables(selection_prepared, recommended_cfg, start, end)
    robustness["external_validation"] = external_validation_table(
        external_prepared, incumbent, challenger_cfg, splits, cost=cost
    )

    events.to_csv(output_dir / "signal_events.csv", index=False)
    split_events.to_csv(output_dir / "signal_events_by_split.csv", index=False)
    missed.to_csv(output_dir / "missed_turns.csv", index=False)
    candidates.to_csv(output_dir / "strategy_candidates.csv", index=False)
    pd.DataFrame(best_trades).to_csv(output_dir / "best_strategy_trades.csv", index=False)
    for name, table in robustness.items():
        table.to_csv(output_dir / f"robustness_{name}.csv", index=False)
    (output_dir / "data_audit.json").write_text(
        json.dumps(data_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_report(data_rows, events, missed, candidates,
                          selection_winner_name, recommended_name, splits,
                          robustness, gates, split_events=split_events,
                          coverage=coverage)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    manifest = _write_manifest(
        output_dir, audit_data_dir, data_rows, symbols, start, end, cost,
        splits, coverage, recommended_name, gates, len(candidates),
        snapshot_inputs, frozen_algorithm_files,
    )
    return {"recommended": recommended_name,
            "reviewed_challenger": selection_winner_name,
            "research_grid_winner": str(candidates.iloc[0]["name"]),
            "adoption_gates": gates, "symbols": list(prepared), "data": data_rows,
            "coverage": coverage,
            "run_id": manifest["run_id"],
            "splits": {key: [str(value[0].date()), str(value[1].date())]
                       for key, value in splits.items()}}


def _date(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value).normalize()
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"无效日期: {value}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description="收藏股票近5年日K信号审计")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/signal-audit-5y"))
    parser.add_argument("--symbols", default=",".join(DEFAULT_AUDIT_SYMBOLS))
    parser.add_argument("--start", type=_date, default=pd.Timestamp("2021-08-30"))
    parser.add_argument("--end", type=_date, default=pd.Timestamp("2026-08-28"))
    parser.add_argument("--cost", type=float, default=0.001)
    parser.add_argument(
        "--snapshot-inputs", action="store_true",
        help="把本次算法源码、行情CSV与元数据复制到报告目录，便于字节级复现",
    )
    args = parser.parse_args(argv)
    symbols = tuple(dict.fromkeys(s.strip().upper() for s in args.symbols.split(",") if s.strip()))
    if not symbols:
        parser.error("symbols 不能为空")
    if args.start >= args.end:
        parser.error("start 必须早于 end")
    if not np.isfinite(args.cost) or not 0 <= args.cost < 1:
        parser.error("cost 必须在 [0,1) 内")
    result = run_audit(
        args.data_dir, args.output_dir, symbols, args.start, args.end,
        args.cost, snapshot_inputs=args.snapshot_inputs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
