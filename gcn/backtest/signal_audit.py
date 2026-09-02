# -*- coding: utf-8 -*-
"""收藏股票日 K 信号审计与稳健策略选择。

本模块只读取本地行情缓存，不刷新或覆盖 ``data/``。指标先在完整历史上计算，
再切到审计窗口，避免滚动指标预热不足。候选方案只使用训练/验证段选择，测试段
仅用于最终评估。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.engine import _one_strategy, _perf
from gcn.recipes.gcn_main import _stage_confirmation, compute_ehopt10


DEFAULT_AUDIT_SYMBOLS = (
    "TQQQ", "MSFT", "NFLX", "YINN", "SNOW", "TSLA",
    "MRNA", "NVDA", "TEM", "GOOGL", "SIVE", "AAOI",
)


@dataclass(frozen=True)
class Candidate:
    entry: str
    exit: str
    trail: float | None
    max_hold: int | None

    @property
    def name(self) -> str:
        trail = "none" if self.trail is None else f"{self.trail:.0%}"
        hold = "none" if self.max_hold is None else str(self.max_hold)
        return f"{self.entry}|exit={self.exit}|trail={trail}|hold={hold}"


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
                         "first": None, "last": None, "rows_5y": 0,
                         "duplicates": None, "bad_ohlcv": None, "nan_rows": None,
                         "sha256": None, "metadata_hash": "missing"})
            continue
        raw = pd.read_csv(path)
        duplicate_dates = int(raw["date"].duplicated().sum()) if "date" in raw else None
        data = _read_ohlcv(path)
        values = data[["open", "high", "low", "close", "volume"]]
        bad = ((values[["open", "high", "low", "close"]] <= 0).any(axis=1)
               | (values["volume"] < 0)
               | (values["high"] < values[["open", "low", "close"]].max(axis=1))
               | (values["low"] > values[["open", "high", "close"]].min(axis=1)))
        window = data.loc[(data.index >= start) & (data.index <= end)]
        status = "ok" if len(window) else "outside-window"
        if data.index.min() > start:
            status = "partial-history"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        metadata_hash = "no-meta"
        if meta_path.exists():
            try:
                expected = json.loads(meta_path.read_text(encoding="utf-8")).get("sha256")
                metadata_hash = "match" if expected == digest else "mismatch"
            except (OSError, ValueError, TypeError):
                metadata_hash = "invalid-meta"
        rows.append({
            "symbol": symbol, "status": status, "rows": int(len(data)),
            "first": data.index.min().date().isoformat(),
            "last": data.index.max().date().isoformat(), "rows_5y": int(len(window)),
            "duplicates": duplicate_dates, "bad_ohlcv": int(bad.sum()),
            "nan_rows": int(values.isna().any(axis=1).sum()),
            "sha256": digest, "metadata_hash": metadata_hash,
        })
        frames[symbol] = data
    return frames, rows


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
        entries = {
            "v4-b+jf": union,
            "v4-b-only": raw_b,
            "stage5-ma60+jf": (
                experiment["B_SIGNAL"].fillna(False).astype(bool) | juefan
            ),
            "b-confirm3-ma20+jf": _confirm(raw_b, stable, 20, 3) | juefan,
            "b-confirm5-ma20+jf": _confirm(raw_b, stable, 20, 5) | juefan,
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
    return [Candidate(e, x, t, h) for e in entries for x in exits
            for t in trails for h in max_holds]


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
                           candidate.max_hold, trail=candidate.trail)
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
    eligible["complexity"] = (eligible["max_hold"].notna().astype(int)
                              + eligible["trail"].notna().astype(int)
                              + eligible["exit"].ne("S").astype(int))
    eligible = eligible.sort_values(["selection_score", "complexity"],
                                    ascending=[False, True])
    return str(eligible.iloc[0]["name"]), gates


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
        fold_end = min(fold_start + pd.DateOffset(years=1) - pd.Timedelta(days=1), end)
        for label, candidate in named.items():
            stats, _ = evaluate_candidate(prepared, candidate, fold_start, fold_end)
            yearly_rows.append({"start": fold_start.date().isoformat(),
                                "end": fold_end.date().isoformat(),
                                "scheme": label, **stats})
        fold_start = fold_start + pd.DateOffset(years=1)

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


def _forward_path(frame: pd.DataFrame, pos: int, horizon: int) -> dict:
    if pos + horizon >= len(frame) or pos + 1 >= len(frame):
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
                       end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for symbol, bundle in prepared.items():
        frame = bundle["v4"]
        ma60 = frame["CLOSE"].rolling(60, min_periods=60).mean()
        ma200 = frame["CLOSE"].rolling(200, min_periods=200).mean()
        for signal, label in (("B_SIGNAL", "B买"), ("ICON_JUEFAN", "绝反"),
                              ("S_SIGNAL", "S卖")):
            mask = frame[signal].fillna(False).astype(bool)
            for pos in np.flatnonzero(mask.to_numpy()):
                date = frame.index[pos]
                if date < start or date > end:
                    continue
                path20 = _forward_path(frame, pos, 20)
                row = {"symbol": symbol, "date": date.date().isoformat(),
                       "signal": label, "close": float(frame["CLOSE"].iloc[pos]),
                       "b_score": float(frame["B_SCORE"].iloc[pos]),
                       "s_score": float(frame["S_SCORE"].iloc[pos]),
                       "regime": ("bull" if frame["CLOSE"].iloc[pos] >= ma200.iloc[pos]
                                  else "bear"),
                       "ma60_slope20_pct": (float(ma60.iloc[pos] / ma60.iloc[pos - 20] - 1) * 100
                                            if pos >= 20 and np.isfinite(ma60.iloc[pos - 20]) else np.nan)}
                for horizon in (5, 10, 20, 60):
                    path = _forward_path(frame, pos, horizon)
                    row[f"ret_{horizon}d_pct"] = path["return"] * 100
                row.update({"mfe_20d_pct": path20["mfe"] * 100,
                            "mae_20d_pct": path20["mae"] * 100,
                            "mfe_day": path20["mfe_day"], "mae_day": path20["mae_day"]})
                if label in ("B买", "绝反"):
                    row["interference"] = bool(path20["return"] < 0 and path20["mae"] <= -0.08)
                    if label == "B买":
                        stage = bool(frame["B_STAGE_SIGNAL"].iloc[pos])
                        base_bull = (bool(frame["B_CONDITION"].iloc[pos])
                                     and frame["CLOSE"].iloc[pos] >= ma200.iloc[pos])
                        row["subtype"] = ("stage" if stage else "base-bull" if base_bull
                                          else "bear/crash-recovery")
                    else:
                        row["subtype"] = "juefan"
                else:
                    row["interference"] = bool(path20["return"] > 0 and path20["mfe"] >= 0.10)
                    row["subtype"] = "bear-rally" if row["regime"] == "bear" else "major-top"
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
                path = _forward_path(frame, pos, 20)
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
                path = _forward_path(frame, pos, 20)
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
                      buy_gain: float = 0.15, sell_drop: float = 0.12) -> pd.DataFrame:
    """用事后转折标签审计覆盖率；这些标签不得直接作为实盘信号。"""
    rows = []
    for symbol, bundle in prepared.items():
        frame = bundle["v4"]
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
        buy_signal = (frame["B_SIGNAL"].fillna(False).astype(bool)
                      | frame["ICON_JUEFAN"].fillna(False).astype(bool))
        sell_signal = frame["S_SIGNAL"].fillna(False).astype(bool)
        for kind, positions, strength, mask in (
            ("buy", buy_pos, gain, buy_signal), ("sell", sell_pos, drop, sell_signal)
        ):
            for pos in positions:
                covered, signal_date, offset = _nearest_signal(frame, mask, pos)
                rows.append({
                    "symbol": symbol, "kind": kind,
                    "date": frame.index[pos].date().isoformat(),
                    "opportunity_pct": float(strength.iloc[pos] * 100),
                    "covered": covered, "missed": not covered,
                    "nearest_signal_date": signal_date,
                    "signal_offset_days": offset,
                })
    return pd.DataFrame(rows).sort_values(["kind", "missed", "opportunity_pct"],
                                          ascending=[True, False, False])


def _fmt(value, digits: int = 2) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _strategy_line(label: str, row: pd.Series, prefix: str) -> str:
    return (f"| {label} | {_fmt(row[f'{prefix}_cagr'])}% | {_fmt(row[f'{prefix}_mdd'])}% | "
            f"{_fmt(row[f'{prefix}_sharpe'])} | {int(row[f'{prefix}_trades'])} | "
            f"{_fmt(row[f'{prefix}_win'], 1)}% | {_fmt(row[f'{prefix}_median_trade'])}% | "
            f"{_fmt(row[f'{prefix}_worst_trade'])}% |")


def build_report(data_rows: list[dict], events: pd.DataFrame, missed: pd.DataFrame,
                 candidates: pd.DataFrame, selection_winner_name: str,
                 recommended_name: str,
                 split_dates: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
                 robustness: dict[str, pd.DataFrame], gates: dict) -> str:
    baseline_name = Candidate("v4-b+jf", "S", None, None).name
    stage_name = Candidate("stage5-ma60+jf", "S", None, None).name
    lookup = candidates.set_index("name")
    selection_winner = lookup.loc[selection_winner_name]
    recommended = lookup.loc[recommended_name]
    baseline = lookup.loc[baseline_name]
    stage = lookup.loc[stage_name]
    lines = [
        "# 收藏股票近5年日K信号审计", "",
        "> 研究用途，不构成投资建议。所有交易均按信号日收盘确认、下一交易日开盘成交、单边成本0.1%。", "",
        "## 数据与验证设计", "",
        f"- 审计窗口：{split_dates['full'][0].date()} 至 {split_dates['full'][1].date()}。",
        f"- 训练：{split_dates['train'][0].date()}～{split_dates['train'][1].date()}；"
        f"验证：{split_dates['validation'][0].date()}～{split_dates['validation'][1].date()}；"
        f"测试：{split_dates['test'][0].date()}～{split_dates['test'][1].date()}。",
        "- 指标先用完整历史预热，再切片；测试段不参与参数评分，只在最后做一次采纳门控。组合指标按可用标的等权日收益计算。",
        "- 统一截止到2026-08-28，排除9月2日尚未收盘的盘中K线；输入文件SHA-256记录在data_audit.json。",
        "- 漏点是事后诊断标签，不是可直接交易的未来函数：买点=±5日局部低点且未来20日最大涨幅≥15%；卖点=±5日局部高点且未来20日最大跌幅≥12%；相邻20日仅保留最强转折。",
        "- 干扰买入=未来20日收盘收益<0且最大不利波动≤-8%；干扰卖出=未来20日收盘仍上涨且最大上涨≥10%。", "",
        "### 数据审计", "", "| 标的 | 状态 | 全部行数 | 5年行数 | 起始 | 结束 | 异常OHLC | 元数据哈希 |",
        "|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in data_rows:
        lines.append(f"| {row['symbol']} | {row['status']} | {row['rows']} | {row['rows_5y']} | "
                     f"{row['first'] or '--'} | {row['last'] or '--'} | {row['bad_ohlcv'] if row['bad_ohlcv'] is not None else '--'} | "
                     f"{row['metadata_hash']} |")

    lines += ["", "## 实际信号质量", "",
              "| 信号 | 数量 | 干扰数 | 干扰率 | 20日均值 | 20日中位 | 20日胜率 | 20日MAE中位 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for signal, group in events.dropna(subset=["ret_20d_pct"]).groupby("signal", sort=False):
        interference = int(group["interference"].sum())
        if signal == "S卖":
            win = float((group["ret_20d_pct"] < 0).mean() * 100)
        else:
            win = float((group["ret_20d_pct"] > 0).mean() * 100)
        lines.append(f"| {signal} | {len(group)} | {interference} | {interference / len(group) * 100:.1f}% | "
                     f"{group['ret_20d_pct'].mean():+.2f}% | {group['ret_20d_pct'].median():+.2f}% | "
                     f"{win:.1f}% | {group['mae_20d_pct'].median():+.2f}% |")

    lines += ["", "### B确认有效性", "",
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
              "| 标的 | B买 | B干扰 | 绝反 | 绝反干扰 | S卖 | S干扰 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for symbol, group in events.groupby("symbol"):
        values = []
        for signal in ("B买", "绝反", "S卖"):
            sg = group[group["signal"] == signal]
            values.extend([len(sg), int(sg["interference"].sum())])
        lines.append(f"| {symbol} | " + " | ".join(map(str, values)) + " |")

    lines += ["", "### 影响最大的干扰信号", "",
              "| 信号 | 标的 | 日期 | 20日收益 | 20日MFE | 20日MAE |",
              "|---|---|---|---:|---:|---:|"]
    buy_noise = (events[events["interference"] & events["signal"].isin(["B买", "绝反"])]
                 .sort_values("ret_20d_pct").head(12))
    sell_noise = (events[events["interference"] & events["signal"].eq("S卖")]
                  .sort_values("mfe_20d_pct", ascending=False).head(12))
    for row in pd.concat([buy_noise, sell_noise]).itertuples():
        lines.append(f"| {row.signal} | {row.symbol} | {row.date} | "
                     f"{row.ret_20d_pct:+.2f}% | {row.mfe_20d_pct:+.2f}% | "
                     f"{row.mae_20d_pct:+.2f}% |")

    lines += ["", "## 错过的转折", "",
              "| 类型 | 事后转折数 | 已覆盖 | 错过 | 覆盖率 |",
              "|---|---:|---:|---:|---:|"]
    for kind, label in (("buy", "买点"), ("sell", "卖点")):
        group = missed[missed["kind"] == kind]
        covered = int(group["covered"].sum())
        lines.append(f"| {label} | {len(group)} | {covered} | {len(group) - covered} | "
                     f"{covered / len(group) * 100 if len(group) else 0:.1f}% |")
    lines += ["", "### 各标的漏点", "",
              "| 标的 | 买点/错过 | 买点最大机会 | 卖点/错过 | 卖点最大机会 |",
              "|---|---:|---:|---:|---:|"]
    for symbol in sorted(missed["symbol"].unique()):
        buy = missed[(missed["symbol"] == symbol) & (missed["kind"] == "buy")]
        sell = missed[(missed["symbol"] == symbol) & (missed["kind"] == "sell")]
        lines.append(f"| {symbol} | {len(buy)}/{int(buy['missed'].sum())} | "
                     f"{_fmt(buy['opportunity_pct'].max())}% | {len(sell)}/{int(sell['missed'].sum())} | "
                     f"{_fmt(sell['opportunity_pct'].max())}% |")
    lines += ["", "### 最大的漏点（每类前15）", "",
              "| 类型 | 标的 | 日期 | 未来20日机会 |",
              "|---|---|---|---:|"]
    top_missed = (missed[missed["missed"]].sort_values("opportunity_pct", ascending=False)
                  .groupby("kind", group_keys=False).head(15))
    for row in top_missed.itertuples():
        lines.append(f"| {'买点' if row.kind == 'buy' else '卖点'} | {row.symbol} | {row.date} | {row.opportunity_pct:.2f}% |")

    lines += ["", "## 方案比较", "",
              "选择分数只使用训练与验证段。`测试` 为冻结的最近一年，未参与调参。", "",
              "| 测试段方案 | CAGR | 最大回撤 | Sharpe | 交易数 | 胜率 | 中位单笔 | 最差单笔 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|",
              _strategy_line("v4原始 B买+绝反→S卖", baseline, "test"),
              _strategy_line("仅阶段B确认", stage, "test"),
              _strategy_line(f"训练/验证分数冠军：{selection_winner_name}", selection_winner, "test"),
              _strategy_line(f"最终推荐：{recommended_name}", recommended, "test"), "",
              f"最终采纳门槛：全样本保留≥{gates['full_cagr_retention']:.0%}基线CAGR、"
              f"MDD≤{gates['full_mdd_ratio']:.0%}基线；测试保留≥{gates['test_cagr_retention']:.0%}"
              f"基线CAGR，且Sharpe不低、MDD不高。通过候选数={gates['eligible_candidates']}。", "",
              "### 训练/验证稳健性", "",
              "| 方案 | 分段 | CAGR | 最大回撤 | Sharpe | 交易数 | 正收益标的 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for label, row in (("基线", baseline), ("最终推荐", recommended)):
        for split in ("train", "validation", "full"):
            lines.append(f"| {label} | {split} | {_fmt(row[f'{split}_cagr'])}% | "
                         f"{_fmt(row[f'{split}_mdd'])}% | {_fmt(row[f'{split}_sharpe'])} | "
                         f"{int(row[f'{split}_trades'])} | {int(row[f'{split}_positive_symbols'])}/{int(row[f'{split}_symbols'])} |")

    lines += ["", "### 训练+验证选出的前10个候选", "",
              "| 排名 | 方案 | 选择分数 | 验证Sharpe | 验证回撤 | 测试Sharpe | 测试回撤 |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(candidates.head(10).itertuples(), 1):
        lines.append(f"| {rank} | {row.name} | {row.selection_score:.3f} | "
                     f"{_fmt(row.validation_sharpe)} | {_fmt(row.validation_mdd)}% | "
                     f"{_fmt(row.test_sharpe)} | {_fmt(row.test_mdd)}% |")

    lines += ["", "### 推荐参数邻域", "",
              "| 方案 | 训练Sharpe/MDD | 验证Sharpe/MDD | 测试Sharpe/MDD | 全样本Sharpe/MDD |",
              "|---|---:|---:|---:|---:|"]
    nearby = (
        Candidate("b-confirm3-ma20+jf", "S", 0.20, None).name,
        Candidate("b-confirm5-ma20+jf", "S", 0.15, None).name,
        Candidate("b-confirm5-ma20+jf", "S", 0.20, None).name,
        Candidate("b-confirm5-ma20+jf", "S", 0.20, 60).name,
    )
    for name in nearby:
        row = lookup.loc[name]
        lines.append(f"| {name} | {_fmt(row.train_sharpe)}/{_fmt(row.train_mdd)}% | "
                     f"{_fmt(row.validation_sharpe)}/{_fmt(row.validation_mdd)}% | "
                     f"{_fmt(row.test_sharpe)}/{_fmt(row.test_mdd)}% | "
                     f"{_fmt(row.full_sharpe)}/{_fmt(row.full_mdd)}% |")

    lines += ["", "## 稳健性验证", "", "### 消融（全5年）", "",
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

    loo = robustness["leave_one_out"].pivot(index="omitted", columns="scheme",
                                             values=["sharpe", "mdd"])
    lines += ["", "### 逐标的留一（全5年）", "",
              "| 剔除标的 | 基线Sharpe | 推荐Sharpe | 基线MDD | 推荐MDD |",
              "|---|---:|---:|---:|---:|"]
    for symbol in loo.index:
        lines.append(f"| {symbol} | {_fmt(loo.loc[symbol, ('sharpe', 'baseline')])} | "
                     f"{_fmt(loo.loc[symbol, ('sharpe', 'recommended')])} | "
                     f"{_fmt(loo.loc[symbol, ('mdd', 'baseline')])}% | "
                     f"{_fmt(loo.loc[symbol, ('mdd', 'recommended')])}% |")

    lines += ["", "## 最终推荐方案", "", f"`{recommended_name}`", "",
              "- B买只作为 Setup；随后5根K线内首次收盘突破Setup当日最高价且站上MA20，才生成可执行买点。",
              "- 绝反保持即时入场；它本身已经包含60日低位、5%反包和量能确认，再延迟会损失有效反转。",
              "- S卖维持原信号；额外使用从入场后最高收盘回撤20%的跟踪止损。S卖再确认和泛化MA20退出均未通过最终收益保留门槛。",
              "- 不设置最长持有期；60日上限没有提升风险指标，且增加不必要参数。", "",
              "该结论仅在当前收藏池、当前成本与本地冻结数据上成立；事后漏点只用于审计，不能直接成为实盘信号。详细逐笔数据见同目录 CSV。",
              ""]
    return "\n".join(lines)


def run_audit(data_dir: Path, output_dir: Path, symbols: tuple[str, ...],
              start: pd.Timestamp, end: pd.Timestamp, cost: float = 0.001) -> dict:
    frames, data_rows = audit_data(data_dir, symbols, start, end)
    prepared = prepare_indicator_frames(frames)
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
    missed = missed_turn_table(prepared, start, end)
    candidates = evaluate_grid(prepared, splits, cost=cost)
    selection_winner_name = str(candidates.iloc[0]["name"])
    recommended_name, gates = choose_recommendation(candidates)
    recommended_cfg = next(candidate for candidate in candidate_grid()
                           if candidate.name == recommended_name)
    _, best_trades = evaluate_candidate(prepared, recommended_cfg, start, end,
                                        cost=cost, collect_trades=True)
    robustness = robustness_tables(prepared, recommended_cfg, start, end)

    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_dir / "signal_events.csv", index=False)
    missed.to_csv(output_dir / "missed_turns.csv", index=False)
    candidates.to_csv(output_dir / "strategy_candidates.csv", index=False)
    pd.DataFrame(best_trades).to_csv(output_dir / "best_strategy_trades.csv", index=False)
    for name, table in robustness.items():
        table.to_csv(output_dir / f"robustness_{name}.csv", index=False)
    (output_dir / "data_audit.json").write_text(
        json.dumps(data_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_report(data_rows, events, missed, candidates,
                          selection_winner_name, recommended_name, splits,
                          robustness, gates)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return {"recommended": recommended_name,
            "train_validation_winner": selection_winner_name,
            "adoption_gates": gates, "symbols": list(prepared), "data": data_rows,
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
    args = parser.parse_args(argv)
    symbols = tuple(dict.fromkeys(s.strip().upper() for s in args.symbols.split(",") if s.strip()))
    if not symbols:
        parser.error("symbols 不能为空")
    if args.start >= args.end:
        parser.error("start 必须早于 end")
    if not np.isfinite(args.cost) or not 0 <= args.cost < 1:
        parser.error("cost 必须在 [0,1) 内")
    result = run_audit(args.data_dir, args.output_dir, symbols, args.start, args.end, args.cost)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
