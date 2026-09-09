"""r35：v5信号的因果质量置信分层研究。"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.historical_research import CORE, SNAPSHOT_SHA, load_snapshot
from gcn.recipes.gcn_main import compute_ehopt10


FEATURE_COLUMNS = (
    "band_position",
    "macd_level",
    "macd_delta",
    "rsi_center",
    "chip_center",
)
SIGNALS = ("b", "jf", "s")
RIDGE_LAMBDA = 8.0
TRAINING_START = pd.Timestamp("2021-08-27")
TRAINING_END = pd.Timestamp("2024-08-26")
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/gcn-historical-r35-20260908/protocol.md"
PROTOCOL_SHA = "04cca422cedba4c74039cc7eb9d7bf80ca64091eb0ca5a10b91b469da9d579b9"


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = denominator.abs().gt(1e-12) & np.isfinite(denominator)
    result = numerator.div(denominator).where(valid, 0.0)
    return result.where(np.isfinite(result), 0.0)


def causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """计算仅依赖当根及历史值的固定五项质量特征。"""
    width = frame["UPPER"] - frame["LOWER"]
    macd_std = frame["MACD"].rolling(60, min_periods=60).std(ddof=0)
    features = pd.DataFrame(
        {
            "band_position": _safe_ratio(frame["CLOSE"] - frame["MID"], width),
            "macd_level": _safe_ratio(frame["MACD"], macd_std),
            "macd_delta": _safe_ratio(frame["MACD"].diff(), macd_std),
            "rsi_center": (frame["RSI1"] - 50.0) / 25.0,
            "chip_center": (frame["获利筹"] - 50.0) / 50.0,
        },
        index=frame.index,
    )
    return features.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-4.0, 4.0)


def build_event_table(snapshot: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """还原原v5三类成熟20根事件及信号日特征。"""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if not start <= end:
        raise ValueError("事件窗口起点必须不晚于终点")
    frames, _ = load_snapshot(Path(snapshot))
    rows: list[dict[str, object]] = []
    signal_columns = {"b": "B_SIGNAL", "jf": "ICON_JUEFAN", "s": "S_SIGNAL"}
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:end], version="v5", diagnostics=True)
        features = causal_features(frame)
        for signal, column in signal_columns.items():
            positions = np.flatnonzero(frame[column].fillna(False).to_numpy(dtype=bool))
            for pos in positions:
                date = frame.index[pos]
                if not start <= date <= end or pos + 20 >= len(frame):
                    continue
                entry_open = float(frame["OPEN"].iloc[pos + 1])
                outcome_close = float(frame["CLOSE"].iloc[pos + 20])
                if not np.isfinite([entry_open, outcome_close]).all() or entry_open <= 0:
                    raise ValueError(f"{symbol} {date.date()}: 非法20根事件价格")
                return20 = outcome_close / entry_open - 1
                mfe20 = float(frame["HIGH"].iloc[pos + 1 : pos + 21].max()) / entry_open - 1
                mae20 = float(frame["LOW"].iloc[pos + 1 : pos + 21].min()) / entry_open - 1
                win = return20 < 0 if signal == "s" else return20 > 0
                interference = (
                    return20 > 0 and mfe20 >= 0.10
                    if signal == "s"
                    else return20 < 0 and mae20 <= -0.08
                )
                rows.append(
                    {
                        "symbol": symbol,
                        "signal": signal,
                        "date": date.date().isoformat(),
                        "year": int(date.year),
                        "reference_open_date": frame.index[pos + 1].date().isoformat(),
                        "outcome_date": frame.index[pos + 20].date().isoformat(),
                        "return20_pct": return20 * 100,
                        "mfe20_pct": mfe20 * 100,
                        "mae20_pct": mae20 * 100,
                        "win": bool(win),
                        "interference": bool(interference),
                        **{
                            feature: float(features.iloc[pos][feature])
                            for feature in FEATURE_COLUMNS
                        },
                    }
                )
    return pd.DataFrame(rows)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def _design_matrix(raw: np.ndarray, signal: np.ndarray) -> np.ndarray:
    is_jf = signal == "jf"
    is_s = signal == "s"
    return np.column_stack(
        [
            np.ones(len(raw)),
            is_jf.astype(float),
            is_s.astype(float),
            raw,
            raw * is_jf[:, None],
            raw * is_s[:, None],
        ]
    )


def fit_quality_models(events: pd.DataFrame) -> dict[str, object]:
    """拟合固定部分池化L2逻辑模型和类型内质量分位。"""
    required = {"signal", "win", *FEATURE_COLUMNS}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"训练事件缺少列: {missing}")
    for signal in SIGNALS:
        selected = events[events["signal"].eq(signal)]
        if len(selected) < 3:
            raise ValueError(f"{signal}: 训练事件不足")
    raw = events[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    target = events["win"].to_numpy(dtype=float)
    signals = events["signal"].astype(str).to_numpy()
    if not np.isfinite(raw).all() or not np.isfinite(target).all():
        raise ValueError("训练事件包含非有限值")
    if set(np.unique(target)) != {0.0, 1.0}:
        raise ValueError("训练标签必须同时包含正反例")
    if not set(np.unique(signals)).issubset(SIGNALS):
        raise ValueError("训练事件包含未知信号类型")
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (raw - mean) / scale
    design = _design_matrix(standardized, signals)
    beta = np.zeros(design.shape[1], dtype=float)
    penalty = np.diag([0.0] + [RIDGE_LAMBDA] * (design.shape[1] - 1))
    for _ in range(100):
        probability = np.clip(_sigmoid(design @ beta), 1e-9, 1 - 1e-9)
        weight = probability * (1 - probability)
        lhs = design.T @ (design * weight[:, None]) + penalty
        rhs = design.T @ (target - probability) - penalty @ beta
        step = np.linalg.solve(lhs, rhs)
        beta = beta + step
        if float(np.max(np.abs(step))) <= 1e-10:
            break
    fitted = _sigmoid(design @ beta)
    thresholds = {}
    for signal in SIGNALS:
        selected = fitted[signals == signal]
        thresholds[signal] = {
            "low": float(np.quantile(selected, 1 / 3)),
            "high": float(np.quantile(selected, 2 / 3)),
        }
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "beta": beta.tolist(),
        "thresholds": thresholds,
    }


def predict_quality(
    events: pd.DataFrame, models: dict[str, object]
) -> pd.DataFrame:
    """用拟合集冻结参数打分；不读取结果标签。"""
    required = {"signal", *FEATURE_COLUMNS}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"评分事件缺少列: {missing}")
    scored = events.copy()
    scored["quality_score"] = np.nan
    scored["quality_tier"] = pd.Series(pd.NA, index=scored.index, dtype="string")
    mean = np.asarray(models["mean"], dtype=float)
    scale = np.asarray(models["scale"], dtype=float)
    beta = np.asarray(models["beta"], dtype=float)
    thresholds = models["thresholds"]
    for signal in SIGNALS:
        if signal not in thresholds:
            raise ValueError(f"缺少{signal}质量模型")
        selected = scored["signal"].eq(signal)
        raw = scored.loc[selected, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError(f"{signal}: 评分事件包含非有限值")
        signal_values = np.full(len(raw), signal, dtype=object)
        design = _design_matrix((raw - mean) / scale, signal_values)
        probability = _sigmoid(design @ beta)
        low = float(thresholds[signal]["low"])
        high = float(thresholds[signal]["high"])
        tier = np.where(probability < low, "low", np.where(probability >= high, "high", "normal"))
        scored.loc[selected, "quality_score"] = probability * 100
        scored.loc[selected, "quality_tier"] = tier
    if scored["quality_score"].isna().any() or scored["quality_tier"].isna().any():
        raise ValueError("事件包含未知信号类型")
    return scored


def cross_validated_scores(events: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """生成扩展时间或留一股票的完全交叉预测。"""
    scored_parts: list[pd.DataFrame] = []
    if scheme == "expanding_time":
        splits = (
            (pd.Timestamp("2022-08-26"), pd.Timestamp("2022-08-27"), pd.Timestamp("2023-08-26")),
            (pd.Timestamp("2023-08-26"), pd.Timestamp("2023-08-27"), pd.Timestamp("2024-08-26")),
        )
        event_dates = pd.to_datetime(events["date"], errors="raise")
        outcome_dates = pd.to_datetime(events["outcome_date"], errors="raise")
        for train_end, predict_start, predict_end in splits:
            fit = events[outcome_dates.le(train_end)]
            held = events[event_dates.between(predict_start, predict_end)]
            model = fit_quality_models(fit)
            scored = predict_quality(held, model)
            scored["cv_scheme"] = scheme
            scored["fold"] = f"{predict_start.date()}..{predict_end.date()}"
            scored["fit_events"] = len(fit)
            scored["fit_outcome_end"] = train_end.date().isoformat()
            scored_parts.append(scored)
    elif scheme == "leave_one_symbol":
        for symbol in CORE:
            held = events[events["symbol"].eq(symbol)]
            if held.empty:
                continue
            fit = events[events["symbol"].ne(symbol)]
            model = fit_quality_models(fit)
            scored = predict_quality(held, model)
            scored["cv_scheme"] = scheme
            scored["fold"] = symbol
            scored["fit_events"] = len(fit)
            scored["fit_outcome_end"] = None
            scored_parts.append(scored)
    else:
        raise ValueError("未知交叉预测方案")
    if not scored_parts:
        raise ValueError("交叉预测没有可评分事件")
    return pd.concat(scored_parts, ignore_index=True)


def _quality_metrics(scored: pd.DataFrame, scheme: str) -> dict[str, object]:
    high = scored[scored["quality_tier"].eq("high")]
    events = len(scored)
    high_events = len(high)
    base_win = 100 * float(scored["win"].mean())
    high_win = 100 * float(high["win"].mean()) if high_events else np.nan
    base_noise = 100 * float(scored["interference"].mean())
    high_noise = 100 * float(high["interference"].mean()) if high_events else np.nan
    coverage = 100 * high_events / events if events else 0.0
    minimum_events = 25 if scheme == "in_sample" else 20
    minimum_symbols = 8 if scheme == "in_sample" else 7
    minimum_win_gain = 8.0 if scheme == "in_sample" else 7.0
    failures = []
    if not 25 <= coverage <= 45:
        failures.append("coverage")
    if high_events < minimum_events:
        failures.append("high_events")
    if int(high["symbol"].nunique()) < minimum_symbols:
        failures.append("symbols")
    if not high_win - base_win >= minimum_win_gain:
        failures.append("win_improvement")
    if not base_noise - high_noise >= 5.0:
        failures.append("interference_reduction")
    high_by_signal = high.groupby("signal").size().to_dict()
    if scheme == "in_sample":
        for signal in SIGNALS:
            if int(high_by_signal.get(signal, 0)) < 5:
                failures.append(f"{signal}_high_events")
    return {
        "scheme": scheme,
        "events": events,
        "high_events": high_events,
        "high_coverage_pct": coverage,
        "high_symbols": int(high["symbol"].nunique()),
        "base_win_rate_pct": base_win,
        "high_win_rate_pct": high_win,
        "win_improvement_pp": high_win - base_win,
        "base_interference_rate_pct": base_noise,
        "high_interference_rate_pct": high_noise,
        "interference_reduction_pp": base_noise - high_noise,
        "b_high_events": int(high_by_signal.get("b", 0)),
        "jf_high_events": int(high_by_signal.get("jf", 0)),
        "s_high_events": int(high_by_signal.get("s", 0)),
        "passed": not failures,
        "failures": ",".join(failures),
    }


def _tier_summary(scored_sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for scheme, scored in scored_sets.items():
        for signal in ("all", *SIGNALS):
            base = scored if signal == "all" else scored[scored["signal"].eq(signal)]
            for tier in ("all", "low", "normal", "high"):
                selected = base if tier == "all" else base[base["quality_tier"].eq(tier)]
                rows.append(
                    {
                        "scheme": scheme,
                        "signal": signal,
                        "tier": tier,
                        "events": len(selected),
                        "symbols": int(selected["symbol"].nunique()),
                        "wins": int(selected["win"].sum()),
                        "win_rate_pct": (
                            100 * float(selected["win"].mean()) if len(selected) else np.nan
                        ),
                        "interference": int(selected["interference"].sum()),
                        "interference_rate_pct": (
                            100 * float(selected["interference"].mean())
                            if len(selected)
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def training_screen(snapshot: Path) -> dict[str, object]:
    """只运行训练及训练内交叉预测，不读取固定验证事件。"""
    events = build_event_table(snapshot, TRAINING_START, TRAINING_END)
    model = fit_quality_models(events)
    in_sample = predict_quality(events, model)
    expanding = cross_validated_scores(events, "expanding_time")
    leave_one = cross_validated_scores(events, "leave_one_symbol")
    scored_sets = {
        "in_sample": in_sample,
        "expanding_time": expanding,
        "leave_one_symbol": leave_one,
    }
    metrics = pd.DataFrame(
        [_quality_metrics(scored, scheme) for scheme, scored in scored_sets.items()]
    )
    passed = bool(metrics["passed"].all())
    decision = {
        "research_version": "gcn-historical-r35",
        "stage": "training_confidence_screen",
        "candidate": "v5-relative-quality" if passed else None,
        "status": "candidate_selected" if passed else "no_candidate",
        "training_passed": passed,
        "validation_run": False,
        "production_changed": False,
    }
    return {
        "events": events,
        "model": model,
        "in_sample": in_sample,
        "expanding_time": expanding,
        "leave_one_symbol": leave_one,
        "metrics": metrics,
        "tier_summary": _tier_summary(scored_sets),
        "decision": decision,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_training_screen(snapshot: Path, output: Path) -> dict[str, object]:
    """运行并归档r35训练筛查；训练失败时严格停止在验证前。"""
    snapshot = Path(snapshot)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output目录必须为空")
    if _sha256(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("r35协议摘要不匹配")
    result = training_screen(snapshot)
    metrics = result["metrics"]
    failed = metrics.loc[~metrics["passed"], ["scheme", "failures"]]
    decision = {
        **result["decision"],
        "reason": "training_cross_validation_failed",
        "failed_schemes": {
            str(row.scheme): str(row.failures) for row in failed.itertuples()
        },
        "quality_fields_implemented": False,
    }
    if result["decision"]["training_passed"]:
        decision["reason"] = "training_gate_passed_validation_pending"

    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "events.csv": result["events"],
        "in_sample.csv": result["in_sample"],
        "expanding_time.csv": result["expanding_time"],
        "leave_one_symbol.csv": result["leave_one_symbol"],
        "metrics.csv": result["metrics"],
        "tier_summary.csv": result["tier_summary"],
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    _write_json(output / "model.json", result["model"])
    _write_json(output / "decision.json", decision)
    shutil.copyfile(PROTOCOL, output / "protocol.md")

    output_names = (*tables, "model.json", "decision.json", "protocol.md")
    manifest = {
        "research_version": "gcn-historical-r35",
        "stage": "training_confidence_screen",
        "window": ["training", "2021-08-27", "2024-08-26"],
        "parent_manifest_sha256": SNAPSHOT_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "ridge_lambda": RIDGE_LAMBDA,
        "feature_columns": list(FEATURE_COLUMNS),
        "algorithm_sources": {
            "gcn/backtest/signal_research_r35.py": _sha256(Path(__file__)),
            "gcn/backtest/historical_research.py": _sha256(
                ROOT / "gcn/backtest/historical_research.py"
            ),
            "gcn/recipes/gcn_main.py": _sha256(ROOT / "gcn/recipes/gcn_main.py"),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "outputs": {name: _sha256(output / name) for name in output_names},
    }
    _write_json(output / "manifest.json", manifest)
    return {**result, "decision": decision, "manifest": manifest}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=ROOT / "reports/signal-audit-v5-review-20260904",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archived = run_training_screen(args.snapshot, args.output)
    print(json.dumps(archived["decision"], indent=2, ensure_ascii=False))
