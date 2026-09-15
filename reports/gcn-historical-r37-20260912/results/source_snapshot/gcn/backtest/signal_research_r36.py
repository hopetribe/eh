"""r36：原v5信号新增指标、收缩概率与时间/股票双隔离回溯研究。"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.historical_research import CORE, SNAPSHOT_SHA, load_snapshot
from gcn.core.indicators import adx, atr, mfi, obv, true_range
from gcn.recipes.gcn_main import compute_ehopt10

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/gcn-historical-r36-20260912/protocol.md"
PROTOCOL_SHA = "50f8a75c982032a3a3d47b01d360bc936140a070d6006944cb9d36719e4ce76d"
FAMILIES = {
    "cmf": ("cmf21", "cmf_delta5"),
    "mfi": ("mfi14", "mfi_delta5"),
    "obv": ("obv_flow20", "obv_delta5"),
    "dmi": ("adx14", "di_balance"),
    "volatility": ("relative_natr", "range_shock"),
}
SIGNALS = {"b": "B_SIGNAL", "jf": "ICON_JUEFAN", "s": "S_SIGNAL"}
FOLDS = tuple((f"{year}-08-27", f"{year+1}-08-{27 if year == 2025 else 26}")
              for year in range(2021, 2026))
RIDGE = 8.0


def indicator_features(frame: pd.DataFrame) -> pd.DataFrame:
    """完整200根预热；缺失仍缺失，保留所有原信号。"""
    h, l, c, v = (frame[name] for name in ("HIGH", "LOW", "CLOSE", "VOLUME"))
    spread = h - l
    multiplier = ((2 * c - h - l) / spread.where(spread.gt(0))).mask(spread.eq(0), 0)
    cmf21 = (multiplier * v).rolling(21).sum() / v.rolling(21).sum().replace(0, np.nan)
    money = (mfi(h, l, c, v, 14) - 50) / 50
    balance = obv(c, v)
    flow = (balance - balance.shift(20)) / v.rolling(20).sum().replace(0, np.nan)
    directional = adx(h, l, c, 14)
    total_di = directional.PDI + directional.MDI
    di = ((directional.PDI - directional.MDI) / total_di.replace(0, np.nan)).mask(total_di.eq(0), 0)
    amplitude = atr(h, l, c, 14)
    natr = amplitude / c
    relative = natr / natr.shift(1).rolling(60).median().replace(0, np.nan)
    shock = true_range(h, l, c) / amplitude.shift(1).replace(0, np.nan)
    result = pd.DataFrame({
        "cmf21": cmf21, "cmf_delta5": cmf21.diff(5),
        "mfi14": money, "mfi_delta5": money.diff(5) / 2,
        "obv_flow20": flow, "obv_delta5": flow.diff(5),
        "adx14": directional.ADX / 100, "di_balance": di,
        "relative_natr": np.log(relative.where(relative.gt(0))),
        "range_shock": np.log(shock.where(shock.gt(0))),
    }, index=frame.index).replace([np.inf, -np.inf], np.nan)
    result.iloc[:199] = np.nan
    return result


def frame_events(symbol: str, frame: pd.DataFrame, trusted: bool) -> pd.DataFrame:
    features = indicator_features(frame)
    rows = []
    for signal, column in SIGNALS.items():
        for pos in np.flatnonzero(frame[column].fillna(False).to_numpy(dtype=bool)):
            date = frame.index[pos]
            if date < pd.Timestamp("2017-01-01"):
                continue
            mature = pos + 20 < len(frame)
            row = {"symbol": symbol, "signal": signal, "date": date,
                   "source_trusted": bool(trusted), "warmup_complete": bool(pos >= 199),
                   "mature": mature, "reference_open": np.nan,
                   "reference_open_date": pd.NaT, "outcome_date": pd.NaT,
                   "return20_pct": np.nan, "direction_net_pct": np.nan,
                   "mfe20_pct": np.nan, "mae20_pct": np.nan,
                   "win": None, "gross_win": None, "interference": None,
                   **features.iloc[pos].to_dict()}
            if pos + 1 < len(frame):
                row.update(reference_open=float(frame.OPEN.iloc[pos + 1]),
                           reference_open_date=frame.index[pos + 1])
            if mature:
                entry = row["reference_open"]
                ratio = float(frame.CLOSE.iloc[pos + 20]) / entry
                mfe = float(frame.HIGH.iloc[pos + 1:pos + 21].max()) / entry - 1
                mae = float(frame.LOW.iloc[pos + 1:pos + 21].min()) / entry - 1
                net = 1 - ratio - .002 if signal == "s" else ratio * .999**2 - 1
                row.update(outcome_date=frame.index[pos + 20], return20_pct=(ratio - 1) * 100,
                           direction_net_pct=net * 100, mfe20_pct=mfe * 100, mae20_pct=mae * 100,
                           win=bool(net > 0), gross_win=bool(ratio < 1 if signal == "s" else ratio > 1),
                           interference=bool(ratio > 1 and mfe >= .10 if signal == "s"
                                             else ratio < 1 and mae <= -.08))
            rows.append(row)
    return pd.DataFrame(rows)


def build_events(snapshot: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, quality = load_snapshot(snapshot)
    tables, checks = [], []
    for symbol in CORE:
        raw = frames[symbol].loc[:"2026-08-27"]
        frame = compute_ehopt10(raw, version="v5", diagnostics=True)
        events = frame_events(symbol, frame, quality[symbol])
        tables.append(events)
        checks.append({"symbol": symbol, "bars": len(raw), "first": raw.index[0],
                       "last": raw.index[-1], "source_trusted": quality[symbol],
                       "events": len(events), "mature": int(events.mature.sum()),
                       "warmup_incomplete": int((~events.warmup_complete).sum()),
                       "pending": int((~events.mature).sum())})
    events = pd.concat(tables, ignore_index=True).sort_values(["date", "symbol", "signal"]).reset_index(drop=True)
    return events, pd.DataFrame(checks)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def fit_model(events: pd.DataFrame, family: str) -> dict:
    """固定Beta(2,2)基率偏移；仅训练两斜率，不自动寻找超参。"""
    columns = FAMILIES[family]
    if len(events) and (events.win.isna().any() or not events.win.isin([True, False]).all()):
        raise ValueError("训练必须使用成熟布尔标签")
    base = (float(events.win.sum()) + 2) / (len(events) + 4)
    raw = events[list(columns)].to_numpy(dtype=float)
    valid = np.isfinite(raw).all(axis=1)
    raw = raw[valid]
    target = events.loc[valid, "win"].to_numpy(dtype=float)
    fallback = len(raw) < 8 or len(np.unique(target)) < 2
    mean = raw.mean(axis=0) if len(raw) else np.zeros(2)
    scale = raw.std(axis=0) if len(raw) else np.ones(2)
    scale = np.where(scale > 1e-12, scale, 1)
    design = np.clip((raw - mean) / scale, -4, 4)
    offset = float(np.log(base / (1 - base)))
    beta = np.zeros(2)
    if not fallback:
        for _ in range(100):
            probability = _sigmoid(offset + design @ beta)
            weight = np.maximum(probability * (1 - probability), 1e-9)
            hessian = design.T @ (design * weight[:, None]) + RIDGE * np.eye(2)
            gradient = design.T @ (target - probability) - RIDGE * beta
            step = np.linalg.solve(hessian, gradient)
            beta += step
            if np.max(np.abs(step)) <= 1e-10:
                break
        else:
            raise ValueError("固定置信度模型未收敛")
    return {"family": family, "columns": list(columns), "base_probability": base,
            "offset": offset, "mean": mean.tolist(), "scale": scale.tolist(),
            "beta": beta.tolist(), "fallback": bool(fallback),
            "fit_events": len(events), "feature_fit_events": int(valid.sum())}


def predict_model(events: pd.DataFrame, model: dict) -> pd.DataFrame:
    scored = events.copy()
    raw = events[model["columns"]].to_numpy(dtype=float)
    available = np.isfinite(raw).all(axis=1)
    probability = np.full(len(raw), model["base_probability"])
    design = np.clip((raw[available] - model["mean"]) / model["scale"], -4, 4)
    probability[available] = _sigmoid(model["offset"] + design @ model["beta"])
    scored["probability"] = probability
    scored["base_probability"] = model["base_probability"]
    scored["feature_available"] = available
    scored["fallback"] = model["fallback"] | ~available
    scored["high"] = available & (probability >= .60) & (probability - model["base_probability"] >= .05)
    return scored


def cross_predictions(events: pd.DataFrame) -> pd.DataFrame:
    """按真实标签成熟日期净化，不从未来训练；time_symbol再剔除预测股。"""
    events = events.copy()
    events["date"] = pd.to_datetime(events.date)
    events["outcome_date"] = pd.to_datetime(events.outcome_date)
    eligible = events.mature.eq(True)
    if "warmup_complete" in events:
        eligible &= events.warmup_complete.eq(True)
    rows = []
    for first, last in FOLDS:
        start, end = pd.Timestamp(first), pd.Timestamp(last)
        historic = events[eligible & events.outcome_date.lt(start)]
        future = events[eligible & events.date.between(start, end)]
        for scheme in ("time", "time_symbol"):
            groups = [(None, future)] if scheme == "time" else list(future.groupby("symbol", sort=True))
            for held_symbol, held in groups:
                fitting = historic if held_symbol is None else historic[historic.symbol.ne(held_symbol)]
                for signal in SIGNALS:
                    target = held[held.signal.eq(signal)]
                    if target.empty:
                        continue
                    train = fitting[fitting.signal.eq(signal)]
                    audit = {"scheme": scheme, "fold_start": first, "fold_end": last,
                             "fit_end": (start - pd.Timedelta(days=1)).date().isoformat(),
                             "fit_max_outcome_date": train.outcome_date.max(),
                             "fit_symbols": json.dumps(sorted(train.symbol.unique())),
                             "fit_events": len(train)}
                    members = []
                    for family in FAMILIES:
                        model = fit_model(train, family)
                        scored = predict_model(target, model)
                        for key, value in audit.items():
                            scored[key] = value
                        scored["candidate"] = family
                        scored["model_json"] = json.dumps(model, sort_keys=True)
                        rows.append(scored)
                        members.append(scored)
                    combined = members[0].copy()
                    combined["candidate"] = "equal_ensemble"
                    combined["model_json"] = json.dumps([json.loads(m.model_json.iloc[0]) for m in members], sort_keys=True)
                    combined["probability"] = np.mean([m.probability.to_numpy() for m in members], axis=0)
                    combined["feature_available"] = np.all([m.feature_available.to_numpy() for m in members], axis=0)
                    combined["fallback"] = np.any([m.fallback.to_numpy() for m in members], axis=0)
                    combined["high"] = (combined.feature_available & combined.probability.ge(.60)
                                        & (combined.probability - combined.base_probability).ge(.05))
                    rows.append(combined)
    if not rows:
        raise ValueError("没有成熟的预测期事件")
    return pd.concat(rows, ignore_index=True)


def _wilson(wins: int, n: int) -> tuple[float, float]:
    if not n:
        return np.nan, np.nan
    p, z = wins / n, 1.959963984540054
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return (center - radius) * 100, (center + radius) * 100


def _losses(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    y = scored.win.to_numpy(dtype=float)
    p = np.clip(scored.probability.to_numpy(dtype=float), 1e-9, 1-1e-9)
    b = np.clip(scored.base_probability.to_numpy(dtype=float), 1e-9, 1-1e-9)
    scored["brier"] = (p-y)**2
    scored["base_brier"] = (b-y)**2
    scored["brier_gain"] = scored.base_brier - scored.brier
    scored["logloss"] = -y*np.log(p) - (1-y)*np.log(1-p)
    scored["base_logloss"] = -y*np.log(b) - (1-y)*np.log(1-b)
    return scored


def _metric_row(group: pd.DataFrame) -> dict:
    high = group[group.high]
    win = float(group.win.astype(float).mean() * 100)
    noise = float(group.interference.astype(float).mean() * 100)
    high_win = float(high.win.astype(float).mean() * 100) if len(high) else np.nan
    high_noise = float(high.interference.astype(float).mean() * 100) if len(high) else np.nan
    low, upper = _wilson(int(high.win.sum()), len(high))
    return {"events": len(group), "symbols": int(group.symbol.nunique()),
            "base_win_pct": win, "base_noise_pct": noise,
            "brier": float(group.brier.mean()), "base_brier": float(group.base_brier.mean()),
            "brier_gain": float(group.brier_gain.mean()),
            "logloss": float(group.logloss.mean()), "base_logloss": float(group.base_logloss.mean()),
            "high_events": len(high), "high_symbols": int(high.symbol.nunique()),
            "coverage_pct": 100 * len(high) / len(group), "high_win_pct": high_win,
            "high_noise_pct": high_noise, "win_gain_pp": high_win - win,
            "noise_reduction_pp": noise - high_noise,
            "high_wilson_low_pct": low, "high_wilson_high_pct": upper}


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = _losses(predictions)
    metrics, breakdown, reliability = [], [], []
    for keys, group in scores.groupby(["scheme", "signal", "candidate"], sort=False):
        identity = dict(zip(("scheme", "signal", "candidate"), keys))
        row = {**identity, **_metric_row(group)}
        cluster = group.groupby("symbol").brier_gain.agg(["sum", "count"])
        rng = np.random.default_rng(360912)
        samples = rng.integers(0, len(cluster), size=(2000, len(cluster)))
        boot = cluster["sum"].to_numpy()[samples].sum(axis=1) / cluster["count"].to_numpy()[samples].sum(axis=1)
        row["brier_gain_cluster_lower"] = float(np.quantile(boot, .05/18))
        row["improved_folds"] = int((group.groupby("fold_start").brier_gain.mean() > 0).sum())
        failures = []
        for key, passed in (
            ("brier", row["brier_gain"] >= .01),
            ("logloss", row["logloss"] <= row["base_logloss"]),
            ("high_events", row["high_events"] >= 12),
            ("high_symbols", row["high_symbols"] >= 5),
            ("coverage", 20 <= row["coverage_pct"] <= 60),
            ("win", row["win_gain_pp"] >= 5),
            ("interference", row["noise_reduction_pp"] >= 5),
            ("time_consistency", row["improved_folds"] >= 3),
            ("cluster_uncertainty", row["brier_gain_cluster_lower"] > 0),
        ):
            if not passed:
                failures.append(key)
        row.update(passed=not failures, failures=",".join(failures))
        metrics.append(row)
        for dimension in ("fold_start", "symbol", "source_trusted"):
            for value, subset in group.groupby(dimension, sort=True):
                breakdown.append({**identity, "dimension": dimension, "value": str(value), **_metric_row(subset)})
        bins = np.minimum((group.probability.to_numpy() * 5).astype(int), 4)
        for bin_i in range(5):
            subset = group.iloc[np.flatnonzero(bins == bin_i)]
            reliability.append({**identity, "bin_low": bin_i/5, "bin_high": (bin_i+1)/5,
                                "events": len(subset), "mean_probability": subset.probability.mean(),
                                "observed_win_rate": subset.win.astype(float).mean()})
    return pd.DataFrame(metrics), pd.DataFrame(breakdown), pd.DataFrame(reliability)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_research(snapshot: Path, output: Path) -> dict:
    """追加式归档，不覆盖旧研究；仅回溯，不改变配方、订单或前端。"""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("研究目录必须为空")
    if _sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("r36协议摘要不匹配")
    sources = ["gcn/backtest/signal_research_r36.py", "gcn/backtest/historical_research.py",
               "gcn/backtest/signal_audit.py", "gcn/backtest/engine.py", "gcn/core/indicators.py",
               "gcn/core/tdx.py", "gcn/core/registry.py", "gcn/recipes/gcn_main.py"]
    source_bytes = {name: (ROOT / name).read_bytes() for name in sources}
    events, data_quality = build_events(snapshot)
    predictions = cross_predictions(events)
    metrics, breakdown, reliability = summarize(predictions)
    selected = {}
    for signal in SIGNALS:
        scored = metrics[metrics.signal.eq(signal)]
        candidates = [name for name in (*FAMILIES, "equal_ensemble")
                      if len(scored[scored.candidate.eq(name)]) == 2
                      and scored.loc[scored.candidate.eq(name), "passed"].all()]
        selected[signal] = max(candidates, key=lambda name: scored.loc[scored.candidate.eq(name), "brier_gain"].mean()) if candidates else None
    decision = {"research_version": "r36", "status": "research_candidate" if any(selected.values()) else "no_candidate",
                "selected_by_signal": selected, "production_changed": False,
                "evidence": "exploratory_historical_walk_forward_not_untouched_or_prospective",
                "all_input_sources_trusted": bool(data_quality.source_trusted.all()),
                "hypotheses": 18, "protocol_sha256": PROTOCOL_SHA}
    for name, content in source_bytes.items():
        if (ROOT/name).read_bytes() != content:
            raise ValueError(f"研究运行期间源码改变: {name}")
    if _sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("研究运行期间协议改变")
    output.mkdir(parents=True, exist_ok=True)
    tables = {"events.csv": events, "data_quality.csv": data_quality,
              "predictions.csv": predictions, "metrics.csv": metrics,
              "breakdown.csv": breakdown, "reliability.csv": reliability}
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    _write_json(output / "decision.json", decision)
    (output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    for name, content in source_bytes.items():
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest = {"research_version": "r36", "parent_manifest_sha256": SNAPSHOT_SHA,
                "protocol_sha256": PROTOCOL_SHA, "folds": FOLDS, "ridge": RIDGE,
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "algorithm_sources": {name: hashlib.sha256(raw).hexdigest() for name, raw in source_bytes.items()},
                "outputs": {name: _sha(output / name) for name in (*tables, "decision.json", "protocol.md")}}
    _write_json(output / "manifest.json", manifest)
    return decision


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "reports/signal-audit-v5-review-20260904")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_research(args.snapshot, args.output), ensure_ascii=False, indent=2))
