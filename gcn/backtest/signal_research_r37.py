"""r37：按730天半衰期更新B/JF/S历史概率，保留独立失败与通过证据。"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r36 import FOLDS, ROOT, SIGNALS, _sha, _write_json, summarize

PROTOCOL = ROOT / "reports/gcn-historical-r37-20260912/protocol.md"
PROTOCOL_SHA = "ff7c0ad2d1d14760c23447a97d771b9e626ae75abb0776b72c7a28c6b20c3141"
PARENT_SHA = "dce812be846291d74a1bb7dcf651749460f6cce22e57a01a0c4909c1a0efc79b"
HALF_LIFE_DAYS = 730


def decayed_base_rate(events: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    cutoff = pd.Timestamp(cutoff)
    dates = pd.to_datetime(events.outcome_date)
    if (dates.isna().any() or dates.ge(cutoff).any() or events.win.isna().any()
            or not events.win.isin([True, False]).all()):
        raise ValueError("必须使用截止日前已成熟的布尔标签")
    days = (cutoff - dates).dt.days.to_numpy(dtype=float)
    weights = np.exp2(-days / HALF_LIFE_DAYS)
    weight_sum = float(weights.sum())
    target = events.win.to_numpy(dtype=float)
    return {"probability": (float(weights @ target) + 2) / (weight_sum + 4),
            "base_probability": (float(target.sum()) + 2) / (len(target) + 4),
            "weighted_events": weight_sum,
            "effective_events": weight_sum**2 / float(weights @ weights) if len(weights) else 0.,
            "fit_events": len(events)}


def cross_predictions(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["date"] = pd.to_datetime(events.date)
    events["outcome_date"] = pd.to_datetime(events.outcome_date)
    eligible = events.mature.eq(True) & events.warmup_complete.eq(True)
    rows = []
    for first, last in FOLDS:
        start = pd.Timestamp(first)
        historic = events[eligible & events.outcome_date.lt(start)]
        future = events[eligible & events.date.between(start, pd.Timestamp(last))]
        for scheme in ("time", "time_symbol"):
            groups = [(None, future)] if scheme == "time" else list(future.groupby("symbol", sort=True))
            for held_symbol, held in groups:
                fitting = historic if held_symbol is None else historic[historic.symbol.ne(held_symbol)]
                for signal in SIGNALS:
                    target = held[held.signal.eq(signal)].copy()
                    if target.empty:
                        continue
                    train = fitting[fitting.signal.eq(signal)]
                    model = decayed_base_rate(train, start)
                    audit = {"scheme": scheme, "candidate": "decay730", "fold_start": first,
                             "fold_end": last, "fit_end": (start-pd.Timedelta(days=1)).date().isoformat(),
                             "fit_max_outcome_date": train.outcome_date.max(),
                             "fit_symbols": json.dumps(sorted(train.symbol.unique())),
                             "feature_available": True, "fallback": train.empty,
                             "high": model["probability"] >= .60 and model["probability"] - model["base_probability"] >= .05,
                             **model}
                    for key, value in audit.items():
                        target[key] = value
                    rows.append(target)
    if not rows:
        raise ValueError("没有成熟的预测期事件")
    return pd.concat(rows, ignore_index=True)


def run_research(parent: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("研究目录必须为空")
    if _sha(parent/"manifest.json") != PARENT_SHA:
        raise ValueError("r36父manifest摘要不匹配")
    parent_manifest = json.loads((parent/"manifest.json").read_bytes())
    # 从校验过的同一份事件字节解析，避免读校验和读数据之间的竞争。
    parent_bytes = {}
    for name, digest in parent_manifest["outputs"].items():
        if Path(name).name != name:
            raise ValueError("r36输出路径不合法")
        content = (parent/name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"r36输出摘要不匹配: {name}")
        parent_bytes[name] = content
    sources = {}
    for name, digest in parent_manifest["algorithm_sources"].items():
        if not name.startswith("gcn/") or ".." in Path(name).parts:
            raise ValueError("r36源码路径不合法")
        content = (parent/"source_snapshot"/name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest or (ROOT/name).read_bytes() != content:
            raise ValueError(f"r36源码摘要不匹配: {name}")
        sources[name] = content
    sources["gcn/backtest/signal_research_r37.py"] = Path(__file__).read_bytes()
    if _sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("r37协议摘要不匹配")
    import io
    events = pd.read_csv(io.BytesIO(parent_bytes["events.csv"]), parse_dates=["date", "outcome_date"])
    scores = cross_predictions(events)
    metrics, breakdown, reliability = summarize(scores)
    selected = {signal: "decay730" if len(metrics[metrics.signal.eq(signal)]) == 2
                and metrics.loc[metrics.signal.eq(signal), "passed"].all() else None for signal in SIGNALS}
    decision = {"research_version": "r37", "status": "research_candidate" if any(selected.values()) else "no_candidate",
                "selected_by_signal": selected, "half_life_days": HALF_LIFE_DAYS,
                "production_changed": False, "new_hypotheses": 3,
                "evidence": "exploratory_historical_algorithm_development_not_independent_validation",
                "parent_manifest_sha256": PARENT_SHA, "protocol_sha256": PROTOCOL_SHA}
    for name, content in sources.items():
        if (ROOT/name).read_bytes() != content:
            raise ValueError(f"运行期间源码变化: {name}")
    if _sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("运行期间协议变化")
    output.mkdir(parents=True, exist_ok=True)
    tables = {"predictions.csv": scores, "metrics.csv": metrics,
              "breakdown.csv": breakdown, "reliability.csv": reliability}
    for name, table in tables.items():
        table.to_csv(output/name, index=False)
    _write_json(output/"decision.json", decision)
    (output/"protocol.md").write_bytes(PROTOCOL.read_bytes())
    for name, content in sources.items():
        target = output/"source_snapshot"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest = {"research_version": "r37", "parent_manifest_sha256": PARENT_SHA,
                "half_life_days": HALF_LIFE_DAYS, "protocol_sha256": PROTOCOL_SHA,
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "algorithm_sources": {name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()},
                "outputs": {name: _sha(output/name) for name in (*tables, "decision.json", "protocol.md")}}
    _write_json(output/"manifest.json", manifest)
    return decision


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT/"reports/gcn-historical-r36-20260912/results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_research(args.parent, args.output), ensure_ascii=False, indent=2))
