"""已有行情的隔离增量研究。结果是回溯证据，不属于前向shadow评估。"""
from __future__ import annotations

import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.recipes.gcn_main import _stage_confirmation, compute_ehopt10
from gcn.backtest.engine import _one_strategy
from gcn.backtest.signal_audit import _portfolio_stats, _forward_path, missed_turn_table, Candidate

RULES = ("v5", "b-cooldown20", "s-cooldown40", "jf-cooldown10",
         "bs-cooldown", "b-momentum", "profit50", "bs-cooldown-profit50")
SNAPSHOT_SHA = "42f1d19b3aacb4d738f01cd6362fa0f68b145115b72449c3dd0d14e02f755094"
CORE = ("TQQQ", "MSFT", "NFLX", "YINN", "SNOW", "TSLA", "MRNA", "NVDA", "GOOGL", "AAOI")


def load_snapshot(root: Path, expected_sha: str = SNAPSHOT_SHA) -> tuple[dict, dict]:
    """从经既有manifest验证的同一份字节解析；来源不可信只标记，不伪造元数据。"""
    raw_manifest = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != expected_sha:
        raise ValueError("父manifest摘要不匹配")
    frames, quality = {}, {}
    for symbol, spec in json.loads(raw_manifest)["inputs"].items():
        csv = (root / spec["snapshot_path"]).read_bytes()
        meta_raw = (root / spec["metadata_snapshot_path"]).read_bytes()
        if (hashlib.sha256(csv).hexdigest() != spec["sha256"]
                or hashlib.sha256(meta_raw).hexdigest() != spec["metadata_sha256"]):
            raise ValueError(f"{symbol}: 输入快照摘要不匹配")
        frame = pd.read_csv(io.BytesIO(csv))
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame = frame.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
        if (frame.empty or frame.index.hasnans or not frame.index.is_unique
                or not frame.index.is_monotonic_increasing
                or not np.isfinite(frame.to_numpy()).all()
                or (frame[["open", "high", "low", "close"]] <= 0).any().any()
                or (frame["volume"] < 0).any()
                or (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any()
                or (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any()):
            raise ValueError(f"{symbol}: 非法OHLCV或日期")
        meta = json.loads(meta_raw)
        frames[symbol] = frame
        quality[symbol] = (meta.get("sha256") == spec["sha256"]
                           and meta.get("adjustment") == "adjusted"
                           and meta.get("source") in {"yahoo", "futu"})
    return frames, quality


def accepted_cooldown(raw: pd.Series, gap: int) -> pd.Series:
    """仅已接受信号启动冷却；被抑制的原始触发不会续期。"""
    accepted = np.zeros(len(raw), dtype=bool)
    last = -gap
    for pos, active in enumerate(raw.fillna(False).astype(bool)):
        if active and pos - last >= gap:
            accepted[pos] = True
            last = pos
    return pd.Series(accepted, index=raw.index)


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """冻结的r1候选。只消费当根或历史值；九转回画列不在输入依赖中。"""
    original = frame[["B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"]]
    b, _ = _stage_confirmation(accepted_cooldown(frame["B_ALL_RAW"], 20),
                               frame["HIGH"], frame["CLOSE"], frame["MID"], 5)
    s = accepted_cooldown(frame["S_RAW"], 40)
    result = {name: original.copy() for name in RULES}
    for name in ("b-cooldown20", "bs-cooldown", "bs-cooldown-profit50"):
        result[name]["B_SIGNAL"] = b
    for name in ("s-cooldown40", "bs-cooldown", "bs-cooldown-profit50"):
        result[name]["S_SIGNAL"] = s
    result["jf-cooldown10"]["ICON_JUEFAN"] = accepted_cooldown(frame["JF_RAW"], 10)
    result["b-momentum"]["B_SIGNAL"] &= frame["MACD"] > frame["MACD"].shift(1)
    return result


def evaluate_rule(prepared: dict, rule: str, start: pd.Timestamp, end: pd.Timestamp,
                  cost: float = 0.001) -> dict:
    """统一0.1%规则费用产生订单，成本压力只对同一订单路径重新计费。"""
    daily, trades, exposure = {}, [], []
    for symbol, bundle in prepared.items():
        frame = bundle["frame"].loc[start:end].copy()
        if len(frame) < 2:
            raise ValueError(f"{symbol}: 评估区间不足两根K线")
        for col in ("B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL"):
            frame[col] = bundle["rules"][rule][col].reindex(frame.index)
        keep = 0.5 if rule in {"profit50", "bs-cooldown-profit50"} else None
        result = _one_strategy(frame, ["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"],
                               0.001, None, trail=0.20, profit_keep=keep)
        counts = np.zeros(len(frame), dtype=int)
        ratio = (1 - cost) / 0.999
        for trade in result["trades"]:
            i, j = trade["i"], trade["j"]
            exit_pos = min(j, len(frame) - 1)
            counts[i] += 1
            counts[exit_pos] += 1
            peak = float(frame["CLOSE"].iloc[i:j].max() / frame["OPEN"].iloc[i] - 1)
            trades.append({"symbol": symbol,
                           "entry_date": frame.index[i].date().isoformat(),
                           "exit_date": frame.index[exit_pos].date().isoformat(),
                           "return_pct": ((1 + trade["ret"]) * ratio**2 - 1) * 100,
                           "hold_days": trade["hold"], "exit_reason": trade["exit_reason"],
                           "peak_close_pct": peak * 100})
        equity = pd.Series(result["equity"] * ratio**counts.cumsum(), index=frame.index)
        returns = equity.pct_change()
        returns.iloc[0] = equity.iloc[0] - 1
        daily[symbol] = returns
        exposure.append(float(np.mean(result["held"])))
    matrix = pd.DataFrame(daily)
    if matrix.empty or matrix.isna().any().any():
        raise ValueError("评估要求完整共同交易日，不允许缺失标的静默改变权重")
    stats = _portfolio_stats(matrix.mean(axis=1))
    rets = np.array([t["return_pct"] for t in trades])
    stats.update({"trades": len(trades), "win": float((rets > 0).mean() * 100) if len(rets) else None,
                  "worst_trade": float(rets.min()) if len(rets) else None,
                  "median_trade": float(np.median(rets)) if len(rets) else None,
                  "exposure": float(np.mean(exposure)),
                  "profit_to_loss": sum(t["peak_close_pct"] > 0 and t["return_pct"] <= 0 for t in trades)})
    return {"stats": stats, "trades": trades, "returns": matrix}


def event_quality(prepared: dict, rule: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """完整20日事件毛收益；entry是B/绝反同日去重后的并集。"""
    events = []
    for symbol, bundle in prepared.items():
        frame, signals = bundle["frame"], bundle["rules"][rule]
        masks = {"entry": signals["B_SIGNAL"] | signals["ICON_JUEFAN"],
                 "b": signals["B_SIGNAL"], "jf": signals["ICON_JUEFAN"],
                 "s": signals["S_SIGNAL"]}
        for kind, mask in masks.items():
            for pos in np.flatnonzero(mask.to_numpy()):
                if not start <= frame.index[pos] <= end:
                    continue
                path = _forward_path(frame, pos, 20, outcome_end=end)
                if not np.isfinite(path["return"]):
                    continue
                win = path["return"] < 0 if kind == "s" else path["return"] > 0
                noise = (path["return"] > 0 and path["mfe"] >= 0.10 if kind == "s"
                         else path["return"] < 0 and path["mae"] <= -0.08)
                events.append({"symbol": symbol, "signal": kind,
                               "date": frame.index[pos].date().isoformat(),
                               "outcome_date": frame.index[pos + 20].date().isoformat(),
                               "ret20_pct": path["return"] * 100, "mfe20_pct": path["mfe"] * 100,
                               "mae20_pct": path["mae"] * 100, "win": bool(win),
                               "interference": bool(noise)})
    stats = {}
    for kind in ("entry", "b", "jf", "s"):
        rows = [r for r in events if r["signal"] == kind]
        stats[kind + "_events"] = len(rows)
        for metric in ("win", "interference"):
            stats[kind + "_" + metric] = (float(np.mean([r[metric] for r in rows]) * 100)
                                          if rows else None)
    return {"stats": stats, "events": events}


def training_failures(row: dict, base: dict) -> list[str]:
    checks = {
        "trade_coverage": row["trades"] >= base["trades"] * 0.85,
        "entry_coverage": row["entry_events"] >= base["entry_events"] * 0.85,
        "entry_win": row["entry_win"] is not None and row["entry_win"] >= base["entry_win"],
        "entry_interference": (row["entry_interference"] is not None
                               and row["entry_interference"] <= base["entry_interference"]),
        "cagr": row["cagr"] >= base["cagr"] * 0.90,
        "mdd": row["mdd"] <= base["mdd"] * 1.10,
    }
    return [name for name, passed in checks.items() if not passed]


def choose_training(rows: list[dict]) -> str | None:
    base = next(r for r in rows if r["rule"] == "v5")
    eligible = [r for r in rows if r["rule"] != "v5" and not training_failures(r, base)
                and np.isfinite(r["calmar"])]
    eligible.sort(key=lambda r: (-r["calmar"], RULES.index(r["rule"])))
    return eligible[0]["rule"] if eligible else None


def validation_failures(row: dict, base: dict) -> list[str]:
    failures = training_failures(row, base)
    for key in ("sharpe", "win"):
        if row[key] is None or not row[key] >= base[key]:
            failures.append(key)
    improves = row["mdd"] <= base["mdd"] * 0.95
    for key in ("entry_win", "b_win", "jf_win", "s_win", "win"):
        improves |= row[key] is not None and base[key] is not None and row[key] > base[key]
    if not improves:
        failures.append("no_material_improvement")
    return failures


def run_research(snapshot: Path, output: Path) -> dict:
    """重放r1；训练只选一个，再验证。压力表不参与重新选型。"""
    if (output / "manifest.json").exists():
        raise FileExistsError("研究版本已固化，请指定新的输出目录")
    frames, quality = load_snapshot(snapshot)
    prepared = {}
    for symbol in CORE + ("TEM",):
        frame = compute_ehopt10(frames[symbol], version="v5", diagnostics=True)
        prepared[symbol] = {"frame": frame, "rules": candidate_signals(frame)}
    core = {s: prepared[s] for s in CORE}
    start, end = pd.Timestamp("2021-08-27"), pd.Timestamp("2026-08-27")
    rows, trades, events = [], [], []

    def compare(case, pool, first, last, rules, cost=0.001):
        result_rows = []
        for rule in rules:
            perf = evaluate_rule(pool, rule, pd.Timestamp(first), pd.Timestamp(last), cost)
            event = event_quality(pool, rule, pd.Timestamp(first), pd.Timestamp(last))
            label = {"case": case, "rule": rule}
            row = {**label, "symbols": ",".join(pool), "start": str(first)[:10],
                   "end": str(last)[:10], "cost": cost, **perf["stats"], **event["stats"]}
            rows.append(row)
            result_rows.append(row)
            if case in {"train", "validation", "full5y", "early8"}:
                trades.extend({**label, **t} for t in perf["trades"])
                events.extend({**label, **e} for e in event["events"])
        return result_rows

    training = compare("train", core, start, "2024-08-26", RULES)
    selected = choose_training(training)
    rules = ("v5", selected) if selected else ("v5",)
    validation = compare("validation", core, "2024-08-27", "2025-08-26", rules)
    failed = validation_failures(validation[1], validation[0]) if selected else ["no_training_candidate"]
    compare("full5y", core, start, end, rules)
    for year in range(2021, 2026):
        last = "2026-08-27" if year == 2025 else f"{year + 1}-08-26"
        compare(f"year{year}", core, f"{year}-08-27", last, rules)
    early = {s: core[s] for s in CORE if s not in {"MRNA", "SNOW"}}
    compare("early8", early, "2017-09-22", "2021-08-26", rules)
    compare("trusted5", {s: b for s, b in core.items() if quality[s]}, start, end, rules)
    compare("unleveraged8", {s: b for s, b in core.items() if s not in {"TQQQ", "YINN"}}, start, end, rules)
    compare("cost025", core, start, end, rules, cost=0.0025)
    compare("TEM_external", {"TEM": prepared["TEM"]}, "2025-06-16", end, rules)
    for symbol in CORE:
        compare("without_" + symbol, {s: b for s, b in core.items() if s != symbol}, start, end, rules)
        compare("only_" + symbol, {symbol: core[symbol]}, start, end, rules)

    turns = []
    for rule in rules:
        bundles = {}
        for symbol, bundle in core.items():
            signals = bundle["rules"][rule]
            bundles[symbol] = {"v4": bundle["frame"],
                               "entries": {"entry": signals["B_SIGNAL"] | signals["ICON_JUEFAN"]},
                               "exits": {"exit": signals["S_SIGNAL"]}}
        # r1唯一已选S候选不改变盈利保护；其他退出规则不得套用这条路径。
        if rule not in {"v5", "s-cooldown40"}:
            raise ValueError("新挑战者需要对应的持仓漏点口径")
        table = missed_turn_table(bundles, start, end, candidate=Candidate("entry", "exit", .20, None))
        table.insert(0, "rule", rule)
        turns.append(table)

    decision = {"research_version": "gcn-historical-r1", "selected": selected,
                "validation_failures": failed, "recommended": "v5" if failed else selected,
                "evidence": "retrospective_only", "production_changed": False,
                "training_rejections": {r["rule"]: training_failures(r, training[0]) for r in training[1:]}}
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {"comparisons.csv": pd.DataFrame(rows), "trades.csv": pd.DataFrame(trades),
                 "events.csv": pd.DataFrame(events), "missed_turns.csv": pd.concat(turns, ignore_index=True)}
    for filename, table in artifacts.items():
        table.to_csv(output / filename, index=False)
    (output / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    root = Path(__file__).resolve().parents[2]
    sources = {}
    for filename in ("gcn/backtest/historical_research.py", "gcn/backtest/signal_audit.py",
                     "gcn/backtest/engine.py", "gcn/recipes/gcn_main.py", "gcn/core/tdx.py"):
        raw = (root / filename).read_bytes()
        destination = output / "source_snapshot" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        sources[filename] = hashlib.sha256(raw).hexdigest()
    manifest = {"research_version": "gcn-historical-r1", "parent_manifest_sha256": SNAPSHOT_SHA,
                "source_quality": quality, "algorithm_sources": sources,
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return decision


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_research(args.snapshot, args.output), ensure_ascii=False, indent=2))
