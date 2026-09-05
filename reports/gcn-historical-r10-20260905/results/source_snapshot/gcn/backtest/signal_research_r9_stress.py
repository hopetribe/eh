"""r9验证后稳健性复核：固定唯一候选，不从压力结果重新选型。"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.historical_research import CORE, SNAPSHOT_SHA, load_snapshot, evaluate_rule, event_quality
from gcn.backtest.signal_research_r3 import executed_turns
from gcn.backtest.signal_research_r4 import _choose_training
from gcn.backtest.signal_research_r7_validation import validation_failures
from gcn.backtest.signal_research_r9 import CHALLENGERS, RULES, PROFIT_KEEPS, candidate_signals
from gcn.recipes.gcn_main import compute_ehopt10


def stress_failures(rows: list[dict]) -> list[str]:
    pairs = {}
    required = ("full5y", "cost025", "early8", *("without_" + s for s in CORE))
    for case in required:
        subset = [r for r in rows if r["case"] == case]
        if len(subset) != 2 or {r["rule"] for r in subset} != {"v5", CHALLENGERS[0]}:
            raise ValueError(f"压力对照缺失或重复：{case}")
        pairs[case] = ({r["rule"]: r for r in subset})
    failures = []
    for case in ("full5y", "without_AAOI", "cost025", "early8"):
        base, row = pairs[case]["v5"], pairs[case][CHALLENGERS[0]]
        floor = base["cagr"] - (1. if case == "early8" else 0.)
        if not np.isfinite(row["cagr"]) or row["cagr"] < floor:
            failures.append(case + "_cagr")
        if not np.isfinite(row["mdd"]) or row["mdd"] > base["mdd"] * 1.10:
            failures.append(case + "_mdd")
    count = sum(np.isfinite(pairs["without_" + s][CHALLENGERS[0]]["cagr"])
                and pairs["without_" + s][CHALLENGERS[0]]["cagr"] >= pairs["without_" + s]["v5"]["cagr"]
                for s in CORE)
    if count < 8:
        failures.append("leave_one_out_cagr")
    return failures


def run_stress(snapshot: Path, training: Path, validation: Path, output: Path) -> dict:
    if (output / "manifest.json").exists():
        raise FileExistsError("压力阶段已固化，请使用新的输出目录")
    root = Path(__file__).resolve().parents[2]
    protocol_path = root / "reports/gcn-historical-r9-20260905/protocol.md"
    protocol = protocol_path.read_bytes()
    environment = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    parents, sources = {}, {}
    for name, folder in (("training", training), ("validation", validation)):
        raw = (folder / "manifest.json").read_bytes()
        meta = json.loads(raw)
        if (meta["research_version"] != "gcn-historical-r9" or meta["parent_manifest_sha256"] != SNAPSHOT_SHA
                or meta["environment"] != environment
                or meta["protocol_sha256"] != hashlib.sha256(protocol).hexdigest()
                or meta.get("profit_keeps") != PROFIT_KEEPS
                or meta.get("entry_profit_enabled_col") != "ENTRY_PROFIT_ENABLED"):
            raise ValueError("压力研究与父阶段身份、协议、环境或保护配置不匹配")
        for relative, digest in meta["outputs"].items():
            if Path(relative).name != relative or hashlib.sha256((folder / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"父阶段工件变化：{relative}")
        for relative, digest in meta["algorithm_sources"].items():
            if not relative.startswith("gcn/") or ".." in Path(relative).parts or not relative.endswith(".py"):
                raise ValueError("父阶段源码路径无效")
            source = (root / relative).read_bytes()
            if hashlib.sha256(source).hexdigest() != digest:
                raise ValueError(f"父阶段后源码变化：{relative}")
            sources[relative] = source
        parents[name] = (folder, raw, meta)
    if parents["validation"][2]["training_manifest_sha256"] != hashlib.sha256(parents["training"][1]).hexdigest():
        raise ValueError("训练与验证链不匹配")
    train_rows = pd.read_csv(training / "training.csv").to_dict("records")
    selected = _choose_training(train_rows, CHALLENGERS)
    valid_rows = pd.read_csv(validation / "comparisons.csv").to_dict("records")
    train_decision = json.loads((training / "decision.json").read_text())
    valid_decision = json.loads((validation / "decision.json").read_text())
    if ([r["rule"] for r in train_rows] != list(RULES) or selected != CHALLENGERS[0]
            or selected != train_decision["selected"] or selected != valid_decision["selected"]
            or [r["rule"] for r in valid_rows] != ["v5", selected]
            or validation_failures(valid_rows[1], valid_rows[0])
            or valid_decision["status"] != "passed_validation_pending_stress"
            or valid_decision["failures"]):
        raise ValueError("没有通过训练和验证的唯一候选")
    own = "gcn/backtest/signal_research_r9_stress.py"
    sources[own] = (root / own).read_bytes()
    frames, quality = load_snapshot(snapshot)
    cache = {}
    rows, trades, events, turns = [], [], [], []

    def compare(case, symbols, first, last, cost=.001, keep=.5):
        if last not in cache:
            cache[last] = {}
        for symbol in symbols:
            if symbol not in cache[last]:
                frame = compute_ehopt10(frames[symbol].loc[:last], version="v5")
                cache[last][symbol] = {"frame": frame, "rules": candidate_signals(frame)}
        pool = {s: cache[last][s] for s in symbols}
        start, end = pd.Timestamp(first), pd.Timestamp(last)
        for rule in ("v5", selected):
            perf = evaluate_rule(pool, rule, start, end, cost,
                                 entry_hard_stop_col="ENTRY_STOP", entry_max_hold_col="ENTRY_LIMIT",
                                 entry_exit_cols=("USE_EXTRA", "EXTRA_EXIT"),
                                 profit_keep=keep if rule == selected else None,
                                 entry_profit_enabled_col="ENTRY_PROFIT_ENABLED",
                                 include_positions=case == "full5y")
            event = event_quality(pool, rule, start, end)
            labels = {"case": case, "rule": rule}
            rows.append({**labels, "symbols": ",".join(symbols), "start": first, "end": last,
                         "cost": cost, "profit_keep": keep if rule == selected else None,
                         **perf["stats"], **event["stats"]})
            if case in {"full5y", "early8", "TEM_external", "cost025"}:
                for trade in perf["trades"]:
                    bundle = pool[trade["symbol"]]
                    pos = bundle["frame"].index.get_loc(pd.Timestamp(trade["entry_date"])) - 1
                    settings = bundle["rules"][rule].iloc[pos]
                    original = bundle["frame"][["B_SIGNAL", "ICON_JUEFAN"]].iloc[pos].any()
                    trades.append({**labels, **trade, "entry_origin": "v5" if original else "additional",
                                   "entry_profit_enabled": bool(settings["ENTRY_PROFIT_ENABLED"]),
                                   "entry_stop_pct": settings["ENTRY_STOP"] * 100,
                                   "entry_limit": settings["ENTRY_LIMIT"]})
            if case in {"full5y", "early8", "TEM_external"}:
                events.extend({**labels, **e} for e in event["events"])
            if case == "full5y":
                missed = executed_turns(pool, perf, start, end)
                for kind in ("buy", "sell"):
                    actionable = missed[(missed.kind == kind) & missed.actionable]
                    rows[-1][kind + "_turns"] = len(actionable)
                    rows[-1][kind + "_covered"] = int(actionable.covered.sum())
                missed.insert(0, "rule", rule)
                turns.append(missed)

    start, end = "2021-08-27", "2026-08-27"
    compare("full5y", CORE, start, end)
    for year in range(2021, 2026):
        last = end if year == 2025 else f"{year + 1}-08-26"
        compare(f"year{year}", CORE, f"{year}-08-27", last)
    compare("early8", tuple(s for s in CORE if s not in {"MRNA", "SNOW"}), "2017-09-22", "2021-08-26")
    compare("trusted5", tuple(s for s in CORE if quality[s]), start, end)
    compare("unleveraged8", tuple(s for s in CORE if s not in {"TQQQ", "YINN"}), start, end)
    compare("cost025", CORE, start, end, cost=.0025)
    compare("TEM_external", ("TEM",), "2025-06-16", end)
    for symbol in CORE:
        compare("without_" + symbol, tuple(s for s in CORE if s != symbol), start, end)
        compare("only_" + symbol, (symbol,), start, end)
    failures = stress_failures(rows)
    if not failures:
        for keep in (.4, .6):
            compare(f"neighbor{int(keep * 100)}", CORE, start, end, keep=keep)
    decision = {"research_version": "gcn-historical-r9", "selected": selected,
                "status": "rejected_keep_v5" if failures else "passed_stress_pending_review",
                "failures": failures, "recommended": "v5", "production_changed": False,
                "evidence": "retrospective_only",
                "neighborhood_status": "not_run_failed_stress" if failures else "diagnostic_only_no_reselection"}
    for relative, raw in sources.items():
        if (root / relative).read_bytes() != raw:
            raise ValueError(f"计算期间源码变化：{relative}")
    for folder, raw, _ in parents.values():
        if (folder / "manifest.json").read_bytes() != raw:
            raise ValueError("计算期间父阶段manifest变化")
    if protocol_path.read_bytes() != protocol:
        raise ValueError("计算期间协议变化")
    output.mkdir(parents=True, exist_ok=True)
    for filename, table in {"comparisons.csv": pd.DataFrame(rows), "trades.csv": pd.DataFrame(trades),
                            "events.csv": pd.DataFrame(events), "missed_turns.csv": pd.concat(turns, ignore_index=True)}.items():
        table.to_csv(output / filename, index=False)
    (output / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    (output / "protocol.md").write_bytes(protocol)
    for relative, raw in sources.items():
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    manifest = {"research_version": "gcn-historical-r9", "stage": "stress",
                **{name + "_manifest_sha256": hashlib.sha256(raw).hexdigest() for name, (_, raw, _) in parents.items()},
                "parent_manifest_sha256": SNAPSHOT_SHA, "source_quality": quality,
                "protocol_sha256": hashlib.sha256(protocol).hexdigest(), "environment": environment,
                "profit_keeps": PROFIT_KEEPS, "entry_profit_enabled_col": "ENTRY_PROFIT_ENABLED",
                "algorithm_sources": {p: hashlib.sha256(raw).hexdigest() for p, raw in sources.items()},
                "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return decision


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--training", type=Path, default=Path("reports/gcn-historical-r9-20260905/results"))
    parser.add_argument("--validation", type=Path, default=Path("reports/gcn-historical-r9-20260905/validation"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_stress(args.snapshot, args.training, args.validation, args.output), indent=2, ensure_ascii=False))
