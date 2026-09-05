"""r10诊断：分离窗口起始持仓状态与期末强平，不选择新策略。"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.engine import _one_strategy
from gcn.backtest.historical_research import CORE, SNAPSHOT_SHA, load_snapshot
from gcn.backtest.signal_audit import _portfolio_stats
from gcn.backtest.signal_research_r9 import CHALLENGERS, PROFIT_KEEPS, candidate_signals
from gcn.recipes.gcn_main import compute_ehopt10

RULES = ("v5", CHALLENGERS[0])
MODES = ("reset_liquidate", "reset_mark", "carry_mark")


def window_replay(bundle: dict, rule: str, anchor: pd.Timestamp, start: pd.Timestamp,
                  end: pd.Timestamp, mode: str) -> dict:
    """连续重放后截取期间收益；不人工重建持仓或使用跨期交易全收益。"""
    if mode not in MODES or rule not in RULES:
        raise ValueError("未知窗口口径或策略")
    if not anchor <= start <= end:
        raise ValueError("窗口日期必须满足anchor <= start <= end")
    first = anchor if mode == "carry_mark" else start
    frame = bundle["frame"].loc[first:end].copy()
    for column in bundle["rules"][rule]:
        frame[column] = bundle["rules"][rule][column].reindex(frame.index)
    if frame.loc[start:end].empty:
        raise ValueError("观察窗口没有K线")

    def simulate(data, terminal_policy="mark"):
        return _one_strategy(data, ["B_SIGNAL", "ICON_JUEFAN"], ["S_SIGNAL"], .001, None,
                             trail=.20, profit_keep=PROFIT_KEEPS.get(rule), terminal_policy=terminal_policy,
                             entry_hard_stop_col="ENTRY_STOP", entry_max_hold_col="ENTRY_LIMIT",
                             entry_exit_cols=("USE_EXTRA", "EXTRA_EXIT"),
                             entry_profit_enabled_col="ENTRY_PROFIT_ENABLED")

    prior = simulate(frame.loc[frame.index < start])["state"]
    result = simulate(frame, "liquidate" if mode == "reset_liquidate" else "mark")
    equity = pd.Series(result["equity"], index=frame.index)
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] - 1.
    prior_entry = frame.index[prior["entry_i"]] if prior["entry_i"] is not None else None
    end_entry = frame.index[result["state"]["entry_i"]] if result["state"]["entry_i"] is not None else None
    trades = []
    for trade in result["trades"]:
        entry_date = frame.index[trade["i"]]
        exit_date = frame.index[min(trade["j"], len(frame) - 1)]
        if exit_date >= start:
            trades.append({"entry_date": str(entry_date.date()), "exit_date": str(exit_date.date()),
                           "exit_reason": trade["exit_reason"], "lifetime_return_pct": trade["ret"] * 100,
                           "entered_before_window": bool(entry_date < start)})
    return {"returns": returns.loc[start:end],
            "held": pd.Series(result["held"], index=frame.index).loc[start:end],
            "prior_state": prior, "end_state": result["state"],
            "prior_entry_date": str(prior_entry.date()) if prior_entry is not None else None,
            "end_entry_date": str(end_entry.date()) if end_entry is not None else None,
            "trades": trades}


def run_diagnostic(snapshot: Path, r9: Path, output: Path) -> dict:
    if (output / "manifest.json").exists():
        raise FileExistsError("诊断阶段已固化，请使用新的输出目录")
    root = Path(__file__).resolve().parents[2]
    protocol_path = root / "reports/gcn-historical-r10-20260905/protocol.md"
    protocol = protocol_path.read_bytes()
    environment = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    sources, parents = {}, {}
    for stage in ("results", "validation", "stress"):
        folder = r9 / stage
        raw = (folder / "manifest.json").read_bytes()
        meta = json.loads(raw)
        if (meta["research_version"] != "gcn-historical-r9"
                or meta["parent_manifest_sha256"] != SNAPSHOT_SHA or meta["environment"] != environment):
            raise ValueError("r9父证据身份或环境不一致")
        for name, digest in meta["outputs"].items():
            if Path(name).name != name or hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"r9父工件变化：{name}")
        for name, digest in meta["algorithm_sources"].items():
            if not name.startswith("gcn/") or ".." in Path(name).parts or not name.endswith(".py"):
                raise ValueError("r9父源码路径无效")
            source = (root / name).read_bytes()
            if hashlib.sha256(source).hexdigest() != digest:
                raise ValueError(f"r9之后算法源码变化：{name}")
            sources[name] = source
        parents[stage] = raw
    for stage in ("validation", "stress"):
        if json.loads(parents[stage])["training_manifest_sha256"] != hashlib.sha256(parents["results"]).hexdigest():
            raise ValueError("r9训练证据链不匹配")
    if json.loads(parents["stress"])["validation_manifest_sha256"] != hashlib.sha256(parents["validation"]).hexdigest():
        raise ValueError("r9验证证据链不匹配")
    own_source = "gcn/backtest/signal_research_r10.py"
    sources[own_source] = (root / own_source).read_bytes()
    frames, quality = load_snapshot(snapshot)
    anchor, last = pd.Timestamp("2021-08-27"), "2026-08-27"
    windows = [("full5y", "2021-08-27", last)] + [
        (f"year{year}", f"{year}-08-27", last if year == 2025 else f"{year + 1}-08-26")
        for year in range(2021, 2026)]
    rows, symbol_rows, boundaries, daily, trades = [], [], [], [], []
    prepared = {}
    for case, first, end in windows:
        if end not in prepared:
            prepared[end] = {}
            for symbol in CORE:
                frame = compute_ehopt10(frames[symbol].loc[:end], version="v5")
                prepared[end][symbol] = {"frame": frame, "rules": candidate_signals(frame)}
        for rule in RULES:
            for mode in MODES:
                returns = {}
                held = []
                for symbol, bundle in prepared[end].items():
                    result = window_replay(bundle, rule, anchor, pd.Timestamp(first), pd.Timestamp(end), mode)
                    returns[symbol] = result["returns"]
                    held.append(float(result["held"].mean()))
                    labels = {"case": case, "rule": rule, "mode": mode, "symbol": symbol}
                    symbol_rows.append({**labels, **_portfolio_stats(result["returns"]), "exposure": held[-1]})
                    trades.extend({**labels, **trade} for trade in result["trades"])
                    if mode == "carry_mark":
                        prior_date = result["prior_entry_date"]
                        origin = protect = None
                        if prior_date is not None:
                            pos = bundle["frame"].index.get_loc(pd.Timestamp(prior_date)) - 1
                            original = bundle["frame"][["B_SIGNAL", "ICON_JUEFAN"]].iloc[pos].any()
                            origin = "v5" if original else "additional"
                            protect = bool(bundle["rules"][rule]["ENTRY_PROFIT_ENABLED"].iloc[pos])
                        boundaries.append({"case": case, "rule": rule, "symbol": symbol,
                                           "start": first, "end": end,
                                           "prior_entry_date": prior_date, "prior_entry_origin": origin,
                                           "prior_entry_profit_enabled": protect,
                                           "end_entry_date": result["end_entry_date"],
                                           **{"prior_" + k: v for k, v in result["prior_state"].items()},
                                           **{"end_" + k: v for k, v in result["end_state"].items()}})
                matrix = pd.DataFrame(returns)
                if matrix.empty or matrix.isna().any().any() or not np.isfinite(matrix.to_numpy()).all():
                    raise ValueError("窗口共同交易日不完整或期间收益非法")
                portfolio = matrix.mean(axis=1)
                labels = {"case": case, "rule": rule, "mode": mode}
                rows.append({**labels, "start": first, "end": end, "bars": len(portfolio),
                             **_portfolio_stats(portfolio), "exposure": float(np.mean(held))})
                daily.extend({**labels, "date": str(date.date()), "return": value} for date, value in portfolio.items())

    table = pd.DataFrame(rows)
    prior_rows = pd.read_csv(r9 / "stress/comparisons.csv")
    reset = table[table["mode"].eq("reset_liquidate")].merge(prior_rows, on=["case", "rule"], suffixes=("", "_r9"))
    if len(reset) != 12 or any(not np.allclose(reset[key], reset[key + "_r9"], rtol=0, atol=1e-10, equal_nan=True)
                               for key in ("total", "cagr", "mdd", "sharpe")):
        raise ValueError("空仓强平口径未还原r9")
    compose_errors = {}
    for rule in RULES:
        selected = table[table.rule.eq(rule)]
        carry = selected[selected["mode"].eq("carry_mark")].set_index("case")
        years = carry.loc[[f"year{year}" for year in range(2021, 2026)]]
        error = float((1 + years.total / 100).prod() - (1 + carry.loc["full5y", "total"] / 100))
        if abs(error) > 1e-12:
            raise ValueError("连续周年收益未复合还原五年盯市收益")
        compose_errors[rule] = error
        mark = selected[selected["mode"].eq("reset_mark")].set_index("case")
        if not np.allclose(mark.loc[["full5y", "year2021"], ["total", "cagr", "mdd"]],
                           carry.loc[["full5y", "year2021"], ["total", "cagr", "mdd"]], rtol=0, atol=1e-12):
            raise ValueError("起始窗口的空仓与连续盯市不一致")
    r9_decision = json.loads((r9 / "stress/decision.json").read_text())
    decision = {"research_version": "gcn-historical-r10", "status": "diagnostic_only",
                "recommended": "v5", "production_changed": False, "r9_status": r9_decision["status"],
                "r9_failures_unchanged": r9_decision["failures"], "reset_matches_r9": True,
                "continuous_compounding_errors": compose_errors, "evidence": "retrospective_only"}
    for name, raw in sources.items():
        if (root / name).read_bytes() != raw:
            raise ValueError(f"诊断期间源码变化：{name}")
    for stage, raw in parents.items():
        if (r9 / stage / "manifest.json").read_bytes() != raw:
            raise ValueError("诊断期间父manifest变化")
    if protocol_path.read_bytes() != protocol:
        raise ValueError("诊断期间协议变化")
    output.mkdir(parents=True, exist_ok=True)
    for filename, content in {"comparisons.csv": table, "by_symbol.csv": pd.DataFrame(symbol_rows),
                              "boundaries.csv": pd.DataFrame(boundaries), "daily_returns.csv": pd.DataFrame(daily),
                              "closed_trades.csv": pd.DataFrame(trades)}.items():
        content.to_csv(output / filename, index=False)
    (output / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    (output / "protocol.md").write_bytes(protocol)
    for name, raw in sources.items():
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    manifest = {"research_version": "gcn-historical-r10", "parent_manifest_sha256": SNAPSHOT_SHA,
                "r9_manifest_sha256": {stage: hashlib.sha256(raw).hexdigest() for stage, raw in parents.items()},
                "source_quality": quality, "protocol_sha256": hashlib.sha256(protocol).hexdigest(),
                "environment": environment, "algorithm_sources": {name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()},
                "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return decision


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--r9", type=Path, default=Path("reports/gcn-historical-r9-20260905"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_diagnostic(args.snapshot, args.r9, args.output), indent=2, ensure_ascii=False))
