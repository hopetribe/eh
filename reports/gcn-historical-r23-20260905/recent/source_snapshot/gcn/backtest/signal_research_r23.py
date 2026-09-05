"""r23：冻结原v5持仓路径的日内压力/恢复诊断，不生成交易信号。"""
from __future__ import annotations

import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.historical_research import CORE, SNAPSHOT_SHA
from gcn.backtest.signal_research_r22 import COMPONENTS, WINDOWS

R22_MANIFESTS = {
    "training": "aee1b37ad0becc5ade4e66bb8a249b502715febb37b5dc01efb20e6b3510b6c4",
    "validation": "abd706a2a5eeee260c5cde656980d95fb79c0760c3033dbfed24333cf0db28ef",
    "recent": "747e3ab581e0181a01152858a956c30a53219b5a3958a71e262da6c0cad18840",
    "full": "c80a0c27b46af61f088ae2fec672a39f0cb98da6b72599a3e5940acd83d39b2f",
}


EVENT_SCHEMA = {
    **{name: "string" for name in ("first_pressure_date", "first_recovery_date", "first_return_date",
                                    "first_return_recovery_date")},
    **{name: "Int64" for name in ("first_pressure_bar", "first_recovery_bar", "first_return_bar",
                                   "first_return_recovery_bar")},
    **{name: "float64" for name in ("first_pressure_net_pct", "first_return_net_pct")},
    **{name: "boolean" for name in ("first_pressure_after_net_positive", "first_return_after_net_positive")},
}
OBS_EXTRA_SCHEMA = {
    **EVENT_SCHEMA,
    **{name: "boolean" for name in ("pressure", "nonnegative", "ever_positive_intraday", "ever_net_positive")},
    "held_bars": "Int64", "pressure_streak": "Int64", "net_return_pct": "float64",
}
AUDIT_SCHEMA = {
    **EVENT_SCHEMA,
    **{name: "boolean" for name in ("pressure_observed", "recovered", "recovery_censored", "return_observed",
                                      "return_censored")},
    **{name: "Int64" for name in ("recovery_bars", "observed_bars_after_pressure", "pressure_bars",
                                   "max_pressure_streak", "return_recovery_bars", "observed_bars_after_return")},
    **{name: "float64" for name in ("post_pressure_peak_intraday_factor", "post_pressure_peak_net_pct",
                                     "post_return_peak_intraday_factor", "post_return_peak_net_pct")},
}


def audit_paths(trades: pd.DataFrame, paths: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """观察表只用当时及以前价格；交易表的后来恢复/峰值/删失属于事后标签。

    pressure严格<1，首次压力恢复>=1；先曾>1后的首次回落<=1，回落恢复须后续>1。
    首次事件的after_net_positive表示截至该CLOSE曾有扣双边各0.1%费用后的净浮盈。
    不设置epsilon或持续天数参数；保持r22原浮点因子与round_trip解析语义。
    """
    observations, extras = [], []
    closes = paths[paths.observation.eq("close")]
    for trade in trades.itertuples(index=False):
        held = closes[closes.symbol.eq(trade.symbol) & closes.trade_id.eq(trade.trade_id)]
        state = {name: None for name in EVENT_SCHEMA}
        ever_positive = ever_net_positive = False
        streak = 0
        rows = []
        for bar, row in enumerate(held.to_dict("records"), 1):
            factor, date = row["running_intraday_factor"], row["date"]
            net = (row["running_gross_factor"] * .999**2 - 1) * 100
            pressure = factor < 1
            streak = streak + 1 if pressure else 0
            ever_net_positive = ever_net_positive or net > 0
            for event, hit in (("pressure", pressure), ("return", ever_positive and factor <= 1)):
                if hit and state["first_" + event + "_date"] is None:
                    state.update({"first_" + event + "_date": date, "first_" + event + "_bar": bar,
                                  "first_" + event + "_net_pct": net,
                                  "first_" + event + "_after_net_positive": ever_net_positive})
            for event, recovery, hit in (("pressure", "recovery", factor >= 1),
                                         ("return", "return_recovery", factor > 1)):
                first = state["first_" + event + "_bar"]
                if first is not None and bar > first and hit and state["first_" + recovery + "_date"] is None:
                    state["first_" + recovery + "_date"] = date
                    state["first_" + recovery + "_bar"] = bar
            ever_positive = ever_positive or factor > 1
            rows.append({**row, **state, "held_bars": bar, "net_return_pct": net,
                         "pressure": pressure, "nonnegative": factor >= 1, "pressure_streak": streak,
                         "ever_positive_intraday": ever_positive, "ever_net_positive": ever_net_positive})
        observations.extend(rows)
        extra = {**state, "pressure_observed": state["first_pressure_date"] is not None,
                 "recovered": state["first_recovery_date"] is not None,
                 "return_observed": state["first_return_date"] is not None,
                 "pressure_bars": sum(row["pressure"] for row in rows),
                 "max_pressure_streak": max((row["pressure_streak"] for row in rows), default=0)}
        for event, recovery, censored in (("pressure", "recovery", "recovery_censored"),
                                          ("return", "return_recovery", "return_censored")):
            first = state["first_" + event + "_bar"]
            recovered = state["first_" + recovery + "_bar"]
            later = [row for row in rows if first is not None and row["held_bars"] > first]
            extra[censored] = first is not None and recovered is None
            extra[recovery + "_bars"] = recovered - first if recovered is not None else None
            extra["observed_bars_after_" + event] = len(later) if first is not None else None
            for label, column in (("intraday_factor", "running_intraday_factor"), ("net_pct", "net_return_pct")):
                extra["post_" + event + "_peak_" + label] = max((row[column] for row in later), default=np.nan)
        extras.append(extra)
    audited = pd.concat([trades.reset_index(drop=True), pd.DataFrame(extras, columns=AUDIT_SCHEMA).astype(AUDIT_SCHEMA)], axis=1)
    observations = pd.DataFrame(observations, columns=[*paths.columns, *OBS_EXTRA_SCHEMA]).astype(
        {**paths.dtypes.to_dict(), **OBS_EXTRA_SCHEMA})
    return audited, observations


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """固定描述分层，恢复率仅以发生过压力的原交易为分母；不拟合持续天数。"""
    rows = []
    def add(scope, group_by, group, subset):
        n = len(subset)
        pressure = subset[subset.pressure_observed]
        recovered = int(subset.recovered.sum())
        rows.append({"scope": scope, "group_by": group_by, "group": group, "trades": n,
                     "symbols": int(subset.symbol.nunique()), "wins": int(subset.return_pct.gt(0).sum()),
                     "win_rate_pct": subset.return_pct.gt(0).mean()*100 if n else np.nan,
                     "mean_return_pct": subset.return_pct.mean(), "median_return_pct": subset.return_pct.median(),
                     "pressure_trades": len(pressure), "pressure_winners": int(pressure.return_pct.gt(0).sum()),
                     "recovered_trades": recovered, "recovery_censored_trades": int(subset.recovery_censored.sum()),
                     "recovery_rate_pct": recovered/len(pressure)*100 if len(pressure) else np.nan,
                     "return_trades": int(subset.return_observed.sum()),
                     "return_censored_trades": int(subset.return_censored.sum()),
                     "final_negative_intraday_trades": int(subset.intraday_factor.lt(1).sum()),
                     "final_negative_intraday_winners": int((subset.intraday_factor.lt(1) & subset.return_pct.gt(0)).sum()),
                     "median_first_pressure_bar": pressure.first_pressure_bar.median() if len(pressure) else np.nan,
                     "median_max_pressure_streak": pressure.max_pressure_streak.median() if len(pressure) else np.nan,
                     "median_recovery_bars": subset.recovery_bars.dropna().median() if subset.recovered.any() else np.nan})
    for scope, subset in (("all", trades), ("B", trades[trades.entry_kind.eq("B")]),
                          ("JF", trades[trades.entry_kind.eq("JF")])):
        add(scope, "all", "all", subset)
        for reason in ("signal", "trail", "terminal"):
            add(scope, "exit_reason", reason, subset[subset.exit_reason.eq(reason)])
        for group, mask in (("win", subset.return_pct.gt(0)), ("nonpositive", subset.return_pct.le(0))):
            add(scope, "outcome", group, subset[mask])
        for group, mask in (("none", ~subset.pressure_observed),
                            ("before_net_positive", subset.first_pressure_after_net_positive.eq(False)),
                            ("after_net_positive", subset.first_pressure_after_net_positive.eq(True))):
            add(scope, "first_pressure", group, subset[mask])
        for group, mask in (("no_pressure", ~subset.pressure_observed), ("recovered", subset.recovered),
                            ("censored", subset.recovery_censored)):
            add(scope, "recovery", group, subset[mask])
        for group, mask in (("no_return", ~subset.return_observed),
                            ("recovered", subset.first_return_recovery_date.notna()),
                            ("censored", subset.return_censored)):
            add(scope, "return_recovery", group, subset[mask])
        for quality in (False, True):
            add(scope, "source_trusted", str(quality).lower(), subset[subset.source_trusted.eq(quality)])
        for symbol in sorted(set(CORE) | set(trades.symbol)):
            add(scope, "symbol", symbol, subset[subset.symbol.eq(symbol)])
    b = trades[trades.entry_kind.eq("B")]
    for col in COMPONENTS:
        add("B", "source", col, b[b[col]])
    add("B", "source", "multiple", b[b[list(COMPONENTS)].sum(axis=1).gt(1)])
    return pd.DataFrame(rows).astype({col: "string" for col in ("scope", "group_by", "group")})


def run_diagnostic(r22: Path, output: Path, *, window: str = "training") -> dict:
    """只读已冻结的对应r22窗口，逐项核验其内容/源码；不重新计算行情或成交。"""
    selected = next((list(spec) for spec in WINDOWS if spec[0] == window), None)
    if selected is None:
        raise ValueError("仅允许协议固定窗口")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("诊断目录非空，请使用新的输出目录")
    root = Path(__file__).resolve().parents[2]
    names = ("gcn/backtest/signal_research_r23.py", "gcn/backtest/signal_research_r22.py",
             "gcn/backtest/historical_research.py")
    sources = {name: (root / name).read_bytes() for name in names}
    protocol_path = root / "reports/gcn-historical-r23-20260905/protocol.md"
    protocol = protocol_path.read_bytes()
    folder = r22 / window
    inputs = {"manifest.json": (folder / "manifest.json").read_bytes()}
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    if digest(inputs["manifest.json"]) != R22_MANIFESTS[window]:
        raise ValueError("r22冻结manifest不匹配")
    parent = json.loads(inputs["manifest.json"])
    if (parent["parent_manifest_sha256"] != SNAPSHOT_SHA or parent["window"] != selected
            or parent["research_version"] != "gcn-historical-r22"):
        raise ValueError("r22父输入或固定窗口不匹配")
    for name, expected in parent["outputs"].items():
        inputs[name] = (folder / name).read_bytes()
        if digest(inputs[name]) != expected:
            raise ValueError(f"r22输入内容变化：{name}")
    input_sources = {}
    for name, expected in parent["algorithm_sources"].items():
        input_sources[name] = (folder / "source_snapshot" / name).read_bytes()
        if digest(input_sources[name]) != expected:
            raise ValueError(f"r22冻结源码变化：{name}")
    read = lambda name: pd.read_csv(io.BytesIO(inputs[name]), float_precision="round_trip")
    original, paths, checks = read("trades.csv"), read("paths.csv"), read("reconciliation.csv")
    original["source_trusted"] = original.symbol.map(parent["source_quality"])
    trades, observations = audit_paths(original, paths)
    checks["held_close_rows"] = checks.symbol.map(observations.groupby("symbol").size()).fillna(0).astype(int)
    expected = checks.symbol.map(trades.groupby("symbol").hold_bars.sum()).fillna(0).astype(int)
    checks["pressure_path_reconciled"] = checks.held_close_rows.eq(expected)
    if not checks.pressure_path_reconciled.all() or set(checks.symbol) != set(CORE):
        raise ValueError("r22原持仓CLOSE数量或核心池不一致")
    tables = {"trades": trades, "observations": observations, "summary": summarize_trades(trades),
              "reconciliation": checks}
    decision = {"research_version": "gcn-historical-r23", "stage": "diagnostic_only", "recommended": "v5",
                "production_changed": False, "core": CORE, "window": selected,
                "input": "frozen r22 original v5 trades and paths; no new simulator or signal calculation",
                "observations": "held CLOSE only; original OPEN exit adds no intraday observation",
                "boundary": "strict factor < 1 pressure; recovery >= 1; prior > 1 then <= 1 return; later > 1 return recovery",
                "net_positive": "ever positive through current CLOSE, including 0.1% cost on each side; not factor 1",
                "precision": "r22 IEEE float factors, round_trip CSV parsing, no epsilon or fitted duration threshold",
                "censoring": "no recovery before original exit is right-censored, not evidence of no later recovery",
                "posthoc": "trade outcomes, recovery and strictly later held peaks are descriptive, not causal filters",
                "window_policy": "independent flat start, terminal liquidation; overlapping windows are not independent"}
    for name, raw in sources.items():
        if (root / name).read_bytes() != raw:
            raise ValueError(f"计算期间源码变化：{name}")
    if protocol_path.read_bytes() != protocol:
        raise ValueError("计算期间协议变化")
    for name, raw in inputs.items():
        if (folder / name).read_bytes() != raw:
            raise ValueError(f"计算期间r22输入变化：{name}")
    for name, raw in input_sources.items():
        if (folder / "source_snapshot" / name).read_bytes() != raw:
            raise ValueError(f"计算期间r22冻结源码变化：{name}")
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if "window" not in table:
            table.insert(0, "window", window)
        table.to_csv(output / (name + ".csv"), index=False)
    (output / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    (output / "protocol.md").write_bytes(protocol)
    for prefix, files in (("source_snapshot", sources), ("input_snapshot", inputs),
                           ("input_source_snapshot", input_sources)):
        for name, raw in files.items():
            target = output / prefix / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    manifest = {"research_version": "gcn-historical-r23", "window": selected,
                "parent_manifest_sha256": SNAPSHOT_SHA, "input_manifest_sha256": R22_MANIFESTS[window],
                "source_quality": parent["source_quality"], "input_environment": parent["environment"],
                "input_files": {name: digest(raw) for name, raw in inputs.items()},
                "input_algorithm_sources": parent["algorithm_sources"],
                "protocol_sha256": digest(protocol),
                "algorithm_sources": {name: digest(raw) for name, raw in sources.items()},
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "outputs": {path.name: digest(path.read_bytes()) for path in sorted(output.iterdir()) if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return decision


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r22", type=Path, default=Path("reports/gcn-historical-r22-20260905"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", choices=[spec[0] for spec in WINDOWS], default="training")
    args = parser.parse_args()
    print(json.dumps(run_diagnostic(args.r22, args.output, window=args.window), indent=2, ensure_ascii=False))
