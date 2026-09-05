"""r25：冻结成交后的空仓信号和真实再入场诊断，不生成新订单。"""
from __future__ import annotations

import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r14 import COMPONENTS, trace_setups
from gcn.backtest.historical_research import CORE, SNAPSHOT_SHA, load_snapshot
from gcn.recipes.gcn_main import compute_ehopt10

WINDOWS = (("training", "2021-08-27", "2024-08-26"), ("validation", "2024-08-27", "2025-08-26"))
R24_MANIFESTS = {
    "training": "6055659eb974ef8cbb82dcb8d446d741fdafd21a40e38c0e1482983dbabc336f",
    "validation": "d6b1ba32e6aac5cebb80bca09495e7909c8ba7d896f8c32ea9cdbae7d513dcf3",
}

SIGNALS = ("B_ALL_RAW", "JF_RAW", "B_SETUP", "B_ENTRY_SIGNAL", "B_SETUP_EXPIRED",
           "B_SIGNAL", "ICON_JUEFAN", "S_SIGNAL")
COUNT_COLUMNS = {"raw_b_rows": "B_ALL_RAW", "raw_jf_rows": "JF_RAW", "setup_rows": "B_SETUP",
                 "confirmed_b_rows": "B_SIGNAL", "expired_rows": "B_SETUP_EXPIRED",
                 "tradable_jf_rows": "ICON_JUEFAN", "s_rows": "S_SIGNAL",
                 "suppressed_b_rows": "raw_b_suppressed", "no_raw_buy_rows": "no_raw_buy"}
COUNT_NAMES = (*COUNT_COLUMNS, "raw_buy_rows", "tradable_buy_rows")

EPISODE_SCHEMA = {
    **{col: "string" for col in ("symbol", "episode_id", "entry_date", "exit_date", "end_kind",
                                  "first_reference_reclaim_date", "next_entry_date", "next_signal_date",
                                  "next_entry_kind", "next_setup_date", "next_exit_date",
                                  "original_exit_date", "original_exit_reason")},
    **{col: "float64" for col in ("reference_price", "return_pct", "next_return_pct",
                                   "original_return_pct", "chain_return_pct")},
    **{col: "Int64" for col in ("flat_bars", "original_horizon_bars", *COUNT_NAMES,
                                 *("original_horizon_" + col for col in COUNT_NAMES))},
    **{col: "boolean" for col in ("source_trusted", "next_entry_b", "next_entry_jf",
                                   "chain_same_original_end", *("next_" + c for c in COMPONENTS))},
}


def signal_states(frame: pd.DataFrame) -> pd.DataFrame:
    """保留全前史；Setup将来结局仅在解决当天出现，不回填此前pending行。"""
    trace = trace_setups(frame)
    for status, column in (("confirmed", "B_ENTRY_SIGNAL"), ("expired", "B_SETUP_EXPIRED")):
        replay = pd.Series(False, index=frame.index, dtype=bool)
        replay.iloc[trace.loc[trace.status.eq(status), "resolution_i"].astype(int).tolist()] = True
        if not replay.equals(frame[column]):
            raise ValueError(f"Setup还原与{column}不一致")
    if not frame.B_SIGNAL.equals(frame.B_ENTRY_SIGNAL):
        raise ValueError("B_SIGNAL与确认不一致")
    result = frame[["OPEN", "CLOSE", "MID", *SIGNALS]].copy()
    result["raw_b_count20"] = frame.B_ALL_RAW.rolling(20, min_periods=1).sum().astype("Int64")
    result["raw_b_suppressed"] = frame.B_ALL_RAW & ~frame.B_SETUP & result.raw_b_count20.gt(1)
    result["no_raw_buy"] = ~(frame.B_ALL_RAW | frame.JF_RAW)
    for col in ("pending_setup_date", "resolved_setup_date", "resolved_setup_status"):
        result[col] = pd.Series(pd.NA, index=frame.index, dtype="string")
    result["pending_setup_age"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for prefix in ("pending_", "resolved_"):
        for col in COMPONENTS:
            result[prefix + col] = False
    for row in trace.itertuples():
        start = int(row.setup_i)
        stop = len(frame) if pd.isna(row.resolution_i) else int(row.resolution_i)
        pending = frame.index[start:stop]
        result.loc[pending, "pending_setup_date"] = row.setup_date
        result.loc[pending, "pending_setup_age"] = np.arange(stop-start)
        for col in COMPONENTS:
            result.loc[pending, "pending_" + col] = getattr(row, col)
        if stop < len(frame):
            date = frame.index[stop]
            result.loc[date, "resolved_setup_date"] = row.setup_date
            result.loc[date, "resolved_setup_status"] = row.status
            for col in COMPONENTS:
                result.loc[date, "resolved_" + col] = getattr(row, col)
    return result


def _signal_counts(view: pd.DataFrame) -> dict:
    return {**{name: int(view[col].sum()) for name, col in COUNT_COLUMNS.items()},
            "raw_buy_rows": int((view.B_ALL_RAW | view.JF_RAW).sum()),
            "tradable_buy_rows": int((view.B_SIGNAL | view.ICON_JUEFAN).sum())}


def audit_symbol(symbol: str, frame: pd.DataFrame, trades: pd.DataFrame, *,
                 source_trusted: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """仅读取已发生退出/再入场；未来结局不进入逐CLOSE观察，原持仓区间另作事后标签。"""
    states = signal_states(frame)
    schema = {
        **{col: "string" for col in ("symbol", "episode_id", "entry_date", "exit_date", "date",
                                      "first_reference_reclaim_date")},
        "flat_bar": "Int64", "reference_price": "float64", "reference_reclaimed": "boolean",
        **states.dtypes.to_dict(),
    }
    episodes, observations = [], []
    own = trades[trades.symbol.eq(symbol)].sort_values("entry_date")
    if own.duplicated(["rule", "entry_date"]).any():
        raise ValueError("重复入场不能唯一配对")
    original = own[own.rule.eq("v5")]
    candidate = own[own.rule.eq("JF-joint-pressure")]
    last = frame.index[-1].date().isoformat() if len(frame) else ""
    for trade in candidate[candidate.exit_reason.eq("joint_pressure") & candidate.exit_date.le(last)].itertuples():
        matched = original[original.entry_date.eq(trade.entry_date)]
        if len(matched) != 1 or matched.exit_date.iloc[0] <= trade.exit_date:
            raise ValueError("共同压力退出与原v5持仓无法配对为严格提前退出")
        old = matched.iloc[0]
        entry = frame.index.get_loc(pd.Timestamp(trade.entry_date))
        exited = frame.index.get_loc(pd.Timestamp(trade.exit_date))
        if (entry == 0 or exited <= entry or trade.entry_b or not trade.entry_jf
                or states.B_SIGNAL.iloc[entry-1] or not states.ICON_JUEFAN.iloc[entry-1]):
            raise ValueError("共同压力事件必须来自实际纯JF持仓及此前信号")
        following = candidate[candidate.entry_date.gt(trade.entry_date)]
        next_trade = following.iloc[0] if len(following) and following.entry_date.iloc[0] <= last else None
        stop = frame.index.get_loc(pd.Timestamp(next_trade.entry_date)) if next_trade is not None else len(frame)
        if stop <= exited:
            raise ValueError("真实再入场不得与原持仓重叠")
        flat = states.iloc[exited:stop]
        reference = float(frame.OPEN.iloc[entry] / .999**2)
        episode_id = symbol + ":" + trade.entry_date + ":" + trade.exit_date
        first = None
        for bar, (date, state) in enumerate(flat.iterrows(), 1):
            date = date.date().isoformat()
            if first is None and state.CLOSE >= reference:
                first = date
            observations.append({"symbol": symbol, "episode_id": episode_id, "entry_date": trade.entry_date,
                                 "exit_date": trade.exit_date, "date": date, "flat_bar": bar,
                                 "reference_price": reference, "reference_reclaimed": state.CLOSE >= reference,
                                 "first_reference_reclaim_date": first, **state.to_dict()})
        old_known = old.exit_date <= last
        row = {"symbol": symbol, "episode_id": episode_id, "entry_date": trade.entry_date,
               "exit_date": trade.exit_date, "source_trusted": source_trusted, "reference_price": reference,
               "return_pct": trade.return_pct, "flat_bars": len(flat), "first_reference_reclaim_date": first,
               "end_kind": "reentry" if next_trade is not None else "right_censored",
               **_signal_counts(flat),
               "original_exit_date": old.exit_date if old_known else None,
               "original_exit_reason": old.exit_reason if old_known else None,
               "original_return_pct": old.return_pct if old_known else np.nan,
               "original_horizon_bars": int(((states.index >= pd.Timestamp(trade.exit_date)) &
                                              (states.index < pd.Timestamp(old.exit_date))).sum()) if old_known else None,
               "next_entry_b": False, "next_entry_jf": False,
               **{"next_" + c: False for c in COMPONENTS}}
        if old_known:
            old_view = states[(states.index >= pd.Timestamp(trade.exit_date)) &
                              (states.index < pd.Timestamp(old.exit_date))]
            row.update({"original_horizon_" + key: value for key, value in _signal_counts(old_view).items()})
        if next_trade is not None:
            signal = states.iloc[stop-1]
            b, jf = bool(signal.B_SIGNAL), bool(signal.ICON_JUEFAN)
            if not (b or jf) or b != next_trade.entry_b or jf != next_trade.entry_jf:
                raise ValueError("下一实际入场与信号日来源不一致")
            completed = next_trade.exit_date <= last
            row.update(next_entry_date=next_trade.entry_date, next_signal_date=states.index[stop-1].date().isoformat(),
                       next_entry_kind="B" if b else "JF", next_entry_b=b, next_entry_jf=jf,
                       next_setup_date=signal.resolved_setup_date if b else None,
                       **{"next_" + c: bool(signal["resolved_" + c]) if b else False for c in COMPONENTS},
                       next_exit_date=next_trade.exit_date if completed else None,
                       next_return_pct=next_trade.return_pct if completed else np.nan,
                       chain_return_pct=((1+trade.return_pct/100)*(1+next_trade.return_pct/100)-1)*100 if completed else np.nan,
                       chain_same_original_end=(next_trade.exit_date == old.exit_date and
                                                (next_trade.exit_reason == "terminal") == (old.exit_reason == "terminal"))
                       if completed and old_known else None)
        episodes.append(row)
    return (pd.DataFrame(episodes, columns=EPISODE_SCHEMA).astype(EPISODE_SCHEMA),
            pd.DataFrame(observations, columns=schema).astype(schema))


def summarize_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    """固定描述分层；多来源可重叠，未来未完成收益不计入胜负分母。"""
    rows = []
    def add(group_by, group, subset):
        aligned = subset.chain_same_original_end.fillna(False)
        rows.append({"group_by": group_by, "group": group, "episodes": len(subset),
                     "symbols": int(subset.symbol.nunique()),
                     "flat_bars": int(subset.flat_bars.sum()),
                     "original_horizon_bars": int(subset.original_horizon_bars.sum()),
                     **{key: int(subset[key].sum()) for key in
                        (*COUNT_NAMES, *("original_horizon_" + c for c in COUNT_NAMES))},
                     "original_completed": int(subset.original_return_pct.notna().sum()),
                     "original_wins": int(subset.original_return_pct.gt(0).sum()),
                     "reference_reclaims": int(subset.first_reference_reclaim_date.notna().sum()),
                     "reference_censored": int(subset.first_reference_reclaim_date.isna().sum()),
                     "has_reentry": int(subset.next_entry_date.notna().sum()),
                     "right_censored": int(subset.end_kind.eq("right_censored").sum()),
                     "next_completed": int(subset.next_return_pct.notna().sum()),
                     "next_wins": int(subset.next_return_pct.gt(0).sum()),
                     "aligned_chains": int(aligned.sum()),
                     "worse_aligned_chains": int((aligned & subset.chain_return_pct.lt(subset.original_return_pct)).sum())})
    add("all", "all", episodes)
    for symbol in sorted(set(CORE) | set(episodes.symbol)):
        add("symbol", symbol, episodes[episodes.symbol.eq(symbol)])
    for group, mask in (("win", episodes.original_return_pct.gt(0)),
                        ("nonpositive", episodes.original_return_pct.le(0)),
                        ("unobserved", episodes.original_return_pct.isna())):
        add("original_outcome", group, episodes[mask])
    for kind in ("B", "JF", "no_reentry"):
        mask = episodes.next_entry_kind.isna() if kind == "no_reentry" else episodes.next_entry_kind.eq(kind)
        add("next_entry", kind, episodes[mask])
    for quality in (False, True):
        add("source_trusted", str(quality).lower(), episodes[episodes.source_trusted.eq(quality)])
    for col in COMPONENTS:
        add("next_source", col, episodes[episodes["next_" + col]])
    sources = episodes[["next_" + c for c in COMPONENTS]].sum(axis=1)
    add("next_source", "multiple", episodes[sources.gt(1)])
    add("next_source", "B_without_source", episodes[episodes.next_entry_b & sources.eq(0)])
    return pd.DataFrame(rows).astype({"group_by": "string", "group": "string"})


def run_diagnostic(snapshot: Path, r24: Path, output: Path, *, window: str = "training") -> dict:
    """绑定两窗原订单/源码及父行情，只重建原生信号状态，不调用成交模拟。"""
    selected = next((list(spec) for spec in WINDOWS if spec[0] == window), None)
    if selected is None:
        raise ValueError("仅允许r25固定训练/验证窗口")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("诊断目录非空，请使用新的输出目录")
    root = Path(__file__).resolve().parents[2]
    captured = {}
    def capture(path):
        raw = path.read_bytes()
        captured[path] = raw
        return raw
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    names = ("gcn/backtest/signal_research_r25.py", "gcn/backtest/signal_research_r14.py",
             "gcn/backtest/historical_research.py", "gcn/backtest/signal_audit.py", "gcn/backtest/engine.py",
             "gcn/recipes/gcn_main.py", "gcn/core/tdx.py", "gcn/core/indicators.py")
    sources = {name: capture(root / name) for name in names}
    protocol = capture(root / "reports/gcn-historical-r25-20260905/protocol.md")
    if digest(protocol) != "91fdea2a82838f83de456d33a4625f83450cd8859034749f7aaa5d89167d8b1e":
        raise ValueError("r25冻结协议变化")
    inputs, input_sources, parents = {}, {}, {}
    for stage, directory in (("training", "results"), ("validation", "validation")):
        folder = r24 / directory
        raw = capture(folder / "manifest.json")
        if digest(raw) != R24_MANIFESTS[stage]:
            raise ValueError("r24冻结manifest不匹配")
        parent = json.loads(raw); parents[stage] = parent
        if (parent["research_version"] != "gcn-historical-r24" or parent["parent_manifest_sha256"] != SNAPSHOT_SHA
                or parent["entry_joint_pressure_col"] != "ENTRY_JOINT_PRESSURE"
                or parent["joint_pressure_reference_cost"] != .001 or parent["joint_pressure_factor_threshold"] != 1.):
            raise ValueError("r24固定输入或配置不一致")
        inputs[stage + "/manifest.json"] = raw
        for name, expected in parent["outputs"].items():
            raw = capture(folder / name)
            if digest(raw) != expected:
                raise ValueError(f"r24输入内容变化：{stage}/{name}")
            inputs[stage + "/" + name] = raw
        for name, expected in parent["algorithm_sources"].items():
            raw = capture(folder / "source_snapshot" / name)
            if digest(raw) != expected:
                raise ValueError(f"r24冻结源码变化：{name}")
            if name in sources and sources[name] != raw:
                raise ValueError(f"原生信号源码与r24不一致：{name}")
            input_sources[stage + "/" + name] = raw
    if parents["validation"]["training_manifest_sha256"] != R24_MANIFESTS["training"]:
        raise ValueError("r24训练/验证链不一致")
    parent_files = {"manifest.json": capture(snapshot / "manifest.json")}
    if digest(parent_files["manifest.json"]) != SNAPSHOT_SHA:
        raise ValueError("父manifest摘要不匹配")
    for spec in json.loads(parent_files["manifest.json"])["inputs"].values():
        for key, sha_key in (("snapshot_path", "sha256"), ("metadata_snapshot_path", "metadata_sha256")):
            name = spec[key]; raw = capture(snapshot / name)
            if digest(raw) != spec[sha_key]:
                raise ValueError(f"父输入快照变化：{name}")
            parent_files[name] = raw
    frames, quality = load_snapshot(snapshot)
    environment = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    if any(parent["source_quality"] != quality or parent["environment"] != environment for parent in parents.values()):
        raise ValueError("r24来源质量或运行环境不一致")
    trades = pd.read_csv(io.BytesIO(inputs[window + "/trades.csv"]), float_precision="round_trip")
    all_episodes, all_observations, checks = [], [], []
    start, end = pd.Timestamp(selected[1]), pd.Timestamp(selected[2])
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:end], version="v5", diagnostics=True)
        episodes, observations = audit_symbol(symbol, frame, trades, source_trusted=quality[symbol])
        own = trades[trades.symbol.eq(symbol)]
        candidate = own[own.rule.eq("JF-joint-pressure")].sort_values("entry_date")
        joint = candidate[candidate.exit_reason.eq("joint_pressure")]
        expected_bars = 0
        for trade in joint.itertuples():
            if not start <= pd.Timestamp(trade.entry_date) < pd.Timestamp(trade.exit_date) <= end:
                raise ValueError("r24订单超出固定窗口")
            next_entries = candidate[candidate.entry_date.gt(trade.entry_date)]
            stop = pd.Timestamp(next_entries.entry_date.iloc[0]) if len(next_entries) else end + pd.Timedelta(days=1)
            expected_bars += int(((frame.index >= pd.Timestamp(trade.exit_date)) & (frame.index < stop)).sum())
        reconciled = len(episodes) == len(joint) and len(observations) == expected_bars
        if not reconciled:
            raise ValueError(f"{symbol}: 原退出及空仓CLOSE数量不一致")
        checks.append({"symbol": symbol, "v5_trades": int(own.rule.eq("v5").sum()),
                       "candidate_trades": len(candidate), "joint_exits": len(joint), "episodes": len(episodes),
                       "flat_closes": len(observations), "expected_flat_closes": expected_bars, "reconciled": reconciled})
        all_episodes.append(episodes); all_observations.append(observations)
    episodes = pd.concat(all_episodes, ignore_index=True)
    observations = pd.concat(all_observations, ignore_index=True)
    if len(episodes) != {"training": 9, "validation": 4}[window]:
        raise ValueError("冻结共同压力退出总数不一致")
    tables = {"episodes": episodes, "observations": observations, "summary": summarize_episodes(episodes),
              "reconciliation": pd.DataFrame(checks)}
    decision = {"research_version": "gcn-historical-r25", "stage": "diagnostic_only", "recommended": "v5",
                "production_changed": False, "window": selected, "core": CORE,
                "input": "both frozen r24 windows and parent prices; no order simulation or new candidate returns",
                "observations": "exit-day CLOSE through the CLOSE before actual next entry OPEN, or window end",
                "signal_state": "full-history raw counts and causal Setup state; no exit-time reset or future backfill",
                "reference": "original entry OPEN / 0.999^2; not a new fill or recovery of realized losses",
                "posthoc": "original horizon and completed trade outcomes belong only to episode-level attribution",
                "secondary_horizon": "exit-date inclusive, original exit-date exclusive, may extend beyond reentry",
                "censoring": "no observed reference reclaim or reentry by episode end is right-censored",
                "stopping": "r24 stays rejected; no later promotion windows or threshold fitting"}
    for path, raw in captured.items():
        if path.read_bytes() != raw:
            raise ValueError(f"计算期间输入或源码变化：{path.name}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("诊断目录非空，请使用新的输出目录")
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / (name + ".csv"), index=False)
    (output / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    (output / "protocol.md").write_bytes(protocol)
    for prefix, files in (("source_snapshot", sources), ("input_snapshot", inputs),
                          ("input_source_snapshot", input_sources), ("parent_snapshot", parent_files)):
        for name, raw in files.items():
            target = output / prefix / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    manifest = {"research_version": "gcn-historical-r25", "window": selected,
                "parent_manifest_sha256": SNAPSHOT_SHA, "input_manifest_sha256": R24_MANIFESTS,
                "source_quality": quality, "input_environment": parents["training"]["environment"],
                "protocol_sha256": digest(protocol), "environment": environment,
                **{key: {name: digest(raw) for name, raw in files.items()} for key, files in
                   (("algorithm_sources", sources), ("input_files", inputs), ("input_algorithm_sources", input_sources),
                    ("parent_files", parent_files))},
                "outputs": {p.name: digest(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return decision


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--r24", type=Path, default=Path("reports/gcn-historical-r24-20260905"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", choices=[spec[0] for spec in WINDOWS], default="training")
    args = parser.parse_args()
    print(json.dumps(run_diagnostic(args.snapshot, args.r24, args.output, window=args.window), indent=2, ensure_ascii=False))
