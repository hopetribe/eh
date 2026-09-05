"""r20：原v5纯绝反结构风险和净保本路径诊断，不生成策略。"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from gcn.backtest.historical_research import CORE, SNAPSHOT_SHA, load_snapshot
from gcn.backtest.signal_research_r14 import WINDOWS
from gcn.backtest.signal_research_r16 import audit_trades
from gcn.core.tdx import LLV
from gcn.recipes.gcn_main import compute_ehopt10


EXTRA_SCHEMA = {
    **{col: "string" for col in ("entry_kind", "risk_status", "observation_state",
                                 "first_net_positive_date", "first_return_to_be_date",
                                 "first_trail_covers_be_date", "post_return_peak_date")},
    **{col: "float64" for col in ("signal_close", "signal_high", "base_low3", "risk_r", "risk_pct",
                                  "entry_gap_pct", "break_even", "post_return_peak_close")},
    "post_return_bars": "Int64", "net_profit_to_loss": "boolean",
}
PATH_SCHEMA = {
    **{col: "string" for col in ("symbol", "trade_id", "date")},
    **{col: "float64" for col in ("close", "running_peak_close", "running_trough_close",
                                  "trail_reference", "close_gross_pct", "close_net_pct",
                                  "mfe_close_r", "mae_close_r")},
    **{col: "boolean" for col in ("above_entry", "net_positive", "ever_net_positive",
                                  "returned_to_be", "trail_covers_be")},
}


def audit_frame(symbol: str, frame: pd.DataFrame, start: pd.Timestamp,
                end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """复用r16订单；路径表仅含收盘当时可知值，交易表含显式事后标签。

    R为实际入场OPEN减信号日原生三根底部；最大有利/不利R为非负距离。
    B（包括B+JF碰撞）不赋予绝反风险，且不生成其逐根路径。
    """
    frame = frame.loc[:end]
    base = frame.loc[start:end]
    original, result = audit_trades(symbol, frame, start, end)
    lows = LLV(frame.LOW, 3)
    extras, paths = [], []
    for row, order in zip(original.to_dict("records"), result["trades"]):
        i, j = order["i"], order["j"]
        terminal = row["exit_reason"] == "terminal"
        state = ("pending_signal" if base.S_SIGNAL.iloc[-1] else
                 "pending_trail" if base.CLOSE.iloc[-1] <= row["peak_close"] * .8 else "open") if terminal else "closed"
        extra = {"entry_kind": "B" if row["entry_b"] else "JF", "risk_status": "not_applicable",
                 "observation_state": state}
        if not row["entry_b"]:
            signal = frame.loc[pd.Timestamp(row["entry_signal_date"])]
            floor = float(lows.loc[pd.Timestamp(row["entry_signal_date"])])
            entry = row["entry_open"]
            risk = entry - floor
            valid = np.isfinite(risk) and risk > 0
            break_even = entry / .999**2
            held = base.CLOSE.iloc[i:j]
            peaks, troughs = held.cummax(), held.cummin()
            positive = held.gt(break_even)
            ever = positive.cummax()
            returns_to_be = ever & held.le(break_even)
            returned = returns_to_be.cummax()
            covers = (peaks * .8).ge(break_even)
            first_positive = held.index[np.flatnonzero(positive)[0]] if positive.any() else None
            first_return = held.index[np.flatnonzero(returns_to_be)[0]] if returns_to_be.any() else None
            first_cover = held.index[np.flatnonzero(covers)[0]] if covers.any() else None
            after_return = held[held.index > first_return] if first_return is not None else held.iloc[:0]
            later_peak_date = after_return.idxmax() if len(after_return) else None
            extra.update(signal_close=float(signal.CLOSE), signal_high=float(signal.HIGH),
                         base_low3=floor, risk_r=risk, risk_pct=risk/entry*100,
                         risk_status="valid" if valid else "nonpositive" if np.isfinite(risk) else "invalid",
                         entry_gap_pct=(entry/float(signal.CLOSE)-1)*100, break_even=break_even,
                         first_net_positive_date=first_positive.date().isoformat() if first_positive is not None else None,
                         first_return_to_be_date=first_return.date().isoformat() if first_return is not None else None,
                         first_trail_covers_be_date=first_cover.date().isoformat() if first_cover is not None else None,
                         post_return_bars=len(after_return) if first_return is not None else None,
                         post_return_peak_close=float(after_return.max()) if len(after_return) else np.nan,
                         post_return_peak_date=later_peak_date.date().isoformat() if later_peak_date is not None else None,
                         net_profit_to_loss=bool(positive.any() and row["return_pct"] <= 0))
            for pos, (date, close) in enumerate(held.items()):
                peak, trough = peaks.iloc[pos], troughs.iloc[pos]
                paths.append({"symbol": symbol, "trade_id": row["trade_id"], "date": date.date().isoformat(),
                              "close": close, "running_peak_close": peak, "running_trough_close": trough,
                              "trail_reference": peak*.8, "close_gross_pct": (close/entry-1)*100,
                              "close_net_pct": (close/entry*.999**2-1)*100,
                              "mfe_close_r": max(peak-entry, 0)/risk if valid else np.nan,
                              "mae_close_r": max(entry-trough, 0)/risk if valid else np.nan,
                              "above_entry": close > entry, "net_positive": positive.iloc[pos],
                              "ever_net_positive": ever.iloc[pos], "returned_to_be": returned.iloc[pos],
                              "trail_covers_be": covers.iloc[pos]})
        extras.append(extra)
    extra_table = pd.DataFrame(extras, columns=list(EXTRA_SCHEMA)).astype(EXTRA_SCHEMA)
    trades = pd.concat([original, extra_table], axis=1)
    path_table = pd.DataFrame(paths, columns=list(PATH_SCHEMA)).astype(PATH_SCHEMA)
    check = {"symbol": symbol, "trades": len(trades), "b_trades": int(trades.entry_b.sum()),
             "jf_trades": int(trades.entry_kind.eq("JF").sum()), "path_bars": len(path_table),
             "reconciled": len(path_table) == int(trades.loc[trades.entry_kind.eq("JF"), "hold_bars"].sum())}
    if not check["reconciled"]:
        raise ValueError(f"{symbol}: 原持仓路径数量不一致")
    return trades, path_table, check


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """固定分层而非阈值搜索；B只提供订单数量和净收益对照。"""
    rows = []

    def add(scope, group_by, group, subset):
        n = len(subset)
        row = {"scope": scope, "group_by": group_by, "group": group, "trades": n,
               "symbols": int(subset.symbol.nunique()), "wins": int(subset.return_pct.gt(0).sum()),
               "win_rate_pct": subset.return_pct.gt(0).mean()*100 if n else np.nan,
               "mean_return_pct": subset.return_pct.mean(), "median_return_pct": subset.return_pct.median(),
               "worst_return_pct": subset.return_pct.min()}
        if scope == "JF":
            valid = subset[subset.risk_status.eq("valid")]
            row.update(valid_risk=len(valid), median_risk_pct=valid.risk_pct.median(),
                       median_entry_gap_pct=subset.entry_gap_pct.median(),
                       ever_net_positive=int(subset.first_net_positive_date.notna().sum()),
                       returned_to_be=int(subset.first_return_to_be_date.notna().sum()),
                       trail_covers_be=int(subset.first_trail_covers_be_date.notna().sum()),
                       net_profit_to_loss=int(subset.net_profit_to_loss.sum()),
                       terminal=int(subset.exit_reason.eq("terminal").sum()),
                       median_peak_close_gain_pct=subset.peak_close_gain_pct.median(),
                       median_giveback_pp=subset.giveback_pp.median())
        rows.append(row)

    jf = trades[trades.entry_kind.eq("JF")]
    for scope, subset in (("all", trades), ("B", trades[trades.entry_kind.eq("B")]), ("JF", jf)):
        add(scope, "all", "all", subset)
    for field, values in (("exit_reason", ("signal", "trail", "terminal")),
                          ("risk_status", ("valid", "nonpositive", "invalid"))):
        for value in values:
            add("JF", field, value, jf[jf[field].eq(value)])
    for group, mask in (("win", jf.return_pct.gt(0)), ("nonpositive", jf.return_pct.le(0))):
        add("JF", "outcome", group, jf[mask])
    for group, mask in (("never_positive", jf.first_net_positive_date.isna()),
                        ("positive_no_return", jf.first_net_positive_date.notna() & jf.first_return_to_be_date.isna()),
                        ("returned", jf.first_return_to_be_date.notna())):
        add("JF", "be_path", group, jf[mask])
    for value in (False, True):
        add("JF", "trail_covers_be", str(value).lower(), jf[jf.first_trail_covers_be_date.notna().eq(value)])
    for symbol in sorted(trades.symbol.unique()):
        add("JF", "symbol", symbol, jf[jf.symbol.eq(symbol)])
    result = pd.DataFrame(rows)
    return result.astype({col: "string" for col in ("scope", "group_by", "group")})


def run_diagnostic(snapshot: Path, output: Path, *, window: str = "training") -> dict:
    """一次只冻结一个事前指定窗口；默认仅训练，机制审计后再披露其余窗口。"""
    selected = next((list(spec) for spec in WINDOWS if spec[0] == window), None)
    if selected is None:
        raise ValueError("仅允许协议固定窗口")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("诊断目录非空，请使用新的输出目录")
    root = Path(__file__).resolve().parents[2]
    names = ("gcn/backtest/signal_research_r20.py", "gcn/backtest/signal_research_r16.py",
             "gcn/backtest/signal_research_r14.py", "gcn/backtest/historical_research.py",
             "gcn/backtest/signal_audit.py", "gcn/backtest/engine.py", "gcn/recipes/gcn_main.py",
             "gcn/core/tdx.py", "gcn/core/indicators.py")
    sources = {name: (root / name).read_bytes() for name in names}
    protocol_path = root / "reports/gcn-historical-r20-20260905/protocol.md"
    protocol = protocol_path.read_bytes()
    frames, quality = load_snapshot(snapshot)
    start, end = pd.Timestamp(selected[1]), pd.Timestamp(selected[2])
    trade_parts, path_parts, checks = [], [], []
    for symbol in CORE:
        frame = compute_ehopt10(frames[symbol].loc[:end], version="v5", diagnostics=True)
        trades, paths, check = audit_frame(symbol, frame, start, end)
        trade_parts.append(trades)
        path_parts.append(paths)
        checks.append(check)
    trades = pd.concat(trade_parts, ignore_index=True)
    paths = pd.concat(path_parts, ignore_index=True)
    tables = {"trades": trades, "paths": paths, "summary": summarize_trades(trades),
              "reconciliation": pd.DataFrame(checks)}
    decision = {"research_version": "gcn-historical-r20", "stage": "diagnostic_only",
                "recommended": "v5", "production_changed": False, "core": CORE, "window": selected,
                "trades": "original v5, next OPEN, trail 0.20, cost 0.001 each side",
                "window_policy": "independent flat start, terminal liquidation; overlapping windows are not independent",
                "risk": "pure JF only; entry OPEN minus signal-day LLV(LOW,3); nonpositive/missing R not divided",
                "break_even": "entry OPEN / 0.999^2; reference price, not an assumed fill",
                "paths": "causal held CLOSE fields only; real OPEN exit day CLOSE excluded",
                "post_return": "strictly after first return to break-even, before original OPEN exit; post-hoc labels",
                "mfe_mae_r": "nonnegative distances from entry to running held CLOSE peak/trough, divided by valid R",
                "giveback_pp": "r16 gross peak gain minus actual exit gross gain; not portfolio performance",
                "observation_state": "closed or terminal at cutoff: open/pending_signal/pending_trail"}
    for name, raw in sources.items():
        if (root / name).read_bytes() != raw:
            raise ValueError(f"计算期间源码变化：{name}")
    if protocol_path.read_bytes() != protocol:
        raise ValueError("计算期间协议变化")
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.insert(0, "window", window)
        table.to_csv(output / (name + ".csv"), index=False)
    (output / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    (output / "protocol.md").write_bytes(protocol)
    for name, raw in sources.items():
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    manifest = {"research_version": "gcn-historical-r20", "window": selected,
                "parent_manifest_sha256": SNAPSHOT_SHA, "source_quality": quality,
                "protocol_sha256": hashlib.sha256(protocol).hexdigest(),
                "algorithm_sources": {name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()},
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "outputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(output.iterdir()) if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return decision


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reports/signal-audit-v5-review-20260904"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", choices=[spec[0] for spec in WINDOWS], default="training")
    args = parser.parse_args()
    print(json.dumps(run_diagnostic(args.snapshot, args.output, window=args.window), indent=2, ensure_ascii=False))
