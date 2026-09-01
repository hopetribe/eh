# -*- coding: utf-8 -*-
"""选股引擎: 结构化条件求值 + 市值硬性过滤 + 结果汇总。"""
from __future__ import annotations

import numpy as np
import time

from gcn.data.service import to_yahoo_symbol
from gcn.screener import fundamentals
from gcn.screener.strategies import FX_TO_CNY, FX_TO_CNY as _FX, get_strategy

GLOBAL_MIN_MKTCAP_CNY = 50e8  # 任务4: 所有选股市值下限 50 亿元


def _cmp(value, op, threshold):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None  # 数据缺失
    if op == "<":
        return value < threshold
    if op == ">":
        return value > threshold
    if op == "between":
        lo, hi = threshold
        return lo <= value <= hi
    if op == "yearly_gt":  # 逐年类由调用方预处理
        return value
    return None


def _eval_condition(cond: dict, m: dict) -> dict:
    """求值单条条件, 返回 {text, value, threshold, passed, note}。"""
    field, op, val = cond["field"], cond["op"], cond["value"]
    note = ""
    if op in ("yearly_gt", "yearly_lt"):
        need = cond.get("need", 3)
        series = (m.get(field) or [])[:need]  # "近N年"只取最近 N 年
        covered = len(series)
        if covered == 0:
            return {"text": cond["text"], "value": None, "threshold": val,
                    "passed": False, "note": f"无数据"}
        compare = (lambda v: v > val) if op == "yearly_gt" else (lambda v: v < val)
        ok_years = sum(1 for v in series if compare(v))
        # 数据不足: 按可得年份全部满足评估, 但标注覆盖不足
        passed = (ok_years == covered) and covered >= min(need, covered)
        if covered < need:
            note = f"数据不足 (覆盖{covered}/{need}年)"
        return {"text": cond["text"], "value": f"{ok_years}/{covered}年达标",
                "threshold": f"每年>{val:.2%}" if op == "yearly_gt" else f"每年<{val:.2%}",
                "passed": passed, "note": note}

    v = m.get(field)
    if op == "between":
        lo, hi = val
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return {"text": cond["text"], "value": None, "threshold": f"{lo}~{hi}",
                    "passed": False, "note": "数据缺失"}
        return {"text": cond["text"], "value": f"{v:.2%}",
                "threshold": f"{lo:.0%}~{hi:.0%}",
                "passed": bool(lo <= v <= hi), "note": ""}
    passed = _cmp(v, op, val)
    if passed is None:
        return {"text": cond["text"], "value": None, "threshold": val,
                "passed": False, "note": "数据缺失"}
    fmt = lambda x: f"{x:.2%}" if isinstance(val, float) and abs(val) <= 5 else f"{x:,.2f}"
    return {"text": cond["text"], "value": fmt(v), "threshold": fmt(val),
            "passed": bool(passed), "note": ""}


def evaluate_symbol(symbol: str, strategy_id: str, count: int = 1300) -> dict:
    """对单个标的执行策略评估 (含全局市值过滤)。"""
    strat = get_strategy(strategy_id)
    ysym = to_yahoo_symbol(symbol)
    try:
        m = fundamentals.compute_metrics(ysym, count=count)
    except Exception as e:  # noqa: BLE001
        n_total = sum(1 for c in strat["conditions"]
                      if c.get("field") != "market_cap_cny") + 1
        return {"symbol": symbol, "yahoo": ysym, "name": symbol,
                "market_cap_cny": None, "passed": False,
                "n_ok": 0, "n_total": n_total,
                "note": f"基本面获取失败: {e}", "conditions": []}

    fx = FX_TO_CNY.get(str(m.get("currency_code") or m.get("currency") or "USD").upper(), 1.0)
    mc = m.get("market_cap")
    mc_cny = mc * fx if mc else None
    m["market_cap_cny"] = mc_cny

    # 全局市值过滤
    floor = max(GLOBAL_MIN_MKTCAP_CNY, strat.get("min_mktcap_cny", 0))
    mcap_pass = mc_cny is not None and mc_cny > floor
    mcap_note = "" if mc_cny else "市值数据缺失"

    conds = [_eval_condition(c, m) for c in strat["conditions"]
             if c.get("field") != "market_cap_cny"]
    conds.append({"text": f"总市值 > {floor / 1e8:.0f} 亿元", "value": mc_cny,
                  "threshold": floor, "passed": mcap_pass, "note": mcap_note})
    all_pass = all(c["passed"] for c in conds)
    n_ok = sum(1 for c in conds if c["passed"])
    return {"symbol": symbol, "yahoo": ysym, "name": m.get("name"),
            "market_cap_cny": mc_cny, "passed": bool(all_pass and mcap_pass),
            "n_ok": n_ok, "n_total": len(conds),
            "conditions": conds,
            "note": "" if m.get("market_cap") else "无基本面数据 (ETF/未覆盖)"}


def run_screen(symbols: list[str], strategy_id: str, count: int = 1300,
               verbose: bool = False, log: bool = True) -> list[dict]:
    """对候选池逐个评估并按通过数排序。"""
    out = []
    for sym in symbols:
        r = evaluate_symbol(sym, strategy_id, count=count)
        time.sleep(1.0)  # 数据源限流保护
        out.append(r)
        if log:
            flag = "✓ PASS" if r["passed"] else f"✗ {r.get('n_ok', 0)}/{r.get('n_total', 0)}"
            print(f"  {sym:<10} {flag:<8} {r.get('note') or ''}")
        if verbose:
            for c in r["conditions"]:
                mark = "✓" if c["passed"] else "✗"
                val = c["value"] if c["value"] is not None else "--"
                print(f"      [{mark}] {c['text']}  = {val}  (阈值 {c['threshold']}) {c['note']}")
    out.sort(key=lambda r: (not r["passed"], -r.get("n_ok", 0)))
    return out
