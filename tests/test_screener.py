# -*- coding: utf-8 -*-
"""选股模块离线测试: 条件求值 / 市值过滤 / 策略定义完整性 (无网络)。"""
from gcn.screener.engine import (_eval_condition, GLOBAL_MIN_MKTCAP_CNY,
                                 evaluate_symbol)
from gcn.screener.strategies import STRATEGIES, get_strategy


def test_strategy_definitions_complete():
    # 八套策略 (大师四套 + 扩展四套) 均已结构化定义且字段齐备
    assert set(STRATEGIES) == {"graham", "growth", "schloss", "buffett",
                               "neff", "lynch", "fisher", "davis"}
    for sid, s in STRATEGIES.items():
        assert s["conditions"] and s["name"] and s["theme"]
        assert s["min_mktcap_cny"] >= GLOBAL_MIN_MKTCAP_CNY
        for c in s["conditions"]:
            assert c["text"] and c["field"] and c["op"] in ("<", ">", "between",
                                                            "yearly_gt", "yearly_lt")


def test_condition_eval_simple_ops():
    m = {"pb_mrq": 0.8, "graham_pe_pb": 30.0, "nan_field": float("nan")}
    assert _eval_condition({"text": "t", "field": "pb_mrq", "op": "<", "value": 1}, m)["passed"]
    assert not _eval_condition({"text": "t", "field": "graham_pe_pb", "op": "<", "value": 22.5}, m)["passed"]
    r = _eval_condition({"text": "t", "field": "nan_field", "op": "<", "value": 1}, m)
    assert r["passed"] is False and r["note"] == "数据缺失"


def test_condition_eval_yearly_window():
    m = {"roe_yearly": [0.2, 0.15, -0.05, 0.3]}  # 按年份降序
    cond = {"text": "t", "field": "roe_yearly", "op": "yearly_gt", "value": 0.10, "need": 3}
    # "近3年"只取最近3年: [0.2, 0.15, -0.05] -> 不通过
    r = _eval_condition(cond, m)
    assert r["passed"] is False and "3年" in r["value"]
    # 全部满足时通过
    m2 = {"roe_yearly": [0.2, 0.15, 0.12]}
    assert _eval_condition(cond, m2)["passed"]
    # 数据不足: 覆盖 < 需求年数仍按可得年份评估但标注
    m3 = {"roe_yearly": [0.2, 0.15]}
    r3 = _eval_condition(cond, m3)
    assert r3["passed"] and "数据不足" in r3["note"]


def test_condition_eval_yearly_lt_uses_less_than():
    cond = {"text": "t", "field": "debt", "op": "yearly_lt", "value": 0.2, "need": 2}
    assert _eval_condition(cond, {"debt": [0.10, 0.15]})["passed"]
    assert not _eval_condition(cond, {"debt": [0.10, 0.25]})["passed"]


def test_global_mktcap_floor():
    assert GLOBAL_MIN_MKTCAP_CNY == 50e8
    assert STRATEGIES["growth"]["min_mktcap_cny"] == 100e8


def test_all_strategies_missing_values_are_json_safe_and_not_passed():
    import json
    import numpy as np
    from unittest.mock import patch
    from gcn.screener import engine
    for strategy_id, strategy in STRATEGIES.items():
        for missing in (None, float("nan"), float("inf"), -float("inf"), np.float32("nan")):
            metrics = {"market_cap": missing, "name": missing}
            for condition in strategy["conditions"]:
                metrics[condition["field"]] = ([missing] * condition.get("need", 3)
                    if condition["op"].startswith("yearly_") else missing)
            with patch.object(engine.fundamentals, "compute_metrics", return_value=metrics):
                result = evaluate_symbol("AAPL", strategy_id)
            json.dumps(result, allow_nan=False)
            assert result["market_cap_cny"] is None
            assert result["name"] == "AAPL"
            assert result["passed"] is False and result["n_ok"] == 0
            assert result["data_status"] == "incomplete"
            assert "数据不足" in result["note"]
            assert all(not condition["passed"] for condition in result["conditions"])


def test_neff_market_cap_condition_receives_converted_value():
    import gcn.screener.engine as engine
    old = engine.fundamentals.compute_metrics
    metrics = {"currency": "USD", "market_cap": 2e9, "name": "Apple",
               "trr": 2, "trailing_pe": 10, "eps_growth": 0.1,
               "static_div_yield": 0.04, "roe_avg_3y": 0.2,
               "fcf_to_net_income": 1, "div_yield_yearly": [0.02] * 5}
    try:
        engine.fundamentals.compute_metrics = lambda *a, **k: metrics.copy()
        result = evaluate_symbol("AAPL", "neff")
    finally:
        engine.fundamentals.compute_metrics = old
    caps = [c for c in result["conditions"] if c["text"] == "总市值 > 100 亿元"]
    assert len(caps) == 1
    cap = caps[0]
    assert cap["passed"] and cap["note"] == ""


def test_evaluate_symbol_failure_has_stable_result_shape():
    import gcn.screener.engine as engine
    old = engine.fundamentals.compute_metrics
    try:
        engine.fundamentals.compute_metrics = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("offline"))
        result = evaluate_symbol("AAPL", "graham")
    finally:
        engine.fundamentals.compute_metrics = old
    assert result["name"] == "AAPL" and result["market_cap_cny"] is None
    assert result["n_ok"] == 0 and result["n_total"] == 8


def test_gross_margin_history_uses_gross_profit_not_cost_of_sales():
    import pandas as pd
    from gcn.screener.fundamentals import _gross_margin_history
    years = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31"])
    revenue = pd.Series([100, 90, 80], index=years)
    gross_profit = pd.Series([60, 50, 40], index=years)
    assert _gross_margin_history(revenue, gross_profit, 3) == [0.6, 50 / 90, 0.5]


def test_dividend_yields_exclude_incomplete_current_year():
    import pandas as pd
    from gcn.screener.fundamentals import _completed_dividend_yields
    idx = pd.to_datetime(["2024-12-31", "2025-12-31", "2026-08-28"])
    prices = pd.DataFrame({"close": [100.0, 100.0, 100.0]}, index=idx)
    dividends = pd.Series([2.0, 3.0, 10.0], index=idx)
    assert _completed_dividend_yields(
        dividends, prices, as_of=pd.Timestamp("2026-08-31")) == [0.03, 0.02]


def test_sales_inventory_spread_requires_both_matching_years():
    import math
    import pandas as pd
    from gcn.screener.fundamentals import _sales_inventory_growth_spread
    years = pd.to_datetime(["2025-12-31", "2024-12-31"])
    revenue = pd.Series([120.0, 100.0], index=years)
    missing_latest_inventory = pd.Series([50.0], index=[years[1]])
    assert math.isnan(_sales_inventory_growth_spread(revenue, missing_latest_inventory))
    inventory = pd.Series([55.0, 50.0], index=years)
    assert abs(_sales_inventory_growth_spread(revenue, inventory) - 0.1) < 1e-12


def test_davis_multiplier_uses_same_period_annual_prices():
    import pandas as pd
    from gcn.screener.fundamentals import _davis_double_play
    px = pd.DataFrame({"close": [100.0, 100.0, 120.0]}, index=pd.to_datetime(
        ["2024-12-31", "2025-12-31", "2026-08-31"]))
    eps = pd.Series([4.0, 2.0], index=pd.to_datetime(["2025-12-31", "2024-12-31"]))
    assert _davis_double_play(px, eps) == 1.0


def test_historical_pe_percentile_uses_period_eps_not_latest_eps_constant():
    import pandas as pd
    from gcn.screener.fundamentals import _historical_pe_percentile
    years = pd.to_datetime([f"{y}-12-31" for y in range(2021, 2026)])
    px = pd.DataFrame({"close": [10, 20, 30, 40, 50]}, index=years)
    eps = pd.Series([10, 2, 3, 4, 1], index=years[::-1])  # descending report order
    assert _historical_pe_percentile(px, eps, current_pe=5.0) == 0.0
def _passing_metrics(strategy):
    metrics = {"market_cap": 1e12, "currency": "CNY", "name": "有效样本"}
    for condition in strategy["conditions"]:
        field, op, value = condition["field"], condition["op"], condition["value"]
        if op.startswith("yearly_"):
            metrics[field] = [value + 1 if op == "yearly_gt" else value - 1] * condition.get("need", 3)
        elif op == "between":
            metrics[field] = sum(value) / 2
        else:
            metrics[field] = value + 1 if op == ">" else value / 2
    return metrics


def test_all_strategies_valid_data_and_each_condition_failure():
    import json
    from unittest.mock import patch
    from gcn.screener import engine
    for strategy_id, strategy in STRATEGIES.items():
        metrics = _passing_metrics(strategy)
        with patch.object(engine.fundamentals, "compute_metrics", side_effect=lambda *a, **kw: metrics.copy()):
            result = evaluate_symbol("GOOD", strategy_id)
            json.dumps(result, allow_nan=False)
            assert result["passed"] and result["data_status"] == "complete", strategy_id
            assert result["n_ok"] == result["n_total"]
            for condition in strategy["conditions"]:
                if condition["field"] == "market_cap_cny":
                    continue
                original = metrics[condition["field"]]
                value = condition["value"]
                if condition["op"].startswith("yearly_"):
                    metrics[condition["field"]] = [value] * condition.get("need", 3)
                elif condition["op"] == "between":
                    metrics[condition["field"]] = value[1] + 1
                else:
                    metrics[condition["field"]] = value
                failed = evaluate_symbol("BAD", strategy_id)
                assert not failed["passed"] and failed["n_ok"] == failed["n_total"] - 1, (strategy_id, condition)
                metrics[condition["field"]] = original
            metrics["market_cap"] = strategy["min_mktcap_cny"]
            assert not evaluate_symbol("SMALL", strategy_id)["passed"]  # 严格 > 边界


def test_yearly_invalid_entry_cannot_pass_or_pull_in_older_year():
    cond = {"text": "年限", "field": "roe", "op": "yearly_gt", "value": 0.1, "need": 3}
    result = _eval_condition(cond, {"roe": [0.2, None, 0.3, 0.4]})
    assert result["passed"] is False and "2/3" in result["note"]
    assert result["value"] == "2/2年达标"


def test_all_strategies_isolate_provider_and_evaluation_failures():
    import json
    from types import SimpleNamespace
    from unittest.mock import patch
    from gcn.screener import engine
    for strategy_id, strategy in STRATEGIES.items():
        def compute(symbol, **kwargs):
            if symbol == "OFFLINE":
                raise RuntimeError("模拟数据源故障")
            if symbol == "INVALID":
                return object()  # 获取成功但评估阶段遇到异常也不应中断整批
            if symbol == "MISSING":
                return {"market_cap": float("nan")}
            return _passing_metrics(strategy)
        with patch.object(engine.fundamentals, "compute_metrics", side_effect=compute), \
             patch.object(engine, "time", SimpleNamespace(sleep=lambda seconds: None)):
            results = engine.run_screen(["OFFLINE", "INVALID", "MISSING", "GOOD"], strategy_id, log=False)
        json.dumps(results, allow_nan=False)
        assert len(results) == 4 and results[0]["symbol"] == "GOOD" and results[0]["passed"]
        assert {row["symbol"] for row in results if row["data_status"] == "error"} == {"OFFLINE", "INVALID"}
        assert next(row for row in results if row["symbol"] == "MISSING")["market_cap_cny"] is None
