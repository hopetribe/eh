# -*- coding: utf-8 -*-
"""选股模块离线测试: 条件求值 / 市值过滤 / 策略定义完整性 (无网络)。"""
from gcn.screener.engine import _eval_condition, GLOBAL_MIN_MKTCAP_CNY
from gcn.screener.strategies import STRATEGIES, get_strategy


def test_strategy_definitions_complete():
    # 四套策略均已结构化定义且字段齐备
    assert set(STRATEGIES) == {"graham", "growth", "schloss", "buffett"}
    for sid, s in STRATEGIES.items():
        assert s["conditions"] and s["name"] and s["theme"]
        assert s["min_mktcap_cny"] >= GLOBAL_MIN_MKTCAP_CNY
        for c in s["conditions"]:
            assert c["text"] and c["field"] and c["op"] in ("<", ">", "yearly_gt", "yearly_lt")


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


def test_global_mktcap_floor():
    assert GLOBAL_MIN_MKTCAP_CNY == 50e8
    assert STRATEGIES["growth"]["min_mktcap_cny"] == 100e8
