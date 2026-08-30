# -*- coding: utf-8 -*-
"""回测引擎不变量测试。"""
import numpy as np

from gcn.backtest.engine import PRESETS, slice_years, run_backtest
from gcn.data.sample import make_sample_data
from gcn.recipes.gcn_main import compute_ehopt10


def _res():
    return compute_ehopt10(make_sample_data(600))


def test_slice_years():
    import pandas as pd
    res = _res()
    s1 = slice_years(res, 1)
    assert s1.index[0] >= res.index[-1] - pd.DateOffset(years=1)
    assert len(s1) < len(res)
    assert slice_years(res, None) is res


def test_run_backtest_shape_and_consistency():
    res = _res()
    rep = run_backtest(res, cost=0.001)
    assert set(rep) >= {"events", "strategies", "equity"}
    assert len(rep["equity"][list(rep["equity"])[0]]) == len(res)
    for s in rep["strategies"]:
        assert set(s) >= {"name", "total", "cagr", "mdd", "sharpe", "trades", "exposure"}
    bh = next(s for s in rep["strategies"] if s["name"].startswith("基准"))
    assert bh["trades"] == 0 and bh["exposure"] == 1.0
    # 事件研究结构
    ev = rep["events"]
    assert any(e["signal"] == "B_SIGNAL" for e in ev)
    assert any(e["signal"] == "_BASE" for e in ev)


def test_preset_columns_missing_skips():
    res = _res()
    presets = [{"name": "不存在列", "entry": ["NOPE_COL"], "exit": ["B_SIGNAL"]},
               {"name": "B买 → S卖", "entry": ["B_SIGNAL"], "exit": ["S_SIGNAL"]}]
    rep = run_backtest(res, cost=0.001, presets=presets)
    assert [s["name"] for s in rep["strategies"]] == ["B买 → S卖", "基准: 买入持有"]
