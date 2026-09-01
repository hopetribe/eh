# -*- coding: utf-8 -*-
"""EHOPT10 配方输入与因果性回归测试。"""
import numpy as np
import pandas as pd

from gcn.recipes.gcn_main import compute_ehopt10


def _rising(n):
    close = np.arange(10.0, 10.0 + n)
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": np.full(n, 1000.0),
    }, index=pd.date_range("2025-01-01", periods=n, freq="D"))


def test_recipe_rejects_unknown_version_and_nonpositive_n():
    df = _rising(30)
    for kwargs in ({"version": "v5"}, {"N": 0}, {"N": -1}):
        try:
            compute_ehopt10(df, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝参数: {kwargs}")


def test_nine_turn_labels_do_not_depend_on_future_rows():
    short = compute_ehopt10(_rising(4), N=1)
    long = compute_ehopt10(_rising(12), N=1).iloc[:4]
    assert short["NINE2_UP_LABEL"].tolist() == [0.0, 1.0, 2.0, 3.0]
    for col in ("NINE2_UP_LABEL", "NINE2_DOWN_LABEL"):
        assert np.array_equal(short[col].to_numpy(), long[col].to_numpy())
