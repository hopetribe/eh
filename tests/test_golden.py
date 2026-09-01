# -*- coding: utf-8 -*-
"""黄金基准: 重构后的配方输出必须与重构前逐值一致。"""
import pickle
from pathlib import Path

import numpy as np

from gcn.recipes.gcn_main import compute_ehopt10
from gcn.data.sample import make_sample_data

GOLDEN = Path(__file__).parent / "golden_v4.pkl"

# 九转 1-8 展示曾通过 BACKSET 在第 9 根后回填，含未来信息；现在刻意改为
# 当根因果计数，因此不再与历史截图型黄金值比较。交易信号列仍完整比较。
CAUSAL_DISPLAY_COLUMNS = {"NINE2_UP_LABEL", "NINE2_DOWN_LABEL"}


def test_golden_equivalence():
    with open(GOLDEN, "rb") as f:
        blob = pickle.load(f)
    df = blob["df"]
    for ver in ("v3", "v4"):
        res = compute_ehopt10(df, version=ver)
        g = blob["golden"][ver]
        assert list(g.columns) == list(res.columns)
        for col in g.columns:
            if col in CAUSAL_DISPLAY_COLUMNS:
                continue
            a = g[col].to_numpy(dtype=float)
            b = res[col].to_numpy(dtype=float)
            assert np.allclose(a, b, equal_nan=True), f"黄金基准不一致: {ver}.{col}"


def test_fresh_sample_matches_golden_pipeline():
    with open(GOLDEN, "rb") as f:
        blob = pickle.load(f)
    df = make_sample_data(len(blob["df"]), seed=7)
    assert (df["close"].to_numpy() == blob["df"]["close"].to_numpy()).all()
