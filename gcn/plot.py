# -*- coding: utf-8 -*-
"""GCN 结果可视化 (可选, 需要 matplotlib)。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gcn.core.tdx import _as_bool

def plot_result(res: pd.DataFrame, OFFSET: int = 15, title: str = "KK2 EHOPT10",
                save_path: str | None = None, show: bool = False):
    """将指标画成图: 布林带 + 收盘价 + 全部信号标注。

    OFFSET 与富途参数一致, 作为九转数字距高低点的偏移(pt)。
    """
    import matplotlib
    if not show and save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 中文字体回退 (★买/★卖/绝反 标注), 找不到则维持默认
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for fname in ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS",
                  "Microsoft YaHei", "SimHei"]:
        if fname in available:
            plt.rcParams["font.family"] = fname
            break
    plt.rcParams["axes.unicode_minus"] = False

    x = np.arange(len(res))
    fig, ax = plt.subplots(figsize=(16, 9))

    ax.plot(x, res["CLOSE"], color="black", lw=1.0, label="CLOSE")
    ax.plot(x, res["MID"], color="#FFAEC9", lw=1.0, label="MID")       # COLORFFAEC9
    ax.plot(x, res["UPPER"], color="#FFC90E", lw=1.0, label="UPPER")   # COLORFFC90E
    ax.plot(x, res["LOWER"], color="#0CAEE6", lw=1.0, label="LOWER")   # COLOR0CAEE6

    def _idx(cond):
        return np.asarray(res.index)[_as_bool(cond).to_numpy()]

    # DRAWICON 7/8 (B/S 信号) 与 34 (绝反)
    ax.scatter(x[_as_bool(res["B_SIGNAL"])], res.loc[_as_bool(res["B_SIGNAL"]), "LOW"] * 0.99,
               marker="^", s=90, color="red", zorder=5, label="B_SIGNAL")
    ax.scatter(x[_as_bool(res["S_SIGNAL"])], res.loc[_as_bool(res["S_SIGNAL"]), "S_POSITION"],
               marker="v", s=90, color="green", zorder=5, label="S_SIGNAL")
    ax.scatter(x[_as_bool(res["ICON_JUEFAN"])], res.loc[_as_bool(res["ICON_JUEFAN"]), "LOW"] * 0.985,
               marker="D", s=45, color="orange", zorder=5, label="绝反")

    # DRAWTEXT ★买/★卖
    for cond, price, txt, color in [
        (res["NINE2_BUY_SIGNAL"], res["LOW"] * 0.98, "★买", "yellow"),
        (res["NINE2_SELL_SIGNAL"], res["HIGH"] * 1.02, "★卖", "magenta"),
    ]:
        sel = _as_bool(cond)
        for i, p in zip(x[sel.to_numpy()], price[sel]):
            ax.annotate(txt, (i, p), ha="center", va="bottom", color=color, fontsize=10)

    # 九转数字标注: 上方 1-8 品红 / 9 绿; 下方 1-8 绿 / 9 品红
    for col, price, side, color in [
        ("NINE2_UP_LABEL", res["HIGH"], 1, "#FF00FF"),
        ("NINE2_UP_9", res["HIGH"], 1, "green"),
        ("NINE2_DOWN_LABEL", res["LOW"], -1, "green"),
        ("NINE2_DOWN_9", res["LOW"], -1, "#FF00FF"),
    ]:
        v = res[col]
        is_nine = v.dtype == bool
        sel = v.to_numpy() if is_nine else (v > 0).to_numpy()
        for i, val in zip(x[sel], v[sel]):
            txt = "9" if is_nine else str(int(val))
            ax.annotate(txt, (i, price.iloc[i]), xytext=(0, OFFSET * side),
                        textcoords="offset points", ha="center", color=color, fontsize=8)

    ax.set_title(title)
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return fig
