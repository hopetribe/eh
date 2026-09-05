"""历史r4：新增P入场的期限与MID失效退出，默认v5不变。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gcn.backtest.signal_research_r3 import candidate_signals as baseline_signals

CHALLENGERS = ("P-stop5-hold10", "P-stop5-hold20", "P-stop5-hold40", "P-stop5-mid2")
RULES = ("v5", "P-stop5") + CHALLENGERS
HOLD_DAYS = dict(zip(CHALLENGERS[:3], (10, 20, 40)))


def candidate_signals(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    baseline = baseline_signals(frame)
    below_mid = frame["CLOSE"] < frame["MID"]
    mid_exit = below_mid & below_mid.shift(1, fill_value=False)
    result = {}
    for rule in RULES:
        signals = baseline["v5" if rule == "v5" else "P-stop5"].copy()
        additional = signals["ENTRY_STOP"].notna()
        signals["ENTRY_LIMIT"] = np.nan
        signals["USE_EXTRA"] = False
        signals["EXTRA_EXIT"] = mid_exit
        if rule in HOLD_DAYS:
            signals.loc[additional, "ENTRY_LIMIT"] = HOLD_DAYS[rule]
        elif rule == "P-stop5-mid2":
            signals["USE_EXTRA"] = additional
        result[rule] = signals
    return result
