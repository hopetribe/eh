# -*- coding: utf-8 -*-
"""兼容入口: GCN 回测引擎 (实现已迁移至 gcn/backtest/)。

历史入口保持可用; 新代码请直接使用 gcn.backtest:
    from gcn.backtest import run_backtest, event_study
"""
from gcn.backtest.cli import *  # noqa: F401,F403
from gcn.backtest.engine import *  # noqa: F401,F403
from gcn.backtest.engine import DEFAULT_SYMBOLS, HORIZONS, PRESETS, SIGNAL_LABELS, TRADING_DAYS  # noqa: F401

if __name__ == "__main__":
    from gcn.backtest.cli import main
    main()
