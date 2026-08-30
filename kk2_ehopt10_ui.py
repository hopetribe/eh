# -*- coding: utf-8 -*-
"""兼容入口: GCN Web 服务与数据服务 (实现已迁移至 gcn/server + gcn/data)。

启动: python3 kk2_ehopt10_ui.py [--port 8642] [--no-browser]
"""
from gcn.data.service import (  # noqa: F401
    DEFAULT_COUNT, df_from_rows, fetch_quote, parse_csv_text,
    to_futu_symbol, to_yahoo_symbol,
)
from gcn.backtest.engine import DEFAULT_SYMBOLS  # noqa: F401
from gcn.server.app import build_payload, main  # noqa: F401

if __name__ == "__main__":
    main()
