# chanlun.py attribution

KK2 uses the analysis core from the pinned `vendor/chanlun.py` submodule:

- Upstream: https://github.com/YuYuKunKun/chanlun.py
- Revision: `2e4fa135b19eaa201fca7bfcc8ca4a86cbde7815`
- Upstream license: MIT; see `vendor/chanlun.py/LICENSE.md`.

The upstream `chan.py` also includes adapted `czsc` signal-framework code under
Apache License 2.0. Its attribution and complete license are retained in
`vendor/chanlun.py/NOTICE` and `vendor/chanlun.py/LICENSES/`.

KK2 does not bundle or serve the upstream FastAPI app, Backtrader workflow, or
TradingView charting-library assets. It feeds KK2's validated OHLCV data to the
pinned analysis core and renders the returned structures with KK2's ECharts UI.
