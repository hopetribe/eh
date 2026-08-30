# -*- coding: utf-8 -*-
"""选股 CLI: python3 -m gcn.screener --strategy graham --market us"""
from __future__ import annotations

import argparse

from gcn.screener.engine import run_screen
from gcn.screener.strategies import SAMPLE_UNIVERSE, STRATEGIES


def main():
    ap = argparse.ArgumentParser(description="GCN 基本面选股")
    ap.add_argument("--strategy", default="graham", choices=list(STRATEGIES))
    ap.add_argument("--market", default="us", choices=["us", "hk", "cn"],
                    help="目标市场 (决定演示候选池)")
    ap.add_argument("--symbols", default=None, help="逗号分隔候选池 (覆盖 --market)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else SAMPLE_UNIVERSE[args.market])
    print(f"策略: {STRATEGIES[args.strategy]['name']}  候选 {len(symbols)} 只\n")
    results = run_screen(symbols, args.strategy, verbose=args.verbose)
    passed = [r for r in results if r["passed"]]
    print(f"\n通过: {len(passed)}/{len(results)}")
    for r in passed:
        print(f"  ✓ {r['symbol']}  {r.get('name') or ''}  市值 {r.get('market_cap_cny', 0) / 1e8:.0f}亿")


if __name__ == "__main__":
    main()
