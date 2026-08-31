# -*- coding: utf-8 -*-
"""机会雷达 CLI: 三大市场市值前100标的日K缓存预热 / 每日增量守护。

  python3 -m gcn.radar --market all          # 立即预热 (只刷陈旧/缺失)
  python3 -m gcn.radar --market us --force   # 强制全量刷新单市场
  python3 -m gcn.radar --loop                # 常驻守护 (每小时巡检, 每日增量)
"""
from __future__ import annotations

import argparse

from gcn.radar.engine import MARKETS, warm_loop, warm_market


def main():
    ap = argparse.ArgumentParser(description="机会雷达 日K缓存预热/守护")
    ap.add_argument("--market", default="all", choices=MARKETS + ["all"],
                    help="目标市场 (默认 all = 美股/港股/A股)")
    ap.add_argument("--force", action="store_true", help="忽略新鲜度全部刷新")
    ap.add_argument("--loop", action="store_true",
                    help="常驻守护模式 (每小时巡检, 陈旧才增量请求)")
    args = ap.parse_args()

    markets = MARKETS if args.market == "all" else [args.market]
    if args.loop:
        warm_loop(markets)
        return
    for m in markets:
        warm_market(m, force=args.force, log=True)


if __name__ == "__main__":
    main()
