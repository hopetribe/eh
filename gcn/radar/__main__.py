# -*- coding: utf-8 -*-
"""机会雷达 CLI: 市值阈值股票池日K预热 / 每日扫描邮件守护。

  python3 -m gcn.radar --market all          # 立即预热 (只刷陈旧/缺失)
  python3 -m gcn.radar --market us --force   # 强制全量刷新单市场
  python3 -m gcn.radar --loop                # 常驻守护 (09:00 扫描发信)
"""
from __future__ import annotations

import argparse
import threading

from gcn.radar.engine import MARKETS, warm_loop, warm_market
from gcn.radar.scheduler import daily_radar_loop


def main():
    ap = argparse.ArgumentParser(description="机会雷达 日K缓存预热/守护")
    ap.add_argument("--market", default="all", choices=MARKETS + ["all"],
                    help="目标市场 (默认 all = 美股/港股/A股)")
    ap.add_argument("--force", action="store_true", help="忽略新鲜度全部刷新")
    ap.add_argument("--loop", action="store_true",
                    help="常驻守护模式 (缓存巡检 + 每天09:00扫描发信)")
    args = ap.parse_args()

    markets = MARKETS if args.market == "all" else [args.market]
    if args.loop:
        threading.Thread(target=warm_loop, args=(markets,), daemon=True).start()
        daily_radar_loop(markets=markets)
        return
    for m in markets:
        warm_market(m, force=args.force, log=True)


if __name__ == "__main__":
    main()
